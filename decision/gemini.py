"""Gemini backend: structured output for decisions a generative model handles well.

Why Gemini is here at all
-------------------------
Clef is a decision model. It is fast, local and calibrated on closed questions,
which is most of what Paytriq asks. It is not a reader: for "given this sponsor
email, is this acceptance, hesitation or a price objection, and why?" a
generative model with a JSON schema is the right instrument, because the
question needs reasoning over text rather than a probability over a fixed set.

So the split is by capability, not by preference:

* **Clef** -> closed questions with a typed answer space and a calibrated
  distribution. Never used for free-text judgement.
* **Gemini** -> intent classification with reasoning, and any judgement whose
  answer space is naturally open. Always with ``response_mime_type =
  "application/json"`` and an explicit schema, so the output is parseable
  without regex.

The honesty rule that shapes this whole file
-------------------------------------------
A generative model returns a *label*, not a distribution. Two ways to bridge
that gap:

* ask for a distribution in the schema and use it, or
* take the single label, give it ``1.0`` and split the remainder evenly.

The second is a **fabrication**. It looks like a calibration, it flows through
every downstream consumer unchanged, and it would let an uncalibrated number
outrank a genuinely calibrated one. So when it happens, ``Decision.degraded`` is
set to ``True`` and ``raw["_distribution"]`` records that the distribution was
synthesised rather than asked for. The decision is still returned -- refusing
would stall the pipeline for a question the model did answer -- but it is
labelled for what it is, and ``Decision.needs_escalation`` is therefore True.

Model fallback
--------------
``settings.gemini_model`` first; ``settings.gemini_fallback_model`` on a client
side error (bad model id, quota for that model, unsupported config). Connection
and server errors are *not* retried against a different model -- the second
model would fail the same way -- they raise ``DecisionUnavailable`` so the
registry falls back to rules instead of burning time on a third attempt.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from core.config import Settings, get_settings
from core.errors import DecisionFailed, DecisionUnavailable
from core.schemas import Decision, DecisionRequest, DecisionSource, QuestionType

from .base import (
    NOUL_LABELS,
    annotate_raw,
    logger,
    make_decision,
    normalise_probabilities,
    snippet,
)

__all__ = [
    "GeminiBackend",
    "build_schema",
    "build_prompt",
    "synthesise_distribution",
    "parse_response",
]

_SYSTEM_INSTRUCTION = (
    "You are the decision layer of an event-sponsorship agent. You answer one "
    "closed question at a time. Read the state, decide, and return only JSON "
    "matching the given schema. Never invent an option that was not offered. "
    "Reason briefly in the 'reasoning' field: it is read by a trace, not by a "
    "human, so keep it to one sentence of evidence."
)

#: Why a uniform tail is labelled degraded, in one sentence, for the trace.
_SYNTHESIS_NOTE = (
    "gemini returned a label but no distribution; mass 1.0 was assigned to the "
    "label and the remainder split evenly across the other options. This is a "
    "synthesised distribution, not a calibrated one."
)


# ==============================================================================
# schema + prompt
# ==============================================================================
def build_schema(request: DecisionRequest) -> dict[str, Any]:
    """JSON schema constraining Gemini's answer to this question's answer space.

    ``probabilities`` is requested but *not* required. Gemini frequently declines
    to invent numbers for a closed set, which is the correct behaviour; when it
    does supply them they are a genuine self-assessment and are used as-is.
    """
    if request.question_type is QuestionType.CHOICE:
        labels = [str(o) for o in request.options]
        answer_field = "choice"
    elif request.question_type is QuestionType.SCORE:
        labels = [str(level) for level in request.rubric]
        answer_field = "level"
    elif request.question_type is QuestionType.NOUL:
        labels = list(NOUL_LABELS)
        answer_field = "yes"
    else:  # pragma: no cover - QuestionType is a closed enum
        raise DecisionFailed(f"gemini: unsupported question type {request.question_type!r}")

    return {
        "type": "object",
        "properties": {
            answer_field: (
                {"type": "boolean", "description": "true iff the state satisfies the question"}
                if request.question_type is QuestionType.NOUL
                else {"type": "string", "enum": labels,
                      "description": "exactly one of the offered labels"}
            ),
            "reasoning": {
                "type": "string",
                "description": "one sentence naming the evidence used",
            },
            "probabilities": {
                "type": "object",
                "description": (
                    "optional self-assessed probability for every label; the values "
                    "should sum to 1. Omit if you cannot justify numbers."
                ),
                "properties": {label: {"type": "number"} for label in labels},
                "additionalProperties": {"type": "number"},
            },
        },
        "required": [answer_field, "reasoning"],
    }


def build_prompt(request: DecisionRequest, schema: Mapping[str, Any]) -> str:
    """The user turn: state, question, closed answer space, schema.

    The answer space is repeated in prose as well as in the schema. Schema
    validation is enforced by the API but *not* by the model's comprehension --
    spelling the options out in the instruction is what stops it answering
    ``"willing_to_negotiate"`` when the schema said ``"pushback"``.

    Tool-directed selection: when ``state`` carries ``tool_schemas`` (function
    definitions built by ``agents/tool_selection.py:describe_tools_for_model``)
    with ``available_tools``/``candidate_count``/``purpose``, the full state is
    already serialised below as JSON *and* the candidate tools are spelled out
    explicitly with their schemas, so the model chooses among real tools rather
    than inventing one.
    """
    lines = [
        "STATE (JSON):",
        json.dumps(request.state, sort_keys=True, ensure_ascii=False, default=str),
        "",
        f"QUESTION: {request.question}",
    ]
    if request.instructions.strip():
        lines.append(f"INSTRUCTIONS: {request.instructions.strip()}")
    if request.question_type is QuestionType.CHOICE:
        lines.append("You must choose exactly one of: "
                     + ", ".join(f'"{o}"' for o in request.options))
    elif request.question_type is QuestionType.SCORE:
        lines.append("You must pick exactly one level of this rubric, in order: "
                     + " < ".join(str(level) for level in request.rubric))
    else:
        lines.append("Answer yes or no.")
    # Explicit tool_schemas/tools forwarding for model-directed tool selection.
    # state_text-equivalent JSON above already carries them; this section names
    # the candidate tools and their function schemas in prose so the generative
    # model cannot miss them.
    tool_schemas = request.state.get("tool_schemas")
    available_tools = request.state.get("available_tools")
    if isinstance(tool_schemas, list) and tool_schemas:
        lines.append("")
        lines.append("AVAILABLE TOOLS (function schemas, JSON):")
        lines.append(json.dumps(tool_schemas, sort_keys=True, ensure_ascii=False,
                                default=str))
        if isinstance(available_tools, list) and available_tools:
            lines.append("Choose exactly one of these tools: "
                         + ", ".join(f'"{t}"' for t in available_tools))
    if request.decision_point:
        lines.append(f"This decision steers the run at: {request.decision_point}")
    lines += ["", "JSON SCHEMA:", json.dumps(schema, sort_keys=True)]
    return "\n".join(lines)


# ==============================================================================
# response handling
# ==============================================================================
def synthesise_distribution(choice: str, options: Sequence[str]) -> dict[str, float]:
    """``choice`` -> 1.0, remainder split evenly. **Not a calibration.**

    Used only when the model returned a label and no distribution. Every caller
    must mark the resulting ``Decision`` degraded; the honest description is a
    tie among the options it did not pick.
    """
    labels = [str(o) for o in options]
    if choice not in labels:
        raise DecisionFailed(
            f"gemini: answered {choice!r}, which is not one of {labels}"
        )
    if len(labels) == 1:
        return {choice: 1.0}
    remainder = (1.0 - 1.0) / (len(labels) - 1)
    return normalise_probabilities({label: (1.0 if label == choice else remainder)
                                    for label in labels})


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise DecisionFailed(f"gemini: expected a JSON object, got {type(value).__name__}")


def parse_response(text: str, request: DecisionRequest,
                   ) -> tuple[str, dict[str, float], float, str, bool]:
    """Read Gemini's JSON answer into ``(choice, probs, confidence, reasoning, synth)``.

    ``synthesised`` is True when the probabilities had to be manufactured
    because the model returned only a label -- the caller must then mark the
    ``Decision`` degraded.
    """
    body = (text or "").strip()
    if not body:
        raise DecisionFailed("gemini: empty response body")
    # Structured output promises bare JSON, but a code fence is cheap to strip
    # and its absence cannot be relied on across model versions.
    if body.startswith("```"):
        lines = [ln for ln in body.splitlines() if not ln.strip().startswith("```")]
        body = "\n".join(lines).strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DecisionFailed(
            f"gemini: response was not JSON ({exc}); body starts: {snippet(body)!r}"
        ) from exc
    data = _as_mapping(payload)

    options = _options_for(request)
    if request.question_type is QuestionType.NOUL:
        if "yes" in data:
            yes = data["yes"]
            if not isinstance(yes, bool):
                raise DecisionFailed(f"gemini: 'yes' was {type(yes).__name__}, expected bool")
            choice = NOUL_LABELS[0] if yes else NOUL_LABELS[1]
        else:
            raise DecisionFailed(f"gemini: noul answer has no 'yes' field; keys={sorted(data)}")
    else:
        field_name = "choice" if request.question_type is QuestionType.CHOICE else "level"
        label = data.get(field_name)
        if not isinstance(label, str) or not label.strip():
            raise DecisionFailed(
                f"gemini: answer has no usable {field_name!r}; keys={sorted(data)}"
            )
        choice = label.strip()
        if choice not in options:
            raise DecisionFailed(
                f"gemini: answered {choice!r}, which is not one of the offered {options}"
            )

    reasoning = data.get("reasoning")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""

    supplied = data.get("probabilities")
    if isinstance(supplied, Mapping) and supplied:
        usable = {str(k): v for k, v in supplied.items() if str(k) in options}
        if not usable:
            raise DecisionFailed(
                f"gemini: probabilities {sorted(str(k) for k in supplied)} share no "
                f"key with the offered {options}"
            )
        probabilities = normalise_probabilities(usable)
        degraded_distribution = False
    else:
        probabilities = synthesise_distribution(choice, options)
        degraded_distribution = True

    confidence = probabilities[choice]
    return choice, probabilities, confidence, reasoning, degraded_distribution


def _options_for(request: DecisionRequest) -> list[str]:
    if request.question_type is QuestionType.CHOICE:
        return [str(o) for o in request.options]
    if request.question_type is QuestionType.SCORE:
        return [str(level) for level in request.rubric]
    return list(NOUL_LABELS)


# ==============================================================================
# backend
# ==============================================================================
class GeminiBackend:
    """``DecisionBackend`` over ``google-genai`` structured output.

    The SDK is imported lazily and its exception classes are resolved once, at
    construction time, into a tuple. Two reasons: importing
    ``google.genai`` at module import would make the whole ``decision`` package
    pay for an optional dependency, and ``except SomeSdkError`` needs a real
    class rather than a bare ``Exception`` -- an unhandled SDK error escaping
    here would break the registry's fallback chain, which is the one mechanism
    this system relies on.
    """

    name = "gemini"

    def __init__(self, settings: Settings | None = None, *,
                 client_factory: Callable[[str], Any] | None = None) -> None:
        self.settings = settings or get_settings()
        self._client_factory = client_factory
        self._client: Any = None
        self._client_model = ""
        self._import_error = ""
        self._sdk_errors: tuple[type[BaseException], ...] = ()
        self._types: Any = None
        self._last_error = ""
        self._last_latency_ms = 0.0
        self._calls = 0
        self._fallbacks = 0

    # -------------------------------------------------------------------- sdk
    def _sdk(self) -> tuple[Any, Any, tuple[type[BaseException], ...]]:
        """``(client, types, sdk_errors)``, or ``DecisionUnavailable`` if absent."""
        if self._client is not None:
            return self._client, self._types, self._sdk_errors
        if not self.settings.gemini_api_key:
            raise DecisionUnavailable(
                "gemini: GEMINI_API_KEY is not set; no key means no model, and a "
                "run without a key degrades to rules rather than pretending"
            )
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:
            self._import_error = f"{type(exc).__name__}: {exc}"
            raise DecisionUnavailable(
                f"gemini: the 'google-genai' package is not installed ({exc}); "
                "pip install google-genai"
            ) from exc

        errors: list[type[BaseException]] = []
        sdk_errors = getattr(genai, "errors", None)
        for attr in ("APIError", "APIConnectionError", "ClientError",
                     "ServerError", "UnknownApiResponseError"):
            candidate = getattr(sdk_errors, attr, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                errors.append(candidate)
        self._sdk_errors = tuple(dict.fromkeys(errors))
        self._types = types

        if self._client_factory is not None:
            self._client = self._client_factory(str(self.settings.gemini_api_key))
        else:
            try:
                self._client = genai.Client(
                    api_key=str(self.settings.gemini_api_key),
                    http_options=types.HttpOptions(
                        timeout=int(self.settings.gemini_timeout_s * 1000)
                    ),
                )
            except ValueError as exc:
                raise DecisionUnavailable(f"gemini: could not build a client ({exc})") from exc
        self._client_model = self.model
        return self._client, self._types, self._sdk_errors

    @property
    def model(self) -> str:
        return self._client_model or self.settings.gemini_model

    @property
    def fallback_model(self) -> str:
        return self.settings.gemini_fallback_model

    # ---------------------------------------------------------------- protocol
    def available(self) -> tuple[bool, str]:
        """A key plus an importable SDK is the whole availability question.

        Deliberately does not spend a token on a probe call: Gemini bills per
        token, and every token spent confirming the key works is a token not
        spent on the run. Unreachable-ness is discovered at call time and
        surfaces as ``DecisionUnavailable``, which is what it is.
        """
        if not self.settings.gemini_api_key:
            return False, "GEMINI_API_KEY is not set"
        try:
            self._sdk()
        except DecisionUnavailable as exc:
            return False, str(exc)
        return True, f"gemini-api key present; model {self.settings.gemini_model}"

    def decide(self, request: DecisionRequest) -> Decision:
        """One structured-output call per ``DecisionRequest``."""
        client, types, sdk_errors = self._sdk()
        schema = build_schema(request)
        prompt = build_prompt(request, schema)
        self._calls += 1
        started = time.perf_counter()

        try:
            text, model_used = self._generate(client, types, prompt, schema, sdk_errors)
            choice, probabilities, confidence, reasoning, synthesised = parse_response(
                text, request
            )
        except (DecisionUnavailable, DecisionFailed) as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self._last_latency_ms = latency_ms
        self._last_error = ""
        raw = annotate_raw(
            {"response_text": text},
            model=model_used,
            transport="google-genai/structured-output",
            question_type=request.question_type.value,
            reasoning=reasoning,
            distribution="synthesised_uniform" if synthesised else "model_supplied",
            http_latency_ms=latency_ms,
        )
        logger.debug(
            "decision.gemini %s answered %s in %.1fms (model=%s, distribution=%s)",
            request.request_id, choice, latency_ms, model_used,
            raw["_distribution"],
        )
        return make_decision(
            request_id=request.request_id,
            question=request.question,
            choice=choice,
            probabilities=probabilities,
            confidence=confidence,
            source=DecisionSource.GEMINI,
            model=model_used,
            latency_ms=latency_ms,
            degraded=synthesised,
            raw=raw,
        )

    # ----------------------------------------------------------------- calling
    def _generate(self, client: Any, types: Any, prompt: str,
                  schema: Mapping[str, Any], sdk_errors: tuple[type[BaseException], ...],
                  ) -> tuple[str, str]:
        """Call the primary model, then the fallback on a model-specific error.

        Returns ``(text, model_actually_used)``.
        """
        attempts: list[tuple[str, bool]] = [(self.settings.gemini_model, False)]
        if self.settings.gemini_fallback_model and \
                self.settings.gemini_fallback_model != self.settings.gemini_model:
            attempts.append((self.settings.gemini_fallback_model, True))

        last_error: Exception | None = None
        for model, is_fallback in attempts:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=dict(schema),
                        temperature=0.0,
                    ),
                )
            except sdk_errors as exc:
                if is_fallback:
                    break
                if type(exc).__name__ in ("ClientError",):
                    # The model id itself is the problem; a different model may work.
                    logger.warning(
                        "decision.gemini %s rejected (%s: %s); trying fallback %s",
                        model, type(exc).__name__, snippet(exc),
                        self.settings.gemini_fallback_model,
                    )
                    self._fallbacks += 1
                    last_error = exc
                    continue
                # Server-side and connection failures would fail identically on
                # the fallback model; going around the registry's rules backend is
                # strictly worse than letting the registry handle it.
                raise DecisionUnavailable(
                    f"gemini: call to {model} failed ({type(exc).__name__}: {snippet(exc)})"
                ) from exc
            except (ValueError, TypeError) as exc:
                raise DecisionFailed(
                    f"gemini: bad request to {model} ({type(exc).__name__}: {snippet(exc)})"
                ) from exc
            except httpx.TimeoutException as exc:
                # google-genai lets transport faults escape unwrapped: a
                # connect timeout is an httpx.TimeoutException, not an
                # APIError, so it has to be named explicitly or it would
                # escape the registry's fallback chain entirely.
                raise DecisionUnavailable(
                    f"gemini: call to {model} timed out after "
                    f"{self.settings.gemini_timeout_s:.0f}s "
                    f"({type(exc).__name__})"
                ) from exc
            except httpx.HTTPError as exc:
                raise DecisionUnavailable(
                    f"gemini: transport failure calling {model} "
                    f"({type(exc).__name__}: {snippet(exc)})"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - third-party boundary, see below
                # Documented, narrow-in-spirit escape hatch: this is the edge
                # between Paytriq and an SDK we do not control, and its exception
                # surface changes between releases. An unrecognised SDK error
                # escaping here would bypass the registry's fallback chain and
                # kill the run -- which is strictly worse than converting it.
                # Nothing is swallowed: the original type and message are carried
                # into the error and the decision is not produced.
                logger.error(
                    "decision.gemini unexpected %s from the %s SDK while calling %s: %s",
                    type(exc).__name__, "google-genai", model, snippet(exc),
                )
                raise DecisionUnavailable(
                    f"gemini: unexpected {type(exc).__name__} from google-genai "
                    f"while calling {model}: {snippet(exc)}"
                ) from exc

            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise DecisionFailed(
                    f"gemini: {model} returned no text "
                    f"(response type {type(response).__name__})"
                )
            if is_fallback:
                self._fallbacks += 1
            self._client_model = model
            return text, model

        assert last_error is not None  # only a ClientError on the primary gets here
        raise DecisionUnavailable(
            f"gemini: model {self.settings.gemini_model} unusable and fallback "
            f"{self.settings.gemini_fallback_model} did not help "
            f"({type(last_error).__name__}: {snippet(last_error)})"
        )

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        ok, reason = self.available()
        return {
            "backend": self.name,
            "available": ok,
            "reason": reason,
            "model": self.model,
            "configured_model": self.settings.gemini_model,
            "fallback_model": self.settings.gemini_fallback_model,
            "fallbacks_used": self._fallbacks,
            "latency_ms": round(self._last_latency_ms, 3),
            "last_error": self._last_error,
            "calls": self._calls,
            "timeout_s": self.settings.gemini_timeout_s,
            "import_error": self._import_error,
        }
