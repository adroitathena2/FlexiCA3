"""Shared machinery for the decision layer.

Why this module exists
----------------------
Three different backends (Clef/System One, Gemini structured output, and a
deterministic rules engine) all have to answer the same two questions safely:

1. *How do I read a probability distribution out of an untrusted payload?*
2. *What does "the backend could not answer" mean, and what may I do about it?*

Answering them once, here, is what makes "no fabricated probabilities" a
property of the whole layer rather than a habit of one author.

Three invariants are enforced here
----------------------------------
* **A distribution is either real or absent.** If a payload does not carry one,
  :func:`choice_from` raises :class:`~core.errors.DecisionFailed`. It never
  invents a distribution, because a fabricated uniform looks exactly like a
  calibrated one to every downstream consumer and destroys the escalation
  signal the whole system depends on.
* **A distribution that is returned sums to 1.0.** :func:`normalise_probabilities`
  coerces, clamps and renormalises, so the pydantic validator in
  ``core.schemas.Decision`` cannot reject it for a rounding artefact.
* **"Unreachable" and "replied with nonsense" are different failures.**
  :class:`DecisionUnavailable` means fall back to another backend.
  :class:`DecisionFailed` means this backend answered and the answer is unusable
  -- also fall back, but never pretend the backend agreed.

A note on the wrong-endpoint trap
---------------------------------
Clef (and Jev) are **System One decision models**, not text generators. The
endpoint is ``POST /v1/systemone`` with ``{"state", "questions"}`` in and
``{"answers", "usage"}`` out. A clef model served by llama.cpp also answers the
OpenAI-compatible ``POST /v1/chat/completions`` -- with *prose*, because that is
the interface the weights were never trained through. Reading probabilities out
of chat logprobs is tempting and is exactly wrong: it destroys the calibration
the model was trained for and yields numbers that look like probabilities to
every consumer downstream. :func:`require_answers` is the guard against it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from core.errors import DecisionFailed, DecisionUnavailable
from core.schemas import Decision, DecisionRequest, QuestionType

__all__ = [
    "logger",
    "NOUL_LABELS",
    "KEY_PREFIX",
    "PROBE_ENDPOINT",
    "ProbeResult",
    "SystemOneClient",
    "normalise_probabilities",
    "choice_from",
    "score_probabilities",
    "noul_probabilities",
    "build_response_key",
    "state_text",
    "criteria_for_choice",
    "require_answers",
    "make_decision",
    "annotate_raw",
    "snippet",
]

#: Package logger. Degradation events are logged here so a trace reader can grep
#: one stream for every moment the system substituted something.
logger = logging.getLogger("paytriq.decision")

#: Labels used to express a ``noul`` question as a two-way distribution, so a
#: yes/no decision has the same shape as every other decision in the system.
NOUL_LABELS: tuple[str, str] = ("true", "false")

#: Prefix for the per-question key inside a System One ``questions`` mapping.
KEY_PREFIX = "q_"

#: Trivial OpenAI-compatible endpoint used only for availability probing and
#: model discovery on llama.cpp servers.
PROBE_ENDPOINT = "/v1/models"

#: Keys Paytriq adds to ``Decision.raw`` are prefixed with an underscore so a
#: reader can always tell "the server said this" from "we said this".
_RAW_META_PREFIX = "_"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SNIPPET_CHARS = 240


# ==============================================================================
# numeric hygiene
# ==============================================================================
def _as_probability(value: Any, *, where: str) -> float:
    """Coerce one raw probability to a float in [0, 1], or refuse.

    Accepts ints, floats and numeric strings (Clef returns ``"0.93"`` more often
    than anyone would like). Rejects booleans, ``None``, ``NaN``, ``inf`` and
    anything non-numeric: a probability we cannot read is a probability we must
    not guess at.
    """
    if isinstance(value, bool) or value is None:
        raise DecisionFailed(f"{where}: {value!r} is not a probability")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise DecisionFailed(
                f"{where}: {value!r} is not a numeric probability"
            ) from exc
    else:
        raise DecisionFailed(f"{where}: {value!r} is not a probability")

    if number != number or number in (float("inf"), float("-inf")):
        # NaN/inf. Notable because a Windows Ollama build with a 32-bit
        # file-offset bug (ollama/ollama#18769) emits non-finite logits, which
        # surface here as non-finite probabilities. Refusing is the point.
        raise DecisionFailed(f"{where}: non-finite probability {value!r}")
    return min(1.0, max(0.0, number))


def normalise_probabilities(raw: Mapping[str, Any]) -> dict[str, float]:
    """Coerce, clamp and renormalise a probability mapping.

    The pydantic validator on ``Decision.probabilities`` demands a sum within
    0.02 of 1.0, but that tolerance is a safety net against a *bad* distribution,
    not permission to ship one. This function lands the sum on 1.0 to within
    1e-6 so a passing validation means the distribution is genuinely normalised.

    Raises :class:`DecisionFailed` for a non-numeric, non-finite, empty, or
    all-zero mapping: every one of those is "no usable distribution", and the
    honest response to that is to fail rather than to synthesise one.
    """
    if not isinstance(raw, Mapping):
        raise DecisionFailed(f"probabilities must be a mapping, got {type(raw).__name__}")
    if not raw:
        raise DecisionFailed("probabilities mapping is empty")

    clamped: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise DecisionFailed(f"probability key {key!r} is not a string")
        clamped[key] = _as_probability(value, where=f"probability {key!r}")

    total = sum(clamped.values())
    if total <= 0.0:
        raise DecisionFailed(
            f"probabilities sum to {total!r}; cannot renormalise a zero distribution "
            f"from {sorted(clamped)}"
        )

    # Divide first so the sum is correct, then round for legibility in a trace,
    # then hand the rounding residual to the largest entry so the invariant
    # holds exactly rather than approximately.
    scaled = {k: v / total for k, v in clamped.items()}
    rounded = {k: round(v, 6) for k, v in scaled.items()}
    residual = 1.0 - sum(rounded.values())
    if residual:
        largest = max(rounded, key=lambda k: rounded[k])
        rounded[largest] = round(rounded[largest] + residual, 6)
    return rounded


def _confidence_from(payload: Mapping[str, Any], probabilities: Mapping[str, float],
                     choice: str) -> float:
    """Reconcile a reported confidence with the probability of the choice.

    ``Decision`` requires ``confidence <= probabilities[choice]``. System One
    reports a separate ``confidence`` field which normally *is* that
    probability, but renormalisation and rounding can push the probability a
    hair below it. Taking the minimum keeps the invariant and never overstates
    how sure anybody was.
    """
    reported = payload.get("confidence")
    ceiling = probabilities[choice]
    if reported is None or isinstance(reported, bool):
        return ceiling
    try:
        value = _as_probability(reported, where="confidence")
    except DecisionFailed:
        return ceiling
    return min(value, ceiling)


# ==============================================================================
# response parsing
# ==============================================================================
def _argmax(probabilities: Mapping[str, float]) -> str:
    """Highest-probability key, ties broken by declaration order (stable)."""
    best_key = ""
    best_value = -1.0
    for key, value in probabilities.items():
        if value > best_value:
            best_key, best_value = key, value
    return best_key


def _infer_type(payload: Mapping[str, Any]) -> QuestionType | None:
    """Recover the question type from the answer body when it is omitted.

    llama.cpp's System One route has shipped both with and without ``type``;
    the discriminating field is unambiguous in every case, so inferring is
    safe. Anything we cannot place returns ``None`` and the caller raises.
    """
    declared = payload.get("type")
    if isinstance(declared, str):
        try:
            return QuestionType(declared.strip().lower())
        except ValueError:
            return None
    if "choice" in payload:
        return QuestionType.CHOICE
    if "score" in payload:
        return QuestionType.SCORE
    if "noul" in payload:
        return QuestionType.NOUL
    return None


def noul_probabilities(payload: Mapping[str, Any], *,
                       labels: tuple[str, str] = NOUL_LABELS) -> dict[str, float]:
    """Two-way distribution for a ``noul`` question.

    ``confidence`` is often absent for ``noul`` answers (the probability of true
    *is* the answer, so a second number would be redundant). It is derived, not
    invented: see :func:`_confidence_from`.
    """
    raw = payload.get("noul")
    if raw is not None:
        p_true = _as_probability(raw, where="noul")
    else:
        probs = payload.get("probabilities")
        if not isinstance(probs, Mapping) or not probs:
            raise DecisionFailed(
                "noul answer carried neither a 'noul' probability nor a "
                "'probabilities' mapping; refusing to invent one"
            )
        # Tolerate {"true": p, "false": 1-p} and {"0": p, "1": 1-p}.
        if labels[0] in probs:
            p_true = _as_probability(probs[labels[0]], where=f"probability {labels[0]!r}")
        else:
            keys = [k for k in probs if str(k) in ("0", "1")]
            if not keys:
                raise DecisionFailed(
                    f"cannot read a noul probability from {sorted(probs)}"
                )
            p_true = _as_probability(probs[sorted(keys)[0]], where="noul")
    return normalise_probabilities({labels[0]: p_true, labels[1]: 1.0 - p_true})


def score_probabilities(payload: Mapping[str, Any], rubric: Sequence[str]) -> dict[str, float]:
    """Distribution over rubric levels for a ``score`` question.

    System One keys score probabilities by *ordinal index* (``"0"``, ``"1"``,
    ``"2"``) and ships the index -> label mapping separately in ``legend``. Both
    shapes are accepted: index-keyed with a legend, and label-keyed directly.
    Anything else raises, because guessing which rubric level ``"2"`` denotes is
    precisely the kind of invention this layer must not perform.
    """
    raw = payload.get("probabilities")
    if not isinstance(raw, Mapping) or not raw:
        raise DecisionFailed(
            "score answer carried no 'probabilities' mapping; a score question "
            "answered without a distribution cannot be turned into a Decision"
        )

    known = list(rubric)
    mapped: dict[str, float] = {}

    if known and all(str(k) in known for k in raw):
        mapped = {str(k): v for k, v in raw.items()}
    else:
        legend = _legend_map(payload.get("legend"))
        for key, value in raw.items():
            label = legend.get(str(key))
            if label is None and known:
                try:
                    label = known[int(str(key))]
                except (ValueError, IndexError):
                    label = None
            if label is None:
                raise DecisionFailed(
                    f"score answer has no rubric level for key {key!r} "
                    f"(legend={sorted(legend)}, rubric={known})"
                )
            if known and label not in known:
                raise DecisionFailed(
                    f"score answer names rubric level {label!r}, which is not in "
                    f"the requested rubric {known}"
                )
            mapped[label] = value

    return normalise_probabilities(mapped)


def _legend_map(legend: Any) -> dict[str, str]:
    """Accept ``legend`` as a mapping or as an ordered list of levels."""
    if isinstance(legend, Mapping):
        return {str(k): str(v) for k, v in legend.items()}
    if isinstance(legend, (list, tuple)):
        return {str(i): str(v) for i, v in enumerate(legend)}
    return {}


def choice_from(payload: Any, options: Sequence[str], *,
                labels: tuple[str, str] = NOUL_LABELS,
                ) -> tuple[str | None, dict[str, float], float]:
    """Read ``(choice, probabilities, confidence)`` out of one System One answer.

    ``payload`` is the per-question object from ``{"answers": {...}}``, e.g.
    ``{"type": "choice", "choice": "billing", "confidence": 0.71,
    "probabilities": {"billing": 0.71, "technical": 0.29}}``.

    Behaviour, in order of preference:

    * **Missing optional fields are tolerated.** A missing ``confidence`` is
      derived from the chosen option's probability. A missing ``type`` is
      inferred from ``choice``/``score``/``noul``. A missing ``choice`` is taken
      as the argmax of the distribution, which is the definition of a choice for
      a System One model, not an invention.
    * **A missing *distribution* is refused.** ``choice`` without
      ``probabilities`` means we would have to make up the alternatives.
      ``DecisionFailed`` is the correct response.
    * **An out-of-set choice is refused.** A choice the request never offered is
      unusable; silently snapping it to the nearest valid option would launder a
      hallucination into a decision.

    Returns ``(None, {}, 0.0)`` only when ``payload`` carries no answer at all
    (not a mapping, or empty), which lets a caller distinguish "nothing here"
    from "malformed here" and decide for itself.
    """
    if not isinstance(payload, Mapping) or not payload:
        return None, {}, 0.0

    qtype = _infer_type(payload)
    if qtype is None:
        raise DecisionFailed(
            f"cannot tell what kind of answer this is: keys={sorted(payload)}"
        )

    known = [str(o) for o in options]

    if qtype is QuestionType.NOUL:
        probs = noul_probabilities(payload, labels=labels)
        choice = labels[0] if probs[labels[0]] >= probs[labels[1]] else labels[1]
        declared = payload.get("choice")
        if isinstance(declared, str) and declared.strip() in probs:
            choice = declared.strip()
        return choice, probs, _confidence_from(payload, probs, choice)

    if qtype is QuestionType.SCORE:
        probs = score_probabilities(payload, known)
        choice = payload.get("score")
        if not isinstance(choice, str) or not choice.strip():
            choice = _argmax(probs)
        choice = choice.strip()
        if choice not in probs:
            raise DecisionFailed(
                f"score answer chose {choice!r} which has no probability in "
                f"{sorted(probs)}"
            )
        return choice, probs, _confidence_from(payload, probs, choice)

    # QuestionType.CHOICE
    raw = payload.get("probabilities")
    if not isinstance(raw, Mapping) or not raw:
        raise DecisionFailed(
            "choice answer carried no 'probabilities' mapping; assigning a "
            "confidence to one option and dividing the rest by fiat would be a "
            "fabricated distribution, so this is refused instead"
        )
    if known:
        filtered = {str(k): v for k, v in raw.items() if str(k) in known}
        dropped = sorted(set(str(k) for k in raw) - set(known))
        if dropped:
            logger.debug(
                "decision.choice_from dropping %d probability key(s) not in the "
                "requested option set: %s", len(dropped), dropped,
            )
        if not filtered:
            raise DecisionFailed(
                f"choice answer probabilities {sorted(str(k) for k in raw)} share "
                f"no key with the requested options {known}"
            )
    else:
        filtered = {str(k): v for k, v in raw.items()}
    probs = normalise_probabilities(filtered)

    choice = payload.get("choice")
    if isinstance(choice, str) and choice.strip():
        choice = choice.strip()
        if choice not in probs:
            raise DecisionFailed(
                f"choice answer chose {choice!r}, which is not one of {sorted(probs)}"
            )
    else:
        choice = _argmax(probs)
    return choice, probs, _confidence_from(payload, probs, choice)


# ==============================================================================
# request construction
# ==============================================================================
def build_response_key(request: DecisionRequest) -> str:
    """Stable key for this request's question inside a System One call.

    Paytriq issues **one question per ``DecisionRequest``** because
    ``core.schemas.Decision`` holds exactly one choice. The key therefore maps
    back to ``request_id`` one-to-one by construction; what it buys is that the
    wire payload and the recorded response stay *legible* -- ``q_which_team_1f3a…``
    rather than ``q_0``.

    It is a pure function of question type and question text, never of
    ``request_id``, because ``request_id`` is random per request and replay
    needs byte-identical requests to produce identical recordings.
    """
    fingerprint = hashlib.sha256(
        f"{request.question_type.value}\x00{request.question}".encode()
    ).hexdigest()[:8]
    slug = _SLUG_RE.sub("_", request.question.lower())
    slug = "_".join(part for part in slug.split("_") if part)[:40].strip("_")
    return f"{KEY_PREFIX}{slug or 'question'}_{fingerprint}"


def state_text(request: DecisionRequest) -> str:
    """Serialise ``DecisionRequest.state`` for the System One ``state`` field.

    JSON rather than prose: the field is documented as "text or JSON", the agent
    that built the state chose those keys, and re-rendering them as English
    would discard the structure while inventing words. ``default=str`` keeps an
    unserialisable leaf from killing a decision; the raw response is recorded
    either way.
    """
    try:
        return json.dumps(request.state, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise DecisionFailed(
            f"request {request.request_id}: state is not JSON-serialisable ({exc})"
        ) from exc


def criteria_for_choice(request: DecisionRequest) -> dict[str, str]:
    """Build the System One ``criteria`` map for a ``choice`` question.

    System One wants ``{option: description}``. ``DecisionRequest.options`` is a
    bare list of labels, so the label doubles as its own description unless the
    agent supplied richer text in ``state["criteria"]`` (a mapping keyed by the
    options, or a plain list in option order).
    """
    options = [str(o) for o in request.options]
    supplied = request.state.get("criteria")
    if isinstance(supplied, Mapping):
        chosen = {str(k): str(v) for k, v in supplied.items() if str(k) in options}
        if chosen and len(chosen) == len(options):
            return chosen
        logger.debug(
            "decision.criteria_for_choice ignoring state['criteria'] for %s: it "
            "covers %d of %d options", request.request_id, len(chosen), len(options),
        )
    elif isinstance(supplied, (list, tuple)) and len(supplied) == len(options):
        return {opt: str(desc) for opt, desc in zip(options, supplied, strict=False)}
    return {opt: opt for opt in options}


def require_answers(payload: Mapping[str, Any], key: str, *,
                    backend: str, endpoint: str) -> Mapping[str, Any]:
    """Return ``payload["answers"]``, or explain precisely what came back instead.

    This is the wrong-endpoint guard. A System One answer always carries an
    ``answers`` object; a chat completion carries ``choices`` and a message; a
    misconfigured proxy carries an HTML error page. All three are common, all
    three are silent if you just index into the body, and the third is the one
    that produces a "decision" which is actually a hallucinated paragraph.
    """
    if not isinstance(payload, Mapping):
        raise DecisionFailed(
            f"{backend}: expected a JSON object from {endpoint}, got "
            f"{type(payload).__name__}"
        )
    if "answers" not in payload:
        shape = ", ".join(sorted(payload)[:8]) or "an empty body"
        detail = ""
        if "choices" in payload:
            detail = (
                " The body has a 'choices' key, i.e. it is an OpenAI-compatible "
                "chat completion: clef is a System One decision model, not a text "
                "generator, and its probabilities only exist on the System One "
                "route."
            )
        if payload.get("success") is False or "errors" in payload:
            detail = (
                detail
                + f" Provider-reported errors: {snippet(payload.get('errors'))}"
            ).strip()
        raise DecisionFailed(
            f"{backend}: response from {endpoint} has no 'answers' key "
            f"(keys: {shape}).{detail} Clef/System One must be called at "
            f"POST {endpoint} with body "
            '{"state": ..., "questions": {...}} -- never at '
            "/v1/chat/completions."
        )
    answers = payload["answers"]
    if not isinstance(answers, Mapping):
        raise DecisionFailed(
            f"{backend}: 'answers' from {endpoint} is a "
            f"{type(answers).__name__}, expected an object keyed by question"
        )
    if key not in answers:
        raise DecisionFailed(
            f"{backend}: asked question {key!r} but the response answered "
            f"{sorted(answers)}"
        )
    answer = answers[key]
    if not isinstance(answer, Mapping):
        raise DecisionFailed(
            f"{backend}: answer for {key!r} is a {type(answer).__name__}, "
            "expected an object"
        )
    return answer


def snippet(value: Any, limit: int = _SNIPPET_CHARS) -> str:
    """One-line, length-capped rendering of an arbitrary value for error text.

    Error messages end up in a trace and in a demo status panel. They must stay
    one line and must not carry a credential that a provider echoed back.
    """
    if isinstance(value, (bytes, bytearray)):
        text = value[:limit].decode("utf-8", "replace")
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def annotate_raw(raw: Mapping[str, Any] | None, **meta: Any) -> dict[str, Any]:
    """Copy a backend payload into ``Decision.raw`` and stamp Paytriq metadata.

    The server's own fields are preserved byte-for-byte; ours go in under
    ``_``-prefixed keys so provenance is never confused with content.
    """
    out: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    for key, value in meta.items():
        out[f"{_RAW_META_PREFIX}{key}"] = value
    return out


def make_decision(**kwargs: Any) -> Decision:
    """Construct a ``Decision``, converting validation errors to ``DecisionFailed``.

    A backend must never leak a ``pydantic.ValidationError`` to the registry:
    the registry's whole contract is that it catches ``DecisionUnavailable`` and
    ``DecisionFailed``, and an unrecognised exception type would escape the
    fallback chain and kill the run.
    """
    try:
        return Decision(**kwargs)
    except ValidationError as exc:
        raise DecisionFailed(f"cannot build a valid Decision: {exc}") from exc


# ==============================================================================
# transport
# ==============================================================================
@dataclass(slots=True)
class ProbeResult:
    """Outcome of one availability probe. Cached by the transport for a TTL."""

    available: bool
    reason: str
    latency_ms: float = 0.0
    model: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 3),
            "model": self.model,
            **self.detail,
        }


#: Process-wide probe cache: ``(endpoint, timeout) -> (monotonic_stamp, ProbeResult)``.
#:
#: Reachability belongs to the endpoint, not to the object asking about it, so a
#: cache scoped to one transport instance re-probes on every construction. That is
#: invisible with a single long-lived registry and pathological when many short-lived
#: ones are created: the evaluation harness builds one per condition per seed, and
#: against a server that is *down* every probe costs the full timeout.
#:
#: Keyed on the timeout as well as the endpoint so a deliberately short-timeout
#: probe (a health check, say) is never served a long-timeout result.
_PROBE_CACHE: dict[tuple[str, float], tuple[float, ProbeResult]] = {}


def clear_probe_cache() -> None:
    """Drop every cached probe. For tests, and after a server is (re)started."""
    _PROBE_CACHE.clear()


class SystemOneClient:
    """HTTP machinery shared by every System One transport.

    Owns the four things that are easy to get subtly wrong and impossible to
    review at a glance: the timeout (every call has one, always), the retry
    policy (transport faults and 5xx/429 are retried, other 4xx are not --
    retrying a 400 just wastes the run's wall clock), the error taxonomy
    (``DecisionUnavailable`` vs ``DecisionFailed``), and the health counters
    that make the demo status panel show something true.

    A fresh :class:`httpx.Client` is created per call. The call volume here is
    tens of calls per run, connection reuse buys nothing measurable, and a
    per-call client cannot leak a socket into an interpreter shutdown warning --
    which matters more in a test suite than in production.
    """

    #: Subclasses set these.
    name: str = "systemone"
    endpoint: str = ""
    model_label: str = "unknown"

    def __init__(self, *, timeout_s: float, max_retries: int = 1,
                 backoff_s: float = 0.25,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.timeout_s = max(0.001, float(timeout_s))
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self._transport = transport
        self._calls = 0
        self._failures = 0
        self._last_latency_ms = 0.0
        self._last_error = ""
        self._last_ok_at: float | None = None
        self._probe_result: ProbeResult | None = None
        self._probe_monotonic: float | None = None
        self._model_discovered: str = ""

    # ------------------------------------------------------------------ probes
    @property
    def model(self) -> str:
        """The loaded model, if discovery has found one; else the configured label."""
        return self._model_discovered or self.model_label

    @property
    def model_discovered(self) -> bool:
        """True when the loaded model was read from the server, not assumed."""
        return bool(self._model_discovered)

    def probe(self, *, ttl_s: float, force: bool = False,
              timeout_s: float | None = None) -> ProbeResult:
        """Probe availability, caching the answer for ``ttl_s`` seconds.

        Availability is never asserted from configuration alone. A configured
        base URL says where a server *would* be; only a probe says whether it is.

        The cache is **process-wide, keyed by endpoint**, not per instance.
        Reachability is a property of the endpoint, not of the object asking, so
        a per-instance cache re-probes on every construction. That is invisible
        with one long-lived registry and pathological when something builds many
        short-lived ones: the evaluation harness constructs a decision invoker
        per condition per seed, and against a server that is *down* each probe
        costs the full timeout -- which turned a 21-second test suite into
        114 seconds. Keying on ``(endpoint, timeout)`` means the second caller
        pays nothing, whether it holds the same instance or a new one.
        """
        now = time.monotonic()
        effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
        cache_key = (self.endpoint, round(float(effective_timeout), 3))

        if not force:
            # Always honour this instance's own cached verdict. A caller that
            # asked twice with the same transport must not hit the network twice
            # just because the process-wide cache was bypassed for injection.
            own = self._probe_result
            if (
                own is not None
                and self._probe_monotonic is not None
                and (now - self._probe_monotonic) < max(0.0, ttl_s)
            ):
                return own

            # Only share a verdict across instances when this one would reach the
            # world the same way. A test that injects its own httpx transport is
            # talking to a mock, not to whatever listens on that endpoint, so
            # sharing across it would hand one caller another caller's answer.
            entry = _PROBE_CACHE.get(cache_key) if self._transport is None else None
            if entry is not None and (now - entry[0]) < max(0.0, ttl_s):
                self._probe_result = entry[1]
                self._probe_monotonic = entry[0]
                if entry[1].model:
                    self._model_discovered = entry[1].model
                return entry[1]

        result = self._do_probe(effective_timeout)
        stamp = time.monotonic()
        if self._transport is None:
            _PROBE_CACHE[cache_key] = (stamp, result)
        self._probe_result = result
        self._probe_monotonic = stamp
        if result.model:
            self._model_discovered = result.model
        return result

    def _do_probe(self, timeout_s: float) -> ProbeResult:
        """Real reachability check. Overridden per transport."""
        raise NotImplementedError

    def invalidate_probe(self) -> None:
        self._probe_result = None
        self._probe_monotonic = None
        # Drop this endpoint's shared entry too, so an explicit invalidation is
        # honoured by every instance rather than only the caller.
        for key in [k for k in _PROBE_CACHE if k[0] == self.endpoint]:
            _PROBE_CACHE.pop(key, None)

    # ------------------------------------------------------------------- stats
    def _ok(self, latency_ms: float) -> None:
        self._last_latency_ms = latency_ms
        self._last_ok_at = time.time()
        self._last_error = ""

    def _bad(self, error: str) -> None:
        self._failures += 1
        self._last_error = error

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "endpoint": self.endpoint,
            "model": self.model,
            "discovered_model": self._model_discovered or None,
            "timeout_s": round(self.timeout_s, 3),
            "max_retries": self.max_retries,
            "calls": self._calls,
            "failures": self._failures,
            "last_latency_ms": round(self._last_latency_ms, 3),
            "last_error": self._last_error,
            "last_ok_at": self._last_ok_at,
            "probe": self._probe_result.as_dict() if self._probe_result else None,
        }

    # ------------------------------------------------------------------ client
    def _new_client(self, timeout_s: float) -> httpx.Client:
        return httpx.Client(timeout=timeout_s, transport=self._transport,
                            follow_redirects=False)

    @staticmethod
    def _retryable(status: int) -> bool:
        return status in (408, 425, 429) or status >= 500

    def _request(self, method: str, url: str, *,
                 json_body: Mapping[str, Any] | None = None,
                 headers: Mapping[str, str] | None = None,
                 timeout_s: float | None = None,
                 ) -> tuple[dict[str, Any], float]:
        """Perform one HTTP call with retries. Returns ``(json_body, latency_ms)``.

        ``latency_ms`` covers every attempt, so a retried call reports the time
        the caller actually waited rather than flattering itself with the
        duration of the successful attempt alone.
        """
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        total_ms = 0.0

        for attempt in range(1, attempts + 1):
            self._calls += 1
            started = time.perf_counter()
            try:
                with self._new_client(timeout) as client:
                    response = client.request(method, url, json=json_body,
                                              headers=dict(headers) if headers else None,
                                              timeout=timeout)
            except httpx.TimeoutException as exc:
                elapsed = (time.perf_counter() - started) * 1000
                total_ms += elapsed
                last_error = DecisionUnavailable(
                    f"{self.name}: {method} {self.endpoint} timed out after "
                    f"{timeout:.1f}s (attempt {attempt}/{attempts}, {type(exc).__name__})"
                )
            except httpx.InvalidURL as exc:
                # A configuration error, never a transient one.
                raise DecisionUnavailable(
                    f"{self.name}: invalid URL {url!r} ({exc})"
                ) from exc
            except httpx.TransportError as exc:
                elapsed = (time.perf_counter() - started) * 1000
                total_ms += elapsed
                last_error = DecisionUnavailable(
                    f"{self.name}: cannot reach {url} (attempt {attempt}/{attempts}, "
                    f"{type(exc).__name__}: {snippet(exc)})"
                )
            except httpx.HTTPError as exc:
                elapsed = (time.perf_counter() - started) * 1000
                total_ms += elapsed
                last_error = DecisionUnavailable(
                    f"{self.name}: transport failure on {method} {url} "
                    f"({type(exc).__name__}: {snippet(exc)})"
                )
            else:
                total_ms += (time.perf_counter() - started) * 1000
                status = response.status_code
                if 200 <= status < 300:
                    try:
                        payload = response.json()
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        last_error = DecisionFailed(
                            f"{self.name}: {self.endpoint} returned HTTP {status} with a "
                            f"non-JSON body ({type(exc).__name__}); a System One server "
                            f"must answer with a JSON object. Body starts: "
                            f"{snippet(response.content)!r}"
                        )
                    else:
                        if not isinstance(payload, Mapping):
                            last_error = DecisionFailed(
                                f"{self.name}: {self.endpoint} returned JSON "
                                f"{type(payload).__name__}, expected an object"
                            )
                        else:
                            self._ok(total_ms)
                            return dict(payload), total_ms
                elif self._retryable(status):
                    last_error = DecisionUnavailable(
                        f"{self.name}: {self.endpoint} returned HTTP {status} "
                        f"(attempt {attempt}/{attempts}): "
                        f"{snippet(response.content)!r}"
                    )
                else:
                    # 4xx: the server understood and refused. Retrying cannot help.
                    self._bad(f"HTTP {status}: {snippet(response.content)!r}")
                    raise DecisionFailed(
                        f"{self.name}: {self.endpoint} returned HTTP {status}: "
                        f"{snippet(response.content)!r}"
                    )
            if attempt < attempts and self.backoff_s > 0:
                # Linear, short. A decision-model call sits inside an agent step
                # budget; exponential backoff would eat the budget to no purpose.
                time.sleep(self.backoff_s * attempt)

        if last_error is None:  # unreachable: attempts >= 1 always sets or raises
            self._bad(f"{self.name}: no attempt was made against {url}")
            raise DecisionUnavailable(f"{self.name}: no attempt was made against {url}")
        self._bad(str(last_error))
        raise last_error
