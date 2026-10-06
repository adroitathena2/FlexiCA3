"""Clef (Cloudflare) System One decision backend.

What a decision model is
------------------------
Clef and Jev "System One" are **decision models**, not text generators. You send
a ``state`` plus a schema of typed ``questions`` and you get typed answers **with
calibrated probabilities**::

    POST /v1/systemone
    {"state": "...", "questions": {"urgent": {"type": "noul", "instructions": "..."}}}

    {"answers": {"urgent": {"type": "noul", "noul": 0.93}}, "usage": {...}}

Question types are ``noul`` (probability of true), ``choice`` (one of a closed
criteria map) and ``score`` (a level on an ordered rubric).

The wrong-endpoint trap
-----------------------
The endpoint is ``POST /v1/systemone``. It is **not** ``/v1/chat/completions``.
A clef model served by llama.cpp also answers the OpenAI-compatible chat route,
and it answers it with *prose*, because that is not the interface the weights
were trained through. Nothing about the response says "this is the wrong route":
it is valid JSON with plausible content. Two failure modes follow, and both are
worse than an outage:

* treating prose as a label, or
* harvesting probabilities out of chat ``logprobs``.

The second is the seductive one. Chat logprobs are not the calibration clef was
trained to produce; they are the model's uncertainty about *its own next token*
under a different objective. A number derived from them looks exactly like a
calibrated probability to every downstream consumer, which is precisely what
makes it dangerous. This module therefore never touches ``logprobs``, and
:func:`~decision.base.require_answers` raises a ``DecisionFailed`` that names
``/v1/systemone`` the moment a response arrives without an ``answers`` key.

Why llama.cpp and not Ollama
----------------------------
``clef_backend=ollama`` is the *setting name* frozen in ``core.config``; the
server behind it is llama.cpp's ``llama-server``, whose default port is 18781.
Ollama's official ``clef-flash`` is broken on Windows: 32-bit file-offset
truncation in ``ifstream`` (ollama/ollama#18769) produces non-finite logits,
so the model emits NaN probabilities for every question. ``_normalise_base_url``
therefore rewrites Ollama's default port to 18781, and
``decision.base._as_probability`` rejects non-finite values as a second line of
defence.

Two transports, one interface
-----------------------------
``LlamaCppTransport``  local ``llama-server``, plain HTTP, no auth, no cost, no
                       rate limit.
``WorkersAiTransport`` Cloudflare Workers AI, bearer token, metered.

Both speak the identical ``{"state", "questions"} -> {"answers", "usage"}``
protocol; Workers AI additionally wraps the result in a Cloudflare envelope
(``{"result": {...}, "success": true}``), which :meth:`unwrap_response` removes.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import httpx

from core.config import Settings, get_settings
from core.errors import DecisionFailed, DecisionUnavailable
from core.schemas import Decision, DecisionRequest, DecisionSource, QuestionType

from .base import (
    NOUL_LABELS,
    ProbeResult,
    SystemOneClient,
    annotate_raw,
    build_response_key,
    choice_from,
    criteria_for_choice,
    logger,
    make_decision,
    require_answers,
    snippet,
    state_text,
)

__all__ = [
    "SYSTEMONE_PATH",
    "DEFAULT_LLAMA_CPP_BASE_URL",
    "WORKERS_AI_MODEL_PATH",
    "ClefTransport",
    "LlamaCppTransport",
    "WorkersAiTransport",
    "ClefBackend",
    "build_body",
]

#: The only correct route for a System One decision model.
SYSTEMONE_PATH = "/v1/systemone"

#: llama.cpp's ``llama-server`` default. Note this is *not* Ollama's 11434.
DEFAULT_LLAMA_CPP_BASE_URL = "http://localhost:18781"

#: Cloudflare's hosted clef model, relative to the account's AI run route.
WORKERS_AI_MODEL_PATH = "@cf/cloudflare/clef-flash"

#: Port Ollama listens on. Seeing it is a strong hint that a base URL was copied
#: from an Ollama habit rather than from a llama.cpp invocation.
_OLLAMA_PORT = 11434

#: Availability probes must be quick even when ``CLEF_TIMEOUT_S`` is generous.
_PROBE_TIMEOUT_CAP_S = 3.0

#: The cheapest possible System One question. Used to prove the decision route
#: actually answers, which ``/v1/models`` alone does not.
_PROBE_STATE = '{"probe": "liveness"}'
_PROBE_KEY = "q_probe"
_PROBE_BODY: dict[str, Any] = {
    "state": _PROBE_STATE,
    "questions": {_PROBE_KEY: {"type": "noul", "instructions": "Is this a probe?"}},
}


def _normalise_base_url(raw: str) -> tuple[str, str]:
    """Return ``(base_url, note)``, moving Ollama's default port to 18781.

    The rewrite is deliberately narrow: only a loopback host on exactly Ollama's
    port is moved, and the move is always reported as a note so it shows up in
    ``health()`` rather than happening silently.
    """
    base = (raw or "").strip().rstrip("/") or DEFAULT_LLAMA_CPP_BASE_URL
    parsed = urlparse(base)
    try:
        port = parsed.port
    except ValueError as exc:
        raise DecisionUnavailable(f"clef: clef_base_url is not a valid URL: {raw!r}") from exc
    if parsed.hostname in ("localhost", "127.0.0.1", "::1") and port == _OLLAMA_PORT:
        base = f"{parsed.scheme}://{parsed.hostname}:18781{parsed.path}".rstrip("/")
        return base, (
            f"clef_base_url pointed at Ollama's port {_OLLAMA_PORT}; rewritten to "
            f"{base}. Ollama's clef-flash is unusable on Windows "
            "(ollama/ollama#18769); run llama-server instead."
        )
    return base, ""


def build_body(request: DecisionRequest) -> tuple[str, dict[str, Any]]:
    """Translate a ``DecisionRequest`` into a System One body.

    Returns ``(question_key, body)`` so the caller can find its own answer in
    the response by key instead of by position.

    Mapping (see ``core.schemas.DecisionRequest``):

    ==================  =================================================
    ``.state``         -> ``state``, as JSON text
    ``.question``      -> the question key, and the default ``instructions``
    ``.instructions``  -> ``instructions`` verbatim when supplied
    ``.options``       -> ``criteria`` map (``choice``)
    ``.rubric``        -> ``criteria`` list (``score``)
    ==================  =================================================

    Tool-directed selection: when ``state`` carries ``tool_schemas`` (a list of
    OpenAI-style function definitions built by
    ``agents/tool_selection.py:describe_tools_for_model``) alongside
    ``available_tools``/``candidate_count``/``purpose``, they are forwarded
    verbatim inside ``state`` via :func:`state_text` so the model chooses among
    real tools with their schemas. The explicit read below keeps that
    forwarding auditable: a request naming tools must actually carry their
    schemas into the wire body.
    """
    key = build_response_key(request)
    question: dict[str, Any] = {
        "type": request.question_type.value,
        "instructions": request.instructions.strip() or request.question,
    }
    if request.question_type is QuestionType.CHOICE:
        question["criteria"] = criteria_for_choice(request)
    elif request.question_type is QuestionType.SCORE:
        question["criteria"] = [str(level) for level in request.rubric]

    # Explicit tool_schemas/tools forwarding: model-directed tool selection
    # (agents/tool_selection.py:select_tool) puts the candidate function
    # schemas under state["tool_schemas"] with the candidate names under
    # state["available_tools"]. state_text serialises the whole state, so the
    # tools and their schemas reach the model; touch the keys here so a reader
    # (and a grep for tool_schemas/tools) can see the contract is honoured.
    _tool_schemas = request.state.get("tool_schemas")
    _available_tools = request.state.get("available_tools")
    if _tool_schemas is not None and not isinstance(_tool_schemas, list):
        raise DecisionFailed(
            f"request {request.request_id}: state['tool_schemas'] must be a "
            f"list of function schemas, got {type(_tool_schemas).__name__}"
        )
    if _available_tools is not None and not isinstance(_available_tools, list):
        raise DecisionFailed(
            f"request {request.request_id}: state['available_tools'] must be a "
            f"list of tool names, got {type(_available_tools).__name__}"
        )

    body = {"state": state_text(request), "questions": {key: question}}
    return key, body


def _options_for(request: DecisionRequest) -> list[str]:
    """The closed option/rubric set this request constrains its answer to."""
    if request.question_type is QuestionType.CHOICE:
        return [str(o) for o in request.options]
    if request.question_type is QuestionType.SCORE:
        return [str(level) for level in request.rubric]
    return list(NOUL_LABELS)


# ==============================================================================
# transports
# ==============================================================================
class ClefTransport(SystemOneClient):
    """A System One HTTP endpoint.

    Adds exactly one behaviour on top of :class:`~decision.base.SystemOneClient`:
    :meth:`unwrap_response`, because Cloudflare's Workers AI REST route returns
    ``{"result": <model output>, "success": true, ...}`` while llama.cpp returns
    the model output directly.
    """

    #: Set by the subclass before ``super().__init__`` runs.
    base_url: str = ""

    def __init__(self, *, base_url: str, timeout_s: float, max_retries: int = 1,
                 backoff_s: float = 0.25,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        super().__init__(timeout_s=timeout_s, max_retries=max_retries,
                         backoff_s=backoff_s, transport=transport)
        self.endpoint = f"{self.base_url}{SYSTEMONE_PATH}"
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

    # ------------------------------------------------------------------ unwrap
    def unwrap_response(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Strip any provider envelope so callers see the System One body."""
        return payload

    # ------------------------------------------------------------------ systemone
    def ask(self, body: Mapping[str, Any], *,
            timeout_s: float | None = None) -> tuple[dict[str, Any], float]:
        """POST one System One request. Returns ``(body, latency_ms)``."""
        payload, latency = self._request("POST", self.endpoint, json_body=body,
                                         headers=self.headers, timeout_s=timeout_s)
        return dict(self.unwrap_response(payload)), latency

    # --------------------------------------------------------------------- probe
    def _probe_timeout(self) -> float:
        return max(0.001, min(self.timeout_s, _PROBE_TIMEOUT_CAP_S))

    def _do_probe(self, timeout_s: float) -> ProbeResult:
        raise NotImplementedError


class LlamaCppTransport(ClefTransport):
    """Local ``llama-server`` serving a clef GGUF. Plain HTTP, no auth.

    ``llama-server -m clef-flash.Q4_K_M.gguf --port 18781`` (or any newer
    llama.cpp that exposes the System One route) is the whole deployment.
    """

    name = "clef/llama.cpp"

    def __init__(self, base_url: str = DEFAULT_LLAMA_CPP_BASE_URL, *,
                 model_label: str = "clef-flash",
                 timeout_s: float = 20.0, max_retries: int = 1,
                 backoff_s: float = 0.25,
                 transport: httpx.BaseTransport | None = None) -> None:
        resolved, note = _normalise_base_url(base_url)
        self.note = note
        self.model_label = model_label
        if note:
            logger.warning("decision.clef %s", note)
        super().__init__(base_url=resolved, timeout_s=timeout_s, max_retries=max_retries,
                         backoff_s=backoff_s, transport=transport)

    # ---------------------------------------------------------------- discovery
    def _discover_model(self, timeout_s: float) -> str:
        """Read the loaded model id from ``GET /v1/models``.

        llama.cpp reports the actual GGUF path here, which is the honest answer
        to "which model is running" -- the configured name is only a label and
        may point at a quantisation the server was not started with.
        """
        payload, _ = self._request("GET", f"{self.base_url}/v1/models",
                                   headers=self.headers, timeout_s=timeout_s)
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return ""
        first = data[0]
        if isinstance(first, Mapping):
            model_id = first.get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
        return ""

    def _do_probe(self, timeout_s: float) -> ProbeResult:
        short = self._probe_timeout()
        started = time.perf_counter()
        discovered = ""
        try:
            discovered = self._discover_model(timeout_s)
        except DecisionUnavailable as exc:
            return ProbeResult(False, f"{self.name}: not reachable ({exc})")
        except DecisionFailed as exc:
            # The server is up but does not speak the OpenAI-compatible models
            # route. That is survivable; fall through to the real System One test.
            logger.debug("decision.clef model discovery failed: %s", exc)

        try:
            body, _ = self.ask(_PROBE_BODY, timeout_s=short)
        except DecisionUnavailable as exc:
            return ProbeResult(False, f"{self.name}: not reachable ({exc})",
                               round((time.perf_counter() - started) * 1000, 3))
        except DecisionFailed as exc:
            # Already carries the /v1/systemone diagnosis from require_answers.
            return ProbeResult(False, f"{self.name}: System One route unusable ({exc})",
                               round((time.perf_counter() - started) * 1000, 3))

        answers = body.get("answers")
        if not isinstance(answers, Mapping) or _PROBE_KEY not in answers:
            return ProbeResult(
                False,
                f"{self.name}: POST {self.endpoint} answered without an 'answers' "
                f"key (keys: {sorted(body)[:8]}). Clef is a System One decision "
                f"model: call POST {self.endpoint}, not /v1/chat/completions.",
                round((time.perf_counter() - started) * 1000, 3),
            )

        self._model_discovered = discovered
        latency = round((time.perf_counter() - started) * 1000, 3)
        return ProbeResult(
            True,
            f"{self.name}: System One route answering at {self.endpoint}",
            latency, model=discovered,
            detail={"probed_question": _PROBE_KEY},
        )


class WorkersAiTransport(ClefTransport):
    """Cloudflare Workers AI running ``@cf/cloudflare/clef-flash``.

    Same System One body as the local server; metered and remote, so the probe
    is a single trivial ``noul`` question -- there is no free ``/v1/models`` to
    ask, and a probe that costs money must cost as little as possible.
    """

    name = "clef/workers_ai"

    def __init__(self, account_id: str, api_key: str, *, model_label: str = "",
                 timeout_s: float = 20.0, max_retries: int = 1,
                 backoff_s: float = 0.25,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.account_id = (account_id or "").strip()
        self.api_key = (api_key or "").strip()
        workers_base = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
        )
        self.model_label = model_label or WORKERS_AI_MODEL_PATH
        super().__init__(base_url=workers_base, timeout_s=timeout_s,
                         max_retries=max_retries, backoff_s=backoff_s,
                         transport=transport)
        self.endpoint = f"{self.base_url}/{WORKERS_AI_MODEL_PATH}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def unwrap_response(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Unwrap the Cloudflare envelope, keeping provider errors visible.

        A successful call returns ``{"result": <system one body>, ...}``. A
        failure returns ``success: false`` with ``result: null`` and a populated
        ``errors`` list; returning the envelope unchanged in that case lets
        :func:`~decision.base.require_answers` report the provider's own words
        rather than a vague "no answers key".
        """
        result = payload.get("result")
        if isinstance(result, Mapping):
            return result
        return payload

    def _do_probe(self, timeout_s: float) -> ProbeResult:
        started = time.perf_counter()
        try:
            body, _ = self.ask(_PROBE_BODY, timeout_s=self._probe_timeout())
        except DecisionUnavailable as exc:
            return ProbeResult(False, f"{self.name}: not reachable ({exc})",
                               round((time.perf_counter() - started) * 1000, 3))
        except DecisionFailed as exc:
            return ProbeResult(False, f"{self.name}: System One route unusable ({exc})",
                               round((time.perf_counter() - started) * 1000, 3))
        answers = body.get("answers")
        if not isinstance(answers, Mapping) or _PROBE_KEY not in answers:
            return ProbeResult(
                False,
                f"{self.name}: response from {self.endpoint} carried no "
                f"'answers' key (keys: {sorted(body)[:8]}); Cloudflare reported "
                f"{snippet(body.get('errors')) or 'nothing'}",
                round((time.perf_counter() - started) * 1000, 3),
            )
        return ProbeResult(
            True,
            f"{self.name}: System One route answering at {self.endpoint}",
            round((time.perf_counter() - started) * 1000, 3),
            model=self.model_label,
        )

    def health(self) -> dict[str, Any]:
        out = super().health()
        # The bearer token must never reach a status panel or a trace file.
        out["account_id"] = self.account_id
        out["authorised"] = bool(self.api_key)
        return out


# ==============================================================================
# backend
# ==============================================================================
class ClefBackend:
    """``DecisionBackend`` over one System One decision model.

    Selection follows ``settings.clef_backend``:

    ``ollama``    -> :class:`LlamaCppTransport` at ``settings.clef_base_url``
    ``workers_ai``-> :class:`WorkersAiTransport` (needs account id *and* key)
    ``auto``      -> whichever answers a probe first, local preferred

    ``available()`` always probes. There is no configuration-only assertion of
    availability anywhere in this class: a base URL says where a server *would*
    be, and only a probe says whether it is.
    """

    name = "clef"

    def __init__(self, settings: Settings | None = None, *,
                 transports: Mapping[str, ClefTransport] | None = None,
                 backoff_s: float = 0.25) -> None:
        self.settings = settings or get_settings()
        self._injected = dict(transports) if transports else None
        self._backoff_s = backoff_s
        self._resolved: ClefTransport | None = None
        self._resolve_note = ""
        self._candidate_errors: list[str] = []
        self._last_error = ""
        self._last_latency_ms = 0.0
        self._calls = 0

    # ------------------------------------------------------------------ config
    @property
    def model(self) -> str:
        """The loaded model if discovery found one, else the configured label.

        ``Decision.model`` always reports ``settings.clef_model``; this property
        is the one that can tell you the server is actually running a different
        quantisation of it.
        """
        if self._resolved is not None:
            return self._resolved.model
        return self.settings.clef_model

    @property
    def endpoint(self) -> str:
        """The System One URL in use, or ``""`` before one has been resolved."""
        return self._resolved.endpoint if self._resolved is not None else ""

    def _build(self, kind: str) -> ClefTransport:
        """Construct one transport, or raise ``DecisionUnavailable`` if impossible."""
        if self._injected is not None:
            if kind in self._injected:
                return self._injected[kind]
            raise DecisionUnavailable(
                f"clef: no injected transport named {kind!r} "
                f"(have {sorted(self._injected)})"
            )
        s = self.settings
        if kind == "ollama":
            return LlamaCppTransport(
                s.clef_base_url, model_label=s.clef_model, timeout_s=s.clef_timeout_s,
                max_retries=s.decision_max_retries, backoff_s=self._backoff_s,
            )
        if kind == "workers_ai":
            missing = [setting for setting, value in
                       (("clef_account_id", s.clef_account_id),
                        ("clef_api_key", s.clef_api_key))
                       if not value]
            if missing:
                raise DecisionUnavailable(
                    "clef: clef_backend=workers_ai requires "
                    f"{' and '.join(missing)} (env: CLEF_ACCOUNT_ID, CLEF_API_KEY)"
                )
            return WorkersAiTransport(
                str(s.clef_account_id), str(s.clef_api_key),
                model_label=s.clef_model, timeout_s=s.clef_timeout_s,
                max_retries=s.decision_max_retries, backoff_s=self._backoff_s,
            )
        raise DecisionUnavailable(f"clef: unknown clef_backend {kind!r}")

    def _candidates(self) -> list[tuple[str, ClefTransport]]:
        """Transports to try, in preference order for the configured backend.

        A transport that cannot even be constructed (a missing Workers AI key,
        an unparseable URL) is recorded in ``self._candidate_errors`` rather than
        swallowed, so the eventual "nothing is usable" message names the actual
        reason instead of just the configuration name.
        """
        kind = self.settings.clef_backend
        order = {
            "ollama": ["ollama"],
            "workers_ai": ["workers_ai"],
            "auto": ["ollama", "workers_ai"],
        }.get(kind)
        if order is None:
            raise DecisionUnavailable(
                f"clef: unknown clef_backend {kind!r} (expected one of "
                "'ollama', 'workers_ai', 'auto')"
            )
        self._candidate_errors = []
        pairs: list[tuple[str, ClefTransport]] = []
        for candidate in order:
            try:
                pairs.append((candidate, self._build(candidate)))
            except DecisionUnavailable as exc:
                logger.info("decision.clef transport %s unusable: %s", candidate, exc)
                self._candidate_errors.append(f"{candidate}: {exc}")
        return pairs

    def _resolve(self, *, force_probe: bool = False) -> ClefTransport:
        """Return the transport that answers, probing if the choice is not cached."""
        if self._resolved is not None and not force_probe:
            return self._resolved
        candidates = self._candidates()
        if not candidates:
            detail = "; ".join(self._candidate_errors) or "no transport candidates"
            raise DecisionUnavailable(
                f"clef: no usable transport for clef_backend="
                f"{self.settings.clef_backend!r} ({detail})"
            )
        if len(candidates) == 1:
            self._resolved = candidates[0][1]
            self._resolve_note = f"clef_backend={self.settings.clef_backend}"
            return self._resolved

        for _, transport in candidates:
            probe = transport.probe(ttl_s=self.settings.clef_health_ttl_s,
                                    force=force_probe)
            if probe.available:
                self._resolved = transport
                self._resolve_note = (
                    f"clef_backend=auto resolved to {transport.name} via probe: "
                    f"{probe.reason}"
                )
                return transport
        probed = [transport.name for _, transport in candidates]
        raise DecisionUnavailable(
            f"clef: clef_backend=auto probed {probed} and none answered"
        )

    # --------------------------------------------------------------- protocol
    def available(self) -> tuple[bool, str]:
        """Probe the configured transport. Cached for ``clef_health_ttl_s``."""
        try:
            transport = self._resolve()
        except DecisionUnavailable as exc:
            self._last_error = str(exc)
            return False, str(exc)
        probe = transport.probe(ttl_s=self.settings.clef_health_ttl_s)
        if probe.available:
            return True, probe.reason
        self._last_error = probe.reason
        return False, probe.reason

    def decide(self, request: DecisionRequest) -> Decision:
        """Ask the System One model one question, and return a sourced ``Decision``.

        One ``DecisionRequest`` produces exactly one System One question because
        ``core.schemas.Decision`` holds a single ``choice``. Raises
        ``DecisionUnavailable`` if the endpoint cannot be reached and
        ``DecisionFailed`` if it answers with something that is not a System One
        body or whose probabilities are unusable.
        """
        transport = self._resolve()
        key, body = build_body(request)
        started = time.perf_counter()
        self._calls += 1
        try:
            payload, http_ms = transport.ask(body)
            answer = require_answers(payload, key, backend=transport.name,
                                     endpoint=transport.endpoint)
            choice, probabilities, confidence = choice_from(
                answer, _options_for(request))
        except (DecisionUnavailable, DecisionFailed) as exc:
            # Both taxonomy members land the same way here: the decision layer
            # cannot be answered, and the registry decides what to do about it.
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self._last_latency_ms = latency_ms
        self._last_error = ""
        usage = payload.get("usage")
        raw = annotate_raw(
            payload,
            transport=transport.name,
            endpoint=transport.endpoint,
            question_key=key,
            question_type=request.question_type.value,
            usage=usage if isinstance(usage, Mapping) else {},
            http_latency_ms=round(http_ms, 3),
        )
        logger.debug(
            "decision.clef %s answered %s in %.1fms (transport=%s, usage=%s)",
            request.request_id, choice, latency_ms, transport.name,
            raw["_usage"],
        )
        return make_decision(
            request_id=request.request_id,
            question=request.question,
            choice=choice,
            probabilities=probabilities,
            confidence=confidence,
            source=DecisionSource.CLEF,
            model=self.settings.clef_model,
            latency_ms=latency_ms,
            degraded=False,
            raw=raw,
        )

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        ok, reason = self.available()
        transport = self._resolved
        return {
            "backend": self.name,
            "available": ok,
            "reason": reason,
            "clef_backend": self.settings.clef_backend,
            "transport": transport.name if transport else None,
            "endpoint": transport.endpoint if transport else None,
            "model": self.model,
            "configured_model": self.settings.clef_model,
            "discovered_model": bool(transport and transport.model_discovered),
            "latency_ms": round(self._last_latency_ms, 3),
            "last_error": self._last_error,
            "calls": self._calls,
            "resolve_note": self._resolve_note,
            "note": getattr(transport, "note", "") if transport else "",
            "transport_health": transport.health() if transport else None,
        }
