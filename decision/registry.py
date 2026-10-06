"""Backend selection, health, and graceful degradation.

This is the single entry point every agent uses: ``ctx.decide(request)``.

The chain
---------
``settings.effective_backend()`` picks the head; the remaining backends follow
in capability order, so a fallback is always the next most capable thing that is
configured:

* ``clef``  -> clef, gemini, rules
* ``gemini``-> gemini, clef, rules
* ``rules`` -> rules, clef, gemini
* ``auto``  -> resolves to clef when it is configured (it always is, for the
  local llama.cpp path, which is why availability is probed rather than assumed)

An explicit ``DECISION_BACKEND`` is honoured as the *head*, not as the whole
chain: setting ``DECISION_BACKEND=gemini`` means Gemini is asked first, and clef
is the second option rather than the rules engine, because clef is a better
answer than rules. It does not mean "never use clef" -- if Gemini is genuinely
unavailable for a run, degrading straight to the keyword heuristics would throw
away a decision model that was sitting right there.

The degradation contract
------------------------
When a backend is unreachable or returns something unusable, the registry falls
down the chain and:

* sets ``Decision.degraded = True``, because a fallback did answer;
* sets ``Decision.source`` to the backend that *actually* answered. Never to the
  one that was configured. Never to ``CLEF`` because clef was supposed to be
  running. A source that does not describe reality is worse than no source,
  because the trace is the evidence and the evidence is the deliverable;
* writes a structured log line naming the backend tried, the error, and the
  fallback used, so the substitution is greppable in a terminal and parsable
  from a log file;
* records the same provenance under ``raw["_degradation"]``.

If every backend in the chain fails -- which for a correct chain cannot happen,
since rules is unconditional -- the registry raises. It does not invent a
decision.

``RunMode.OFFLINE`` truncates the chain to rules alone: in offline mode a
network call is a bug, not a degradation, and paying a timeout to discover that
on every decision would be an expensive one.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from core.config import Settings, get_settings
from core.errors import DecisionFailed, DecisionUnavailable
from core.protocols import DecisionBackend
from core.schemas import Decision, DecisionRequest

from .base import annotate_raw, logger
from .clef import ClefBackend
from .gemini import GeminiBackend
from .rules import RulesBackend

__all__ = ["DecisionRegistry", "BackendStatus", "build_registry", "CHAIN_ORDER"]

#: Fallback order below the head of the chain, most capable first.
CHAIN_ORDER: tuple[str, ...] = ("clef", "gemini", "rules")


class BackendStatus(dict):
    """One backend's availability snapshot, shaped for the status panel.

    A ``dict`` subclass so it serialises straight into a trace event or a JSON
    status endpoint without a conversion step, with ``__getattr__`` so the call
    sites read as ``status.name`` rather than ``status["name"]``. Both styles
    work because they are the same data.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - attribute typos only
            raise AttributeError(item) from exc

    @classmethod
    def of(cls, backend: DecisionBackend, *, probe: bool = True) -> BackendStatus:
        if probe:
            ok, reason = backend.available()
        else:
            ok, reason = True, "not probed"
        return cls(
            name=backend.name,
            model=backend.model,
            available=bool(ok),
            reason=reason,
        )


class DecisionRegistry:
    """``DecisionBackend`` over an ordered chain, with honest provenance."""

    name = "registry"
    model = "chain"

    def __init__(self, settings: Settings | None = None, *,
                 clef: DecisionBackend | None = None,
                 gemini: DecisionBackend | None = None,
                 rules: DecisionBackend | None = None) -> None:
        self.settings = settings or get_settings()
        #: Injected backends win, which is how tests substitute fakes and how a
        #: future REPLAY backend slots in without touching the chain logic.
        self._backends: dict[str, DecisionBackend] = {
            "clef": clef if clef is not None else ClefBackend(self.settings),
            "gemini": gemini if gemini is not None else GeminiBackend(self.settings),
            "rules": rules if rules is not None else RulesBackend(self.settings),
        }
        self._calls = 0
        self._degradations = 0
        self._last_error = ""
        self._last_error_backend = ""
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ chain
    def chain(self) -> list[DecisionBackend]:
        """The backends to try, in order, for this configuration."""
        if self.settings.is_offline:
            # Offline means offline: probing a network backend would spend a
            # timeout per decision to learn what the run mode already says.
            return [self._backends["rules"]]
        head = self.settings.effective_backend()
        names = [head] + [name for name in CHAIN_ORDER if name != head]
        return [self._backends[name] for name in names if name in self._backends]

    # --------------------------------------------------------------- protocol
    def available(self) -> tuple[bool, str]:
        """Available if *any* backend in the chain is. Always true in practice.

        The rules backend is unconditional, so a registry is always usable; the
        reason string is what matters, because it names what is really standing
        behind the decisions.
        """
        available = [
            status["name"] for status in
            (BackendStatus.of(backend) for backend in self.chain())
            if status["available"]
        ]
        if available:
            return True, "chain available: " + ", ".join(available)
        return False, "no backend in the chain is available"

    def decide(self, request: DecisionRequest) -> Decision:
        """Answer ``request`` with the first backend in the chain that can.

        Raises :class:`~core.errors.DecisionUnavailable` only if every link in
        the chain failed, which with rules present means the rules backend itself
        refused the question -- which it does rather than guess.
        """
        self._calls += 1
        chain = self.chain()
        attempted: list[dict[str, Any]] = []
        last: Exception | None = None

        for position, backend in enumerate(chain):
            ok, reason = backend.available()
            if not ok:
                attempted.append({"backend": backend.name, "model": backend.model,
                                  "error": f"unavailable: {reason}"})
                self._record_failure(backend.name, reason, attempted)
                last = DecisionUnavailable(
                    f"{backend.name}: {reason}"
                )
                continue
            try:
                decision = backend.decide(request)
            except (DecisionUnavailable, DecisionFailed) as exc:
                attempted.append({
                    "backend": backend.name, "model": backend.model,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                self._record_failure(backend.name, str(exc), attempted)
                last = exc
                continue

            if position > 0:
                # A fallback answered. Say so, on the Decision itself and in the
                # log, and leave source exactly as the answering backend set it.
                decision.degraded = True
                decision.raw = annotate_raw(
                    decision.raw,
                    degradation={
                        "configured_head": self.settings.effective_backend(),
                        "tried": attempted,
                        "answered_by": decision.source.value,
                        "answered_by_model": decision.model,
                    },
                )
                self._degradations += 1
                self._log_degradation(
                    request=request, attempted=attempted,
                    backend=backend, decision=decision,
                )
            self._last_error = ""
            self._last_error_backend = ""
            return decision

        self._last_error = str(last) if last else "empty decision chain"
        self._last_error_backend = chain[-1].name if chain else ""
        raise DecisionUnavailable(
            f"decision {request.request_id}: every backend in the chain "
            f"{[b.name for b in chain]} failed; last error: {self._last_error}"
        )

    # --------------------------------------------------------------- logging
    def _record_failure(self, backend_name: str, error: str,
                        attempted: Sequence[Mapping[str, Any]]) -> None:
        self._last_error = error
        self._last_error_backend = backend_name
        self._history.append({
            "at": round(time.time(), 3),
            "backend": backend_name,
            "error": error,
            "tried_so_far": [dict(a) for a in attempted],
        })
        del self._history[:-50]

    def _log_degradation(self, *, request: DecisionRequest,
                         attempted: Sequence[Mapping[str, Any]],
                         backend: DecisionBackend, decision: Decision) -> None:
        """One greppable, parsable line per degradation.

        The payload is JSON-encoded into the message *and* attached as
        ``extra``, so it survives a structured handler and remains readable in a
        plain terminal. A degradation that only exists in memory is a degradation
        nobody can audit.
        """
        payload = {
            "event": "decision_degraded",
            "request_id": request.request_id,
            "decision_point": request.decision_point,
            "question": request.question,
            "configured_head": self.settings.effective_backend(),
            "tried": [dict(a) for a in attempted],
            "fallback_used": backend.name,
            "source": decision.source.value,
            "model": decision.model,
            "confidence": round(decision.confidence, 4),
            "confidence_threshold": self.settings.confidence_threshold,
            "needs_escalation": decision.needs_escalation,
        }
        logger.warning("decision_degraded %s",
                       json.dumps(payload, sort_keys=True, default=str),
                       extra={"decision_degradation": payload})
        logger.debug("decision registry fell back to %s for %s",
                     backend.name, request.request_id)

    # ---------------------------------------------------------------- health
    def health(self, *, probe: bool = True) -> dict[str, Any]:
        """Aggregate every backend for the demo's status panel.

        ``probe=False`` reports configuration and cached state without spending
        a network call, which is what a trace-recorded status line wants.
        """
        chain = self.chain()
        chain_names = {id(b) for b in chain}
        statuses: list[BackendStatus] = []
        for key in CHAIN_ORDER:
            backend = self._backends.get(key)
            if backend is None:
                continue
            status = BackendStatus.of(backend, probe=probe)
            status["in_chain"] = id(backend) in chain_names
            status["configured"] = (
                key == self.settings.effective_backend() and not self.settings.is_offline
            )
            statuses.append(status)

        return {
            "backend": self.name,
            "configured_backend": self.settings.decision_backend,
            "effective_backend": self.settings.effective_backend(),
            "run_mode": self.settings.run_mode.value,
            "offline": self.settings.is_offline,
            "confidence_threshold": self.settings.confidence_threshold,
            "escalation_threshold": self.settings.escalation_threshold,
            "chain": [b.name for b in chain],
            "backends": [dict(s) for s in statuses],
            "calls": self._calls,
            "degradations": self._degradations,
            "last_error": self._last_error,
            "last_error_backend": self._last_error_backend,
            "settings": self.settings.redacted(),
        }

    def describe(self) -> dict[str, Any]:
        """The chain and its availability, in the shape a demo panel prints.

        Separate from :meth:`health` because ``describe`` is about *shape* (what
        would be tried, in what order) while ``health`` is about *state*
        (latency, errors, counters). A status panel wants both; this returns the
        first without dragging the second along.
        """
        chain = self.chain()
        described: list[dict[str, Any]] = []
        for position, backend in enumerate(chain):
            ok, reason = backend.available()
            described.append({
                "position": position,
                "name": backend.name,
                "model": backend.model,
                "role": ("primary" if position == 0 else "fallback"),
                "available": bool(ok),
                "reason": reason,
            })
        return {
            "configured": self.settings.decision_backend,
            "effective": self.settings.effective_backend(),
            "order": [b.name for b in chain],
            "chain": described,
            "confidence_threshold": self.settings.confidence_threshold,
            "escalation_threshold": self.settings.escalation_threshold,
            "run_mode": self.settings.run_mode.value,
            "first_available": next(
                (d["name"] for d in described if d["available"]), None
            ),
        }


def build_registry(settings: Settings | None = None, **kwargs: Any) -> DecisionRegistry:
    """Construct the process-wide decision registry.

    Mirrors ``tools.registry.build_registry()``: the wiring layer asks for one
    call and gets a fully constructed chain, so no agent ever constructs a
    backend itself.
    """
    return DecisionRegistry(settings or get_settings(), **kwargs)
