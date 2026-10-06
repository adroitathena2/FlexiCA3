"""Lazy, fault-tolerant wiring for the Paytriq API.

Why this module exists
----------------------
``core.protocols`` fixes the *dependency direction* of the project::

    core -> nothing
    observability, decision -> core
    blackboard, tools -> core, observability
    agents -> core, decision, blackboard, tools, observability
    graph -> agents
    api -> graph, agents' collaborators

Eight packages are being written at once. The API is the only one that touches
every other package, so it is the only one that will regularly be asked to import
a sibling that does not exist yet, is half-written, or raises on import. If
``api`` imported ``graph`` at module scope it would fail to start -- and a
backend that will not start cannot demonstrate anything.

So every dependency here is built **inside a function**, behind a
:class:`_Lazy` holder that caches one of two things:

* the constructed instance, permanently; or
* the failure reason, plus the moment it was observed.

Both are reported rather than hidden. A handler that needs a subsystem and
cannot have it raises :class:`SubsystemUnavailable`, which the route layer turns
into ``503 {"error": ..., "unavailable": true, "reason": ...}`` -- never into a
plausible-looking success.

Failures are re-probed, not cached forever
------------------------------------------
A sibling package that is broken *right now* will very plausibly be working in
two minutes, and a long-lived ``uvicorn --reload`` process should notice. So a
cached failure expires after :data:`_RETRY_AFTER_S` and the factory is retried.
A success is cached for the life of the process. The retry interval is a
compromise: long enough that a genuinely broken dependency is not hammered, short
enough that the demo does not need a manual restart when a file lands.

Per-run context
---------------
``RunContext`` bundles the collaborators that belong to *one* run: its
blackboard, its tracer, and its gate ledger. Per-run rather than process-wide for
two reasons that are not stylistic:

1. ``graph.build.collect_board_delta`` advances a high-water sequence mark keyed
   by thread id, so two concurrent runs sharing a board would project each
   other's entries into their own state.
2. ``OtelTracer.configure(run_id)`` points the exporter at
   ``traces/<run_id>.jsonl``. One tracer means one open trace file, and a run's
   evidence living in another run's file would be worse than no evidence.

The module-level singletons (``get_board``, ``get_tracer``) therefore return the
collaborators of the *current* run, which is what ``/health`` and the demo panel
want; the run-scoped accessors (:func:`get_run`, :meth:`RunContext.board`) are
what the route handlers use.

The gate ledger
---------------
:class:`GateLedger` is the API's own record of which side-effecting actions a
human has authorised. It exists because the single most important property of
this system is that **no send, counter-offer or MoU release happens without a
matching approval row**, and a property enforced only inside the graph is a
property that disappears the moment a handler takes a shortcut. Every gated
endpoint asks the ledger first, so the refusal is decided by the same table the
approval was written to.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import Settings
from core.errors import PaytriqError
from core.ids import new_id, utcnow
from core.ids import run_id as new_run_id
from core.protocols import Blackboard, Tool, Tracer
from core.schemas import (
    Approval,
    DecisionSource,
    GateKind,
    GateOutcome,
    HumanDecision,
    HumanGate,
    TraceEvent,
    TraceKind,
)

__all__ = [
    "SubsystemUnavailable",
    "GateLedger",
    "RunContext",
    "get_settings",
    "set_settings",
    "get_board",
    "get_tracer",
    "get_decision_registry",
    "get_tool_registry",
    "get_graph_runner",
    "get_run",
    "current_run",
    "list_runs",
    "set_current_run",
    "health_payload",
    "reset_deps",
    "unavailable_payload",
]

log = logging.getLogger("paytriq.api.deps")

#: Seconds a cached construction failure is honoured before the factory is
#: retried. See the module docstring for why failures expire at all.
_RETRY_AFTER_S = 2.0

#: The three ``DecisionSource`` values that are not models the API itself
#: produced. Anything reported as a decision source is one of these four.
_KNOWN_DECISION_SOURCES = tuple(s.value for s in DecisionSource)


class SubsystemUnavailable(PaytriqError):
    """A dependency the handler needs does not exist or could not be built.

    Carries the *reason* and the module that failed, because "503" alone tells a
    reader nothing about whether the system is broken or merely unfinished.
    """

    def __init__(self, name: str, reason: str, module: str = "") -> None:
        self.name = name
        self.reason = reason
        self.module = module
        detail = f"{name} is unavailable: {reason}"
        if module:
            detail += f" (module {module})"
        super().__init__(detail)


def unavailable_payload(exc: SubsystemUnavailable | str, *, where: str = "") -> dict[str, Any]:
    """The one error shape this API uses for a missing subsystem.

    ``unavailable: true`` is the machine-readable flag; ``reason`` is the human
    one. ``where`` names the endpoint that wanted the subsystem so a reader of a
    log can tell which feature is dark.
    """
    if isinstance(exc, SubsystemUnavailable):
        reason, name, module = exc.reason, exc.name, exc.module
    else:
        reason, name, module = str(exc), "subsystem", ""
    payload: dict[str, Any] = {
        "error": reason,
        "unavailable": True,
        "reason": reason,
        "subsystem": name,
    }
    if module:
        payload["module"] = module
    if where:
        payload["where"] = where
    return payload


# ============================================================================== lazy
class _Lazy:
    """Thread-safe lazy singleton that caches success *or* failure.

    Deliberately not ``functools.lru_cache``: that caches the return value but
    has nowhere to record *why* construction failed, which is the only thing the
    status panel and the 503 body need.
    """

    __slots__ = ("name", "module", "factory", "retry_after_s",
                 "_lock", "_value", "_error", "_failed_at")

    def __init__(self, name: str, module: str, factory: Callable[[], Any], *,
                 retry_after_s: float = _RETRY_AFTER_S) -> None:
        self.name = name
        self.module = module
        self.factory = factory
        self.retry_after_s = retry_after_s
        self._lock = threading.Lock()
        self._value: Any = None
        self._error: str = ""
        self._failed_at: float = 0.0

    def _build(self) -> tuple[Any, str]:
        try:
            return self.factory(), ""
        except SubsystemUnavailable as exc:
            return None, exc.reason
        except Exception as exc:  # noqa: BLE001 - the point is to report, not crash
            # Any exception type is acceptable here: a half-written sibling can
            # fail with SyntaxError, TypeError or AttributeError as easily as
            # ImportError, and "it raised X" is the diagnostic. Nothing is
            # swallowed -- it is logged and carried in the 503 body.
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("subsystem %s unavailable (%s): %s", self.name, self.module, reason)
            return None, reason

    def get(self) -> Any:
        """The instance, or raise :class:`SubsystemUnavailable`."""
        with self._lock:
            if self._value is not None:
                return self._value
            if self._error and (time.monotonic() - self._failed_at) < self.retry_after_s:
                raise SubsystemUnavailable(self.name, self._error, self.module)
            value, error = self._build()
            self._value, self._error, self._failed_at = value, error, time.monotonic()
            if value is None:
                raise SubsystemUnavailable(self.name, error or "unknown failure", self.module)
            return value

    def peek(self) -> Any:
        """The instance if it is already built, else ``None``. Never raises."""
        with self._lock:
            return self._value

    def state(self, *, probe: bool = True) -> dict[str, Any]:
        """Availability record for the status panel.

        Probing here rather than reading a cached flag is what lets ``/health``
        report a sibling that has been fixed since the last request. The probe
        result is cached by :meth:`get`, so this is not a per-request rebuild.
        """
        if probe:
            try:
                self.get()
            except SubsystemUnavailable:
                pass  # the reason below is the point of this method
        with self._lock:
            built, error, at = self._value is not None, self._error, self._failed_at
        if built:
            return {"available": True, "module": self.module, "reason": "",
                    "cached": True, "retry_in_s": 0.0}
        age = round(time.monotonic() - at, 2) if error else 0.0
        return {
            "available": False,
            "module": self.module,
            "reason": error or "not probed",
            "retry_in_s": (max(0.0, round(self.retry_after_s - age, 2))
                           if error else 0.0),
        }

    def reset(self) -> None:
        with self._lock:
            self._value, self._error, self._failed_at = None, "", 0.0


# ============================================================================== gates
class GateLedger:
    """Which side-effecting actions a human has authorised, and on what instruction.

    Three actions require an approval record, and only three, matching
    :class:`~core.schemas.GateKind`:

    ``SEND``     before any outbound mail
    ``COUNTER``  before answering a counter-offer
    ``MOU``      before releasing an MoU

    ``ESCALATION`` is also a gate but it authorises nothing irreversible, so it
    is recorded the same way and never satisfies :meth:`authorises`.

    ``authorises`` is the single choke point every gated handler calls. The
    check is by ``(gate_id, kind)``, not by gate id alone: an approval to send
    mail must not release a contract, because a request that can name any gate id
    is otherwise a way to smuggle one authorisation into another action.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gates: dict[str, HumanGate] = {}
        self._decisions: dict[str, HumanDecision] = {}
        self._approvals: dict[str, Approval] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------- write
    def raise_gate(self, kind: GateKind, *, event_id: str, run_id: str,
                   question: str, payload_preview: str = "",
                   options: Sequence[GateOutcome] | None = None,
                   raised_at: Any | None = None) -> HumanGate:
        """Record a pending approval request. Recording is not approving."""
        gate = HumanGate(
            gate_id=new_id("gat"),
            kind=kind,
            event_id=event_id,
            run_id=run_id,
            question=question,
            payload_preview=payload_preview,
            options=list(options or (GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE)),
        )
        if raised_at is not None:
            gate = gate.model_copy(update={"raised_at": raised_at})
        with self._lock:
            self._gates[gate.gate_id] = gate
            self._order.append(gate.gate_id)
        return gate

    def record(self, gate: HumanGate, decision: HumanDecision,
               approval: Approval) -> None:
        """Store an answered gate. ``approval`` is required -- never optional.

        Making the approval a required positional argument is the enforcement:
        there is no signature by which a decision can be recorded without the
        durable row that proves the side effect was authorised.
        """
        if approval.gate_id != gate.gate_id or decision.gate_id != gate.gate_id:
            raise PaytriqError(
                f"refusing to record a mismatched gate triple: gate={gate.gate_id} "
                f"decision={decision.gate_id} approval={approval.gate_id}"
            )
        with self._lock:
            self._gates.setdefault(gate.gate_id, gate)
            self._decisions[decision.gate_id] = decision
            self._approvals[approval.gate_id] = approval
            if gate.gate_id not in self._order:
                self._order.append(gate.gate_id)

    # -------------------------------------------------------------------- read
    def gate(self, gate_id: str) -> HumanGate | None:
        with self._lock:
            return self._gates.get(gate_id)

    def decision(self, gate_id: str) -> HumanDecision | None:
        with self._lock:
            return self._decisions.get(gate_id)

    def approval(self, gate_id: str) -> Approval | None:
        with self._lock:
            return self._approvals.get(gate_id)

    def gates(self, *, run_id: str = "") -> list[HumanGate]:
        with self._lock:
            rows = [self._gates[g] for g in self._order if g in self._gates]
        return [g for g in rows if not run_id or g.run_id == run_id]

    def outstanding(self, *, run_id: str = "") -> list[HumanGate]:
        """Gates raised and not yet answered."""
        with self._lock:
            answered = set(self._decisions)
            rows = [self._gates[g] for g in self._order
                    if g in self._gates and g not in answered]
        return [g for g in rows if not run_id or g.run_id == run_id]

    def approvals(self, *, run_id: str = "") -> list[Approval]:
        with self._lock:
            rows = [self._approvals[g] for g in self._order if g in self._approvals]
        return [a for a in rows if not run_id or self._gates.get(a.gate_id) is None
                or self._gates[a.gate_id].run_id == run_id]

    def decisions(self, *, run_id: str = "") -> list[HumanDecision]:
        with self._lock:
            rows = [self._decisions[g] for g in self._order if g in self._decisions]
        return [d for d in rows if not run_id or self._gates.get(d.gate_id) is None
                or self._gates[d.gate_id].run_id == run_id]

    # -------------------------------------------------------------- the check
    def authorises(self, kind: GateKind, gate_id: str) -> tuple[bool, str]:
        """``(permitted, reason)`` for one side effect of one kind.

        ``permitted`` is true only for an explicit ``APPROVE`` on a gate of
        exactly ``kind``. ``REVISE`` and ``REJECT`` never authorise the effect
        they were asked about -- a revision instruction is an instruction about
        *what* to send, not permission to send it.
        """
        with self._lock:
            gate = self._gates.get(gate_id)
            approval = self._approvals.get(gate_id)
        if gate is None:
            return False, (
                f"no approval record exists for gate {gate_id!r}; nothing was "
                f"authorised. Raise a gate and answer it with POST /api/gates/{{gate_id}}"
            )
        if approval is None:
            return False, (
                f"gate {gate_id} ({gate.kind.value}) is still awaiting a decision; "
                f"nothing was authorised"
            )
        if approval.kind is not kind:
            return False, (
                f"approval {approval.approval_id} is for a {approval.kind.value} "
                f"gate, not a {kind.value} gate; one approval never authorises "
                f"two different side effects"
            )
        if approval.outcome is not GateOutcome.APPROVE:
            return False, (
                f"gate {gate_id} was answered {approval.outcome.value} by "
                f"{approval.decided_by}; only 'approve' authorises a "
                f"{kind.value}"
            )
        return True, (
            f"approved by {approval.decided_by} at {approval.at.isoformat()} "
            f"({approval.approval_id})"
        )

    def summary(self, *, run_id: str = "") -> dict[str, Any]:
        """Counts for the demo panel, with human and auto kept apart."""
        approvals = self.approvals(run_id=run_id)
        human = [a for a in approvals if str(a.decided_by).strip().lower() != "auto"]
        auto = [a for a in approvals if str(a.decided_by).strip().lower() == "auto"]
        by_kind: dict[str, int] = {}
        for row in approvals:
            by_kind[row.kind.value] = by_kind.get(row.kind.value, 0) + 1
        return {
            "run_id": run_id or "(all)",
            "raised": len(self.gates(run_id=run_id)),
            "outstanding": len(self.outstanding(run_id=run_id)),
            "approvals": len(approvals),
            "human_approvals": len(human),
            "auto_approvals": len(auto),
            "by_kind": by_kind,
        }


# ============================================================================ runs
@dataclass
class RunContext:
    """Everything that belongs to one run: board, tracer, gates, last state.

    The board and the tracer are per-run on purpose -- see the module docstring.
    ``finished`` is the flag the SSE loop watches to know it may emit its
    terminal ``summary`` event.
    """

    run_id: str
    event_id: str = ""
    settings: Settings | None = None
    board: Blackboard | None = None
    tracer: Any | None = None
    gates: GateLedger = field(default_factory=GateLedger)
    #: LangGraph thread this run resumes on. Defaults to ``run_id`` on first
    #: use so a gate retry continues the parked run instead of forking a new
    #: thread per stage call.
    thread_id: str = ""
    board_error: str = ""
    tracer_error: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    degraded: bool = False
    notes: list[str] = field(default_factory=list)
    finished: bool = False
    created_at: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------ events
    def events(self) -> list[TraceEvent]:
        """Trace events recorded so far, or ``[]`` when there is no tracer."""
        if self.tracer is None:
            return []
        getter = getattr(self.tracer, "events", None)
        if isinstance(getter, list):
            return list(getter)
        return []

    def trace_path(self) -> str | None:
        if self.tracer is None:
            return None
        path = getattr(self.tracer, "trace_path", None)
        return str(path) if path is not None else None

    def board_high_water(self) -> int:
        if self.board is None:
            return 0
        try:
            history = self.board.history()
        except (AttributeError, TypeError):
            return 0
        return max((int(getattr(e, "seq", 0)) for e in history), default=0)

    def note(self, message: str) -> str:
        with self.lock:
            self.notes.append(f"[{utcnow().isoformat()}] {message}")
        log.info("run %s: %s", self.run_id, message)
        return message

    def emit(self, kind: TraceKind, name: str, **attrs: Any) -> TraceEvent | None:
        """Best-effort trace emission; a tracer fault must not fail a request."""
        if self.tracer is None:
            return None
        try:
            return self.tracer.event(kind, name, **attrs)
        except Exception as exc:  # noqa: BLE001 - durability beats tracing
            log.warning("tracer rejected event %r: %s: %s", name, type(exc).__name__, exc)
            return None

    def finish(self) -> dict[str, Any]:
        """Flush this run's trace. Safe to call twice; never raises."""
        if self.tracer is None:
            return {"run_id": self.run_id, "finished": False,
                    "reason": self.tracer_error or "no tracer was built"}
        with self.lock:
            if self.finished:
                return {"run_id": self.run_id, "finished": True, "already_finished": True}
            self.finished = True
        try:
            result = self.tracer.finish()
        except Exception as exc:  # noqa: BLE001 - shutdown must complete
            log.error("tracer.finish() failed for run %s: %s: %s",
                      self.run_id, type(exc).__name__, exc)
            return {"run_id": self.run_id, "finished": False,
                    "reason": f"{type(exc).__name__}: {exc}"}
        return {"run_id": self.run_id, "finished": True, **(result or {})}

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "status": self.status,
            "degraded": self.degraded,
            "age_s": round(time.monotonic() - self.created_at, 2),
            "board_entries": self.board_high_water(),
            "trace_events": len(self.events()),
            "trace_file": self.trace_path(),
            "gates": self.gates.summary(run_id=self.run_id),
            "notes": list(self.notes[-8:]),
            "board_error": self.board_error,
            "tracer_error": self.tracer_error,
        }


_RUNS: dict[str, RunContext] = {}
_RUN_BY_EVENT: dict[str, str] = {}
_CURRENT: dict[str, str] = {"run_id": ""}
_RUNS_LOCK = threading.RLock()


def get_run(run_id: str) -> RunContext | None:
    """An existing run context, or ``None``. Never creates one."""
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def current_run() -> RunContext | None:
    with _RUNS_LOCK:
        return _RUNS.get(_CURRENT["run_id"]) if _CURRENT["run_id"] else None


def set_current_run(run_id: str) -> None:
    with _RUNS_LOCK:
        _CURRENT["run_id"] = run_id


def list_runs() -> list[dict[str, Any]]:
    with _RUNS_LOCK:
        contexts = list(_RUNS.values())
    return [c.describe() for c in sorted(contexts, key=lambda c: c.created_at)]


def iter_runs() -> list[RunContext]:
    """Every live run context, oldest first.

    Public counterpart of the private lookup in :mod:`api.routes_gates`. Exists
    because boards and tracers are per run, so "read everything on the board"
    means "read every run's board" -- and a handler that forgets that reads only
    the most recent run and then reports it as everything.
    """
    with _RUNS_LOCK:
        contexts = list(_RUNS.values())
    return sorted(contexts, key=lambda c: c.created_at)


def create_run(*, event_id: str = "", settings: Settings | None = None,
               run: str | None = None) -> RunContext:
    """Build (or rebuild) the context for one run.

    **The board and the tracer are constructed fresh per run**, not shared. Three
    things break if they are shared, and all three were caught by running the
    demo rather than by reading the code:

    1. ``OtelTracer.configure(run_id)`` re-points one tracer at one file, so a
       second run silently takes over the first run's exporter. The first run then
       has a trace file with no summary sidecar, and
       ``/api/runs/{first_run}/trace`` answers with the second run's events.
    2. ``graph.build.collect_board_delta`` advances a high-water sequence mark
       keyed by thread id, so two runs sharing a board project each other's
       entries into their own state.
    3. ``finish()`` is idempotent on one tracer, so only the last run to finish
       gets a summary sidecar.

    Board and tracer construction is individually guarded: a run with a board but
    no tracer is still useful for ``/board``, and vice versa, so one missing
    collaborator must not make the run unusable. Each failure is recorded on the
    context and surfaced by ``/health`` rather than thrown.
    """
    cfg = settings or get_settings()
    rid = run or new_run_id()
    context = RunContext(run_id=rid, event_id=event_id, settings=cfg)

    # The import probes raise first if the sibling package is absent at all;
    # otherwise a fresh instance is built for this run.
    try:
        _board_singleton().get()
        context.board = _build_board()
    except SubsystemUnavailable as exc:
        context.board_error = exc.reason
        context.degraded = True
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        context.board_error = f"{type(exc).__name__}: {exc}"
        context.degraded = True

    try:
        _tracer_singleton(cfg).get()
        context.tracer = _build_tracer(cfg)
    except SubsystemUnavailable as exc:
        context.tracer_error = exc.reason
        context.degraded = True
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        context.tracer_error = f"{type(exc).__name__}: {exc}"
        context.degraded = True

    if context.tracer is not None:
        try:
            context.tracer.configure(rid, service=cfg.app_name, event_id=event_id)
        except Exception as exc:  # noqa: BLE001 - a tracer that will not configure
            context.tracer_error = f"{type(exc).__name__}: {exc}"
            context.tracer = None
            context.degraded = True
        else:
            if event_id:
                binder = getattr(context.tracer, "bind_event", None)
                if callable(binder):
                    try:
                        binder(event_id)
                    except Exception as exc:  # noqa: BLE001
                        context.note(f"tracer.bind_event failed: "
                                     f"{type(exc).__name__}: {exc}")

    if context.board is not None and event_id:
        context.note(f"run created for event {event_id}")

    with _RUNS_LOCK:
        _RUNS[rid] = context
        _CURRENT["run_id"] = rid
        if event_id:
            _RUN_BY_EVENT[event_id] = rid
    return context


def run_for_event(event_id: str, *, settings: Settings | None = None,
                  create: bool = True) -> RunContext | None:
    """The run context for an event, creating one on first reference.

    An event and its run are created together: the pipeline endpoints are all
    ``/api/events/{id}/<stage>``, so resolving an event has to resolve the run
    that operates on it. Refusing here would mean every endpoint had to grow its
    own "create a run if needed" branch.
    """
    with _RUNS_LOCK:
        rid = _RUN_BY_EVENT.get(event_id)
        existing = _RUNS.get(rid) if rid else None
    if existing is not None:
        return existing
    if not create:
        return None
    return create_run(event_id=event_id, settings=settings)


def _reset_runs(finish: bool = True) -> None:
    with _RUNS_LOCK:
        contexts = list(_RUNS.values())
        _RUNS.clear()
        _RUN_BY_EVENT.clear()
        _CURRENT["run_id"] = ""
    if finish:
        for context in contexts:
            context.finish()


# ========================================================================== settings
_SETTINGS_LOCK = threading.Lock()
_SETTINGS_OVERRIDE: Settings | None = None


def set_settings(settings: Settings | None) -> None:
    """Install a process-wide settings override (used by ``create_app``)."""
    global _SETTINGS_OVERRIDE
    with _SETTINGS_LOCK:
        _SETTINGS_OVERRIDE = settings


def get_settings() -> Settings:
    """The settings this process runs on.

    Named identically to :func:`core.config.get_settings` on purpose: the wiring
    layer has exactly one settings object and two spellings of "the current one"
    in a codebase is a bug waiting to happen.
    """
    with _SETTINGS_LOCK:
        if _SETTINGS_OVERRIDE is not None:
            return _SETTINGS_OVERRIDE
    from core.config import get_settings as core_get_settings

    return core_get_settings()


# ======================================================================= the lazy
_LAZIES: dict[str, _Lazy] = {}


def _lazy(key: str, name: str, module: str, factory: Callable[[], Any]) -> _Lazy:
    """Register a lazy singleton on first use and return it."""
    existing = _LAZIES.get(key)
    if existing is None:
        existing = _Lazy(name, module, factory)
        _LAZIES[key] = existing
    return existing


def _board_singleton() -> _Lazy:
    """Import probe for ``blackboard``.

    Probes importability; the *instance* is per run (see :func:`create_run`). A
    board shared across runs would let ``graph.build.collect_board_delta`` project
    one run's entries into another run's state, because its high-water mark is
    keyed by thread id rather than by board.
    """
    return _lazy("board", "blackboard", "blackboard", _build_board)


def _build_board() -> Blackboard:
    from blackboard import InMemoryBlackboard

    return InMemoryBlackboard()


def _tracer_singleton(cfg: Settings | None = None) -> _Lazy:
    """Import probe for ``observability``; the tracer *instance* is per run.

    Keyed on the settings' traces dir so a test that changes ``traces_dir`` gets a
    fresh probe rather than one reporting the previous directory's status.
    """
    resolved = cfg or get_settings()
    key = f"tracer::{getattr(resolved, 'traces_dir', '')}"
    return _lazy(key, "tracer", "observability", lambda: _build_tracer(resolved))


def _build_tracer(cfg: Settings) -> Tracer:
    from observability import OtelTracer

    return OtelTracer(cfg)


def _decision_singleton(cfg: Settings) -> _Lazy:
    key = f"decision::{getattr(cfg, 'decision_backend', '')}"
    return _lazy(key, "decision", "decision", lambda: _build_decision(cfg))


def _build_decision(cfg: Settings) -> Any:
    from decision.registry import build_registry

    return build_registry(cfg)


def _tools_singleton(cfg: Settings) -> _Lazy:
    key = f"tools::{getattr(cfg, 'tools_live', False)}::{getattr(cfg, 'tools_enabled', True)}"
    return _lazy(key, "tools", "tools", lambda: _build_tools(cfg))


def _build_tools(cfg: Settings) -> dict[str, Tool]:
    from tools.registry import build_registry

    return build_registry(cfg)


def _graph_singleton() -> _Lazy:
    return _lazy("graph", "graph", "graph", _build_graph_probe)


def _build_graph_probe() -> Any:
    """Import-check ``graph.build`` without compiling anything.

    Compiling a LangGraph is expensive and needs a checkpointer, so the lazy
    singleton for the graph holds a *probe* (the module object) and the runner
    itself is built per run in :func:`get_graph_runner`. Health therefore reports
    "the graph package imports" rather than "a graph was compiled", which is the
    question ``/health`` is actually asked.
    """
    import importlib

    return importlib.import_module("graph.build")


# ===================================================================== public API
def get_board() -> Blackboard:
    """The blackboard of the current run.

    Bootstraps a run if none exists, so ``/health`` and the demo work against a
    bare process with no requests made yet.
    """
    with _RUNS_LOCK:
        context = _RUNS.get(_CURRENT["run_id"]) if _CURRENT["run_id"] else None
    if context is None:
        context = create_run()
    if context.board is None:
        raise SubsystemUnavailable(
            "board", context.board_error or "no board was built", "blackboard")
    return context.board


def get_tracer() -> Tracer:
    """The tracer of the current run (see :func:`get_board`)."""
    with _RUNS_LOCK:
        context = _RUNS.get(_CURRENT["run_id"]) if _CURRENT["run_id"] else None
    if context is None:
        context = create_run()
    if context.tracer is None:
        raise SubsystemUnavailable(
            "tracer", context.tracer_error or "no tracer was built", "observability")
    return context.tracer


def get_decision_registry() -> Any:
    """The decision chain (``clef`` -> ``gemini`` -> ``rules``).

    A registry is available even when both models are not -- the rules backend
    is unconditional -- so "available" here means "there is something that can
    answer", and the reason string names what is really behind the answers.
    """
    return _decision_singleton(get_settings()).get()


def get_tool_registry() -> dict[str, Tool]:
    """name -> tool instance, ordered by ``tools.registry.TOOL_ORDER``."""
    return _tools_singleton(get_settings()).get()


def get_graph_runner(context: RunContext | None = None, *,
                     interactive: bool | None = None,
                     thread_id: str | None = None) -> Any:
    """Compile a :class:`~graph.build.PaytriqGraph` for one run.

    Raises :class:`SubsystemUnavailable` when ``graph.build`` will not import, so
    the caller answers 503 with the real reason rather than pretending to have
    run a pipeline.
    """
    _graph_singleton().get()  # raises SubsystemUnavailable with the import error
    ctx = context or current_run()
    if ctx is None:
        ctx = create_run()
    cfg = ctx.settings or get_settings()
    from graph.build import build_graph  # re-import after the probe succeeded

    tools: dict[str, Tool] = {}
    tools_note = "tools registry unavailable; agents run with no tools"
    try:
        tools = get_tool_registry()
        tools_note = f"{len(tools)} tool(s) wired"
    except SubsystemUnavailable as exc:
        ctx.note(f"{tools_note} ({exc.reason})")

    decide = None
    decide_note = "decision registry unavailable; the graph's rules_decide() will answer"
    try:
        decide = get_decision_registry().decide
        decide_note = f"decision chain: {get_decision_registry().health(probe=False)['chain']}"
    except SubsystemUnavailable as exc:
        ctx.note(f"{decide_note} ({exc.reason})")

    policy_interactive = (cfg.human_gates_interactive if interactive is None
                          else bool(interactive))
    # One stable thread per run: a retry resumes the parked run instead of
    # forking a fresh thread per stage call. The thread is passed per-invoke
    # (runner.run/resume(thread_id=...)); this just records the stable key.
    tid = (thread_id or "").strip() or ctx.thread_id.strip() or ctx.run_id
    ctx.thread_id = tid
    runner = build_graph(
        cfg,
        board=ctx.board,
        tracer=ctx.tracer,
        tools=tools,
        decide=decide,
        interactive=policy_interactive,
        mode=cfg.run_mode,
    )
    ctx.note(f"graph compiled: {', '.join(runner.describe()['nodes'])}")
    ctx.note(f"wiring — {tools_note}; {decide_note}")
    return runner


# ======================================================================== health
def _probe_backend_rows(probe: bool) -> tuple[list[dict[str, Any]], list[str], str]:
    """Per-backend decision rows for the status panel, plus the chain and any error."""
    try:
        registry = get_decision_registry()
    except SubsystemUnavailable as exc:
        return [], [], exc.reason
    try:
        health = registry.health(probe=probe)
    except Exception as exc:  # noqa: BLE001 - a backend probe may hang or throw
        return [], [], f"{type(exc).__name__}: {exc}"
    return (list(health.get("backends") or []),
            list(health.get("chain") or []),
            "")


def _tool_rows() -> tuple[list[dict[str, Any]], str]:
    try:
        registry = get_tool_registry()
    except SubsystemUnavailable as exc:
        return [], exc.reason
    try:
        from tools.registry import describe as describe_registry

        return list(describe_registry(registry)), ""
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def _frontend_status() -> dict[str, Any]:
    """Whether a ``frontend/index.html`` exists, checked here rather than assumed.

    ``/health`` is the first thing anyone pokes when the demo shows a blank page,
    so "the frontend is not in this checkout" belongs in the health payload with
    the same weight as "the graph will not import".
    """
    if os.getenv("PAYTRIQ_SKIP_FRONTEND"):
        return {"present": False, "path": None,
                "reason": "PAYTRIQ_SKIP_FRONTEND is set"}
    root = Path(get_settings().repo_root)
    candidate = root / "frontend"
    if candidate.is_dir() and (candidate / "index.html").exists():
        return {"present": True, "path": str(candidate), "reason": ""}
    return {
        "present": False,
        "path": str(candidate),
        "reason": "no frontend/index.html in this checkout; the API is complete "
                  "without it and /openapi.json describes every endpoint",
    }


def health_payload(*, probe_backends: bool = False) -> dict[str, Any]:
    """Everything the demo status panel needs, in one JSON-safe dict.

    ``probe_backends`` defaults to ``False`` on purpose: ``ClefBackend.available()``
    opens a socket to a local llama.cpp server and takes seconds to time out, and
    a health endpoint that takes four seconds is a health endpoint nobody calls.
    The panel still shows which backend is *configured* and which answered last;
    ``/health?probe=true`` asks the expensive question when someone wants it.
    """
    cfg = get_settings()
    board_state = _board_singleton().state()
    tracer_state = _tracer_singleton(cfg).state()
    decision_state = _decision_singleton(cfg).state()
    tools_state = _tools_singleton(cfg).state()
    graph_state = _graph_singleton().state()

    backends, chain, backends_error = _probe_backend_rows(probe_backends)
    tool_rows, tools_error = _tool_rows()

    live_tools = [r["name"] for r in tool_rows if r.get("live_gate_open")]
    fixture_tools = [r.get("mode") for r in tool_rows if not r.get("live_gate_open")]

    subsystems = {
        "board": board_state,
        "tracer": tracer_state,
        "decision": decision_state,
        "tools": tools_state,
        "graph": graph_state,
    }
    unavailable = sorted(name for name, s in subsystems.items() if not s["available"])
    core_missing = [n for n in ("board", "tracer") if n in unavailable]

    if core_missing:
        status = "unavailable"
    elif unavailable:
        status = "degraded"
    else:
        status = "ok"

    context = current_run()
    return {
        "status": status,
        "ok": not core_missing,
        "unavailable_subsystems": unavailable,
        "app": {
            "name": cfg.app_name,
            "version": cfg.app_version,
            "environment": cfg.environment,
            "git_commit": cfg.git_commit,
            "run_mode": cfg.run_mode.value,
            "tools_live": cfg.tools_live,
            "traces_dir": str(cfg.traces_dir),
        },
        "settings": cfg.redacted(),
        "subsystems": subsystems,
        "decision": {
            "available": decision_state["available"],
            "reason": decision_state["reason"],
            "configured_backend": cfg.decision_backend,
            "effective_backend": cfg.effective_backend(),
            "chain": chain,
            "known_sources": list(_KNOWN_DECISION_SOURCES),
            "probed": probe_backends,
            "error": backends_error,
            "backends": backends,
        },
        "tools": {
            "available": tools_state["available"],
            "reason": tools_state["reason"] or tools_error,
            "live_enabled": cfg.tools_live,
            "count": len(tool_rows),
            "live_capable": sorted(live_tools),
            "serving": sorted({m for m in fixture_tools if m}),
            "rows": tool_rows,
        },
        "current_run": context.describe() if context is not None else None,
        "runs": list_runs(),
        "frontend": _frontend_status(),
    }


def reset_deps(*, finish_runs: bool = True) -> None:
    """Drop every cached instance and every run context.

    Tests call this between cases; a test that leaked a tracer would otherwise
    write its trace file into the *next* test's temp directory.
    """
    _reset_runs(finish=finish_runs)
    for lazy in _LAZIES.values():
        lazy.reset()
    _LAZIES.clear()
    set_settings(None)
