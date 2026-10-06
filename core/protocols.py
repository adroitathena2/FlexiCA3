"""Interfaces every Paytriq module is written against.

This file is the reason eight modules can be developed in parallel without
merge conflicts: each one depends only on the *protocols* declared here, never
on another module's internals. Concrete implementations live in their own
packages and are injected at wiring time.

Dependency direction (strict, enforced by ``tests/unit/test_contracts.py``):

    core  ->  nothing
    observability, decision  ->  core
    blackboard, tools  ->  core, observability
    agents  ->  core, decision, blackboard, tools, observability
    graph  ->  agents (registration only)
    api  ->  graph            eval  ->  everything

All protocols are ``runtime_checkable`` so tests can assert conformance without
instantiating anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .schemas import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    GateOutcome,
    Handoff,
    HumanDecision,
    Intent,
    Message,
    RunMode,
    SpanStatus,
    ToolStatus,
    TraceEvent,
    TraceKind,
)

__all__ = [
    "ToolResult", "Tool", "TOOL_REGISTRY",
    "DecisionBackend", "Blackboard", "BoardEntry", "Tracer",
    "AgentContext", "Plan", "ActResult", "Observation", "Reflection", "Agent",
    "ReplayStore", "HumanGateProvider",
]


# ============================================================================== tool
@dataclass(slots=True)
class ToolResult:
    """Uniform return type for every tool.

    ``status`` is mandatory and honest. A tool that could not reach its real
    backend returns ``UNAVAILABLE`` with ``degraded=True`` and a reason — it
    never returns fabricated data shaped like success.
    """

    ok: bool
    data: Any = None
    status: ToolStatus = ToolStatus.OK
    source: str = ""
    degraded: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    evidence_url: str | None = None

    @classmethod
    def success(cls, data: Any, *, source: str = "", latency_ms: float = 0.0,
                evidence_url: str | None = None) -> ToolResult:
        return cls(ok=True, data=data, status=ToolStatus.OK, source=source,
                   latency_ms=latency_ms, evidence_url=evidence_url)

    @classmethod
    def cached(cls, data: Any, *, source: str = "cache") -> ToolResult:
        return cls(ok=True, data=data, status=ToolStatus.CACHED, source=source,
                   degraded=True, reason="served from cache; live backend not called")

    @classmethod
    def unavailable(cls, reason: str, *, source: str = "") -> ToolResult:
        return cls(ok=False, data=None, status=ToolStatus.UNAVAILABLE, source=source,
                   degraded=True, reason=reason)

    @classmethod
    def failed(cls, reason: str, *, source: str = "") -> ToolResult:
        return cls(ok=False, data=None, status=ToolStatus.FAILED, source=source,
                   degraded=True, reason=reason)


@runtime_checkable
class Tool(Protocol):
    """A side-effecting or fact-fetching capability.

    Implementations must be import-time cheap and must never raise for an
    environmental problem — they return ``ToolResult.unavailable(...)`` instead,
    so a missing API key degrades the run instead of crashing it.
    """

    name: str
    description: str

    def available(self) -> tuple[bool, str]:
        """``(is_available, reason_if_not)``. Checked before every live call."""
        ...

    def run(self, **kwargs: Any) -> ToolResult:
        ...


#: name -> Tool. Populated by ``tools.registry.build_registry()``.
TOOL_REGISTRY: dict[str, Tool] = {}


# ======================================================================== decision
@runtime_checkable
class DecisionBackend(Protocol):
    """A source of calibrated decisions.

    Implementations must be honest about degradation: when a backend is
    unreachable they raise ``DecisionUnavailable`` rather than returning a
    fabricated answer, and the registry substitutes the rules backend while
    recording ``DecisionSource.RULES`` and ``degraded=True``.
    """

    name: str
    model: str

    def available(self) -> tuple[bool, str]:
        ...

    def decide(self, request: DecisionRequest) -> Decision:
        ...

    def health(self) -> dict[str, Any]:
        """Diagnostics for the demo's status panel: latency, model, errors."""
        ...


# ===================================================================== blackboard
@dataclass(slots=True)
class BoardEntry:
    """One append-only blackboard entry (Nii 1986, Hayes-Roth 1985)."""

    entry_id: str
    zone: str
    kind: str
    author: AgentId
    payload: dict[str, Any] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: DecisionSource = DecisionSource.RULES
    seq: int = 0
    at: str = ""


@runtime_checkable
class Blackboard(Protocol):
    """Typed, append-only shared workspace.

    Append-only is the point: entries are never mutated or deleted, so the
    history of how a decision was reached survives the run. Agents interact
    *solely* by posting and reading entries.
    """

    def post(self, zone: str, kind: str, author: AgentId, payload: dict[str, Any],
             *, refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> BoardEntry:
        ...

    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[BoardEntry]:
        ...

    def latest(self, zone: str, kind: str | None = None) -> BoardEntry | None:
        ...

    def zones(self) -> list[str]:
        ...

    def history(self) -> list[BoardEntry]:
        """Full ordered entry list — the audit trail."""
        ...


# ==================================================================== observability
@runtime_checkable
class Tracer(Protocol):
    """Emits spans and events to the trace. Context-manager based."""

    def configure(self, run_id: str, *, service: str = "paytriq") -> None:
        ...

    def agent(self, agent: AgentId, name: str, **attrs: Any) -> Any:
        """Context manager yielding an OTel span (INTERNAL kind)."""
        ...

    def llm(self, agent: AgentId, model: str, *, provider: str = "gemini",
            **attrs: Any) -> Any:
        """Context manager yielding an OTel span (CLIENT kind)."""
        ...

    def tool(self, agent: AgentId, tool_name: str, **attrs: Any) -> Any:
        """Context manager yielding an OTel span (INTERNAL kind)."""
        ...

    def decision(self, agent: AgentId, decision: Decision, **attrs: Any) -> Any:
        ...

    def handoff(self, handoff: Handoff) -> None:
        ...

    def message(self, message: Message) -> None:
        """Record an agent-to-agent message as both an event and a span.

        Mirrors :meth:`handoff`, with the ``Message`` model (fields
        ``from_agent``, ``to_agent``, ``decision_source``, ``confidence``).
        """
        ...

    def event(self, kind: TraceKind, name: str, *, agent: AgentId | None = None,
              **attrs: Any) -> TraceEvent:
        ...

    def finish(self) -> dict[str, Any]:
        """Close the run, write the trace file and its summary sidecar."""
        ...

    def replay_log(self) -> list[dict[str, Any]]:
        """Recorded model outputs, for ``RunMode.REPLAY``."""
        ...


# ============================================================================= agent
@dataclass(slots=True)
class AgentContext:
    """Everything an agent is allowed to see and do for one step.

    Deliberately narrow: an agent cannot reach the blackboard, the decision
    layer, or the tracer except through these handles. That is what keeps the
    dependency graph acyclic.
    """

    run_id: str
    event_id: str
    agent: AgentId
    board: Blackboard
    decide: Any                      # callable(DecisionRequest) -> Decision
    tracer: Tracer
    tools: dict[str, Tool]
    mode: RunMode = RunMode.LIVE
    step_budget: int = 6
    deadline_s: float = 60.0
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Plan:
    """An agent's chosen next action. Must be schema-valid to be executable."""

    goal: str
    steps: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    source: DecisionSource = DecisionSource.RULES
    stop: bool = False


@dataclass(slots=True)
class ActResult:
    """The outcome of executing a plan."""

    ok: bool
    output: Any = None
    produced: list[BoardEntry] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    degraded: bool = False


@dataclass(slots=True)
class Observation:
    """What the agent saw after acting."""

    summary: str
    facts: dict[str, Any] = field(default_factory=dict)
    sufficient: bool = True
    gaps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Reflection:
    """A post-step lesson. Persisted to the blackboard by the caller."""

    lesson_trigger: str
    correction: str
    rule: str
    confidence: float = 0.5


@runtime_checkable
class Agent(Protocol):
    """A reasoning agent: plan -> act -> observe -> reflect.

    The loop is explicit rather than hidden in a framework so that each phase is
    independently visible in the trace and independently testable.
    """

    id: AgentId
    role: str

    def plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        ...

    def act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        ...

    def observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        ...

    def reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        ...


# ========================================================================== replay
@runtime_checkable
class ReplayStore(Protocol):
    """Recorded model outputs, keyed by a stable hash of the request.

    Replay must return *recorded bytes*, never synthesised plausible text. This
    is the line between an honest offline demo and a fabrication.
    """

    def get(self, key: str) -> dict[str, Any] | None:
        ...

    def put(self, key: str, value: dict[str, Any]) -> None:
        ...

    @staticmethod
    def key(namespace: str, payload: Any) -> str:
        ...


# ============================================================== human gate provider
@runtime_checkable
class HumanGateProvider(Protocol):
    """Surfaces a ``HumanGate`` and returns the human's answer."""

    def request(self, gate_kind: str, question: str, *, preview: str = "",
                run_id: str = "", event_id: str = "") -> HumanDecision:
        ...


def classify_intent_fallback(text: str) -> Intent:
    """Deterministic last-resort classifier used only when no model is reachable.

    Order matters and is deliberate: refusal and pushback are tested *before*
    assent, because the previous prototype tested ``"yes"`` first and so
    classified "yesterday we thought the price was too high" as acceptance.
    """
    t = (text or "").lower()
    if not t.strip():
        return Intent.UNKNOWN
    if any(k in t for k in ("not interested", "no thanks", "pass on", "we decline",
                            "cannot", "can't", "not proceeding", "reject")):
        return Intent.NO
    if any(k in t for k in ("too high", "too expensive", "expensive", "discount",
                            "budget", "reduce", "cheaper", "negotiat", "lower",
                            "afford", "costly")):
        return Intent.PUSHBACK
    if any(k in t for k in ("yes", "agree", "approved", "confirm", "let's proceed",
                            "lets proceed", "deal", "sign", "onboard")):
        return Intent.YES
    if any(k in t for k in ("interested", "tell me more", "share", "deck",
                            "call", "meeting", "sounds good", "curious")):
        return Intent.INTERESTED
    return Intent.NEUTRAL


__all__ += ["classify_intent_fallback", "REASONING_AGENT_IDS", "GateOutcome",
            "SpanStatus", "DecisionSource"]

#: Convenience tuple mirroring ``schemas.REASONING_AGENTS`` for cheap membership tests.
REASONING_AGENT_IDS = frozenset(a.value for a in AgentId if a.is_reasoning_agent)
