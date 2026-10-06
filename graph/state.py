"""The LangGraph state contract, its reducers, and the run-scoped runtime.

Everything that crosses a node boundary in Paytriq is declared here. Two
constraints shaped the design and both are deliberate:

**1. Parallel branches must merge without loss.**
The graph is not a chain. After pricing, outreach runs *and* the compliance
lookahead runs; after a dispute opens, the debate and the arbiter's
preparation both fire. LangGraph runs same-superstep nodes concurrently and
merges their writes, so a field without a reducer is silently last-write-wins
and two parallel branches overwrite each other. Every append-only artefact
(``handoffs``, ``messages``, ``disputes``, ``lessons``, ``approvals``, ``trace_events``,
``risk_flags``) therefore uses :func:`append`, and every *collection of domain
objects that different agents author for the same entity* (brands, offers,
threads, MoUs) uses :func:`upsert_by` so two agents pricing the same brand
converge on one record instead of two.

**2. State must survive the checkpointer.**
``SqliteSaver`` serialises channels with msgpack. Pydantic models round-trip,
but only after an "unregistered type" warning, and only because they are
importable at the same path on the next run. Domain artefacts are therefore
stored as plain JSON-shaped ``dict`` via ``model_dump(mode="json")``; the
accessors at the bottom of this file re-hydrate them into the frozen schemas.
That keeps a checkpoint written today readable by a future refactor, which is
the whole point of having a checkpointer.

Magentic-One ledgers
--------------------
Following Fourney et al., *Magentic-One: A Generalist Multi-Agent System for
Solving Complex Tasks* (Microsoft Research, MSR-TR-2024-47), the state carries
two ledgers:

* :class:`TaskLedger` — the *plan of record*: what is known, what is assumed,
  what to do next. A replan rewrites it; nothing else does.
* :class:`ProgressLedger` — whether the plan is producing movement. ``route_progress``
  reads it to decide between finishing, continuing, and re-planning.

They are separated because conflating "what should we do" with "is it working"
is exactly how an agent loop turns into a spinner: the planner keeps declaring
the same next step because it cannot observe that the step changed nothing.
"""
from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from core import (
    AgentId,
    Approval,
    Blackboard,
    BrandLead,
    ConfigError,
    Decision,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    GateOutcome,
    Handoff,
    HumanDecision,
    MoU,
    Offer,
    RiskFlag,
    ROIReport,
    RunMode,
    Settings,
    Thread,
    Tool,
    ToolStatus,
    Tracer,
    new_id,
)

try:
    from core.schemas import Message
except ImportError:  # Track 1 (core.schemas Message) has not landed yet.
    # Resolve the same placeholder class that backs KIND_MODELS["message"],
    # so board validation and state rehydration agree. Once core.schemas
    # provides the real model this branch never runs.
    from blackboard.zones import KIND_MODELS as _KIND_MODELS

    Message = _KIND_MODELS["message"]

__all__ = [
    # reducers
    "append", "keep_last", "merge_dicts", "upsert_by", "REDUCERS",
    # ledgers
    "TaskLedger", "ProgressLedger", "new_task_ledger", "new_progress_ledger",
    "read_task_ledger", "read_progress_ledger", "ledger_is_complete",
    "ledger_is_progressing", "stall_reason",
    # state
    "PaytriqState", "PHASE_PIPELINE", "PHASE_DONE",
    "coerce_json", "as_brand_leads", "as_offers", "as_threads", "as_mous",
    "as_risk_flags", "as_roi_report", "as_event_profile", "as_handoffs",
    "as_messages",
    "as_disputes", "as_approvals", "as_human_decisions", "as_notes",
    "bidder_agents", "active_threads", "blocking_flags", "latest_offer_for",
    # runtime
    "GraphRuntime", "GatePolicy", "runtime_context", "StepMemo", "RuntimeScratch",
    # fallback collaborators
    "InMemoryBoardLike", "NullTracer", "rules_decide",
]

log = logging.getLogger("paytriq.graph.state")


# =============================================================================== reducers
def append(left: Sequence[Any] | None, right: Sequence[Any] | None) -> list[Any]:
    """Concatenate two append-only lists. LangGraph's ``operator.add`` made explicit.

    ``None`` is treated as empty because a node returning ``{"handoffs": []}``
    for an absent key must not clobber a sibling branch's handoffs.
    """
    return list(left or []) + list(right or [])


def keep_last(left: Any, right: Any) -> Any:
    """Last-write-wins for scalars.

    Written as a named function rather than relying on the absence of a
    reducer so the intent is readable at the channel definition, and so
    ``None`` from a node that "has nothing to report" does not erase a real
    value from a parallel branch.
    """
    return left if right is None else right


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Shallow key-wise merge. Used for ledgers and scalar-ish report blobs.

    Shallow on purpose: nested dicts inside a report are replaced wholesale by
    the branch that computed them, never spliced, because a half-merged
    ROI report is worse than a missing key.
    """
    out = dict(left or {})
    out.update(right or {})
    return out


def upsert_by(key: str) -> Callable[[Any, Any], list[Any]]:
    """Build a merge reducer keyed on an id field of the dicts being merged.

    Two agents can legitimately produce an ``offer`` for the same brand in
    parallel. ``operator.add`` would leave two offers with the same brand and
    the downstream thread-creation step would mail the sponsor twice. Keyed
    upsert makes the later write win *per entity* while preserving every other
    entity either branch contributed.

    Dicts without the key (or with a falsy one) are appended rather than
    dropped, so an artefact that genuinely has no identity is never lost.
    """
    field_name = key

    def reducer(left: Sequence[Any] | None, right: Sequence[Any] | None) -> list[Any]:
        merged: list[Any] = []
        index: dict[Any, int] = {}
        for item in list(left or []) + list(right or []):
            ident = item.get(field_name) if isinstance(item, dict) else None
            if not ident:
                merged.append(item)
                continue
            if ident in index:
                merged[index[ident]] = item
            else:
                index[ident] = len(merged)
                merged.append(item)
        return merged

    reducer.__name__ = f"upsert_by_{field_name}"
    reducer.__doc__ = f"Merge JSON dicts keyed on {field_name!r}; later write wins."
    return reducer


#: Reducers referenced by name from tests and from other graph modules.
REDUCERS = {
    "append": append,
    "keep_last": keep_last,
    "merge_dicts": merge_dicts,
}


# ================================================================================ ledgers
class TaskLedger(TypedDict, total=False):
    """Magentic-One's task ledger (MSR-TR-2024-47, §3 "ledger" component).

    ``facts`` are things a tool or the board established. ``guesses`` are
    things nobody has verified but the plan leans on. Keeping them in separate
    lists is the point: a replan may legitimately keep a guess, but it may
    never promote a guess to a fact without new evidence, and the ledger's
    ``open_questions`` list is what makes that omission visible.
    """

    facts: list[str]
    guesses: list[str]
    plan: list[str]
    open_questions: list[str]
    revision: int
    written_by: str


class ProgressLedger(TypedDict, total=False):
    """Magentic-One's progress ledger.

    ``is_complete``/``is_progress`` are *claims made by a decision*, not
    booleans the orchestrator sets. Every value written here is traceable to a
    ``Decision`` that produced it; :func:`route_progress
    <graph.edges.route_progress>` records that decision's source and confidence
    on the resulting :class:`~core.schemas.Handoff`.
    """

    is_complete: bool
    is_complete_reason: str
    is_progress: bool
    progress_reason: str
    changed: str
    observed_by: str
    source: str


def new_task_ledger(*, facts: Sequence[str] = (), guesses: Sequence[str] = (),
                   plan: Sequence[str] = (), revision: int = 0,
                   written_by: str = "graph") -> TaskLedger:
    """A fresh task ledger. ``revision`` increments on every replan."""
    return TaskLedger(
        facts=list(facts),
        guesses=list(guesses),
        plan=list(plan),
        open_questions=[],
        revision=int(revision),
        written_by=written_by,
    )


def new_progress_ledger(*, is_complete: bool = False, is_progress: bool = True,
                        is_complete_reason: str = "", progress_reason: str = "",
                        changed: str = "", observed_by: str = "graph",
                        source: str = "unset") -> ProgressLedger:
    """A fresh progress ledger. Defaults to "not done, but moving"."""
    return ProgressLedger(
        is_complete=bool(is_complete),
        is_complete_reason=is_complete_reason,
        is_progress=bool(is_progress),
        progress_reason=progress_reason,
        changed=changed,
        observed_by=observed_by,
        source=source,
    )


def read_task_ledger(state: dict) -> TaskLedger:
    """State is ``total=False``: a missing ledger means an untouched run."""
    value = state.get("task_ledger")
    return new_task_ledger() if not value else dict(value)  # type: ignore[return-value]


def read_progress_ledger(state: dict) -> ProgressLedger:
    value = state.get("progress_ledger")
    if not value:
        return new_progress_ledger()
    ledger = new_progress_ledger()
    ledger.update(dict(value))  # type: ignore[arg-type]
    return ledger


def ledger_is_complete(state: dict) -> bool:
    return bool(read_progress_ledger(state).get("is_complete"))


def ledger_is_progressing(state: dict) -> bool:
    return bool(read_progress_ledger(state).get("is_progress"))


def stall_reason(state: dict) -> str:
    """Human-readable justification for a stall, suitable for a trace attribute."""
    ledger = read_progress_ledger(state)
    why = str(ledger.get("progress_reason") or "no observable change in world state")
    changed = str(ledger.get("changed") or "(nothing)")
    return f"{why}; last observed change: {changed}"


# ================================================================================= state
#: Nominal phases, in pipeline order. ``status`` carries the finer-grained value;
#: ``phase`` is the coarse one the API groups SSE frames by.
PHASE_PIPELINE = "pipeline"
PHASE_DONE = "done"


class PaytriqState(TypedDict, total=False):
    """The single shared state object every node reads from and writes to.

    ``total=False`` because LangGraph treats the annotation set as the channel
    specification: a missing key means "no write this step", not "reset to
    default". Reducers decide how two concurrent writes combine.

    Identity
    `````````
    event_id, run_id, phase, agent_id, status, pending_gate

    World model (all JSON dicts — see the module docstring)
    `````````````````````````````````````````````````````````
    event_profile, brands, offers, threads, mous, risk_flags, roi_report

    Coordination
    `````````````
    handoffs, messages, disputes, lessons, bids, approvals, trace_events, notes

    Magentic-One
    `````````````
    task_ledger, progress_ledger, stall_count, replan_count, discovery_radius_km
    """

    # ---- identity / control (last-write-wins) --------------------------------
    event_id: Annotated[str, keep_last]
    run_id: Annotated[str, keep_last]
    phase: Annotated[str, keep_last]
    #: The agent whose node produced the most recent write. Routers read this to
    #: build a ``Handoff`` that does not violate its no-self-handoff rule.
    agent_id: Annotated[str | None, keep_last]
    status: Annotated[str, keep_last]
    #: The gate most recently answered, or ``None`` before any gate has run.
    #: Retained after the fact on purpose: the conditional edge that follows a gate
    #: reads it to decide where to go next, so a node that answered the gate must
    #: not clear the field on its way out.
    pending_gate: Annotated[dict | None, keep_last]
    compliance_score: Annotated[float | None, keep_last]
    roi_report: Annotated[dict | None, keep_last]
    event_profile: Annotated[dict | None, keep_last]
    task_ledger: Annotated[TaskLedger, merge_dicts]
    progress_ledger: Annotated[ProgressLedger, merge_dicts]
    stall_count: Annotated[int, keep_last]
    replan_count: Annotated[int, keep_last]
    #: Concessions granted on the sponsor's counter-offer. Counted explicitly by
    #: the ``revise`` node rather than inferred from ``Offer.version``: the version
    #: only advances when A2 actually revises, so inferring from it would make the
    #: loop bound depend on whether an agent is registered.
    negotiation_rounds: Annotated[int, keep_last]
    discovery_radius_km: Annotated[float, keep_last]

    # ---- append-only coordination artefacts ---------------------------------
    handoffs: Annotated[list[dict], append]
    messages: Annotated[list[dict], append]
    disputes: Annotated[list[dict], append]
    lessons: Annotated[list[dict], append]
    approvals: Annotated[list[dict], append]
    trace_events: Annotated[list[dict], append]
    #: Human-readable lines written by the orchestration layer itself.
    notes: Annotated[list[str], append]
    bids: Annotated[list[dict], append]
    #: Protocol-level events (announcements, awards, expedites, run summaries)
    #: projected off the board. Kept separate from ``notes`` so ``notes`` stays
    #: ``list[str]`` and a caller can render it without type-checking every item.
    run_events: Annotated[list[dict], append]

    # ---- world model, keyed upsert (two agents may author the same entity) ---
    brands: Annotated[list[dict], upsert_by("lead_id")]
    offers: Annotated[list[dict], upsert_by("offer_id")]
    threads: Annotated[list[dict], upsert_by("thread_id")]
    mous: Annotated[list[dict], upsert_by("mou_id")]
    #: Risk flags are pure appends — A5 raises objections, nobody withdraws them.
    risk_flags: Annotated[list[dict], append]
    #: Human decisions are append-only; one row per gate answered.
    human_decisions: Annotated[list[dict], append]
    #: Names of steps already executed, for idempotency across node re-entry.
    steps_done: Annotated[list[str], append]


# ------------------------------------------------------------------------- coercions
def coerce_json(value: Any) -> dict:
    """Best-effort conversion of a schema artefact (or dict) to a JSON dict.

    Used defensively at the board->state boundary: the blackboard payload is
    ``dict[str, Any]`` by contract but a misbehaving writer may have posted a
    model instance, so normalise rather than crash mid-run.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return {k: v for k, v in as_dict.items() if not k.startswith("_")}
    return {"value": str(value)}


def _as_models(values: Any, model: type) -> list[Any]:
    out: list[Any] = []
    for item in values or []:
        data = coerce_json(item)
        try:
            out.append(model(**data))
        except Exception as exc:  # noqa: BLE001 - a malformed record must not kill a run
            log.warning("dropping unparseable %s from state: %s", model.__name__, exc)
    return out


def as_brand_leads(state: dict) -> list[BrandLead]:
    return _as_models(state.get("brands"), BrandLead)


def as_offers(state: dict) -> list[Offer]:
    return _as_models(state.get("offers"), Offer)


def as_threads(state: dict) -> list[Thread]:
    return _as_models(state.get("threads"), Thread)


def as_mous(state: dict) -> list[MoU]:
    return _as_models(state.get("mous"), MoU)


def as_risk_flags(state: dict) -> list[RiskFlag]:
    return _as_models(state.get("risk_flags"), RiskFlag)


def as_handoffs(state: dict) -> list[Handoff]:
    return _as_models(state.get("handoffs"), Handoff)


def as_messages(state: dict) -> list[Message]:
    return _as_models(state.get("messages"), Message)


def as_disputes(state: dict) -> list[Any]:
    from core import Dispute

    return _as_models(state.get("disputes"), Dispute)


def as_approvals(state: dict) -> list[Approval]:
    return _as_models(state.get("approvals"), Approval)


def as_human_decisions(state: dict) -> list[HumanDecision]:
    return _as_models(state.get("human_decisions"), HumanDecision)


def as_notes(state: dict) -> list[str]:
    return [str(n) for n in (state.get("notes") or [])]


def as_roi_report(state: dict) -> ROIReport | None:
    data = state.get("roi_report")
    if not data:
        return None
    try:
        return ROIReport(**coerce_json(data))
    except Exception as exc:  # noqa: BLE001 - report is advisory; never fatal
        log.warning("ROIReport in state is incomplete (%s); reporting partial", exc)
        return None


def as_event_profile(state: dict) -> EventProfile | None:
    data = state.get("event_profile")
    if not data:
        return None
    try:
        return EventProfile(**coerce_json(data))
    except Exception as exc:  # noqa: BLE001 - malformed profile is a caller bug
        raise ConfigError(f"state.event_profile is not a valid EventProfile: {exc}") from exc


# ------------------------------------------------------------------ state queries
def bidder_agents(state: dict) -> list[AgentId]:
    """Agents that may bid for an outreach task: those with a contract on a lead.

    A bid from an agent with nothing on the board is noise, so eligibility is
    derived from the state rather than declared in config.
    """
    return _agent_ids_present(state)


def active_threads(state: dict) -> list[dict]:
    """Threads that are not closed. Closed threads stop generating work."""
    return [t for t in (state.get("threads") or [])
            if t.get("status") not in ("closed_won", "closed_lost")]


def blocking_flags(state: dict) -> list[dict]:
    """Unresolved BLOCKING risk flags. A5's veto, materialised as data."""
    out = []
    for flag in state.get("risk_flags") or []:
        severity = str(flag.get("severity") or "").rsplit(".", 1)[-1]
        if severity == "blocking" and not flag.get("resolved"):
            out.append(flag)
    return out


def latest_offer_for(state: dict, brand: str) -> dict | None:
    """Highest-version offer for a brand, or ``None``.

    Pushback handling revises rather than replaces: the sponsor's reaction to
    version 1 is only interpretable against version 1, so the revision must
    supersede it rather than sit beside it.
    """
    candidates = [o for o in (state.get("offers") or []) if o.get("brand") == brand]
    if not candidates:
        return None
    return max(candidates, key=lambda o: (int(o.get("version") or 1), str(o.get("offer_id"))))


def _agent_ids_present(state: dict) -> list[AgentId]:
    ids: list[AgentId] = []
    for source in (state.get("brands") or [], state.get("offers") or [],
                   state.get("risk_flags") or [], state.get("threads") or []):
        for item in source:
            raw = item.get("author") or item.get("raised_by")
            if not raw:
                continue
            try:
                aid = AgentId(str(raw).rsplit(".", 1)[-1])
            except ValueError:
                continue
            if aid.is_reasoning_agent and aid not in ids:
                ids.append(aid)
    return ids


# ============================================================================== runtime
@dataclass
class GatePolicy:
    """How human gates behave for one build of the graph.

    Separated from :class:`~core.config.Settings` because the *policy* is what
    the API overrides per run (a web request is interactive; a scripted demo is
    not), while the settings object is process-wide.
    """

    interactive: bool = True
    #: Outcome used when not interactive. Stored as a string because
    #: :class:`GateOutcome` validation happens at construction time.
    auto_outcome: str = "approve"
    #: Overrides the resume value for a *non*-interactive run only, for tests.
    scripted_resume: dict | None = None

    def outcome(self) -> GateOutcome:
        try:
            return GateOutcome(str(self.auto_outcome))
        except ValueError as exc:
            raise ConfigError(
                f"auto_approve_outcome={self.auto_outcome!r} is not one of "
                f"{[o.value for o in GateOutcome]}"
            ) from exc


@dataclass
class StepMemo:
    """Remembered result of one agent step.

    LangGraph **discards a node's state writes when that node raises
    ``interrupt``**. On resume the node re-executes from its first line, so any
    agent run above the interrupt would be paid for twice and would post twice
    to the blackboard. The memo is kept on the runtime — outside the
    checkpointed channel — so the second execution replays the first one's
    result instead of repeating it.
    """

    key: str
    delta: dict
    ok: bool = True
    degraded: bool = False
    summary: str = ""


@dataclass
class RuntimeScratch:
    """Process-local, per-build memo store for node re-entry safety."""

    steps: dict[str, StepMemo] = field(default_factory=dict)
    pending_handoffs: list[Handoff] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def memo(self, key: str) -> StepMemo | None:
        return self.steps.get(key)

    def remember(self, memo: StepMemo) -> StepMemo:
        self.steps[memo.key] = memo
        return memo

    def queue_handoff(self, handoff: Handoff) -> Handoff:
        self.pending_handoffs.append(handoff)
        return handoff

    def take_handoffs(self) -> list[dict]:
        """Drain queued handoffs as JSON dicts for the next node's state update."""
        if not self.pending_handoffs:
            return []
        drained = [h.model_dump(mode="json") for h in self.pending_handoffs]
        self.pending_handoffs.clear()
        return drained


def _emit_decision_traces(tracer: Any, agent_id: Any, request: Any,
                          decision: Any, elapsed_ms: float) -> None:
    """Record one agent decision as ``decision`` + (when model-backed) ``llm`` spans.

    Every ``ctx.decide()`` call flows through here via
    :meth:`GraphRuntime.context_for`, so agent routing decisions are no longer
    invisible while router decisions (which already emit ``tracer.decision``)
    stay the only other source. ``rules`` decisions emit only the ``decision``
    span: no model was called, so an ``llm`` span would be a fabrication.
    """
    try:
        mgr = tracer.decision(
            agent_id, decision,
            decision_point=getattr(request, "decision_point", ""),
            request_id=getattr(request, "request_id", ""),
            latency_ms=elapsed_ms,
        )
        if hasattr(mgr, "__enter__"):
            with mgr:
                pass
    except Exception as exc:  # noqa: BLE001 - tracing must not break the agent
        log.debug("tracer.decision emit failed for %s: %s",
                  getattr(agent_id, "value", agent_id), exc)
    try:
        source = getattr(getattr(decision, "source", None), "value",
                         str(getattr(decision, "source", ""))).lower()
        if source in ("clef", "gemini"):
            model = getattr(decision, "model", "") or source
            mgr = tracer.llm(
                agent_id, str(model),
                provider=source,
                decision_point=getattr(request, "decision_point", ""),
                request_id=getattr(request, "request_id", ""),
                source=source,
                confidence=float(getattr(decision, "confidence", 0.0) or 0.0),
                latency_ms=elapsed_ms,
            )
            if hasattr(mgr, "__enter__"):
                with mgr:
                    pass
    except Exception as exc:  # noqa: BLE001 - tracing must not break the agent
        log.debug("tracer.llm emit failed for %s: %s",
                  getattr(agent_id, "value", agent_id), exc)


def _wrap_decide_with_tracing(original: Callable[[Any], Any], tracer: Any,
                              agent_id: Any) -> Callable[[Any], Any]:
    """Return a ``decide`` callable that traces every answer it gives."""
    def decide(request: Any) -> Any:
        started = time.perf_counter()
        decision = original(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _emit_decision_traces(tracer, agent_id, request, decision, elapsed_ms)
        return decision
    return decide


def _accepted_keywords(runner: Any) -> frozenset[str] | None:
    """Keyword names ``runner`` accepts, or ``None`` to pass everything through.

    Mirrors ``eval.integration._accepted_keywords``: ``None`` means the
    callable wants everything (``**kwargs`` or C-implemented with no
    introspectable signature). A fixed signature returns its accepted names so
    callers passing a superset (location/query/address/name aliases) do not
    crash the tool with an unexpected keyword.
    """
    if not callable(runner):
        return frozenset()
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return None
    params = signature.parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return frozenset(
        p.name for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    )


class _TracedTool:
    """A :class:`~core.protocols.Tool` proxy that emits ``tracer.tool`` per call.

    Wrapping here (in :meth:`GraphRuntime.context_for`) covers every agent at
    once: each agent's local ``invoke(tool, args)`` helper receives the proxy
    from ``ctx.tools`` and calls ``run()`` on it, so no per-agent edit is
    needed and no call site can silently bypass tracing.
    """

    def __init__(self, inner: Any, tracer: Any, agent_id: Any) -> None:
        self._inner = inner
        self._tracer = tracer
        self._agent_id = agent_id
        self.name = getattr(inner, "name", "unknown_tool")
        self.description = getattr(inner, "description", "")
        # Introspect the wrapped tool so geocode/search aliases filter
        # correctly: the agents' ``invoke`` sees this proxy's ``run`` (which
        # takes ``**kwargs`` and would look like "accepts everything") and
        # would otherwise forward location/query/address/name plus
        # radius_m/limit/lat/lon extras the inner ``run`` rejects. Filtering
        # here keeps the proxy transparent to signature-based callers (same
        # pattern as eval.integration.CountingTool).
        self._accepted = _accepted_keywords(getattr(inner, "run", None))

    def __getattr__(self, item: str) -> Any:
        if item in ("_inner", "_tracer", "_agent_id", "_accepted",
                    "name", "description"):
            raise AttributeError(item)
        return getattr(self._inner, item)

    def available(self) -> tuple[bool, str]:
        probe = getattr(self._inner, "available", None)
        if not callable(probe):
            return True, ""
        return probe()

    def run(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        # Forward only what the wrapped tool accepts; extra aliases would
        # otherwise raise TypeError inside geocode/search tools.
        call_kwargs = (
            kwargs if self._accepted is None
            else {k: v for k, v in kwargs.items() if k in self._accepted}
        )
        try:
            result = self._inner.run(**call_kwargs)
        except Exception:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            try:
                mgr = self._tracer.tool(
                    self._agent_id, str(self.name),
                    latency_ms=elapsed_ms, tool_status=ToolStatus.FAILED,
                )
                if hasattr(mgr, "__enter__"):
                    with mgr:
                        pass
            except Exception as exc:  # noqa: BLE001 - tracing is best-effort
                log.debug("tracer.tool error emit failed for %s: %s",
                          self.name, exc)
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        try:
            status = getattr(result, "status", None)
            attrs: dict[str, Any] = {
                "latency_ms": elapsed_ms,
                "ok": bool(getattr(result, "ok", False)),
            }
            degraded = getattr(result, "degraded", None)
            if degraded is not None:
                attrs["degraded"] = bool(degraded)
            reason = getattr(result, "reason", "")
            if reason:
                attrs["reason"] = str(reason)[:300]
            if status is not None:
                attrs["tool_status"] = status
            mgr = self._tracer.tool(self._agent_id, str(self.name), **attrs)
            if hasattr(mgr, "__enter__"):
                with mgr:
                    pass
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort
            log.debug("tracer.tool emit failed for %s: %s", self.name, exc)
        return result


@dataclass
class GraphRuntime:
    """Run-scoped wiring handed to every node and every router.

    This is the ``context_schema`` of the compiled graph. It exists because the
    frozen :class:`~core.protocols.AgentContext` is per-agent and per-step,
    while routing and coordination need the run's collaborators (board, tracer,
    tools, decision callable) without an agent identity. LangGraph passes it as
    ``runtime.context`` and never checkpoints it, which is exactly right: it
    holds live handles, not state.

    Kept in ``state`` rather than ``build`` so ``edges``/``contract_net``/
    ``debate``/``interrupts`` can share it without importing each other.
    """

    settings: Settings
    board: Blackboard
    tracer: Tracer
    decide: Callable[[DecisionRequest], Decision]
    agents: dict[AgentId, Any] = field(default_factory=dict)
    tools: dict[str, Tool] = field(default_factory=dict)
    mode: RunMode = RunMode.LIVE
    gate_policy: GatePolicy = field(default_factory=GatePolicy)
    scratch: RuntimeScratch = field(default_factory=RuntimeScratch)
    thread_id: str = ""
    run_id: str = ""
    event_id: str = ""
    #: Populated by ``build.build_graph`` with names of agents not found.
    absent_agents: tuple[str, ...] = ()
    #: Free-form slot for the API layer (request id, user, resume tokens).
    extras: dict[str, Any] = field(default_factory=dict)

    def agent(self, agent_id: AgentId) -> Any | None:
        """The registered agent instance, or ``None`` if it was never built."""
        return self.agents.get(agent_id)

    def note(self, message: str, where: str = "") -> str:
        """Emit a run note and return it, so callers can append it to state."""
        text = f"[{where or 'graph'}] {message}"
        self.scratch.notes.append(text)
        log.info("%s", text)
        return text

    def context_for(self, agent_id: AgentId, event_id: str, run_id: str,
                    *, step_budget: int | None = None,
                    deadline_s: float | None = None) -> Any:
        """Build the frozen :class:`~core.protocols.AgentContext` for one agent step.

        The ``decide`` callable and every tool are wrapped so agent work is
        structurally traced: decisions emit ``tracer.decision`` (plus
        ``tracer.llm`` when a model answered) and tool executions emit
        ``tracer.tool``. Router paths already trace; without this, agent paths
        never do and ``llm_call_count``/``tool_call_count`` stay zero.
        """
        from core import AgentContext

        # Propagate graph-answered gates so an agent re-running after resume can
        # see the human's answer even though its own gate_id differs. The agent
        # still enforces its own gate; this only supplies the evidence.
        graph_approvals = list(self.extras.get("graph_approvals") or [])
        traced_decide = _wrap_decide_with_tracing(self.decide, self.tracer,
                                                  agent_id)
        traced_tools = {name: _TracedTool(tool, self.tracer, agent_id)
                        for name, tool in (self.tools or {}).items()}
        return AgentContext(
            run_id=run_id or self.run_id,
            event_id=event_id or self.event_id,
            agent=agent_id,
            board=self.board,
            decide=traced_decide,
            tracer=self.tracer,
            tools=traced_tools,
            mode=self.mode,
            step_budget=int(step_budget or self.settings.agent_step_budget),
            deadline_s=float(deadline_s or self.settings.agent_deadline_s),
            scratch={"thread_id": self.thread_id,
                     "graph_approvals": graph_approvals},
        )

    def take_handoffs(self) -> dict[str, Any]:
        """State delta draining router-queued handoffs."""
        handoffs = self.scratch.take_handoffs()
        return {"handoffs": handoffs} if handoffs else {}


def runtime_context(runtime: Any) -> GraphRuntime:
    """Accept either a LangGraph ``Runtime`` or a bare :class:`GraphRuntime`.

    LangGraph 1.x injects ``Runtime[ContextT]`` as a node's second positional
    argument. Unit tests call routers and nodes directly with a plain
    ``GraphRuntime``. Supporting both here means no test has to construct a
    ``Runtime``, and no node has to care which caller it has.
    """
    if isinstance(runtime, GraphRuntime):
        return runtime
    context = getattr(runtime, "context", None)
    if isinstance(context, GraphRuntime):
        return context
    raise ConfigError(
        "expected a GraphRuntime or a langgraph Runtime carrying one; got "
        f"{type(runtime).__name__}. Build the graph with "
        "graph.build.build_graph(settings, agents, runtime=...) so the runtime "
        "schema matches."
    )


# ====================================================== fallback collaborators
# These three exist so ``build_graph`` never fails for want of a collaborator.
# During concurrent development ``blackboard/``, ``observability/`` and
# ``decision/`` may not be importable yet, and an orchestrator that cannot be
# constructed cannot be demonstrated or tested. Each fallback logs *once*, names
# what it is standing in for, and marks what it emits as degraded, so a run on
# fallbacks is never mistaken for a run on the real thing.


class InMemoryBoardLike:
    """A minimal append-only blackboard satisfying the :class:`Blackboard` protocol.

    Same semantics as the real one — append-only, monotonic ``seq``, never
    mutated or deleted — because those semantics are what the coordination layer's
    arguments rest on. It is not a substitute for the real board in production;
    it exists so the graph is testable without it.
    """

    def __init__(self) -> None:
        self._entries: list[Any] = []
        self._seq = 0

    def post(self, zone: str, kind: str, author: AgentId,
             payload: dict[str, Any], *, refs: list[str] | None = None,
             confidence: float = 1.0,
             source: Any = DecisionSource.RULES) -> Any:
        from core import BoardEntry, utcnow  # local: avoids a module-level cycle

        self._seq += 1
        entry = BoardEntry(
            entry_id=new_id("be", 12),
            zone=zone,
            kind=kind,
            author=author,
            payload=dict(payload or {}),
            refs=list(refs or []),
            confidence=float(confidence),
            source=source,
            seq=self._seq,
            at=utcnow().isoformat(),
        )
        self._entries.append(entry)
        return entry

    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[Any]:
        rows = [e for e in self._entries if e.zone == zone
                and (kind is None or e.kind == kind)]
        return rows[-limit:] if limit else rows

    def latest(self, zone: str, kind: str | None = None) -> Any | None:
        rows = self.read(zone, kind=kind)
        return rows[-1] if rows else None

    def zones(self) -> list[str]:
        return sorted({e.zone for e in self._entries})

    def history(self) -> list[Any]:
        return list(self._entries)


class NullTracer:
    """A tracer that records into a list and writes nowhere.

    Implements the :class:`Tracer` protocol's surface used by the graph. Spans
    come from :mod:`contextlib`, and every emission is tagged ``degraded`` so a
    run with no real tracer is identifiable after the fact.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.handoffs: list[Any] = []
        self.replay_log_: list[dict[str, Any]] = []
        self.run_id = ""

    def configure(self, run_id: str, *, service: str = "paytriq") -> None:
        self.run_id = run_id

    def _span(self, name: str, **attrs: Any) -> Any:
        return _NullSpan(self, name, attrs)

    def agent(self, agent: AgentId, name: str, **attrs: Any) -> Any:
        return self._span(name, agent=getattr(agent, "value", str(agent)), **attrs)

    def llm(self, agent: AgentId, model: str, *, provider: str = "gemini",
            **attrs: Any) -> Any:
        return self._span(name=f"llm.{model}", agent=getattr(agent, "value", str(agent)),
                          provider=provider, **attrs)

    def tool(self, agent: AgentId, tool_name: str, **attrs: Any) -> Any:
        return self._span(name=f"tool.{tool_name}",
                          agent=getattr(agent, "value", str(agent)), **attrs)

    def decision(self, agent: AgentId, decision: Decision, **attrs: Any) -> None:
        self.events.append({"kind": "decision", "degraded": True,
                            "agent": getattr(agent, "value", str(agent)),
                            "source": decision.source.value,
                            "confidence": decision.confidence,
                            "attrs": attrs})

    def handoff(self, handoff: Any) -> None:
        self.handoffs.append(handoff)

    def event(self, kind: Any, name: str, *, agent: AgentId | None = None,
              **attrs: Any) -> Any:
        from core import TraceEvent

        record = TraceEvent(
            seq=len(self.events), run_id=self.run_id, event_id="", trace_id=new_id("trc"),
            span_id=new_id("spn"), kind=kind, name=name, agent=agent,
            attributes={**attrs, "degraded": True},
        )
        self.events.append(record)
        return record

    def finish(self) -> dict[str, Any]:
        return {"degraded": True, "reason": "NullTracer writes nothing",
                "event_count": len(self.events), "handoff_count": len(self.handoffs)}

    def replay_log(self) -> list[dict[str, Any]]:
        return list(self.replay_log_)


class _NullSpan:
    """Context manager yielded by :class:`NullTracer`. Records entry and exit."""

    __slots__ = ("_tracer", "_name", "_attrs")

    def __init__(self, tracer: NullTracer, name: str, attrs: dict[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._attrs = attrs

    def __enter__(self) -> _NullSpan:
        self._tracer.events.append({"kind": "span", "name": self._name,
                                    "status": "started", **self._attrs})
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._tracer.events.append({
            "kind": "span", "name": self._name,
            "status": "error" if exc is not None else "ok", **self._attrs})
        return False


def rules_decide() -> Callable[[DecisionRequest], Decision]:
    """A rules-only decision callable: answers from the option list, labelled.

    Used when ``decision/`` is not available. It never fabricates a preference:
    it returns the first option with a flat probability and
    ``DecisionSource.RULES``/``degraded=True``. A graph running on this answers
    every routing question "deterministically, no model", which is exactly what an
    offline demo should say.
    """

    def decide(request: DecisionRequest) -> Decision:
        options = list(request.options)
        if not options:
            options = ["yes", "no"]
        share = round(1.0 / len(options), 4)
        probabilities = {option: share for option in options}
        total = sum(probabilities.values()) or 1.0
        probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}
        choice = options[0]
        return Decision(
            request_id=request.request_id,
            question=request.question,
            choice=choice,
            probabilities=probabilities,
            confidence=probabilities[choice],
            source=DecisionSource.RULES,
            model="graph.state.rules_decide",
            degraded=True,
            raw={"reason": "no decision backend was supplied",
                 "decision_point": request.decision_point},
        )

    return decide
