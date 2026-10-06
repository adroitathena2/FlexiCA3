"""``StateGraph`` construction, compilation, and the run/stream conveniences.

Topology
--------
::

    START -> discover --route_after_discovery--> [replan -> discover | price]
    price --route_after_proposal--> [auction | debate -> adjudicate]
    auction --route_after_auction--> outreach --(SEND gate)--> reply
    reply --route_after_reply--> [contract | revise -> outreach | outreach | archive]
    contract --(MOU gate)--> compliance --route_after_compliance--> [audit | adjudicate]
    adjudicate --route_after_adjudication--> [audit | escalate --(ESCALATION gate)--> ...]
    audit --route_progress--> [finalize | replan | finalize(give-up) | revise] -> END
    (revise from audit is the budgeted A6->A2 ROI critique: revision_request
    Message, negotiation_rounds counter against max_replans/max_debate_rounds)

Two structural choices worth defending:

**``finalize`` is the only route to ``END``.** Routers cannot write state, so each
one queues its :class:`~core.schemas.Handoff` on the runtime for the *next node*
to drain (:func:`graph.edges.flush_handoffs`). If a router could target ``END``
directly, the handoff explaining that final routing decision would never reach
the state or the trace. Funnelling through ``finalize`` means the last decision a
run makes is as durable as the first.

**Missing agents become logged no-op nodes, not absent ones.** If A2 has not
landed yet, ``price`` still exists — it emits a note, sets ``status``, and
returns. Skipping the node entirely would delete its conditional edge, break the
``path_map``, and change the architecture diagram depending on who had finished
coding. A stable topology means the diagram in the design document is the
diagram the code compiles to, even mid-development. What *is* absent is the
reasoning: the note says which agent was missing, and ``absent_agents`` is on
the runtime for the status panel.

Interrupt safety
----------------
Every node that can be re-entered follows the same shape::

    flush handoffs -> (guarded agent run) -> build -> interrupt -> apply

A node's state writes are discarded when it raises ``interrupt``, and the node
re-runs from its first line on resume. So the agent run above a gate is memoised
on the runtime (:class:`~graph.state.StepMemo`) and the board/tracer writes
below the gate happen exactly once. See :mod:`graph.interrupts`.

The diagram
-----------
:meth:`PaytriqGraph.mermaid` returns ``get_graph().draw_mermaid()`` — drawn from
the *compiled* object, not from a hand-maintained string. The published
architecture is therefore generated from the code by construction.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from core import (
    REASONING_AGENTS,
    AgentId,
    Blackboard,
    Decision,
    DecisionRequest,
    DecisionSource,
    DisputeZone,
    EventProfile,
    HumanGateRequired,
    RunMode,
    Settings,
    Tool,
    Tracer,
    new_id,
)
from core import (
    run_id as new_run_id,
)

from . import contract_net, debate, interrupts
from .checkpointer import STATE_KEY, CheckpointSetup, build_checkpointer, default_config
from .edges import (
    NODE,
    PATH_MAPS,
    RADIUS_STEPS,
    flush_handoffs,
    flush_messages,
    flush_pending_disputes,
    route_after_adjudication,
    route_after_auction,
    route_after_compliance,
    route_after_discovery,
    route_after_proposal,
    route_after_reply,
    route_gate_outcome,
    route_progress,
)
from .registry import build_agents, registry_status
from .state import (
    GatePolicy,
    GraphRuntime,
    PaytriqState,
    ProgressLedger,
    RuntimeScratch,
    StepMemo,
    TaskLedger,
    blocking_flags,
    new_progress_ledger,
    new_task_ledger,
    read_progress_ledger,
    read_task_ledger,
    runtime_context,
)

__all__ = [
    "PaytriqGraph", "build_graph", "build_runtime", "initial_state",
    "collect_board_delta", "NODE_OWNERS",
]

log = logging.getLogger("paytriq.graph.build")

#: Which agent each node runs. Nodes not listed here are graph-owned
#: coordination steps (auction, debate, replan, archive, gates, finalize).
NODE_OWNERS: dict[str, AgentId] = {
    NODE.DISCOVER: AgentId.A1_DISCOVERY,
    NODE.PRICE: AgentId.A2_PRICING,
    NODE.REVISE: AgentId.A2_PRICING,
    NODE.OUTREACH: AgentId.A3_OUTREACH,
    NODE.REPLY: AgentId.A3_OUTREACH,
    NODE.ARCHIVE: AgentId.A3_OUTREACH,
    NODE.CONTRACT: AgentId.A4_CONTRACT,
    NODE.COMPLIANCE: AgentId.A5_COMPLIANCE,
    NODE.AUDIT: AgentId.A6_AUDIT,
    NODE.ADJUDICATE: AgentId.A7_ARBITER,
    NODE.ESCALATE: AgentId.A7_ARBITER,
}

#: Ordered precedence of the board->state classifier. First match wins, so the
#: specific zones must precede the generic ones: an ``audit`` *finding* belongs
#: in ``roi_report``, but an ``audit`` *summary* belongs in nothing, and
#: ``risk_flag`` must beat ``compliance`` because A5 posts both.
#:
#: **One record, one source of truth.** Only *agent-authored* artefacts are listed
#: here. The orchestration layer's own records — handoffs, human gates, human
#: decisions, approvals, disputes, auction bids and awards — are written straight
#: into state from the schema objects it already holds, because it is the thing
#: that made them. Listing them here as well would write each one twice: once
#: directly and once by projection. On an append-only channel that duplicate is
#: silent, and a reviewer counting handoffs would be counting an artefact of the
#: plumbing rather than of the system's reasoning.
_CLASSIFY_RULES: tuple[tuple[str, str], ...] = (
    ("risk_flag", "risk_flags"),
    ("risk", "risk_flags"),
    ("brand_lead", "brands"),
    ("lead", "brands"),
    ("offer", "offers"),
    ("proposal", "offers"),
    ("pricing", "offers"),
    ("revision", "offers"),
    ("thread", "threads"),
    ("outreach", "threads"),
    ("reply", "threads"),
    ("mou", "mous"),
    ("contract", "mous"),
    # Audit and compliance records land in ``run_events`` as raw appends. They are
    # *not* projected into ``roi_report``: that channel is a single report dict,
    # and appending to it would turn it into a list. The owning nodes
    # (:func:`_node_compliance`, :func:`_node_audit`) pick the newest record and
    # write the scalar form, which is the only place a summary may be formed.
    ("compliance", "run_events"),
    ("audit", "run_events"),
    ("roi", "run_events"),
    ("lesson", "lessons"),
    ("message", "messages"),
)

#: Zones the orchestration layer owns. Asserted in the tests: if one of these ever
#: appears in :data:`_CLASSIFY_RULES`, the double-write bug is back.
ORCHESTRATION_OWNED_ZONES = ("handoffs", "gates", "debate", "tasks", "runs")


# ============================================================== board -> state bridge
def _classify_entry(zone: str, kind: str) -> str | None:
    """Which state field a board entry belongs to, by zone then kind.

    Zone first, then kind, because agents name their zones consistently
    (``risk_flags``) and their kinds more freely (``risk_flag`` vs ``flag``).
    Anything unrecognised is dropped rather than guessed into a field: a
    mis-filed artefact is worse than an unfiled one.
    """
    for needle, field_name in _CLASSIFY_RULES:
        if needle in zone:
            return field_name
    for needle, field_name in _CLASSIFY_RULES:
        if needle in kind:
            return field_name
    return None


def collect_board_delta(rt: GraphRuntime, *, since_seq: int = 0) -> tuple[dict[str, Any], int]:
    """Project blackboard entries written after ``since_seq`` into a state delta.

    The blackboard is the agents' real output; the graph state is a projection of
    it. Reading a *delta* rather than the whole board keeps repeated projections
    idempotent and keeps the reducer semantics meaningful: with ``operator.add``
    on append-only fields, re-reading the whole board on every node would
    duplicate every handoff.

    Returns ``(delta, high_water_seq)``. The high-water mark is stored in
    ``rt.extras`` so successive projections advance instead of restarting.
    """
    marker_key = f"board_seq:{rt.thread_id or 'default'}"
    start = max(int(since_seq or 0), int(rt.extras.get(marker_key, -1)))
    try:
        history = rt.board.history()
    except Exception as exc:  # noqa: BLE001 - a board without history still runs
        log.warning("board.history() unavailable (%s); state will not project entries", exc)
        return {}, start

    delta: dict[str, list[Any]] = {}
    high = start
    for entry in history:
        if int(getattr(entry, "seq", 0)) <= start:
            continue
        high = max(high, int(getattr(entry, "seq", 0)))
        field_name = _classify_entry(str(entry.zone or ""), str(entry.kind or ""))
        if field_name is None:
            continue
        payload = getattr(entry, "payload", None)
        record = dict(payload) if isinstance(payload, dict) else {"value": str(payload)}
        # Tag every projected record with its provenance so a downstream reader
        # can trace a state artefact back to a board entry id. ``author`` matters
        # beyond bookkeeping: ``graph.state.bidder_agents`` derives auction
        # eligibility from who wrote the artefacts, so without this the board's
        # authorship — which lives on the *entry*, not in the payload — would be
        # lost at projection time.
        record.setdefault("_entry_id", getattr(entry, "entry_id", ""))
        record.setdefault("_zone", getattr(entry, "zone", ""))
        record.setdefault("_kind", getattr(entry, "kind", ""))
        record.setdefault("_seq", int(getattr(entry, "seq", 0)))
        author = getattr(entry, "author", None)
        record.setdefault("author", getattr(author, "value", None) or str(author or ""))
        delta.setdefault(field_name, []).append(record)

    rt.extras[marker_key] = high
    return delta, high


def _newest_scored_record(rt: GraphRuntime, *, keys: tuple[str, ...],
                          zones: tuple[str, ...]) -> dict | None:
    """The most recent board payload in ``zones`` carrying any of ``keys``.

    Scores and reports are *read*, never recomputed here: they belong to A5 and
    A6, and this layer's job is to carry the number forward, not to invent a
    second one that could disagree.
    """
    try:
        for entry in reversed(rt.board.history()):
            if str(getattr(entry, "zone", "")) not in zones:
                continue
            payload = getattr(entry, "payload", None)
            if not isinstance(payload, dict):
                continue
            if any(k in payload for k in keys):
                return dict(payload)
    except Exception as exc:  # noqa: BLE001 - scoring is advisory
        log.warning("could not read %s from the board: %s", keys[0], exc)
    return None


def _compliance_score_from_board(rt: GraphRuntime) -> float | None:
    """The compliance score A5 last published, normalised to 0-100.

    ``None`` means A5 never published one, and callers must keep that distinct
    from ``0.0``: "not measured" and "measured and found nothing" are different
    facts, and the routers' messages depend on the difference.
    """
    record = _newest_scored_record(
        rt, keys=("compliance_score", "score", "percent_complete"),
        zones=("audit", "compliance", "risk_flags"))
    if not record:
        return None
    for key in ("compliance_score", "score", "percent_complete"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            score = float(value)
            return round(score, 4) if score <= 1.0 else round(score, 1)
    return None


def _roi_report_from_board(rt: GraphRuntime) -> dict | None:
    """The ROI report A6 last published, as a single dict."""
    return _newest_scored_record(
        rt, keys=("report_id", "roi_multiple", "total_sponsored_inr"),
        zones=("audit", "roi"))


# ================================================================= initial state
def initial_state(*, event: EventProfile | None = None,
                  run_id: str | None = None,
                  event_id: str | None = None,
                  discovery_radius_km: float = 10.0,
                  extra: dict[str, Any] | None = None) -> PaytriqState:
    """A fully-populated starting state, with both Magentic-One ledgers seeded.

    ``total=False`` TypedDicts give LangGraph no defaults, so a state missing
    ``stall_count`` would make ``int(state.get("stall_count") or 0)`` the only
    thing standing between a fresh run and a ``TypeError`` deep inside a router.
    Every channel is seeded explicitly here instead.
    """
    profile = event or EventProfile(
        event_id=event_id or new_id("evt"),
        name="Unnamed Campus Event",
        location="Pune, Maharashtra",
        footfall=5000,
        date="2026-11-15",
        audience="college students 18-24",
    )
    rid = run_id or new_run_id()
    state: PaytriqState = {
        "event_id": event_id or profile.event_id,
        "run_id": rid,
        "phase": "pipeline",
        "agent_id": None,
        "status": "created",
        "pending_gate": None,
        "compliance_score": None,
        "roi_report": None,
        "event_profile": profile.model_dump(mode="json"),
        "task_ledger": new_task_ledger(
            facts=[f"event {profile.event_id}: {profile.name} at {profile.location}"],
            guesses=["brand leads are discoverable near the venue"],
            plan=["discover leads", "price proposals", "contact sponsors",
                  "contract", "verify compliance", "audit"],
            written_by="graph.build.initial_state",
        ),
        "progress_ledger": new_progress_ledger(
            progress_reason="run has just started",
            observed_by="graph.build.initial_state",
            source="rules",
        ),
        "stall_count": 0,
        "replan_count": 0,
        "negotiation_rounds": 0,
        "discovery_radius_km": float(discovery_radius_km),
        "handoffs": [],
        "messages": [],
        "disputes": [],
        "lessons": [],
        "approvals": [],
        "human_decisions": [],
        "trace_events": [],
        "notes": [],
        "bids": [],
        "run_events": [],
        "brands": [],
        "offers": [],
        "threads": [],
        "mous": [],
        "risk_flags": [],
        "steps_done": [],
    }
    if extra:
        state.update(extra)  # type: ignore[typeddict-item]
    return state


# ================================================================== node internals
def _memo_key(base: str, state: dict) -> str:
    """Versioned memo key so replan and loop re-entry actually re-run.

    Constant keys (``"discover:A1"``) replay the first delta forever: after a
    replan widens ``discovery_radius_km`` and bumps the task-ledger revision,
    or after a pushback loop revisits ``revise``/``outreach``, the memo still
    hits and the agent never re-executes. The key therefore carries the
    ledger revision, the radius, the replan/negotiation counters and the
    visit count. On ``interrupt`` resume the state is unchanged (writes were
    discarded) so the key is identical and the memo still replays exactly
    once; on any real re-entry at least one component differs and the agent
    re-runs.
    """
    try:
        rev = int(read_task_ledger(state).get("revision", 0))
    except Exception:
        rev = 0
    try:
        radius = float(state.get("discovery_radius_km") or 0.0)
    except (TypeError, ValueError):
        radius = 0.0
    replan = int(state.get("replan_count") or 0)
    neg = int(state.get("negotiation_rounds") or 0)
    steps = len(state.get("steps_done") or [])
    return f"{base}|rev{rev}|r{radius:.2f}|replan{replan}|neg{neg}|steps{steps}"


def _step(rt: GraphRuntime, *, key: str, owner: AgentId, state: dict,
          node: str, gate: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Run one agent step, project its board output, and optionally gate it.

    This is the single place agents are invoked, which is what makes the
    idempotency guarantee checkable: :func:`graph.state.RuntimeScratch.memo`
    keyed on ``f"{node}:{owner}"`` means a node re-entered after ``interrupt``
    replays the memoised delta instead of re-running the agent.
    """
    memo = rt.scratch.memo(key)
    if memo is not None:
        log.debug("replaying memoised step %s (node re-entered after interrupt)", key)
        delta = dict(memo.delta)
    else:
        delta = _run_agent_step(rt, owner=owner, state=state, node=node)
        rt.scratch.remember(StepMemo(key=key, delta=delta,
                                    summary=f"{owner.value} ran in {node}"))
    if gate is not None:
        # ``gate`` must call interrupt() itself so it stays the last thing before
        # the side effects it authorises. See graph.interrupts for why.
        gate_result = gate()
        delta.update(_gate_delta(rt, state, gate_result, node=node))
    return delta


def _rerun_after_approval(rt: GraphRuntime, *, first_key: str, state: dict,
                          delta: dict[str, Any], owner: AgentId, node: str,
                          gate_id: str) -> dict[str, Any]:
    """Re-run one agent after its gate was approved, merging new board output.

    The first pass parks at its own gate (no approval exists yet) and is
    memoised under ``first_key`` so interrupt resume does not double-post. The
    graph gate then approves and propagates the answer via
    ``rt.extras["graph_approvals"]``. Without this second pass the approval
    would be recorded while the agent never re-executes -- the memoised replay
    returns the parked delta and ``_find_human_decision`` never runs again --
    so the trace shows an approved send that was never attempted.

    The second pass uses a distinct memo key (``first_key`` + gate id) so it
    runs exactly once per approval and is itself resume-safe. Only artefact
    lists and notes are merged; gate fields (approvals, pending_gate, status)
    set by the caller are preserved.
    """
    second_key = f"{first_key}|post-gate:{gate_id}"
    second = _step(rt, key=second_key, owner=owner, state=state, node=node)
    for field_name, values in second.items():
        if field_name in ("agent_id", "status", "progress_ledger",
                          "approvals", "human_decisions", "pending_gate",
                          "phase", "steps_done"):
            continue
        if isinstance(values, list) and values:
            existing = delta.setdefault(field_name, [])
            if isinstance(existing, list):
                existing.extend(values)
            else:
                delta[field_name] = list(values)
    for note in second.get("notes") or []:
        existing_notes = delta.setdefault("notes", [])
        if note not in existing_notes:
            existing_notes.append(note)
    return second


def _run_agent_step(rt: GraphRuntime, *, owner: AgentId, state: dict, node: str) -> dict[str, Any]:
    """Invoke one agent and project what it wrote to the blackboard."""
    agent = rt.agent(owner)
    delta: dict[str, Any] = {"agent_id": owner.value, "status": f"{node}_running"}

    if agent is None:
        note = (f"{node}: agent {owner.value} ({owner.label}) is not registered; "
                f"step skipped and the graph continued without it")
        delta["notes"] = [rt.note(note, node)]
        delta["agent_id"] = state.get("agent_id") or owner.value
        delta["status"] = f"{node}_skipped_no_agent"
        log.warning(note)
        return delta

    ctx = rt.context_for(owner, str(state.get("event_id") or ""),
                         str(state.get("run_id") or ""))
    try:
        result = agent.run(ctx)
    except HumanGateRequired as exc:
        # Not a failure -- a *park*. An agent reached a side effect that needs a
        # human and refused to perform it, which is the governance working. The
        # gate has already been written to the board by the agent; the node's own
        # ``gate=`` callable (when present) turns it into a real ``interrupt()``
        # that suspends the run until a person answers.
        #
        # Previously this fell into the generic handler below and was recorded as
        # an agent *error*, which mislabelled correct behaviour and left the run
        # reporting success having sent nothing.
        log.info("%s: %s parked on a human gate (%s)", node, owner.value, exc)
        delta["notes"] = [rt.note(
            f"{node}: {owner.value} stopped at a human gate and performed no side "
            f"effect: {exc}", node)]
        delta["status"] = f"{node}_parked_at_gate"
        delta["agent_id"] = owner.value
        delta["gate_pending"] = True
        projection, _seq = collect_board_delta(rt)
        for field_name, values in projection.items():
            delta.setdefault(field_name, []).extend(values)
        return delta
    except Exception as exc:  # noqa: BLE001 - one agent's failure must not kill the run
        note = (f"{node}: agent {owner.value} raised {type(exc).__name__}: {exc}; "
                f"the run continues and the failure is recorded")
        log.error("%s", note, exc_info=True)
        delta["notes"] = [rt.note(note, node)]
        delta["status"] = f"{node}_agent_error"
        delta["agent_id"] = owner.value
        return delta

    projection, _seq = collect_board_delta(rt)
    for field_name, values in projection.items():
        delta.setdefault(field_name, []).extend(values)
    roi = _roi_report_from_board(rt)
    if roi is not None:
        delta["roi_report"] = roi
    score = _compliance_score_from_board(rt)
    if score is not None:
        delta["compliance_score"] = score

    notes = [
        f"{node}: {owner.label} finished in {result.steps} step(s) "
        f"({result.duration_ms:.0f} ms), sufficient={result.observation.sufficient}"
        + (f", degraded={result.degraded}" if result.degraded else "")
    ]
    notes.extend(f"{node}: {owner.value}: {n}" for n in result.notes)
    delta["notes"] = [rt.note(n, node) for n in notes]
    delta["agent_id"] = owner.value
    delta["status"] = f"{node}_done"
    # The ledger is written here, from the projection this node just collected. A
    # node cannot ask for it afterwards: the high-water mark has already moved
    # past its own records, so a second collection would report no change for a
    # step that produced several.
    delta["progress_ledger"] = _progress_from_projection(
        node, projection, _artefact_counts(state),
        observed=(result.observation.summary if result.observation else "")[:200],
        agent_claim=_agent_claim(result))
    return delta


def _agent_claim(result: Any) -> dict[str, Any]:
    """The agent's own verdict, as a partial progress-ledger claim.

    The agent knows whether *it* reached a conclusion; only it can say so. This
    records that claim verbatim and lets ``route_progress`` decide whether to
    believe it — a node that decided for itself that its step succeeded would make
    the progress router unfalsifiable.
    """
    observation = getattr(result, "observation", None)
    sufficient = bool(getattr(observation, "sufficient", False))
    degraded = bool(getattr(result, "degraded", False))
    gaps = list(getattr(observation, "gaps", []) or [])
    if not sufficient:
        reason = f"agent reported gaps: {'; '.join(gaps[:3])}"
    elif degraded:
        # "We answered, but only from the fallback" is not a finished job.
        reason = "agent answered but degraded; verdict is provisional"
    else:
        reason = "agent reached a verdict"
    return {
        "is_complete": sufficient and not gaps and not degraded,
        "is_complete_reason": (str(getattr(observation, "summary", "") or "")
                               if sufficient and not degraded else ""),
        "is_progress": sufficient and not degraded,
        "progress_reason": reason,
        "observed_by": str(getattr(result, "agent", "agent")),
    }


def _propagate_approval(rt: GraphRuntime, result: Any) -> None:
    """Make a graph-answered gate visible to the agent that raised its own gate.

    A3/A4 enforce their own ``HumanGate`` (``agents/a3_outreach.py``,
    ``agents/a4_contract.py``) with an exact ``gate_id`` match, while the graph
    raises a *different* gate in ``gates/``. Without a bridge the agent stays
    ``pending_approval`` forever and the node then claims ``outreach_sent`` /
    ``mou_approved`` on state while the board says otherwise. The bridge is two
    parts: the decision is mirrored into the agent-visible ``approvals/`` zone
    (best effort, never fatal) and stashed in ``rt.extras["graph_approvals"]``
    so :meth:`GraphRuntime.context_for` can hand it to the next agent run via
    scratch. Agents still enforce their own gate; this only gives them the
    evidence to do so on resume.
    """
    try:
        approvals = rt.extras.setdefault("graph_approvals", [])
        approvals.append({
            "gate_id": result.gate.gate_id,
            "kind": result.gate.kind.value,
            "outcome": result.decision.outcome.value,
            "decided_by": result.decision.decided_by,
            "human": bool(result.human),
            "instruction": result.decision.instruction,
        })
    except Exception:
        pass
    try:
        rt.board.post("approvals", "human_decision", AgentId.A3_OUTREACH,
                      result.decision.model_dump(mode="json"),
                      confidence=1.0 if result.human else 0.0,
                      source=DecisionSource.RULES)
    except Exception:
        pass
    try:
        rt.board.post("approvals", "approval", AgentId.A3_OUTREACH,
                      result.approval.model_dump(mode="json"),
                      confidence=1.0,
                      source=DecisionSource.RULES)
    except Exception:
        pass


def _gate_delta(rt: GraphRuntime, state: dict, result: Any, *, node: str) -> dict[str, Any]:
    """State delta produced by an answered gate.

    ``pending_gate`` is left holding the gate that was just answered rather than
    being cleared. The router that runs next
    (:func:`graph.edges.route_gate_outcome`) is the component that turns the
    outcome into a destination, and it has nothing to read if this node empties the
    field on its way out. The field's contract is therefore "the gate most
    recently answered", not "a gate currently blocking" — a gate that is still
    blocking has not been answered yet, so the run has not returned from
    ``interrupt`` and there is no state write to make.
    """
    return {
        "approvals": [result.approval.model_dump(mode="json")],
        "human_decisions": [result.decision.model_dump(mode="json")],
        "pending_gate": {
            "gate_id": result.gate.gate_id,
            "kind": result.gate.kind.value,
            "question": result.gate.question,
            "outcome": result.decision.outcome.value,
            "decided_by": result.decision.decided_by,
            "human": result.human,
            "instruction": result.decision.instruction,
            "answered_at_node": node,
        },
        "notes": [rt.note(
            f"{node}: {result.gate.kind.value} gate {result.gate.gate_id} -> "
            f"{result.outcome.value} decided_by={result.decision.decided_by}"
            f"{'' if result.human else ' (auto)'}",
            node,
        )],
        "status": f"{node}_gate_{result.outcome.value}",
    }


def _prep(rt: GraphRuntime, state: dict, *, node: str, owner: AgentId | None = None,
          summary: str = "") -> dict[str, Any]:
    """Common node preamble: drain queued handoffs, set the phase, count the visit.

    ``steps_done`` records every visit, not just the ones where an agent ran, so
    ``route_after_reply`` can count how many outreach touchpoints an event has had
    even when A3 was never registered. Counting only successful agent runs would
    make the follow-up budget depend on which agents happen to exist, which is
    exactly the kind of coupling that makes a run irreproducible.

    Track 4: also drains router-queued ``Message`` records (into ``run_events``)
    and router-filed blocking ``Dispute`` rows (into ``disputes``). Routers
    cannot write state, so both queue in ``rt.extras``; the next node carries
    them. Draining here keeps every forcing disagreement visible in state even
    when the board zone itself rejects the post.
    """
    delta: dict[str, Any] = dict(flush_handoffs(rt))
    try:
        delta.update(flush_messages(rt))
    except Exception:
        pass
    try:
        pending = flush_pending_disputes(rt)
        if pending.get("disputes"):
            existing = delta.get("disputes")
            if isinstance(existing, list):
                existing.extend(pending["disputes"])
            else:
                delta["disputes"] = list(pending["disputes"])
    except Exception:
        pass
    delta["phase"] = node
    delta["steps_done"] = [node]
    if owner is not None:
        delta["agent_id"] = owner.value
    if summary:
        delta["notes"] = [rt.note(f"{node}: {summary}", node)]
    return delta


# ============================================================================ nodes
def _seed_event_on_board(rt: GraphRuntime, state: dict) -> None:
    """Put the run's ``EventProfile`` on the blackboard, once.

    The event is the run's real input, but it arrived in LangGraph *state*, and
    the agents do not read state -- they read the blackboard. So A1 looked for an
    event profile, found none, and correctly refused: *"no event profile on the
    board; A1 refuses to invent a location"*. The refusal was right and the run
    still produced nothing, which is the worst combination: every layer behaving
    correctly while the pipeline did no work.

    Seeding here rather than in ``build_graph`` keeps it idempotent against
    re-entry (the discover node is reached again on every replan) by checking
    whether the zone already holds an entry for this event id.
    """
    event = state.get("event_profile") or {}
    if not isinstance(event, dict) or not event:
        return
    event_id = str(event.get("event_id") or state.get("event_id") or "")
    if not event_id:
        return
    try:
        for entry in rt.board.history():
            if str(getattr(entry, "zone", "")) != "event":
                continue
            payload = getattr(entry, "payload", None)
            if isinstance(payload, dict) and str(payload.get("event_id") or "") == event_id:
                return
        rt.board.post("event", "event_profile", AgentId.A1_DISCOVERY,
                      dict(event), confidence=1.0, source=DecisionSource.RULES)
    except Exception as exc:  # noqa: BLE001 - a board that refuses must not kill the run
        log.warning("could not seed the event profile onto the board: %s: %s",
                    type(exc).__name__, exc)


def _node_discover(state: dict, runtime: Any) -> dict[str, Any]:
    """A1: find brand leads inside the current search radius."""
    rt = runtime_context(runtime)
    _seed_event_on_board(rt, state)
    ledger = read_task_ledger(state)
    radius = float(state.get("discovery_radius_km") or 10.0)
    delta = _step(rt, key=_memo_key("discover:A1", state), owner=AgentId.A1_DISCOVERY, state=state,
                  node=NODE.DISCOVER)
    delta["phase"] = NODE.DISCOVER
    delta["agent_id"] = AgentId.A1_DISCOVERY.value
    delta["status"] = delta.get("status", "discover_done")
    leads = len(state.get("brands") or [])
    delta["task_ledger"] = {
        **ledger,
        "revision": int(ledger.get("revision", 0)),
        "plan": [f"discover leads within {radius:.1f} km",
                 "price viable leads", "contact sponsors"],
    }
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"discover: {leads} lead(s) in state after A1 at radius {radius:.1f} km",
                NODE.DISCOVER)]
    return delta


def _node_replan(state: dict, runtime: Any) -> dict[str, Any]:
    """Rewrite the Task Ledger and widen the radius. The Magentic-One replan.

    Deliberately *not* an agent step: a planner that re-plans by asking the model
    "what now?" will propose what it proposed last time. Here the ledger is
    rewritten from the stall the router actually observed, the radius is widened
    through a fixed, visible ladder, and the *reason* is recorded so the next
    progress check can tell whether widening helped.
    """
    rt = runtime_context(runtime)
    ledger = read_task_ledger(state)
    progress = read_progress_ledger(state)
    replan_count = int(state.get("replan_count") or 0)
    stall_count = int(state.get("stall_count") or 0) + 1
    radius = float(state.get("discovery_radius_km") or 10.0)
    factor = RADIUS_STEPS[min(replan_count, len(RADIUS_STEPS) - 1)]
    new_radius = round(radius * factor, 2)

    reason = (progress.get("progress_reason") or "no reason recorded")[:300]
    new_ledger: TaskLedger = {
        "facts": list(ledger.get("facts") or []),
        "guesses": [g for g in (ledger.get("guesses") or [])
                    if "radius" not in g.lower()],
        "plan": [
            f"re-widen discovery from {radius:.1f} km to {new_radius:.1f} km",
            "re-price any newly viable leads",
            "re-contact sponsors that had not been reached",
        ],
        "open_questions": list(ledger.get("open_questions") or []) + [
            f"why did the previous plan stall ({replan_count + 1}/{max_replans_of(rt)})?"],
        "revision": int(ledger.get("revision", 0)) + 1,
        "written_by": "graph.build._node_replan",
    }
    note = (f"replan #{replan_count + 1}: ledger revision "
            f"{new_ledger['revision']}; radius {radius:.1f} -> {new_radius:.1f} km; "
            f"stall reason: {reason}")
    log.info("%s", note)
    return {
        **_prep(rt, state, node=NODE.REPLAN),
        "task_ledger": new_ledger,
        "progress_ledger": new_progress_ledger(
            is_complete=False,
            is_progress=True,
            progress_reason=f"replanned (revision {new_ledger['revision']}): {reason}",
            changed=f"radius {radius:.1f} -> {new_radius:.1f} km",
            observed_by="graph.build._node_replan",
            source=DecisionSource.RULES.value,
        ),
        "discovery_radius_km": new_radius,
        "replan_count": replan_count + 1,
        "stall_count": stall_count,
        "status": "replanned",
        "notes": [rt.note(note, NODE.REPLAN)],
    }


def max_replans_of(rt: GraphRuntime) -> int:
    """The replan budget, named so the open question reads correctly."""
    return int(rt.settings.max_replans)


def _node_price(state: dict, runtime: Any) -> dict[str, Any]:
    """A2: produce priced proposals for viable leads."""
    rt = runtime_context(runtime)
    delta = _step(rt, key=_memo_key("price:A2", state), owner=AgentId.A2_PRICING, state=state,
                  node=NODE.PRICE)
    delta.update(_prep(rt, state, node=NODE.PRICE, owner=AgentId.A2_PRICING))
    delta["agent_id"] = AgentId.A2_PRICING.value
    delta["status"] = delta.get("status", "priced")
    return delta


def _node_revise(state: dict, runtime: Any) -> dict[str, Any]:
    """A2: revise an offer after pushback, behind the COUNTER gate."""
    rt = runtime_context(runtime)
    brand = _last_brand(state)

    delta: dict[str, Any] = {}
    if rt.gate_policy.interactive:
        delta = _step(rt, key=_memo_key("revise:A2", state), owner=AgentId.A2_PRICING, state=state,
                      node=NODE.REVISE)
        delta.update(_prep(rt, state, node=NODE.REVISE, owner=AgentId.A2_PRICING))
    else:
        # Non-interactive runs still get an approval row, so the audit trail is
        # the same shape; only ``decided_by`` differs. See graph.interrupts.
        delta = _prep(rt, state, node=NODE.REVISE, owner=AgentId.A2_PRICING)

    # The counter-offer preview is built from the merged state *and* delta,
    # because A2's revision only exists in ``delta`` at this point. Reading the
    # pre-node state would show the reviewer the price they are about to replace,
    # which is the opposite of useful.
    merged = _merged(state, delta)
    gate_result = interrupts.gate_counter(
        rt, state,
        question=(f"A sponsor pushed back on {brand}. Reply with a revised "
                  f"proposal, or hold the original price?"),
        preview=_counter_preview(merged),
        raised_by=AgentId.A3_OUTREACH,
        context={"brand": brand, "offer": _last_offer(merged)},
    )
    delta.update(_gate_delta(rt, state, gate_result, node=NODE.REVISE))
    _propagate_approval(rt, gate_result)
    delta["agent_id"] = AgentId.A2_PRICING.value
    delta["status"] = f"revised_gate_{gate_result.outcome.value}"
    delta["negotiation_rounds"] = int(state.get("negotiation_rounds") or 0) + 1
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"revise: counter-offer {gate_result.outcome.value}; "
                f"instruction={gate_result.instruction or '(none)'}", NODE.REVISE)]
    return delta


def _node_auction(state: dict, runtime: Any) -> dict[str, Any]:
    """Contract Net: who is accountable for the outreach task, and on what evidence.

    Single-executor design: only A3 holds a send transport, so OUTREACH always
    runs as A3; the award names the accountable contractor (or manager
    self-execution when uncontracted) with utility evidence, not an executor
    switch. See route_after_auction and PATH_MAPS[NODE.AUCTION].
    """
    rt = runtime_context(runtime)
    delta = _prep(rt, state, node=NODE.AUCTION, owner=AgentId.A2_PRICING,
                  summary="contract-net announcement for the outreach task")
    offers = state.get("offers") or []
    brands = state.get("brands") or []
    title = f"contact {len(offers)} sponsor(s) about {len(brands)} discovered lead(s)"
    result = contract_net.run_auction(
        rt, state,
        title=title,
        description=("Run the day-0 outreach wave: send the approved proposal to "
                     "each viable sponsor and classify the first replies."),
        manager=AgentId.A2_PRICING,
        requirements=[f"{len(offers)} priced proposal(s) awaiting a contact",
                      "every send must pass the SEND human gate"],
    )
    # Bids and the award row are written from the objects this node already holds,
    # not projected. See :data:`_CLASSIFY_RULES`: the orchestration layer owns its
    # own records, and projecting them would write each one twice.
    delta["bids"] = [b.model_dump(mode="json") for b in result.bids]
    delta["run_events"] = [result.to_payload()]
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"auction {result.task.task_id}: winner="
                f"{result.winner.value if result.winner else 'none'} "
                f"({'; '.join(result.notes)})", NODE.AUCTION),
    ]
    delta["status"] = "auction_done"
    ledger = read_task_ledger(state)
    delta["task_ledger"] = {
        **ledger,
        "facts": list(ledger.get("facts") or []) + [
            f"contract-net awarded the outreach task to "
            f"{result.winner.value if result.winner else 'the manager'} "
            f"(utility {result.winning_bid.utility if result.winning_bid else 0:.3f})",
        ],
    }
    delta["progress_ledger"] = new_progress_ledger(
        is_complete=False,
        is_progress=True,
        progress_reason=f"{len(result.bids)} bid(s) collected and awarded",
        changed=f"auction round {result.rounds}",
        observed_by="graph.build._node_auction",
        source=DecisionSource.RULES.value,
    )
    return delta


def _node_outreach(state: dict, runtime: Any) -> dict[str, Any]:
    """A3 outreach, behind the SEND gate."""
    rt = runtime_context(runtime)
    first_key = _memo_key("outreach:A3", state)
    delta = _step(rt, key=first_key, owner=AgentId.A3_OUTREACH, state=state,
                  node=NODE.OUTREACH)
    delta.update(_prep(rt, state, node=NODE.OUTREACH, owner=AgentId.A3_OUTREACH))
    # Built from the merged view: A3's threads exist in ``delta``, not in
    # ``state``, and a SEND gate whose preview says "no threads" while the run is
    # about to mail four sponsors is worse than no gate at all.
    merged = _merged(state, delta)
    preview = _outreach_preview(merged)
    targets = len(merged.get("threads") or [])

    gate_result = interrupts.gate_send(
        rt, state,
        question=(f"Send outreach to {targets} sponsor thread(s)? "
                  f"Each send is irreversible."),
        preview=preview,
        raised_by=AgentId.A3_OUTREACH,
        context={"threads": [{"brand": t.get("brand"), "status": t.get("status")}
                             for t in (merged.get("threads") or [])[:8]]},
    )
    first_status = str(delta.get("status") or "")
    first_parked = bool(delta.get("gate_pending")) or "parked_at_gate" in first_status
    first_errored = "agent_error" in first_status
    delta.update(_gate_delta(rt, state, gate_result, node=NODE.OUTREACH))
    _propagate_approval(rt, gate_result)
    if gate_result.approved and (first_parked or first_errored):
        # The agent parked (or errored) before the approval existed; run it
        # again with the propagated answer so the approved send is actually
        # attempted. Without this the memoised parked delta replays and the
        # approval never reaches the transport. A first pass that already
        # succeeded is left alone so resume does not double-post.
        _rerun_after_approval(rt, first_key=first_key, state=state,
                              delta=delta, owner=AgentId.A3_OUTREACH,
                              node=NODE.OUTREACH,
                              gate_id=gate_result.gate.gate_id)
        merged = _merged(state, delta)
    delta["agent_id"] = AgentId.A3_OUTREACH.value
    # Honesty: a graph gate approval is not a delivery. A3 enforces its own
    # SEND gate (exact gate_id match) and only marks delivered on ToolStatus.OK,
    # so "outreach_sent" is claimed only when the merged threads show a real
    # send; otherwise the gate approval is recorded but the send stays pending.
    _delivered = any(bool(t.get("delivered")) and str(t.get("status")) == "sent"
                     for t in (merged.get("threads") or []))
    if gate_result.approved and _delivered:
        delta["status"] = "outreach_sent"
    elif gate_result.approved:
        delta["status"] = "outreach_gate_approved_pending_send"
    else:
        delta["status"] = f"outreach_blocked_{gate_result.outcome.value}"
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"outreach: SEND gate {gate_result.outcome.value} "
                f"(decided_by={gate_result.decision.decided_by})", NODE.OUTREACH)]
    # The gate outcome is folded into the ledger's *observation*, not its
    # judgement: "the SEND gate was approved" is a fact, and whether that counts
    # as progress is route_progress's call.
    delta["progress_ledger"] = {
        **dict(delta.get("progress_ledger") or {}),
        "changed": f"SEND gate {gate_result.outcome.value} "
                   f"(decided_by={gate_result.decision.decided_by})",
    }
    return delta


def _node_reply(state: dict, runtime: Any) -> dict[str, Any]:
    """Ingest the latest sponsor reply so ``route_after_reply`` has text to classify.

    The classification itself belongs to the *router* (it is the routing decision),
    but a router cannot mutate state, so this node's only job is to lift the reply
    text and the classified intent out of the environment and into the state.
    """
    rt = runtime_context(runtime)
    delta = _prep(rt, state, node=NODE.REPLY, owner=AgentId.A3_OUTREACH)
    reply = _newest_reply(state)
    brand = _last_brand(state)
    delta["status"] = "reply_ingested" if reply else "reply_absent"
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"reply: {len(reply)} char(s) from {brand or 'unknown sponsor'}"
                if reply else "reply: no sponsor reply is present; routing on nothing",
                NODE.REPLY)]
    delta["progress_ledger"] = _progress_from_board(
        rt, state, node=NODE.REPLY,
        observed=f"reply from {brand or 'unknown'}: {reply[:120]!r}" if reply
        else "no reply yet")
    return delta


def _node_archive(state: dict, runtime: Any) -> dict[str, Any]:
    """Close a declined thread. A4 records a lesson so the next run differs."""
    rt = runtime_context(runtime)
    delta = _prep(rt, state, node=NODE.ARCHIVE, owner=AgentId.A3_OUTREACH)
    brand = _last_brand(state)
    threads = [t for t in (state.get("threads") or [])
               if str(t.get("status")) not in ("closed_won", "closed_lost")]
    for thread in threads:
        thread["status"] = "closed_lost"
    delta["threads"] = threads

    lesson = None
    if brand:
        from core import Lesson

        lesson = Lesson(
            lesson_id=new_id("les"),
            event_id=str(state.get("event_id") or ""),
            author=AgentId.A3_OUTREACH,
            trigger=f"{brand} declined the proposal",
            correction=f"the proposal to {brand} was declined, not countered",
            rule=(f"for {brand}, do not re-pitch within this run; record the decline "
                  f"and redirect effort to an uncontacted lead"),
            confidence=0.7,
        )
        delta["lessons"] = [lesson.model_dump(mode="json")]
    delta["status"] = "archived"
    delta["progress_ledger"] = new_progress_ledger(
        is_complete=False,
        is_progress=True,
        progress_reason=f"closed {len(threads)} declined thread(s)",
        changed=f"archived {brand or 'thread'}",
        observed_by="graph.build._node_archive",
        source=DecisionSource.RULES.value,
    )
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"archive: closed {len(threads)} thread(s); lesson recorded"
                if lesson else f"archive: closed {len(threads)} thread(s)",
                NODE.ARCHIVE)]
    return delta


def _node_debate(state: dict, runtime: Any) -> dict[str, Any]:
    """Multi-agent debate over the contested pricing position."""
    rt = runtime_context(runtime)
    delta = _prep(rt, state, node=NODE.DEBATE, owner=AgentId.A7_ARBITER,
                  summary="debate opened over the contested pricing position")

    claimant, opponent = _debate_parties(state)
    result = debate.run_debate(
        rt, state,
        claimant=claimant,
        opponent=opponent,
        topic=("the price and deliverable set currently on the table for "
               f"{_last_brand(state) or 'the top-ranked sponsor'}"),
        zone=DisputeZone.PRICING,
    )
    delta["disputes"] = [result.dispute.model_dump(mode="json")]
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"debate {result.dispute.dispute_id}: {claimant.value} vs "
                f"{opponent.value}, {result.rounds} round(s), {result.status.value} "
                f"{'- ' + result.resolution if result.resolution else ''}", NODE.DEBATE),
    ]
    delta["status"] = "debated"
    ledger = read_task_ledger(state)
    delta["task_ledger"] = {
        **ledger,
        "guesses": list(ledger.get("guesses") or []) + [
            f"dispute {result.dispute.dispute_id} over pricing is unsettled"],
    }
    delta["progress_ledger"] = new_progress_ledger(
        is_complete=False,
        is_progress=True,
        progress_reason=f"debate ran {result.rounds} round(s), {result.status.value}",
        changed=f"dispute {result.dispute.dispute_id} {result.status.value}",
        observed_by="graph.build._node_debate",
        source=DecisionSource.RULES.value,
    )
    return delta


def _node_adjudicate(state: dict, runtime: Any) -> dict[str, Any]:
    """A7: adjudicate the newest dispute via the Arbiter agent's resolve().

    The Arbiter agent (``agents.a7_arbiter.ArbiterAgent.resolve``) judges the
    dispute against its cited board evidence with real ``ctx.decide`` calls;
    the debate-protocol ``adjudicate`` is only the fallback when A7 is absent
    or unusable. Role claims ("A7 adjudicates") therefore describe executed
    code, not a parallel implementation.
    """
    rt = runtime_context(runtime)
    delta = _prep(rt, state, node=NODE.ADJUDICATE, owner=AgentId.A7_ARBITER,
                  summary="arbitration of the open dispute")

    disputes = list(state.get("disputes") or [])
    result: debate.DebateResult | None = None
    newest = disputes[-1] if disputes else None
    if newest:
        try:
            dispute_obj = debate.Dispute(**{k: v for k, v in newest.items()
                                           if not k.startswith("_")})
        except Exception as exc:  # noqa: BLE001 - a malformed dispute escalates
            log.warning("cannot rehydrate dispute %r for adjudication: %s",
                        newest.get("dispute_id"), exc)
            dispute_obj = None
        if dispute_obj is not None:
            result = debate.DebateResult(
                dispute=dispute_obj,
                claimant=dispute_obj.claimant,
                opponent=dispute_obj.opponent,
                rounds=int(dispute_obj.rounds or 1),
                status=dispute_obj.status,
                resolution=dispute_obj.resolution or "",
            )
            if result.settled:
                result.adjudicated_by = dispute_obj.resolved_by
            else:
                # Prefer the real A7 agent with the dispute's cited evidence.
                _resolved_via_a7 = False
                try:
                    arbiter = rt.agent(AgentId.A7_ARBITER)
                    resolve = getattr(arbiter, "resolve", None)
                    if callable(resolve):
                        ctx = rt.context_for(
                            AgentId.A7_ARBITER,
                            str(state.get("event_id") or ""),
                            str(state.get("run_id") or ""))
                        outcome = resolve(ctx, dispute_obj)
                        # ArbiterOutcome -> DebateResult shape for the ledger.
                        from core import DisputeStatus as _DS
                        _status = _DS.ESCALATED if outcome.escalated else _DS.RESOLVED
                        result.status = _status
                        result.resolution = outcome.rationale
                        result.adjudicated_by = AgentId.A7_ARBITER
                        result.dispute.status = _status
                        result.dispute.resolution = outcome.rationale
                        result.dispute.resolved_by = AgentId.A7_ARBITER if not outcome.escalated else None
                        # resolve() already posted dispute/decision/lesson rows to
                        # the board; project them so state sees the agent's work.
                        try:
                            _proj, _ = collect_board_delta(rt)
                            for _k, _v in _proj.items():
                                if _k in ("brands", "offers", "threads", "mous",
                                          "risk_flags", "disputes", "lessons", "bids"):
                                    delta.setdefault(_k, []).extend(_v)
                        except Exception:
                            pass
                        _resolved_via_a7 = True
                except Exception as exc:  # noqa: BLE001 - fallback stays honest
                    log.warning("A7 resolve() failed (%s: %s); using debate protocol",
                                type(exc).__name__, exc)
                    _resolved_via_a7 = False
                if not _resolved_via_a7:
                    debate.adjudicate(rt, state, result)
                    result.dispute.status = result.status

    if result is not None:
        delta["disputes"] = [result.dispute.model_dump(mode="json")]
        if result.resolution:
            delta["notes"] = list(delta.get("notes") or []) + [
                rt.note(f"adjudicate: {result.resolution}", NODE.ADJUDICATE)]
        delta["status"] = f"adjudicated_{result.status.value}"
        delta["progress_ledger"] = new_progress_ledger(
            is_complete=result.settled,
            is_complete_reason=result.resolution,
            is_progress=True,
            progress_reason=f"arbiter returned {result.status.value}",
            changed=result.resolution[:200],
            observed_by="graph.build._node_adjudicate",
            source=DecisionSource.RULES.value,
        )
    else:
        # No dispute to adjudicate: adjudicate the blocking compliance flags,
        # which is the other thing A7 exists for.
        flags = blocking_flags(state)
        delta["status"] = "adjudicated_no_dispute"
        delta["progress_ledger"] = new_progress_ledger(
            is_complete=False,
            is_progress=True,
            progress_reason=(f"{len(flags)} blocking flag(s) carry to the escalation "
                             f"gate" if flags else "nothing blocking; continuing"),
            changed=f"{len(flags)} blocking flag(s)",
            observed_by="graph.build._node_adjudicate",
            source=DecisionSource.RULES.value,
        )
        if flags:
            delta["notes"] = list(delta.get("notes") or []) + [
                rt.note("adjudicate: " + ", ".join(
                    f"{f.get('code')} ({f.get('brand')})" for f in flags[:4]),
                    NODE.ADJUDICATE)]
    delta["agent_id"] = AgentId.A7_ARBITER.value
    return delta


def _node_escalate(state: dict, runtime: Any) -> dict[str, Any]:
    """The ESCALATION gate. Everything below the interrupt is the side effect."""
    rt = runtime_context(runtime)
    pending = state.get("pending_gate") or {}
    flags = blocking_flags(state)
    preview = "; ".join(f"{f.get('code')} ({f.get('brand')}): "
                        f"{str(f.get('message'))[:120]}" for f in flags[:4])

    gate_result = interrupts.gate_escalation(
        rt, state,
        question=("The arbiter could not resolve a contested area with the "
                  "evidence on the board. A person must decide whether to proceed."),
        preview=preview or str(pending.get("question") or "unresolved dispute"),
        raised_by=AgentId.A7_ARBITER,
        context={"blocking_flags": [str(f.get("code")) for f in flags],
                 "disputes": [str(d.get("dispute_id"))
                              for d in (state.get("disputes") or [])[-4:]]},
    )
    delta = _prep(rt, state, node=NODE.ESCALATE, owner=AgentId.A7_ARBITER)
    delta.update(_gate_delta(rt, state, gate_result, node=NODE.ESCALATE))
    _propagate_approval(rt, gate_result)
    delta["agent_id"] = AgentId.A7_ARBITER.value
    delta["status"] = f"escalated_{gate_result.outcome.value}"
    delta["progress_ledger"] = new_progress_ledger(
        is_complete=False,
        is_progress=True,
        progress_reason=f"escalation gate answered {gate_result.outcome.value}",
        changed=f"human gate {gate_result.gate.gate_id} {gate_result.outcome.value}",
        observed_by=f"human:{gate_result.decision.decided_by}",
        source=DecisionSource.RULES.value,
    )
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"escalate: {gate_result.outcome.value} by "
                f"{gate_result.decision.decided_by}"
                f"{'' if gate_result.human else ' (auto)'} "
                f"instruction={gate_result.instruction or '(none)'}", NODE.ESCALATE)]
    return delta


def _node_contract(state: dict, runtime: Any) -> dict[str, Any]:
    """A4: draft the MoU, behind the MOU gate."""
    rt = runtime_context(runtime)
    first_key = _memo_key("contract:A4", state)
    delta = _step(rt, key=first_key, owner=AgentId.A4_CONTRACT, state=state,
                  node=NODE.CONTRACT)
    delta.update(_prep(rt, state, node=NODE.CONTRACT, owner=AgentId.A4_CONTRACT))
    # A4's draft MoU is in ``delta``, not ``state``; the reviewer must be shown
    # the document that is actually about to be released.
    brand = _last_brand(_merged(state, delta))
    mous = _merged(state, delta).get("mous") or []
    preview = (f"{brand}: MoU at {mous[-1].get('amount_inr') if mous else 'n/a'} INR "
               f"with {len(mous[-1].get('deliverables') or []) if mous else 0} "
               f"deliverable(s)")

    gate_result = interrupts.gate_mou(
        rt, state,
        question=f"Release the MoU for {brand} to the sponsor?",
        preview=preview,
        raised_by=AgentId.A4_CONTRACT,
        context={"brand": brand, "mous": [{"mou_id": m.get("mou_id"),
                                           "status": m.get("status")} for m in mous[-3:]]},
    )
    delta.update(_gate_delta(rt, state, gate_result, node=NODE.CONTRACT))
    _propagate_approval(rt, gate_result)
    # NOTE: _gate_delta overwrote status; recover the pre-gate agent outcome
    # from the memoised first pass instead. Re-read the memo: it holds the
    # agent-only delta before gate fields were merged in.
    memo_first = rt.scratch.memo(first_key)
    memo_status = str((memo_first.delta if memo_first else {}).get("status") or "")
    first_parked = "parked_at_gate" in memo_status or bool((memo_first.delta if memo_first else {}).get("gate_pending"))
    first_errored = "agent_error" in memo_status
    if gate_result.approved and (first_parked or first_errored):
        # Same memoised-replay reason as outreach: the agent parked before the
        # approval existed, so run it again with the propagated answer.
        _rerun_after_approval(rt, first_key=first_key, state=state,
                              delta=delta, owner=AgentId.A4_CONTRACT,
                              node=NODE.CONTRACT,
                              gate_id=gate_result.gate.gate_id)
        mous = _merged(state, delta).get("mous") or []
    # Honesty: a graph MOU approval does not release the document. A4 enforces
    # its own MOU gate and only marks approved after its own HumanDecision, so
    # the graph flips the status only when the merged MoUs already show an
    # agent-performed release; otherwise the approval is recorded and the MoU
    # stays pending.
    _released = any(str(m.get("status")) in ("approved", "signed") for m in mous)
    if gate_result.approved and _released:
        delta["status"] = "mou_approved"
    elif gate_result.approved and mous:
        delta["status"] = "mou_gate_approved_pending_release"
    elif gate_result.approved:
        delta["status"] = "mou_gate_approved_no_mou"
    else:
        delta["status"] = f"mou_{gate_result.outcome.value}"
    delta["agent_id"] = AgentId.A4_CONTRACT.value
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"contract: MOU gate {gate_result.outcome.value} "
                f"(decided_by={gate_result.decision.decided_by})", NODE.CONTRACT)]
    delta["progress_ledger"] = {
        **dict(delta.get("progress_ledger") or {}),
        "changed": f"MOU gate {gate_result.outcome.value} "
                   f"(decided_by={gate_result.decision.decided_by})",
    }
    return delta


def _node_compliance(state: dict, runtime: Any) -> dict[str, Any]:
    """A5: verify deliverable commitments and raise flags (with veto power)."""
    rt = runtime_context(runtime)
    delta = _step(rt, key=_memo_key("compliance:A5", state), owner=AgentId.A5_COMPLIANCE, state=state,
                  node=NODE.COMPLIANCE)
    delta.update(_prep(rt, state, node=NODE.COMPLIANCE, owner=AgentId.A5_COMPLIANCE))
    delta["agent_id"] = AgentId.A5_COMPLIANCE.value
    delta["status"] = delta.get("status", "compliance_done")
    flags = blocking_flags(_merged(state, delta))
    total = len(_merged(state, delta).get("risk_flags") or [])
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"compliance: {total} flag(s), {len(flags)} blocking; "
                f"score={_merged(state, delta).get('compliance_score')}",
                NODE.COMPLIANCE)]
    return delta


def _node_audit(state: dict, runtime: Any) -> dict[str, Any]:
    """A6: the ROI audit, plus an A7 arbiter review on every pass.

    A7 is invoked here — not only in adjudicate — because the happy path never
    visits adjudicate (it is reached only via debate/compliance-blocking), so
    without this review A7 would never execute on a default run. The review
    runs the registered A7 agent via the memoised step helper (idempotent on
    interrupt resume via its own memo key) and merges its board projection
    (disputes/lessons) and notes into this node's delta. It never changes this
    node's ``roi_report``/``compliance_score``/``progress_ledger``/``status``:
    the audit's numbers stay A6's. When A7 is absent the review is a logged
    no-op.
    """
    rt = runtime_context(runtime)
    delta = _step(rt, key=_memo_key("audit:A6", state), owner=AgentId.A6_AUDIT, state=state,
                  node=NODE.AUDIT)
    delta.update(_prep(rt, state, node=NODE.AUDIT, owner=AgentId.A6_AUDIT))
    delta["agent_id"] = AgentId.A6_AUDIT.value
    delta["status"] = delta.get("status", "audited")
    # Read the merged view: A6's report is in ``delta``, not in ``state``, and
    # reporting a stale or absent ROI after the audit node just ran would be wrong.
    roi = _merged(state, delta).get("roi_report") or {}
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"audit: roi_multiple={roi.get('roi_multiple')} "
                f"sponsored={roi.get('total_sponsored_inr')} "
                f"assumptions={len(roi.get('assumptions') or [])}", NODE.AUDIT)]
    # A7 review: always execute so the arbiter reasons on every audit pass.
    try:
        a7_delta = _step(rt, key=_memo_key("audit:A7-review", state),
                         owner=AgentId.A7_ARBITER, state=state, node=NODE.AUDIT)
        for _key in ("disputes", "lessons", "bids", "run_events", "risk_flags",
                     "brands", "offers", "threads", "mous"):
            _values = a7_delta.get(_key)
            if _values:
                delta.setdefault(_key, []).extend(list(_values))
        _a7_notes = [n for n in (a7_delta.get("notes") or [])
                     if n not in (delta.get("notes") or [])]
        if _a7_notes:
            delta["notes"] = list(delta.get("notes") or []) + list(_a7_notes)
        else:
            delta["notes"] = list(delta.get("notes") or []) + [
                rt.note("audit: A7 arbiter review ran (no new disputes/lessons)",
                        NODE.AUDIT)]
    except Exception as exc:  # noqa: BLE001 - the review must never break the audit
        log.warning("audit: A7 review failed (%s: %s); continuing with A6 audit",
                    type(exc).__name__, exc)
    return delta


def _node_finalize(state: dict, runtime: Any) -> dict[str, Any]:
    """Close the run. The only node that drains handoffs straight to ``END``.

    Everything here is bookkeeping. No new reasoning happens in ``finalize``,
    which is what makes it safe for it to be the terminal target of every router:
    the decisions are already recorded by the time it runs.
    """
    rt = runtime_context(runtime)
    delta = _prep(rt, state, node=NODE.FINALIZE, owner=AgentId.A6_AUDIT)
    approvals = interrupts.gate_summary(state.get("approvals") or [])
    roi = state.get("roi_report") or {}
    # Provenance: stamped from the environment when set, "unknown" otherwise —
    # never fabricated. Mirrors Settings (GIT_COMMIT/CODE_SHA256, with
    # RENDER_GIT_COMMIT/GITHUB_SHA fallbacks for CI/Render) so the run
    # summary identifies its own build even when the trace was captured without
    # env vars. A zero-commit checkout records "uncommitted-working-tree".
    import os as _os
    def _env_sha(*names: str) -> str:
        for _n in names:
            _v = str(_os.getenv(_n, "") or "").strip()
            if _v:
                return _v
        return ""
    _commit = _env_sha("GIT_COMMIT", "RENDER_GIT_COMMIT", "GITHUB_SHA") or str(
        getattr(rt.settings, "git_commit", "") or "").strip() or "unknown"
    _digest = _env_sha("CODE_SHA256", "RENDER_GIT_COMMIT", "GITHUB_SHA") or str(
        getattr(rt.settings, "code_sha256", "") or "").strip() or "unknown"
    summary = {
        "event_id": state.get("event_id"),
        "run_id": state.get("run_id"),
        "status": "complete",
        "brands": len(state.get("brands") or []),
        "offers": len(state.get("offers") or []),
        "threads": len(state.get("threads") or []),
        "mous": len(state.get("mous") or []),
        "risk_flags": len(state.get("risk_flags") or []),
        "blocking_flags": len(blocking_flags(state)),
        "handoffs": len(state.get("handoffs") or []),
        "disputes": len(state.get("disputes") or []),
        "bids": len(state.get("bids") or []),
        "approvals": approvals["total"],
        "human_approvals": approvals["human_count"],
        "auto_approvals": approvals["auto_count"],
        "compliance_score": state.get("compliance_score"),
        "roi_multiple": roi.get("roi_multiple"),
        "stall_count": state.get("stall_count"),
        "replan_count": state.get("replan_count"),
        "task_ledger_revision": read_task_ledger(state).get("revision"),
        "git_commit": _commit,
        "code_sha256": _digest,
    }
    delta["status"] = "complete"
    delta["phase"] = "done"
    delta["notes"] = list(delta.get("notes") or []) + [
        rt.note(f"finalize: run complete — {summary['handoffs']} handoff(s), "
                f"{summary['approvals']} approval(s) "
                f"({summary['human_approvals']} human / {summary['auto_approvals']} auto), "
                f"{summary['stall_count']} stall(s), {summary['replan_count']} replan(s)",
                NODE.FINALIZE)]
    try:
        rt.board.post("runs", "run_summary", AgentId.A6_AUDIT, summary,
                      source=DecisionSource.RULES)
    except Exception as exc:  # noqa: BLE001 - summary is best-effort
        log.warning("could not post the run summary: %s", exc)
    return delta


# ------------------------------------------------------------------------- helpers
def _last_brand(state: dict) -> str:
    for thread in reversed(list(state.get("threads") or [])):
        if thread.get("brand"):
            return str(thread["brand"])
    for offer in reversed(list(state.get("offers") or [])):
        if offer.get("brand"):
            return str(offer["brand"])
    for lead in reversed(list(state.get("brands") or [])):
        if lead.get("name"):
            return str(lead["name"])
    return ""


def _last_offer(state: dict) -> dict:
    offers = list(state.get("offers") or [])
    return offers[-1] if offers else {}


def _newest_reply(state: dict) -> str:
    for thread in reversed(list(state.get("threads") or [])):
        text = thread.get("reply_text")
        if text:
            return str(text)
    return ""


def _outreach_preview(state: dict) -> str:
    threads = state.get("threads") or []
    if not threads:
        return "(no outreach threads are ready to send)"
    return "; ".join(f"{t.get('brand')}: {t.get('status')}" for t in threads[:6])


def _counter_preview(state: dict) -> str:
    """What the reviewer is about to authorise sending as a counter-reply."""
    brand = _last_brand(state)
    offer = _last_offer(state)
    if not offer:
        return f"counter-reply to {brand or 'the sponsor'}: (no offer on the board)"
    return (f"counter-reply to {brand}: revised offer "
            f"{offer.get('amount_inr')} INR (version {offer.get('version')}, "
            f"{len(offer.get('deliverables') or [])} deliverable(s))")


def _merged(state: dict, delta: dict) -> dict:
    """State as it will be *after* ``delta`` is applied, for previews and counts.

    A node's own writes are not in ``state`` yet — LangGraph applies the delta
    after the node returns. Reading ``state`` alone to build a gate preview would
    therefore show the reviewer what was true before the step, which for a SEND or
    MOU gate means showing them the wrong document.

    List channels are concatenated because that is what ``operator.add`` will do
    on apply; scalars take the delta's value when present.
    """
    merged = dict(state)
    for key, value in delta.items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = [*merged[key], *value]
        elif key.startswith("_"):
            continue
        else:
            merged[key] = value
    return merged


def _debate_parties(state: dict) -> tuple[AgentId, AgentId]:
    """Pick the two sides of the debate from who is actually involved.

    A fixed pair would be a fabricated conflict. If the state shows a blocked
    MoU, compliance argues with contract; otherwise pricing argues with outreach,
    because those are the two that disagree about value in every run.
    """
    for dispute in reversed(list(state.get("disputes") or [])):
        try:
            return AgentId(str(dispute.get("claimant"))), AgentId(str(dispute.get("opponent")))
        except (ValueError, TypeError):
            continue
    if blocking_flags(state):
        return AgentId.A5_COMPLIANCE, AgentId.A4_CONTRACT
    return AgentId.A2_PRICING, AgentId.A3_OUTREACH


def _progress_from_projection(node: str, projection: dict[str, list[Any]],
                             before: dict[str, int], *, observed: str,
                             agent_claim: dict[str, Any] | None = None
                             ) -> ProgressLedger:
    """A progress-ledger claim derived from what actually changed on the board.

    The honesty rule: the claim says *what changed*, never *that it went well*.
    ``route_progress`` is the component that judges whether the change was
    progress; a node that concluded "went well" itself would make the router
    unfalsifiable.

    Two independent sources are combined, and which one said what is preserved:
    the **board delta** (did anything actually get written?) and the **agent's own
    verdict** (did it reach a conclusion?). They can disagree — an agent can
    produce records and still report gaps — and when they do, both facts are kept
    rather than one overwriting the other.

    ``before`` is the artefact count as the node found it. Comparing against it
    rather than an empty baseline is what lets the ledger say "brands: 0->3"
    instead of "produced 3 records", which is the same fact but hides whether the
    run started empty.
    """
    after = dict(before)
    for field_name, values in projection.items():
        after[field_name] = after.get(field_name, 0) + len(values)
    changed = ", ".join(
        f"{k}: {before.get(k, 0)}->{v}" for k, v in sorted(after.items())
        if v != before.get(k, 0)
    )
    produced = sum(len(v) for v in projection.values())
    board_reason = (f"{node} produced {produced} board record(s)"
                    if produced else f"{node} produced no new board record")
    ledger = new_progress_ledger(
        is_complete=False,
        is_complete_reason="",
        is_progress=produced > 0,
        progress_reason=board_reason,
        changed=changed or f"observed: {observed}",
        observed_by=f"graph.build.{node}",
        source=DecisionSource.RULES.value,
    )
    if not agent_claim:
        return ledger
    # The agent's completeness claim is the stronger signal when it is positive;
    # the board delta is the stronger signal when it is negative. Neither silently
    # erases the other.
    merged = {**agent_claim,
              "progress_reason": f"{agent_claim['progress_reason']}; {board_reason}",
              "changed": f"{ledger['changed']} | {observed}"
                         if observed else str(ledger["changed"])}
    return new_progress_ledger(**merged)  # type: ignore[arg-type]


def _progress_from_board(rt: GraphRuntime, state: dict, *, node: str,
                         observed: str) -> ProgressLedger:
    """Same as :func:`_progress_from_projection`, collecting its own projection.

    Only for graph-owned nodes, which do not run an agent and therefore have no
    projection to reuse. An agent node must not call this: the projection it needs
    was already consumed by :func:`_run_agent_step`, so a second call would report
    "no new board record" for a step that produced several.
    """
    projection, _seq = collect_board_delta(rt)
    return _progress_from_projection(node, projection, _artefact_counts(state),
                                     observed=observed)


def _artefact_counts(state: dict) -> dict[str, int]:
    keys = ("brands", "offers", "threads", "mous", "risk_flags", "disputes",
            "lessons", "bids")
    return {k: len(state.get(k) or []) for k in keys}


# ============================================================================ graph
@dataclass
class PaytriqGraph:
    """A compiled LangGraph plus the runtime it was built with.

    Holds both because ``stream(..., context=runtime)`` has to be called with the
    *same* runtime the graph was compiled against: the runtime schema is part of
    the compiled object, and mixing two runtimes would put one run's board behind
    another run's nodes.
    """

    compiled: Any
    runtime: GraphRuntime
    settings: Settings
    nodes: tuple[str, ...] = ()
    agents_present: tuple[str, ...] = ()
    agents_absent: tuple[str, ...] = ()
    checkpointer: CheckpointSetup | None = None
    checkpointer_owned: bool = False
    #: The runtime of the most recent :meth:`run`, kept so :meth:`resume` can
    #: continue it. Single-run by design; a server handling concurrent requests
    #: must build one graph per request, not share this object.
    _active_runtime: GraphRuntime | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ diagram
    def mermaid(self) -> str:
        """The architecture diagram, drawn from the compiled graph.

        Generated rather than written down, so the published diagram cannot drift
        from the topology that actually runs.
        """
        return self.compiled.get_graph().draw_mermaid()

    def nodes_present(self) -> list[str]:
        return sorted(str(n) for n in self.compiled.get_graph().nodes)

    def describe(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes_present(),
            "agents_present": list(self.agents_present),
            "agents_absent": list(self.agents_absent),
            "checkpointer": self.checkpointer.status() if self.checkpointer else None,
            "registry": registry_status(),
        }

    # ---------------------------------------------------------------------- run
    def fresh_runtime(self) -> GraphRuntime:
        """A copy of this graph's runtime with an empty memo store and board.

        A built graph is a *single-run* object. ``StepMemo`` exists to make one
        node's re-execution idempotent, which is exactly wrong for a second run:
        the second run would replay the first run's agent deltas and post nothing
        new. So a fresh run gets a fresh runtime, and the expensive collaborators
        (checkpointer, settings, agents) are shared.

        The board is preserved, not replaced. A caller who seeded the board before
        calling ``run()`` seeded it *on purpose* (seeded leads, a pre-loaded
        sponsor reply); swapping in an empty board would silently discard that.
        The memo store, the projection high-water mark and the queue are reset,
        because those describe the previous run rather than the environment.
        """
        base = self.runtime
        clone = replace_runtime(base, scratch=RuntimeScratch(),
                                thread_id="", run_id="", event_id="")
        clone.extras = {}
        return clone

    def run(self, *, event: EventProfile | None = None,
            state: PaytriqState | None = None,
            thread_id: str | None = None,
            config: dict[str, Any] | None = None,
            recursion_limit: int = 60,
            reuse_runtime: bool = False,
            **state_overrides: Any) -> dict[str, Any]:
        """Invoke to completion, or until the next gate parks the run.

        Returns the state as persisted in the checkpointer rather than the last
        streamed value. With ``stream_mode="values"`` a parked run emits nothing,
        so reading the snapshot afterwards is the only way to report the
        interrupted state honestly — and it is also the state a *resume* will
        continue from, which is what a caller actually needs.
        """
        runtime = self.runtime if reuse_runtime else self.fresh_runtime()
        # The run's runtime is stashed so :meth:`resume` continues with it. Resume
        # *must* reuse it: the memo store is what stops the agent above a gate from
        # running twice when the node re-executes, and a fresh runtime would lose
        # exactly that.
        self._active_runtime = runtime
        cfg = config or default_config(thread_id or runtime.thread_id)
        payload = self._payload(state, event, state_overrides, runtime=runtime)
        # Drain: the run's progress is not this method's return value. The
        # ``pass`` is a deliberate discard, not a swallowed error.
        for _node_delta in self._drive(payload, cfg, recursion_limit, runtime):
            pass
        return self.state_of(cfg)

    def resume(self, value: Any, *, thread_id: str | None = None,
               config: dict[str, Any] | None = None,
               recursion_limit: int = 60) -> dict[str, Any]:
        """Resume a parked run with ``Command(resume=value)`` on the same thread.

        Uses the runtime of the run that parked, not a fresh one: the memo store is
        what keeps the agent above the gate from running a second time.
        """
        runtime = self._active_runtime or self.runtime
        cfg = config or default_config(thread_id or runtime.thread_id)
        for _node_delta in self._drive(Command(resume=value), cfg, recursion_limit,
                                       runtime):
            pass
        return self.state_of(cfg)

    def state_of(self, config: dict[str, Any] | None = None,
                 thread_id: str | None = None) -> dict[str, Any]:
        """The persisted state for a thread, straight from the checkpointer."""
        cfg = config or default_config(thread_id or self.runtime.thread_id)
        snapshot = self.compiled.get_state(cfg)
        return dict(getattr(snapshot, "values", {}) or {})

    def pending_interrupts(self, *, thread_id: str | None = None,
                           config: dict[str, Any] | None = None) -> list[Any]:
        """The gates currently blocking a thread, for the API's poll endpoint."""
        cfg = config or default_config(thread_id or self.runtime.thread_id)
        snapshot = self.compiled.get_state(cfg)
        return list(getattr(snapshot, "interrupts", ()) or ())

    def _drive(self, input_value: Any, config: dict[str, Any],
               recursion_limit: int, runtime: GraphRuntime) -> Iterator[Any]:
        """Yield per-node update deltas until the run ends or parks.

        ``updates`` mode is used internally rather than ``values`` for one
        reason: it is the mode that reports ``__interrupt__``, and this is the
        code that has to know whether the run paused.
        """
        stream = self.compiled.stream(input_value, config=config,
                                      stream_mode="updates",
                                      recursion_limit=recursion_limit,
                                      context=runtime)
        for chunk in stream:
            yield chunk
            if "__interrupt__" in chunk:
                log.info("run parked at a human gate on thread %s",
                         (config.get("configurable") or {}).get(STATE_KEY))
                return

    def stream(self, *, event: EventProfile | None = None,
               state: PaytriqState | None = None,
               thread_id: str | None = None,
               config: dict[str, Any] | None = None,
               recursion_limit: int = 60,
               stream_mode: str = "values",
               **state_overrides: Any) -> Iterable[Any]:
        """Stream the run, wiring the runtime into every node and router.

        ``stream_mode="updates"`` yields the per-node delta, which is what the API
        converts into SSE frames; ``"values"`` yields the accumulated state after
        each superstep. Both are the *real* execution trace, which is the point:
        the demo cannot show a timeline the engine did not produce.
        """
        runtime = self.fresh_runtime()
        self._active_runtime = runtime
        cfg = config or default_config(thread_id or runtime.thread_id)
        payload = self._payload(state, event, state_overrides, runtime=runtime)
        return self.compiled.stream(payload, config=cfg, stream_mode=stream_mode,
                                    recursion_limit=recursion_limit,
                                    context=runtime)

    def stream_state(self, input_value: Any, *, config: dict[str, Any] | None = None,
                     thread_id: str | None = None, recursion_limit: int = 60,
                     stream_mode: str = "updates") -> Iterable[Any]:
        """Like :meth:`stream` but accepts a raw input (``Command``, ``None``, dict)."""
        cfg = config or default_config(thread_id or self.runtime.thread_id)
        return self.compiled.stream(input_value, config=cfg, stream_mode=stream_mode,
                                    recursion_limit=recursion_limit,
                                    context=self.runtime)

    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.run(*args, **kwargs)

    def _payload(self, state: PaytriqState | None, event: EventProfile | None,
                 overrides: dict[str, Any], *,
                 runtime: GraphRuntime | None = None) -> PaytriqState:
        rt = runtime or self.runtime
        if state is not None:
            merged = dict(state)
            merged.update(overrides)
            return merged  # type: ignore[return-value]
        payload = initial_state(event=event, **overrides)
        rt.run_id = str(payload.get("run_id") or "")
        rt.event_id = str(payload.get("event_id") or "")
        if not rt.thread_id:
            rt.thread_id = default_config("")["configurable"]["thread_id"]
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"PaytriqGraph(nodes={len(self.nodes)}, "
                f"agents={len(self.agents_present)}/{len(self.agents_present) + len(self.agents_absent)})")


# ---------------------------------------------------------------------- assembly
def build_runtime(settings: Settings, *, board: Blackboard | None = None,
                  tracer: Tracer | None = None,
                  tools: dict[str, Tool] | None = None,
                  decide: Callable[[DecisionRequest], Decision] | None = None,
                  agents: dict[AgentId, Any] | None = None,
                  mode: RunMode | None = None,
                  interactive: bool | None = None,
                  thread_id: str = "",
                  refresh: bool = False,
                  ) -> GraphRuntime:
    """Assemble a :class:`~graph.state.GraphRuntime` from whatever exists.

    Every collaborator is optional and every absence is visible:
    ``board``/``tracer`` fall back to the in-module stubs that keep the run going
    and log once; ``decide`` falls back to a rules callable that answers from the
    option list. Nothing here raises for a missing collaborator, because a graph
    that cannot be constructed cannot be demonstrated.
    """
    from .state import InMemoryBoardLike, NullTracer

    resolved_agents = dict(agents) if agents is not None else build_agents(
        settings, refresh=refresh)

    # `board or InMemoryBoardLike()` looked right and was wrong.
    # ``InMemoryBlackboard`` implements ``__len__``, so an *empty* board is falsy
    # and the `or` silently substituted the in-module stub -- the caller passed a
    # board, the runtime used a different one, and every artefact an agent posted
    # vanished with nothing logged. Test `is None` instead: the question is
    # whether a board was supplied, not whether it happens to be empty.
    resolved_board = board if board is not None else InMemoryBoardLike()
    resolved_tracer = tracer if tracer is not None else NullTracer()

    # Decision layer. Previously this defaulted straight to ``rules_decide()``,
    # which silently downgraded every run to deterministic keywords even when a
    # real decision model was configured and reachable. That is the single failure
    # this project exists to prevent: the run looked fine, every number was
    # internally consistent, and every decision quietly came from an ``if``.
    #
    # The rule is now: an explicit ``decide`` wins; otherwise ask the real
    # ``decision`` package whether it has a usable backend, and only fall back to
    # rules when it genuinely cannot answer. The fallback still records
    # ``DecisionSource.RULES`` with ``degraded=True`` inside each Decision, so a
    # degraded run is labelled rather than merely intended.
    resolved_decide = decide if decide is not None else _real_decide(settings)

    # Tool layer. Previously ``dict(tools or {})``, so a graph built without an
    # explicit registry gave its agents *no tools at all* -- and A1, whose entire
    # job is a tool call, silently produced zero leads with no error anywhere. The
    # registry is cheap to build and every tool already degrades honestly on its
    # own (fixture or unavailable), so building it by default is strictly better
    # than handing an agent an empty toolbox.
    if tools is not None:
        resolved_tools = dict(tools)
    else:
        try:
            from tools.registry import build_registry as _build_tools
            resolved_tools = dict(_build_tools(settings))
        except Exception as exc:  # noqa: BLE001 - agents must still run without tools
            log.warning("tool registry unavailable (%s: %s); agents get no tools",
                        type(exc).__name__, exc)
            resolved_tools = {}

    policy = GatePolicy(
        interactive=bool(settings.human_gates_interactive if interactive is None
                         else interactive),
        auto_outcome=str(settings.auto_approve_outcome),
    )
    absent = tuple(a.value for a in sorted(
        (a for a in _all_expected_agents() if a not in resolved_agents),
        key=lambda a: a.value,
    ))
    return GraphRuntime(
        settings=settings,
        agents=resolved_agents,
        board=resolved_board,
        tracer=resolved_tracer,
        tools=dict(resolved_tools),
        decide=resolved_decide,
        mode=mode or settings.run_mode,
        gate_policy=policy,
        thread_id=thread_id,
        absent_agents=absent,
    )


def _all_expected_agents() -> tuple[AgentId, ...]:
    return tuple(REASONING_AGENTS)


def _real_decide(settings: Settings) -> Callable[[DecisionRequest], Decision]:
    """The project's real decision layer, or a labelled rules fallback.

    ``decision.registry.build_registry`` already owns the whole fallback chain
    (clef -> gemini -> rules) and records the provenance of any degradation, so
    this only has to answer one question: can that package be used at all?

    ``decide`` is deliberately still a plain callable taking a ``DecisionRequest``
    and returning a ``Decision``, because that is the contract ``AgentContext``
    and every router are written against.
    """
    try:
        from decision.registry import build_registry as _build_decision_registry
    except ImportError as exc:  # pragma: no cover - sibling always present here
        log.warning("decision layer unavailable (%s); using rules fallback", exc)
        from .state import rules_decide
        return rules_decide()

    try:
        registry = _build_decision_registry(settings)
    except Exception as exc:  # noqa: BLE001 - a bad config must not kill the run
        log.warning("decision registry could not be built (%s: %s); using rules fallback",
                    type(exc).__name__, exc)
        from .state import rules_decide
        return rules_decide()

    available, why = registry.available()
    if not available:
        log.warning("decision chain unavailable (%s); using rules fallback", why)
    else:
        chain = registry.describe().get("chain") or []
        live = [f"{c['name']}/{c['model']}" for c in chain if c.get("available")]
        log.info("decision layer: %s", ", ".join(live) or why)
    return registry.decide


def build_graph(settings: Settings | None = None,
                agents: dict[AgentId, Any] | None = None,
                *, board: Blackboard | None = None,
                tracer: Tracer | None = None,
                tools: dict[str, Tool] | None = None,
                decide: Callable[[DecisionRequest], Decision] | None = None,
                runtime: GraphRuntime | None = None,
                checkpointer: Any | None = None,
                checkpointer_path: str | None = None,
                interactive: bool | None = None,
                mode: RunMode | None = None,
                refresh: bool = False,
                ) -> PaytriqGraph:
    """Wire, compile, and return a :class:`PaytriqGraph`.

    ``checkpointer`` accepts a ``CheckpointSetup``, a bare saver, or ``None``
    (which builds the SQLite one under ``settings.traces_dir``). A graph compiled
    *without* any checkpointer cannot park at a human gate, so the default here
    is always a real checkpointer.
    """
    cfg = settings or Settings()
    rt = runtime or build_runtime(cfg, board=board, tracer=tracer, tools=tools,
                                  decide=decide, agents=agents, mode=mode,
                                  interactive=interactive, refresh=refresh)

    setup, owned = _resolve_checkpointer(cfg, checkpointer, checkpointer_path)
    rt.thread_id = rt.thread_id or default_config("")["configurable"]["thread_id"]

    builder: StateGraph = StateGraph(PaytriqState, context_schema=GraphRuntime)

    nodes: dict[str, Callable[[dict, Any], dict[str, Any]]] = {
        NODE.DISCOVER: _node_discover,
        NODE.REPLAN: _node_replan,
        NODE.PRICE: _node_price,
        NODE.AUCTION: _node_auction,
        NODE.OUTREACH: _node_outreach,
        NODE.REPLY: _node_reply,
        NODE.REVISE: _node_revise,
        NODE.ARCHIVE: _node_archive,
        NODE.DEBATE: _node_debate,
        NODE.ADJUDICATE: _node_adjudicate,
        NODE.ESCALATE: _node_escalate,
        NODE.CONTRACT: _node_contract,
        NODE.COMPLIANCE: _node_compliance,
        NODE.AUDIT: _node_audit,
        NODE.FINALIZE: _node_finalize,
    }
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, NODE.DISCOVER)
    builder.add_edge(NODE.REPLAN, NODE.DISCOVER)
    builder.add_edge(NODE.AUCTION, NODE.OUTREACH)
    builder.add_edge(NODE.OUTREACH, NODE.REPLY)
    builder.add_edge(NODE.REVISE, NODE.OUTREACH)
    builder.add_edge(NODE.ARCHIVE, NODE.AUDIT)
    builder.add_edge(NODE.DEBATE, NODE.ADJUDICATE)
    builder.add_edge(NODE.CONTRACT, NODE.COMPLIANCE)
    builder.add_edge(NODE.FINALIZE, END)

    builder.add_conditional_edges(NODE.DISCOVER, route_after_discovery, PATH_MAPS[NODE.DISCOVER])
    builder.add_conditional_edges(NODE.PRICE, route_after_proposal, PATH_MAPS[NODE.PRICE])
    builder.add_conditional_edges(NODE.AUCTION, route_after_auction, PATH_MAPS[NODE.AUCTION])
    builder.add_conditional_edges(NODE.REPLY, route_after_reply, PATH_MAPS[NODE.REPLY])
    builder.add_conditional_edges(NODE.COMPLIANCE, route_after_compliance, PATH_MAPS[NODE.COMPLIANCE])
    builder.add_conditional_edges(NODE.ADJUDICATE, route_after_adjudication, PATH_MAPS[NODE.ADJUDICATE])
    builder.add_conditional_edges(NODE.ESCALATE, route_gate_outcome, PATH_MAPS[NODE.ESCALATE])
    builder.add_conditional_edges(NODE.AUDIT, route_progress, PATH_MAPS[NODE.AUDIT])

    compiled = builder.compile(checkpointer=setup.saver if setup else None)

    if rt.absent_agents:
        log.warning("building graph with %d/%d agents; absent: %s",
                    len(rt.agents), len(_all_expected_agents()),
                    ", ".join(rt.absent_agents))

    return PaytriqGraph(
        compiled=compiled,
        runtime=rt,
        settings=cfg,
        nodes=tuple(nodes),
        agents_present=tuple(sorted((a.value for a in rt.agents), key=str)),
        agents_absent=tuple(rt.absent_agents),
        checkpointer=setup,
        checkpointer_owned=owned,
    )


def replace_runtime(base: GraphRuntime, **changes: Any) -> GraphRuntime:
    """A copy of ``base`` with selected fields replaced.

    Implemented with :func:`dataclasses.replace` rather than a hand-written
    ``__init__`` copy so a field added to :class:`~graph.state.GraphRuntime` later
    is carried over automatically. A forgotten field here would silently reset a
    run's collaborators, which is the kind of bug that only shows up under load.
    """
    return dataclasses.replace(base, **changes)


def _resolve_checkpointer(settings: Settings, supplied: Any | None,
                          path: str | None) -> tuple[CheckpointSetup | None, bool]:
    """Accept a setup, a bare saver, or nothing. Returns ``(setup, we_own_it)``."""
    if isinstance(supplied, CheckpointSetup):
        return supplied, False
    if supplied is not None:
        # A bare saver: wrap it so ``.status()`` and ``.close()`` still work, and
        # do not close a saver the caller owns.
        return CheckpointSetup(saver=supplied, kind="external"), False
    setup = build_checkpointer(path, settings=settings)
    return setup, True
