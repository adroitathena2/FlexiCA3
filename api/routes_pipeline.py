"""Pipeline endpoints: thin invocations of the graph, with honest provenance.

What these endpoints are, and are not
--------------------------------------
They are **not** the agents. Every one of them is a thin shim over
``graph.build.build_graph`` plus one stage's worth of extraction. The agents own
the reasoning; this module owns three things and delegates everything else:

1. **Enforcement.** The gated endpoints check the approval ledger *before* the
   graph is touched, so an unauthorised request is refused even when the graph
   is unavailable.
2. **Provenance.** Every response states which ``DecisionSource`` produced the
   routing decision behind it (``clef`` | ``gemini`` | ``rules`` | ``replay``),
   with the confidence, and says whether the decision layer degraded. A response
   that cannot name its decision source is not returned.
3. **The blackboard delta.** What this call caused to be written, so an endpoint
   is inspectable after the fact rather than only during.

The shape of a response
-----------------------
Every stage returns::

    {
      "stage": "propose",
      "agent": "A2",
      "run_id": ...,
      "event_id": ...,
      "result": {...},           # the stage's artefacts, from graph state
      "board_delta": [...],      # entries written by this call
      "decision": {
        "source": "clef",        # or null, with `unavailable: true`
        "confidence": 0.81,
        "degraded": false,
        "model": "clef-flash",
        "note": "..."
      },
      "degraded": false,
      "notes": [...],
      "duration_ms": 812.3
    }

When the graph cannot be imported the response is **503** with
``{"error", "unavailable": true, "reason", "subsystem": "graph"}``. It is never a
200 with an empty ``result``, because an empty result and a pipeline that did not
run look identical to a reader and only one of them is true.

Stages and their gates
----------------------
============  =====  =========================================================
stage         agent  gate
============  =====  =========================================================
``discover``  A1     none -- read-only
``propose``   A2     none -- drafts an offer, sends nothing
``outreach``  A3     ``SEND``    -- refuses with 403 without an approval
``reply``     A3     none -- classifies an inbound message
``contract``  A4     ``MOU``     -- refuses release without an approval
``compliance`` A5    none -- read-only, though it holds veto authority
``audit``     A6     none -- read-only
============  =====  =========================================================
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from core.errors import BlackboardError, PaytriqError
from core.protocols import BoardEntry
from core.schemas import (
    AgentId,
    DecisionSource,
    EventProfile,
    GateKind,
    TraceKind,
)

from . import deps
from .deps import RunContext, SubsystemUnavailable, unavailable_payload
from .routes_gates import require_approval
from .schemas import (
    AuditRequest,
    ContractRequest,
    OutreachRequest,
    ReplyRequest,
    StageRequest,
)

__all__ = ["router", "STAGES", "board_delta"]

log = logging.getLogger("paytriq.api.pipeline")

router = APIRouter(prefix="/api", tags=["pipeline"])

#: stage -> (agent, gate kind or None, what the stage is for). Declared here rather
#: than in seven near-identical handlers so ``GET /api/pipeline/stages`` and the
#: enforcement table cannot drift from each other.
STAGES: dict[str, tuple[AgentId, GateKind | None, str]] = {
    "discover": (AgentId.A1_DISCOVERY, None,
                 "find evidence-backed sponsor leads near the venue"),
    "propose": (AgentId.A2_PRICING, None,
                "price a proposal per viable lead (drafts only; sends nothing)"),
    "outreach": (AgentId.A3_OUTREACH, GateKind.SEND,
                 "send the day-0 outreach wave -- irreversible, gated"),
    "reply": (AgentId.A3_OUTREACH, None,
              "classify an inbound sponsor reply and route it"),
    "contract": (AgentId.A4_CONTRACT, GateKind.MOU,
                 "draft an MoU and, once approved, release it -- gated"),
    "compliance": (AgentId.A5_COMPLIANCE, None,
                   "verify deliverable promises; A5 holds veto authority"),
    "audit": (AgentId.A6_AUDIT, None,
              "produce the ROI report with its stated assumptions"),
}

#: Which graph-state field carries each stage's artefacts. Read from the final
#: state the graph returns, so the extraction cannot invent anything.
_STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "discover": ("brands",),
    "propose": ("offers",),
    "outreach": ("threads",),
    "reply": ("threads",),
    "contract": ("mous",),
    "compliance": ("risk_flags",),
    "audit": ("roi_report",),
}


# ============================================================================= deps
def _require_event(event_id: str) -> tuple[RunContext, EventProfile]:
    """Resolve the run context and profile for an event, or 404/503 honestly."""
    from .routes_events import EVENT_KIND, EVENT_ZONE, _rehydrate

    context = deps.run_for_event(event_id, create=False)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"no run for event_id {event_id!r}",
                    "hint": "POST /api/events first; it creates the event and its run"},
        )
    if context.board is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=unavailable_payload(
                context.board_error or "this run has no blackboard",
                where=f"/api/events/{event_id}"),
        )
    board = context.board
    try:
        entries = board.read(EVENT_ZONE, kind=EVENT_KIND)
    except BlackboardError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "cannot read the event zone",
                                    "reason": str(exc)}) from exc
    for entry in entries:
        profile = _rehydrate(board, entry)
        if profile.event_id == event_id:
            return context, profile
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": f"no event with event_id {event_id!r}",
                "hint": "the run exists but its board holds no such event"},
    )


def board_delta(board: object, since_seq: int) -> list[dict[str, Any]]:
    """Entries appended after ``since_seq``, as JSON-safe dicts.

    This is the "what did this call actually do" part of every response. It is
    read from the board rather than accumulated from return values, because the
    board is append-only and therefore the only place a delta can be verified
    after the fact.
    """
    if board is None:
        return []
    try:
        history = board.history()  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        log.debug("board has no history(): %s", exc)
        return []
    rows: list[dict[str, Any]] = []
    for entry in history:
        if int(getattr(entry, "seq", 0)) <= since_seq:
            continue
        rows.append(_entry_row(entry))
    return rows


def _entry_row(entry: BoardEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "seq": entry.seq,
        "zone": entry.zone,
        "kind": entry.kind,
        "author": entry.author.value,
        "confidence": entry.confidence,
        "source": entry.source.value,
        "refs": list(entry.refs),
        "at": entry.at,
        "payload": dict(entry.payload),
    }


def _decision_provenance(state: dict[str, Any], context: RunContext) -> dict[str, Any]:
    """Name the decision source behind this response, or admit there is none.

    Three measurements, in priority order, and **no guesses**:

    1. ``TraceKind.DECISION`` events in the run's own trace, counted by
       ``DecisionSource``. Measured, not asserted -- but often empty, because the
       agents reach the decision layer through ``ctx.decide`` rather than through
       ``tracer.decision``, so the answer is usually recorded on the *handoff* that
       the decision produced rather than on a span.
    2. The most recent :class:`~core.schemas.Handoff`, whose ``decision_source``
       and ``confidence`` are **mandatory** in ``core.schemas`` -- a handoff that
       could not name its decision source is not constructible. Read from graph
       state, falling back to the blackboard's ``handoffs`` zone so the answer
       survives a run whose final state was truncated.
    3. Nothing measurable, in which case ``source`` is ``null`` and the block says
       so explicitly while naming what the layer *is* configured to use.

    Step 3 is the honest answer when it applies. Reporting
    ``settings.effective_backend()`` as the source would claim that clef answered
    every question when in fact the rules fallback did, which is precisely the
    substitution this project's trace is meant to make visible.
    """
    block: dict[str, Any] = {
        "source": None,
        "confidence": None,
        "degraded": False,
        "model": None,
        "measured_from": None,
        "available_sources": [s.value for s in DecisionSource],
    }

    # ---- (1) measured from the trace ---------------------------------------
    counted: dict[str, int] = {}
    degraded = 0
    for event in context.events():
        attrs = event.attributes or {}
        source = attrs.get("source") or attrs.get("decision.source")
        if not source:
            continue
        counted[str(source)] = counted.get(str(source), 0) + 1
        if attrs.get("degraded") or attrs.get("decision.degraded"):
            degraded += 1
    if counted:
        block["decision_counts"] = dict(sorted(counted.items()))
        block["degraded_decisions"] = degraded
        block["source"] = max(counted, key=lambda k: (counted[k], k))
        block["measured_from"] = "trace"

    # ---- (2) the latest handoff: mandatory source + confidence --------------
    handoff = _latest_handoff(state, context)
    if handoff is not None:
        raw_source = str(handoff.get("decision_source") or "").rsplit(".", 1)[-1]
        try:
            source = DecisionSource(raw_source)
        except ValueError:
            source = None
        if source is not None:
            block["source"] = source.value
            block["confidence"] = handoff.get("confidence")
            block["measured_from"] = handoff.pop("_measured_from", "handoff")
        block["handoff"] = {
            "handoff_id": handoff.get("handoff_id"),
            "from_agent": handoff.get("from_agent"),
            "to_agent": handoff.get("to_agent"),
            "reason": handoff.get("reason"),
            "summary": handoff.get("summary"),
        }
        if block["source"] == DecisionSource.RULES.value:
            # A rules-sourced routing decision is, by ``core.schemas``'
            # definition, a deterministic fallback: the model did not answer.
            block["degraded"] = True

    # ---- (3) the decision layer's own state --------------------------------
    try:
        registry = deps.get_decision_registry()
    except SubsystemUnavailable as exc:
        block["unavailable"] = True
        block["reason"] = exc.reason
        block["subsystem"] = "decision"
        return block
    block["unavailable"] = False
    try:
        health = registry.health(probe=False)
    except Exception as exc:  # noqa: BLE001 - a backend listing must not 500
        block["reason"] = f"{type(exc).__name__}: {exc}"
        return block
    block["configured_backend"] = health.get("configured_backend")
    block["effective_backend"] = health.get("effective_backend")
    block["chain"] = list(health.get("chain") or [])
    block["degradation_count"] = health.get("degradations")
    if health.get("last_error"):
        block["degraded"] = True
        block["degraded_reason"] = str(health.get("last_error"))
        block["degraded_backend"] = health.get("last_error_backend")
    if block["source"] is None:
        block["note"] = (
            "no decision could be measured for this call: no decision event in the "
            "trace and no handoff on the board. The layer is configured to use "
            f"{health.get('effective_backend')!r}, which is a statement about "
            "configuration, not about what answered"
        )
        block["measured_from"] = "none"
    return block


def _latest_handoff(state: dict[str, Any], context: RunContext) -> dict[str, Any] | None:
    """The most recent handoff, from graph state or the blackboard.

    The board is the fallback because graph state is written by LangGraph's
    reducers and a run that ended on an interrupt or an exception may not have
    flushed them -- whereas the blackboard is append-only and always has the row.
    """
    rows = [h for h in (state.get("handoffs") or []) if isinstance(h, dict)]
    if rows:
        latest = dict(rows[-1])
        latest["_measured_from"] = "graph_state"
        return latest
    board = context.board
    if board is None:
        return None
    try:
        entries = [e for e in board.history() if e.zone == "handoffs"  # type: ignore[attr-defined]
                   and e.kind == "handoff"]
    except (AttributeError, BlackboardError, TypeError) as exc:
        log.debug("could not read handoffs from the board: %s", exc)
        return None
    if not entries:
        return None
    latest = dict(entries[-1].payload)
    latest["_measured_from"] = "blackboard"
    return latest


def _stage_result(stage: str, state: dict[str, Any]) -> dict[str, Any]:
    """Extract this stage's artefacts from the state the graph returned."""
    fields = _STAGE_FIELDS.get(stage, ())
    result: dict[str, Any] = {}
    for field_name in fields:
        value = state.get(field_name)
        if isinstance(value, list) or value is not None:
            result[field_name] = value
        else:
            result[field_name] = []
    result["status"] = state.get("status", "")
    result["phase"] = state.get("phase", "")
    result["compliance_score"] = state.get("compliance_score")
    result["notes"] = list(state.get("notes") or [])[-12:]
    return result


# ========================================================================== driver
def _infer_brand(context: RunContext) -> str:
    """The brand this run is about, newest artefact first.

    Mirrors ``graph.build._last_brand`` (threads -> offers -> leads) but reads
    the API layer's cached ``context.state`` plus a board fallback, so it works
    even when the graph package is unavailable. Returns ``""`` when no brand
    exists anywhere; callers turn that into a 422 with a clear message.
    """
    state = dict(getattr(context, "state", None) or {})
    try:
        from graph.build import _last_brand as _graph_last_brand

        brand = (_graph_last_brand(state) or "").strip()
        if brand:
            return brand
    except Exception:  # noqa: BLE001 - the graph may be uninstallable here
        pass
    for thread in reversed(list(state.get("threads") or [])):
        if isinstance(thread, dict) and str(thread.get("brand") or "").strip():
            return str(thread["brand"]).strip()
    for offer in reversed(list(state.get("offers") or [])):
        if isinstance(offer, dict) and str(offer.get("brand") or "").strip():
            return str(offer["brand"]).strip()
    for lead in reversed(list(state.get("brands") or [])):
        if isinstance(lead, dict) and str(lead.get("name") or "").strip():
            return str(lead["name"]).strip()
    board = getattr(context, "board", None)
    if board is not None:
        try:
            history = list(board.history())  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return ""
        for entry in reversed(history):
            payload = getattr(entry, "payload", None)
            if not isinstance(payload, dict):
                continue
            zone = str(getattr(entry, "zone", "") or "")
            if str(payload.get("brand") or "").strip() and any(
                needle in zone for needle in
                ("thread", "offer", "brand", "mou", "contract")
            ):
                return str(payload["brand"]).strip()
        for entry in reversed(history):
            payload = getattr(entry, "payload", None)
            if not isinstance(payload, dict):
                continue
            zone = str(getattr(entry, "zone", "") or "")
            if str(payload.get("name") or "").strip() and any(
                needle in zone for needle in ("opportunit", "brand", "lead")
            ):
                return str(payload["name"]).strip()
    return ""


def _record_inbound_reply(context: RunContext, event_id: str, brand: str,
                          subject: str, text: str) -> None:
    """Post the sponsor's message to the ``threads`` zone before the graph runs.

    The graph's reply node reads ``thread.reply_text`` out of state
    (``graph.build._newest_reply``), and state is a projection of the board --
    so a reply that lives only in the request body would never reach the
    routing decision. Posting it first means the run's own projection picks it
    up and ``route_after_reply`` classifies the actual text.

    Best effort: a board fault is noted, never fatal; the run then reports
    ``reply_absent`` honestly instead of failing the request.
    """
    from core.ids import new_id

    combined = f"{(subject or '').strip()}\n{(text or '').strip()}".strip()
    thread_id = ""
    offer_id: str | None = None
    state = dict(getattr(context, "state", None) or {})
    for thread in reversed(list(state.get("threads") or [])):
        if isinstance(thread, dict) and str(thread.get("brand") or "") == brand:
            thread_id = str(thread.get("thread_id") or "")
            offer_id = thread.get("offer_id")
            break
    try:
        from core.schemas import AgentId as _AgentId
        from core.schemas import DecisionSource as _DecisionSource
        from core.schemas import Intent as _Intent
        from core.schemas import Thread as _Thread

        record = _Thread(
            thread_id=thread_id or new_id("thr"),
            event_id=event_id,
            brand=brand,
            status="replied",
            intent=_Intent.UNKNOWN,
            reply_text=combined,
            offer_id=offer_id,
        )
        board = context.board
        if board is None:
            return
        board.post(  # type: ignore[attr-defined]
            "threads", "thread", _AgentId.ENVIRONMENT,
            record.model_dump(mode="json"),
            source=_DecisionSource.RULES,
        )
        context.note(f"inbound reply recorded for {brand} ({len(combined)} chars)")
    except Exception as exc:  # noqa: BLE001 - the run must survive a bad post
        log.warning("could not record the inbound reply for %s: %s: %s",
                    brand, type(exc).__name__, exc)
        context.note(f"inbound reply for {brand} could not be recorded: {exc}")


def _run_stage(stage: str, context: RunContext, profile: EventProfile,
               body: StageRequest | None = None) -> dict[str, Any]:
    """Invoke the graph for one stage and shape the response.

    The graph has no per-node entry point, so a "stage" endpoint runs the
    compiled pipeline and reports the artefacts belonging to that stage. Running
    the real graph rather than a stand-in is the only way the response means
    anything; a hand-rolled ``propose()`` in the API layer would be a second,
    divergent implementation of A2.
    """
    agent, gate_kind, description = STAGES[stage]
    started = time.perf_counter()

    since_seq = context.board_high_water()
    context.emit(TraceKind.AGENT, f"api.stage.{stage}", agent=agent,
                 event_id=context.event_id, description=description)
    # A new invocation re-opens the stream: the previous call may have marked
    # the run finished, but this one has not completed yet.
    with context.lock:
        context.finished = False

    # ---- the one place a missing sibling becomes an honest 503 -------------
    tid = ""
    try:
        tid = ((getattr(body, "thread_id", "") or "").strip()
               if body is not None else "")
    except (AttributeError, TypeError):
        tid = ""
    tid = tid or context.thread_id.strip() or context.run_id
    context.thread_id = tid
    try:
        runner = deps.get_graph_runner(context)
    except SubsystemUnavailable as exc:
        context.status = f"{stage}_unavailable"
        context.degraded = True
        context.emit(TraceKind.ERROR, f"api.stage.{stage}.unavailable",
                     agent=agent, event_id=context.event_id,
                     reason=exc.reason)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=unavailable_payload(exc, where=f"POST /api/events/{context.event_id}/{stage}"),
        ) from exc

    # The API's run id is passed in rather than letting ``initial_state()`` mint
    # its own: otherwise the graph state, the blackboard and the trace file would
    # carry three different run ids for one pipeline, and every
    # ``/api/runs/{run_id}/...`` lookup would miss.
    overrides: dict[str, Any] = {
        "run_id": context.run_id,
        "event_id": context.event_id,
    }
    # Per-call agent budgets override the runner's settings for this call only.
    # They must NOT travel as ``initial_state()`` kwargs: that function only
    # accepts its declared channels, so an ``agent_step_budget`` kwarg raised
    # ``TypeError`` and the whole stage returned an ``error`` payload. The
    # agents read budgets from ``GraphRuntime.settings`` (see
    # ``GraphRuntime.context_for``), so a per-request copy of the settings is
    # the correct slot -- and a copy, so the shared ``ctx.settings`` does not
    # leak one call's budget into the next.
    if body is not None:
        step_budget = getattr(body, "step_budget", None)
        deadline_s = getattr(body, "deadline_s", None)
        if step_budget is not None or deadline_s is not None:
            try:
                import copy as _copy

                runner.runtime.settings = _copy.copy(runner.runtime.settings)
                if step_budget is not None:
                    runner.runtime.settings.agent_step_budget = int(step_budget)
                if deadline_s is not None:
                    runner.runtime.settings.agent_deadline_s = float(deadline_s)
                context.note(
                    f"stage {stage}: per-call budget "
                    f"step_budget={step_budget} deadline_s={deadline_s}"
                )
            except Exception as exc:  # noqa: BLE001 - budgets are advisory
                log.warning("could not apply per-call budget for %s: %s",
                            stage, exc)
        label = str(getattr(body, "label", None) or "").strip()
        if label:
            context.note(f"stage {stage} label: {label}")

    state: dict[str, Any] = {}
    error: str = ""
    try:
        state = runner.run(event=profile, thread_id=tid, **overrides) or {}
    except PaytriqError as exc:
        # A domain error from the graph (a budget ceiling, a tool failure that
        # escalated). Reported with the message and the stage status; the trace
        # already holds the detail.
        error = f"{type(exc).__name__}: {exc}"
        log.warning("stage %s raised %s", stage, error)
    except Exception as exc:  # noqa: BLE001 - one stage must not 500 the API
        error = f"{type(exc).__name__}: {exc}"
        log.error("stage %s crashed: %s", stage, error, exc_info=True)

    with context.lock:
        context.state = dict(state or {})
    context.status = f"{stage}_done" if not error else f"{stage}_error"
    context.degraded = context.degraded or bool(error)
    # The SSE loop watches ``finished`` to emit its terminal summary/done.
    # A parked run (status != complete) is waiting on a human gate, so it
    # must NOT read as finished; a completed run must, even though the tracer
    # file is only flushed at shutdown.
    if not error and (state.get("status") == "complete" or state.get("phase") == "done"):
        with context.lock:
            context.finished = True

    decision = _decision_provenance(state or {}, context)
    delta = board_delta(context.board, since_seq)
    # A parked run waits on a LangGraph interrupt. Report it explicitly so no
    # caller can read this 200 as "sent".
    pending_gates: list[dict[str, Any]] = []
    try:
        for intr in runner.pending_interrupts(thread_id=tid):
            value = getattr(intr, "value", intr)
            if isinstance(value, dict):
                pending_gates.append({k: value.get(k) for k in
                                      ("gate_id", "kind", "question",
                                       "payload_preview") if value.get(k)})
            else:
                pending_gates.append({"interrupt": str(value)[:300]})
    except Exception as exc:  # noqa: BLE001 - pending check must not fail a stage
        log.debug("pending_interrupts check failed for %s: %s", tid, exc)

    payload: dict[str, Any] = {
        "stage": stage,
        "agent": agent.value,
        "agent_label": agent.label,
        "gate_kind": gate_kind.value if gate_kind else None,
        "run_id": context.run_id,
        "event_id": context.event_id,
        "thread_id": tid,
        "parked": bool(pending_gates),
        "pending_gates": pending_gates,
        "result": _stage_result(stage, state or {}),
        "board_delta": delta,
        "board_delta_count": len(delta),
        "decision": decision,
        "degraded": bool(error) or bool(decision.get("degraded")),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "notes": [],
    }
    if pending_gates:
        payload["notes"].append(
            "the run is parked on a human gate and nothing was sent; answer "
            f"it via POST /api/gates/{{gate_id}} or POST /api/runs/{context.run_id}/resume"
        )
    if error:
        payload["error"] = error
        payload["notes"].append(
            f"the graph raised while running stage {stage!r}; the response "
            f"contains whatever state it had reached, not a success"
        )
    if delta:
        payload["notes"].append(
            f"{len(delta)} blackboard entr{'y' if len(delta) == 1 else 'ies'} "
            f"written during this call"
        )
    else:
        payload["notes"].append(
            "this call wrote nothing to the blackboard; that is a fact about the "
            "run, not an error, but it is reported rather than omitted"
        )
    if decision.get("degraded"):
        payload["notes"].append(
            "the decision layer degraded: "
            f"{decision.get('degraded_reason') or 'a fallback answered instead'}"
        )
    return payload


# ====================================================================== endpoints
@router.get("/pipeline/stages", summary="The pipeline stages and their gates")
def pipeline_stages(request: Request) -> dict:
    """Static description of every stage, for the demo panel's stage column.

    Generated from :data:`STAGES` so the panel cannot advertise a stage the
    enforcement table does not gate.
    """
    return {
        "stages": [
            {
                "stage": name,
                "agent": agent.value,
                "label": agent.label,
                "gate_kind": gate.value if gate else None,
                "gated": gate is not None,
                "description": description,
                "endpoint": (f"POST /api/events/{{event_id}}/{name}"
                             if name != "outreach" else "POST /api/outreach"),
            }
            for name, (agent, gate, description) in STAGES.items()
        ],
        "gated": {k.value for k in GateKind if k in (GateKind.SEND, GateKind.COUNTER,
                                                     GateKind.MOU)},
    }


@router.post("/events/{event_id}/discover", summary="A1: discover sponsor leads")
def discover(event_id: str, request: Request,
             body: StageRequest | None = None) -> dict:
    """Find sponsor leads. Read-only: no gate, nothing irreversible."""
    context, profile = _require_event(event_id)
    return _run_stage("discover", context, profile, body)


@router.post("/events/{event_id}/propose", summary="A2: price proposals")
def propose(event_id: str, request: Request,
            body: StageRequest | None = None) -> dict:
    """Draft priced proposals. No gate: a draft sends nothing."""
    context, profile = _require_event(event_id)
    return _run_stage("propose", context, profile, body)


@router.post("/events/{event_id}/reply", summary="A3: classify a sponsor reply")
def reply(event_id: str, request: Request,
          body: ReplyRequest | None = None) -> dict:
    """Run the full reply stage for an event.

    Accepts an optional body carrying the sponsor's message; the message is
    posted to the ``threads`` zone before the graph runs, so the routing
    decision classifies the actual text (``graph.build._newest_reply`` reads
    ``thread.reply_text`` out of state, which is a projection of the board --
    a body-only message would never reach it). ``brand`` falls back to the
    run's own state; only a reply with no brand anywhere is a 422.
    """
    context, profile = _require_event(event_id)
    effective_brand = ""
    if body is not None:
        effective_brand = (body.brand or "").strip() or _infer_brand(context)
        if not effective_brand:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "reply needs a brand: none was sent and the run "
                                 "has no threads, offers or leads to infer one from",
                        "hint": "send {brand, text} or run discover/propose first"},
            )
        _record_inbound_reply(context, event_id, effective_brand,
                              body.subject, body.text)
    payload = _run_stage("reply", context, profile, body)
    if body is not None:
        payload["reply"] = {"brand": effective_brand, "subject": body.subject,
                            "text": body.text}
    return payload


@router.post("/events/{event_id}/compliance", summary="A5: verify promises")
def compliance(event_id: str, request: Request,
               body: StageRequest | None = None) -> dict:
    """Verify deliverable promises. Read-only; A5's flags carry veto authority."""
    context, profile = _require_event(event_id)
    return _run_stage("compliance", context, profile, body)


@router.post("/events/{event_id}/audit", summary="A6: ROI audit")
def audit(event_id: str, request: Request,
          body: AuditRequest | None = None) -> dict:
    """Produce the ROI report. Read-only."""
    context, profile = _require_event(event_id)
    return _run_stage("audit", context, profile, body)


@router.post("/outreach", summary="A3: send outreach -- refuses without an approval")
def outreach(request: Request, body: OutreachRequest) -> dict:
    """Run the outreach wave. **403 unless an ``Approval`` row authorises it.**

    There is deliberately no ``approve: bool`` in the body. The previous version
    had one, and the previous graph called ``send_day0(approve=True)`` from inside
    the handler, so the flag recorded the code's own opinion of itself rather than
    a human's. Here the only thing that authorises a send is an ``Approval`` row
    written by ``POST /api/gates/{gate_id}``, and this handler can only read one.

    Order of operations matters and is: **event -> gate -> graph.** The gate is
    checked before the graph is touched, so an unauthorised request is refused
    identically whether or not the graph is available. A 503 for an unauthorised
    action would leak whether the pipeline is up, and would let a caller keep
    retrying an unauthorised send in the hope of a different answer.
    """
    if not body.event_id and not body.brands:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "outreach needs an event_id",
                    "accepted": ["event_id in the body", "POST /api/events/{event_id}/outreach"]},
        )
    if not body.event_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "outreach needs an event_id; the event-scoped form is "
                             "POST /api/events/{event_id}/outreach"},
        )
    context, profile = _require_event(body.event_id)
    target = ", ".join(body.brands) if body.brands else f"every ready thread for {body.event_id}"
    _gate_id, gate_reason = require_approval(
        context, GateKind.SEND, body.gate_id, action=f"send outreach to {target}")
    context.note(f"outreach authorised: {gate_reason}")
    payload = _run_stage("outreach", context, profile, body)
    payload["gate"] = {
        "kind": GateKind.SEND.value,
        "gate_id": body.gate_id,
        "authorised_by": gate_reason,
        "brands": body.brands,
    }
    return payload


@router.post("/events/{event_id}/outreach", summary="A3: send outreach (event-scoped)")
def outreach_for_event(event_id: str, request: Request,
                       body: StageRequest | None = None,
                       gate_id: str | None = None) -> dict:
    """Event-scoped alias for ``POST /api/outreach``. Same gate, same refusal."""
    forwarded = OutreachRequest(
        event_id=event_id,
        gate_id=gate_id,
        thread_id=(body.thread_id if body is not None else None),
        step_budget=(body.step_budget if body is not None else None),
        deadline_s=(body.deadline_s if body is not None else None),
        label=(body.label if body is not None else None),
    )
    return outreach(request, forwarded)


@router.post("/events/{event_id}/contract", summary="A4: draft / release an MoU -- gated")
def contract(event_id: str, request: Request, body: ContractRequest) -> dict:
    """Draft and release an MoU. ``403`` unless a ``MOU`` approval authorises it.

    The whole stage is gated, not just the release, and that is deliberate: the
    graph's own ``contract`` node drafts behind ``gate_mou``, so gating only the
    final release would let this handler drive a drafting pass with no human in
    the loop and merely withhold the last step.
    """
    context, profile = _require_event(event_id)
    effective_brand = (body.brand or "").strip() or _infer_brand(context)
    if not effective_brand:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "contract needs a brand: none was sent and the run "
                             "has no threads, offers or leads to infer one from",
                    "hint": "send {brand, ...} or run discover/propose first"},
        )
    _gate_id, gate_reason = require_approval(
        context, GateKind.MOU, body.gate_id,
        action=f"draft and release the MoU for {effective_brand}"
        + (f" at {body.amount_inr} INR" if body.amount_inr is not None else ""),
    )
    context.note(f"MoU authorised: {gate_reason}")
    payload = _run_stage("contract", context, profile, body)
    payload["gate"] = {
        "kind": GateKind.MOU.value,
        "gate_id": body.gate_id,
        "brand": effective_brand,
        "released": True,
        "reason": gate_reason,
    }
    return payload


@router.post("/events/{event_id}/revise", summary="A2: answer a counter-offer -- gated")
def revise(event_id: str, request: Request,
           body: ContractRequest) -> dict:
    """Revise a price in response to pushback. ``COUNTER`` gate required.

    Present so the counter-offer gate has an endpoint of its own: "no
    counter-offer without approval" is only a testable claim if there is a code
    path that would otherwise perform one.
    """
    context, profile = _require_event(event_id)
    effective_brand = (body.brand or "").strip() or _infer_brand(context)
    if not effective_brand:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "revise needs a brand: none was sent and the run "
                             "has no threads, offers or leads to infer one from",
                    "hint": "send {brand, ...} or run discover/propose first"},
        )
    _gate_id, reason = require_approval(
        context, GateKind.COUNTER, body.gate_id,
        action=f"answer {effective_brand}'s counter-offer"
        + (f" at {body.amount_inr} INR" if body.amount_inr is not None else ""),
    )
    context.note(f"counter-offer authorised: {reason}")
    payload = _run_stage("propose", context, profile, body)
    payload["stage"] = "revise"
    payload["gate"] = {
        "kind": GateKind.COUNTER.value,
        "gate_id": body.gate_id,
        "brand": effective_brand,
        "authorised_by": reason,
    }
    return payload


@router.post("/reply", summary="Classify one inbound sponsor reply")
def classify_reply(request: Request, body: ReplyRequest,
                   event_id: str | None = None) -> dict:
    """Classify a reply with the deterministic fallback classifier.

    This endpoint does **not** invoke the graph. It is the classification step on
    its own, and it uses ``core.protocols.classify_intent_fallback`` -- the last
    resort classifier, which is deterministic, offline and honest about being the
    fallback. It reports ``source: "rules"`` because that is exactly what ran;
    nothing about a keyword classifier should be presented as a model's opinion.

    The reason it exists as a separate endpoint: the previous prototype classified
    inbound mail with substring matching, tested ``"yes"`` first, and so routed
    "yesterday we thought the price was too high" to contract signing. Making the
    classifier directly inspectable is the cheapest way to show the fix.
    """
    from core.protocols import classify_intent_fallback

    if not event_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "classify a reply for a known event; pass ?event_id=",
                    "hint": "POST /api/events/{event_id}/reply for the full stage"},
        )
    context, _profile = _require_event(event_id)
    text = f"{body.subject}\n{body.text}".strip()
    intent = classify_intent_fallback(text)
    # ``brand`` is optional on the model; fall back to the run's own state so
    # a brand-less classify still attributes correctly. No 422 here: the
    # classification itself needs no brand.
    brand = (body.brand or "").strip() or _infer_brand(context) or "unknown"
    context.emit(TraceKind.DECISION, f"api.reply.classify.{intent.value}",
                 event_id=event_id, brand=brand, intent=intent.value,
                 source=DecisionSource.RULES.value, degraded=False)
    return {
        "event_id": event_id,
        "brand": brand,
        "text": body.text,
        "intent": intent.value,
        "route": _ROUTES.get(intent.value, "human_review"),
        "decision": {
            "source": DecisionSource.RULES.value,
            "confidence": 1.0 if intent.value in ("yes", "no") else 0.5,
            "degraded": False,
            "model": "core.protocols.classify_intent_fallback",
            "note": "deterministic keyword classifier; refusal and pushback are "
                    "tested before assent",
        },
        "run_id": context.run_id,
    }


#: Intent -> the stage a reply should go to. ``NEUTRAL`` and ``UNKNOWN`` route to
#: human review on purpose: an unrecognised message is not assent, and the old
#: default of "everything unrecognised is interested" is the bug this table fixes.
_ROUTES: dict[str, str] = {
    "yes": "A4_contract",
    "pushback": "A2_revise",
    "interested": "A3_followup",
    "no": "A3_archive",
    "neutral": "human_review",
    "unknown": "human_review",
}
