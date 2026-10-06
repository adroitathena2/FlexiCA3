"""Human approval gates.

The control surface of the entire system
----------------------------------------
Three actions are irreversible once taken: sending mail, answering a
counter-offer, and releasing an MoU. :class:`core.schemas.HumanGate` exists for
exactly those three (plus escalation, which authorises nothing). Every one of
them must pass a gate, and this module is where a gate is raised, listed and
answered.

Two endpoints
-------------
``GET  /api/gates/{run_id}``     outstanding gates for a run (answered ones too,
                                 clearly separated, so an unattended run can never
                                 be presented as an attended one)
``POST /api/gates/{gate_id}``    answer a gate: approve / reject / revise

The invariant
-------------
Every ``POST`` here writes, in this order and without exception:

1. a :class:`~core.schemas.HumanDecision` -- what the human said;
2. an :class:`~core.schemas.Approval` -- the durable row that proves the gate was
   honoured;
3. a blackboard entry of kind ``human_decision`` **and** one of kind
   ``approval``, each citing the gate;
4. a ``TraceKind.HUMAN`` trace event.

:func:`record_decision` is the only function in this module that may write (2)
and it takes the approval as a *required* argument, so there is no signature by
which a decision can be recorded without the proof. The other endpoints
(``/api/outreach``, ``/api/events/{id}/contract``) then consult
:meth:`~api.deps.GateLedger.authorises` before doing anything irreversible, and
refuse with ``403 {"gated": true, ...}``.

That is the fix for the previous version's bug: the graph called
``send_day0(approve=True)`` from inside a request handler, so the "approval" was
the code's own opinion of itself. Here an approval is a row in a table that the
handler has to look up by id and cannot write for itself.

Zone naming
-----------
Two zones accept human-gate rows (``blackboard/zones.py``), with deliberately
different writers: ``gates`` is where the orchestration layer
(``graph/interrupts.py``, ``ZONE_GATES``) records its interrupt gates, while
``approvals`` is where the revenue agents (A3/A4) and this API record the
decisions they consume. Keeping the request (graph interrupt) apart from the
grant (agent/API approval) is what makes it possible to ask "was this action
approved, and by whom?" without conflating the two -- and an approval names
exactly one gate kind, so a send approval can never authorise an MoU. The
constant below is this module's side of that split, checked against the
registry at import time rather than discovered at the first approval.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from core.errors import BlackboardError, PaytriqError
from core.ids import new_id
from core.schemas import (
    AgentId,
    Approval,
    DecisionSource,
    GateKind,
    GateOutcome,
    HumanDecision,
    HumanGate,
    TraceKind,
    utcnow,
)

from .deps import GateLedger, RunContext
from .schemas import GateActionResponse, GateDecisionRequest

__all__ = ["router", "GATE_ZONE", "record_decision"]

log = logging.getLogger("paytriq.api.gates")

router = APIRouter(prefix="/api/gates", tags=["gates"])

#: The blackboard zone that accepts ``human_gate``, ``human_decision`` and
#: ``approval`` entries. Verified against the zone registry below.
GATE_ZONE = "approvals"
KIND_GATE = "human_gate"
KIND_DECISION = "human_decision"
KIND_APPROVAL = "approval"

#: Who is recorded as raising a gate raised by an API caller rather than by an
#: agent. ``ENVIRONMENT`` is not a reasoning agent, so an API-raised gate does
#: not appear in any agent's authorship statistics.
API_RAISER = AgentId.ENVIRONMENT


def _verify_zone() -> str:
    """Fail loudly at import if the zone registry does not match this module."""
    try:
        from blackboard.zones import zone as resolve_zone
    except ImportError as exc:  # pragma: no cover - blackboard is a hard sibling
        log.warning("cannot verify the %r zone at import: %s", GATE_ZONE, exc)
        return GATE_ZONE
    try:
        allowed = resolve_zone(GATE_ZONE).allowed_kinds
    except PaytriqError as exc:
        raise RuntimeError(
            f"api.routes_gates.GATE_ZONE={GATE_ZONE!r} is not a registered "
            f"blackboard zone: {exc}"
        ) from exc
    missing = [k for k in (KIND_GATE, KIND_DECISION, KIND_APPROVAL) if k not in allowed]
    if missing:
        raise RuntimeError(
            f"blackboard zone {GATE_ZONE!r} does not accept {missing}; this "
            f"module cannot record approvals without them"
        )
    return GATE_ZONE


_verify_zone()


def _context_for(run_id: str) -> RunContext:
    from .deps import get_run

    context = get_run(run_id)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"unknown run_id {run_id!r}",
                    "hint": "POST /api/events creates a run and returns its run_id"},
        )
    return context


@router.get("/{run_id}", summary="Outstanding human gates for one run")
def list_gates(run_id: str, request: Request) -> dict:
    """Every gate raised by a run, split into outstanding and answered.

    The split is the deliverable. A run in which every gate auto-resolved with
    ``decided_by="auto"`` and a run in which a person clicked approve produce the
    same *count*; only the split distinguishes them, so a viewer can always see
    how much of a run a human actually authorised.
    """
    context = _context_for(run_id)
    ledger = context.gates
    outstanding = ledger.outstanding(run_id=run_id)
    answered = [g for g in ledger.gates(run_id=run_id)
                if ledger.decision(g.gate_id) is not None]
    return {
        "run_id": run_id,
        "event_id": context.event_id,
        "outstanding": [_gate_row(ledger, gate) for gate in outstanding],
        "answered": [_gate_row(ledger, gate) for gate in answered],
        "summary": ledger.summary(run_id=run_id),
        "policy": {
            "interactive": bool(getattr(context.settings, "human_gates_interactive", True)),
            "auto_approve_outcome": getattr(context.settings, "auto_approve_outcome", ""),
            "gated_kinds": [k.value for k in (GateKind.SEND, GateKind.COUNTER,
                                              GateKind.MOU)],
        },
    }


def _gate_row(ledger: GateLedger, gate: HumanGate) -> dict:
    """One gate plus whatever decision and approval it has accumulated."""
    decision = ledger.decision(gate.gate_id)
    approval = ledger.approval(gate.gate_id)
    return {
        "gate": gate.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json") if decision else None,
        "approval": approval.model_dump(mode="json") if approval else None,
        "answered": decision is not None,
    }


@router.post(
    "/{gate_id}",
    response_model=GateActionResponse,
    summary="Answer a human gate (approve / reject / revise)",
)
def decide_gate(gate_id: str, payload: GateDecisionRequest,
                request: Request) -> GateActionResponse:
    """Record a human's answer and the approval that authorises (or forbids) it.

    Idempotent by design: re-answering an already-answered gate overwrites the
    decision and approval rather than appending a second one, so a reviewer who
    changes their mind does not leave two contradictory rows behind. The blackboard
    is append-only, so the earlier rows remain in the board's history -- which is
    the point of an append-only board.
    """
    ledger: GateLedger | None = None
    context: RunContext | None = None
    run_id = ""
    # A gate id is globally unique (``new_id("gat")``), so exactly one run's
    # ledger can hold it. Searching the registry avoids requiring the caller to
    # know which run raised the gate.
    for candidate in _iter_contexts():
        if candidate.gates.gate(gate_id) is not None:
            ledger, context = candidate.gates, candidate
            run_id = candidate.run_id
            break
    if ledger is None or context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"unknown gate_id {gate_id!r}",
                    "hint": "raise one by POSTing the gated action without an "
                            "approval; the refusal names the new gate_id"},
        )

    gate = ledger.gate(gate_id)
    assert gate is not None  # guaranteed by the search above
    outcome = GateOutcome(payload.outcome)
    decided_at = utcnow()

    decision = HumanDecision(
        gate_id=gate.gate_id,
        kind=gate.kind,
        outcome=outcome,
        decided_by=payload.decided_by,
        instruction=payload.instruction,
        at=decided_at,
    )
    action_taken = (
        f"{gate.kind.value} gate answered {outcome.value} by {payload.decided_by}"
        + (f": {payload.instruction}" if payload.instruction else "")
    )
    approval = Approval(
        approval_id=new_id("apr"),
        gate_id=gate.gate_id,
        kind=gate.kind,
        outcome=outcome,
        decided_by=payload.decided_by,
        action_taken=action_taken,
        at=decided_at,
    )

    # The single write path. ``record_decision`` requires the approval.
    entry_ids = record_decision(context, gate, decision, approval)

    if payload.close_gate and outcome is GateOutcome.REJECT:
        context.note(
            f"gate {gate_id} rejected by {payload.decided_by}; the action was "
            f"not taken and no approval authorises it"
        )

    authorises = None
    if gate.kind in (GateKind.SEND, GateKind.COUNTER, GateKind.MOU):
        permitted, reason = ledger.authorises(gate.kind, gate_id)
        authorises = (
            f"this approval now authorises the {gate.kind.value} action"
            if permitted else
            f"this approval does not authorise the {gate.kind.value} action: {reason}"
        )

    context.emit(
        TraceKind.HUMAN, f"gate.{gate.kind.value}",
        gate_id=gate.gate_id, outcome=outcome.value,
        decided_by=decision.decided_by,
        human=str(decision.decided_by).strip().lower() != "auto",
        approval_id=approval.approval_id,
        instruction=decision.instruction,
        event_id=gate.event_id,
    )

    return GateActionResponse(
        gate=gate.model_dump(mode="json"),
        decision=decision.model_dump(mode="json"),
        approval=approval.model_dump(mode="json"),
        run_id=run_id,
        human=str(decision.decided_by).strip().lower() != "auto",
        authorises=authorises,
        board_entry_ids=entry_ids,
        notes=[action_taken],
    )


def record_decision(context: RunContext, gate: HumanGate, decision: HumanDecision,
                    approval: Approval) -> dict[str, str]:
    """Persist a gate decision and its approval. The only writer of approvals.

    ``approval`` is a required parameter, not an option. That is the whole
    enforcement mechanism for "no side effect without a matching approval row":
    there is no call site that can record a decision and forget the proof.

    Writes, in order: the ``human_decision`` entry, then the ``approval`` entry
    citing it, then the ``HumanDecision`` trace event. If the board refuses, the
    ledger still holds both objects and the caller reports the board failure --
    an in-memory decision with an unwritable board is a degraded audit trail, not
    a lost one, and saying so beats pretending it succeeded.
    """
    ledger = context.gates
    entry_ids: dict[str, str] = {}
    board = context.board
    if board is None:
        ledger.record(gate, decision, approval)
        context.note(
            f"gate {gate.gate_id} answered {decision.outcome.value}; recorded in "
            f"the ledger only (no blackboard on this run)"
        )
        return entry_ids
    try:
        gate_entry_id = entry_ids.get("gate") or _gate_entry_id(board, gate)
        decision_entry = board.post(
            GATE_ZONE, KIND_DECISION, API_RAISER,
            decision.model_dump(mode="json"),
            refs=[gate_entry_id] if gate_entry_id else None,
            # 1.0 for a person, 0.0 for ``decided_by="auto"``: the board's
            # confidence is about *authority*, not about the outcome.
            confidence=1.0 if str(decision.decided_by).strip().lower() != "auto" else 0.0,
            source=DecisionSource.RULES,
        )
        entry_ids["decision"] = decision_entry.entry_id
        approval_entry = board.post(
            GATE_ZONE, KIND_APPROVAL, API_RAISER,
            approval.model_dump(mode="json"),
            refs=[e for e in (gate_entry_id, decision_entry.entry_id) if e],
            confidence=1.0,
            source=DecisionSource.RULES,
        )
        entry_ids["approval"] = approval_entry.entry_id
    except BlackboardError as exc:
        context.note(
            f"gate {gate.gate_id} answered {decision.outcome.value} but the "
            f"blackboard refused the audit row: {exc}"
        )
        ledger.record(gate, decision, approval)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "the approval was recorded in memory but the blackboard "
                         "refused the durable audit row",
                "reason": str(exc),
                "gate_id": gate.gate_id,
                "approval_id": approval.approval_id,
                "note": "the approval row is still authoritative for "
                        "authorisation; the board write is a separate failure",
            },
        ) from exc

    ledger.record(gate, decision, approval)
    return entry_ids


def _gate_entry_id(board: object, gate: HumanGate) -> str:
    """The board entry for a gate, if one was posted; ``""`` if not.

    The ledger may know about a gate the board never saw (the board could be
    absent, or the gate was raised by a path that does not post). Citations are
    therefore optional: a dangling id would make the justification chain
    unfalsifiable, so a missing one is simply omitted.
    """
    try:
        for entry in board.read(GATE_ZONE, kind=KIND_GATE):  # type: ignore[attr-defined]
            if entry.payload.get("gate_id") == gate.gate_id:
                return entry.entry_id
    except (BlackboardError, AttributeError, TypeError) as exc:
        log.debug("could not look up a gate entry id for %s: %s", gate.gate_id, exc)
    return ""


def _iter_contexts() -> list[RunContext]:
    """Every live run context. Used to find the ledger that owns a gate id.

    Delegates to the public :func:`api.deps.iter_runs` rather than reaching into
    the module's private lock, so the registry's thread-safety stays in one place.
    """
    from .deps import iter_runs

    return iter_runs()


def raise_gate(context: RunContext, kind: GateKind, *, question: str,
               payload_preview: str = "",
               options: list[GateOutcome] | None = None) -> HumanGate:
    """Raise a gate on a run and post it to the blackboard.

    Posting the ``human_gate`` entry here (rather than only when it is answered)
    means ``GET /api/gates/{run_id}`` reflects a pending question, and so does
    the board -- a reviewer watching the board sees the question before the answer.
    """
    gate = context.gates.raise_gate(
        kind,
        event_id=context.event_id,
        run_id=context.run_id,
        question=question,
        payload_preview=payload_preview,
        options=options,
    )
    board = context.board
    if board is not None:
        try:
            entry = board.post(GATE_ZONE, KIND_GATE, API_RAISER,
                               gate.model_dump(mode="json"),
                               confidence=1.0, source=DecisionSource.RULES)
            context.note(f"gate {gate.gate_id} ({kind.value}) raised: {question}")
            context.emit(TraceKind.HUMAN, f"gate.raised.{kind.value}",
                         gate_id=gate.gate_id, question=question,
                         board_entry=entry.entry_id, event_id=context.event_id)
        except BlackboardError as exc:
            context.note(
                f"gate {gate.gate_id} ({kind.value}) raised but the blackboard "
                f"refused the entry: {exc}"
            )
    else:
        context.note(f"gate {gate.gate_id} ({kind.value}) raised (no blackboard)")
    return gate


def require_approval(context: RunContext, kind: GateKind, gate_id: str | None,
                     *, action: str) -> tuple[str, str]:
    """Authorise one irreversible action, or raise ``403``.

    Returns ``(gate_id, reason)`` on success. Raises :class:`HTTPException` with
    ``403 {"gated": true, ...}`` otherwise, **including the id of a freshly
    raised gate** so the caller has something concrete to approve.

    Failing closed is the only safe direction here. The alternative -- treating
    "no approval found" as "probably fine" -- is exactly the bug that let the
    previous version send mail from a code path that had approved itself.
    """
    ledger = context.gates
    if gate_id:
        permitted, reason = ledger.authorises(kind, gate_id)
        if permitted:
            return gate_id, reason
        gate = ledger.gate(gate_id)
        detail: dict = {
            "gated": True,
            "error": reason,
            "reason": reason,
            "gate_id": gate_id,
            "gate_kind": kind.value,
            "action": action,
            "authorise_with": f"POST /api/gates/{gate_id}",
        }
        if gate is not None:
            detail["gate"] = gate.model_dump(mode="json")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

    gate = raise_gate(
        context, kind,
        question=f"Approve: {action}?",
        payload_preview=action,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "gated": True,
            "error": f"no approval record exists for the {kind.value} action "
                     f"'{action}'; nothing was done",
            "reason": f"the {kind.value} gate requires an explicit approval; "
                      f"no approval was supplied",
            "gate_id": gate.gate_id,
            "gate_kind": kind.value,
            "gate": gate.model_dump(mode="json"),
            "action": action,
            "authorise_with": f"POST /api/gates/{gate.gate_id}",
        },
    )
