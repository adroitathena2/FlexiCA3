"""Human-in-the-loop gates built on LangGraph v1 ``interrupt`` / ``Command(resume=...)``.

Why ``interrupt`` and not ``interrupt_before`` / ``interrupt_after``
--------------------------------------------------------------------
In LangGraph 1.x the static breakpoints are deprecated for human-in-the-loop
use and are kept for debugging only. The dynamic primitive is
:func:`langgraph.types.interrupt`, which:

* raises :class:`langgraph.types.GraphInterrupt` to park the run, carrying a
  payload the caller can render, and
* returns the caller's resume value on the *next* execution of the same node,
  driven by ``Command(resume=value)`` with the **same** ``thread_id``.

Two consequences drive the whole design of this module.

1. **A checkpointer and a ``thread_id`` are mandatory.** Without them there is
   nothing to park against. :func:`graph.checkpointer.build_checkpointer` and
   :func:`graph.checkpointer.default_config` exist for exactly this.
2. **The node re-executes from its first line on resume.** *Any* code above the
   ``interrupt()`` call runs a second time. So every function here is written
   as three clearly separated regions, in this order:

   ````
   build (pure, no I/O)  ->  interrupt(...)  ->  apply (all side effects)
   ````

   The :class:`~core.schemas.HumanGate` is *constructed* before the interrupt
   and *posted to the blackboard* after it. Constructing it before is what lets
   the parked run show the reviewer a fully-formed question; posting it after is
   what keeps the board from gaining a duplicate row every time the node is
   re-entered. Same reasoning applies to the agent run that precedes a gate in a
   node: :class:`~graph.state.StepMemo` on the runtime caches it, because a
   node's state writes are discarded when the node raises ``interrupt``.

Gates
-----
One per side-effecting action, matching :class:`~core.schemas.GateKind`:

===========  ========================================================
``SEND``     before any outbound mail
``COUNTER``  before accepting or answering a counter-offer
``MOU``      before releasing an MoU
``ESCALATION`` when the arbiter could not resolve a dispute
===========  ========================================================

There is no code path in the graph that performs any of these without a gate
first, and the gate's :class:`~core.schemas.Approval` row is written *after* the
decision and *before* the effect, so the trace shows approval preceding action.

Auto-resolution is labelled, never disguised
--------------------------------------------
When :attr:`~graph.state.GatePolicy.interactive` is false the gate resolves
itself from ``settings.auto_approve_outcome`` and records
``decided_by="auto"``. That string is the whole point: an unattended demo run
produces the same schema-valid approvals as an attended one, and the two remain
distinguishable in the trace forever.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command, interrupt

from core import (
    AgentId,
    Approval,
    DecisionSource,
    GateKind,
    GateOutcome,
    HumanDecision,
    HumanGate,
    new_id,
    utcnow,
)

from .state import GraphRuntime, runtime_context

__all__ = [
    "GateRequest", "GateResult", "run_gate", "gate_send", "gate_counter",
    "gate_mou", "gate_escalation", "gate_summary", "resume_command",
    "coerce_resume", "ZONE_GATES", "KIND_GATE", "KIND_DECISION", "KIND_APPROVAL",
    "AUTO_DECIDER",
]

log = logging.getLogger("paytriq.graph.interrupts")

ZONE_GATES = "gates"
KIND_GATE = "human_gate"
KIND_DECISION = "human_decision"
KIND_APPROVAL = "approval"

#: The literal written to ``HumanDecision.decided_by`` for a non-human
#: resolution. A module-level constant so a test can assert on it and a reader
#: can grep for every place a decision could have been faked.
AUTO_DECIDER = "auto"


@dataclass(slots=True)
class GateRequest:
    """What a caller wants approved. Pure data — no I/O, no interrupt."""

    kind: GateKind
    question: str
    #: Human-readable summary of exactly what will happen on approve.
    payload_preview: str = ""
    #: ``action_taken`` recorded on the Approval row. Must describe the effect
    #: precisely enough that an auditor can check it against the trace.
    action_taken: str = ""
    raised_by: AgentId = AgentId.A7_ARBITER
    options: tuple[GateOutcome, ...] = (
        GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE)
    #: Extra context handed to a ``HumanGateProvider`` or a web client.
    context: dict[str, Any] = field(default_factory=dict)
    #: Free-form resume value used when the policy is non-interactive. Lets a
    #: scripted demo drive a REJECT or REVISE without editing settings.
    scripted_resume: dict[str, Any] | None = None


@dataclass(slots=True)
class GateResult:
    """Everything a gated action needs in order to proceed."""

    gate: HumanGate
    decision: HumanDecision
    approval: Approval
    #: ``True`` when a person answered, ``False`` when the policy auto-resolved.
    human: bool
    #: The raw resume value, kept for nodes that accept free-form instructions.
    resume_value: Any = None

    @property
    def outcome(self) -> GateOutcome:
        return self.decision.outcome

    @property
    def approved(self) -> bool:
        return self.decision.outcome is GateOutcome.APPROVE

    @property
    def revise(self) -> bool:
        return self.decision.outcome is GateOutcome.REVISE

    @property
    def rejected(self) -> bool:
        return self.decision.outcome is GateOutcome.REJECT

    @property
    def instruction(self) -> str:
        return self.decision.instruction


# ------------------------------------------------------------------------ coercion
def coerce_resume(value: Any, *, kind: GateKind,
                  options: Sequence[GateOutcome]) -> tuple[GateOutcome, str, str]:
    """Normalise whatever a human (or a client) supplied into an outcome.

    Accepted shapes, because every HTTP client sends something different:

    * ``GateOutcome`` / ``"approve"`` / ``"APPROVE"``
    * ``{"outcome": "reject", "instruction": "drop the discount"}``
    * ``{"action": "approve", "note": "..."}``
    * a bare string instruction, which means REVISE — a human who wrote a note
      without picking an option is asking for a change, not a rubber stamp.

    An unrecognised value becomes REJECT. Failing closed is the only safe
    default for a control that guards sending mail and signing agreements: if we
    cannot tell what was approved, we must not proceed.
    """
    instruction = ""
    outcome_value: Any = value

    if isinstance(value, dict):
        for key in ("outcome", "action", "decision", "gate_outcome"):
            if key in value:
                outcome_value = value[key]
                break
        else:
            instruction = str(value.get("instruction") or value.get("note") or "")
            outcome_value = "revise"
        for key in ("instruction", "note", "comment", "reason"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                instruction = text.strip()
                break
    elif isinstance(value, GateOutcome):
        outcome_value = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return GateOutcome.REJECT, "", "empty resume value; failing closed"
        lowered = stripped.lower()
        if any(lowered.startswith(o.value) for o in options):
            outcome_value = stripped
        else:
            return GateOutcome.REVISE, stripped, "free-text instruction treated as REVISE"

    if isinstance(outcome_value, GateOutcome):
        outcome = outcome_value
    else:
        text = str(outcome_value or "").strip().lower()
        outcome = next((o for o in options if text.startswith(o.value)), None)  # type: ignore[assignment]
        if outcome is None:
            log.warning("gate %s: unrecognised resume value %r; failing closed to REJECT",
                        kind.value, outcome_value)
            return (GateOutcome.REJECT, instruction,
                    f"unrecognised outcome {outcome_value!r}; defaulting to reject")
    if kind is GateKind.ESCALATION and outcome is GateOutcome.REJECT:
        # Rejecting an escalation means "do not act on the disputed area".
        instruction = instruction or "escalation rejected; disputed area must not proceed"
    return outcome, instruction, ""


def resume_command(value: Any) -> Command:
    """Build the resume command for a parked run.

    Thin wrapper so the call site reads as intent rather than as LangGraph
    plumbing, and so there is exactly one place in the codebase that knows a
    resume is a ``Command``.
    """
    return Command(resume=value)


# ---------------------------------------------------------------------------- gate
def run_gate(rt: GraphRuntime, state: dict, request: GateRequest, *,
             interrupt_fn: Callable[[Any], Any] | None = None) -> GateResult:
    """Raise a gate, park the run, and apply the answer. See the module docstring.

    ``interrupt_fn`` is injectable so unit tests can drive a gate without a
    checkpointer; production always uses :func:`langgraph.types.interrupt`.
    """
    context = runtime_context(rt)
    policy = context.gate_policy
    event_id = str(state.get("event_id") or context.event_id or "")
    run_id = str(state.get("run_id") or context.run_id or "")

    # ---- region 1: build (pure) ------------------------------------------------
    gate = HumanGate(
        gate_id=new_id("gat"),
        kind=request.kind,
        event_id=event_id,
        run_id=run_id,
        question=request.question,
        payload_preview=request.payload_preview,
        options=list(request.options),
        raised_at=utcnow(),
    )
    payload = {
        "gate_id": gate.gate_id,
        "kind": gate.kind.value,
        "question": gate.question,
        "payload_preview": gate.payload_preview,
        "options": [o.value for o in gate.options],
        "raised_by": request.raised_by.value,
        "event_id": event_id,
        "run_id": run_id,
        "context": dict(request.context),
    }

    # ---- region 2: park -------------------------------------------------------
    raw: Any
    if not policy.interactive:
        raw = request.scripted_resume if request.scripted_resume is not None else (
            policy.scripted_resume if policy.scripted_resume is not None
            else {"outcome": policy.outcome().value, "decided_by": AUTO_DECIDER}
        )
        log.info("gate %s (%s) auto-resolving: interactive=%s",
                 gate.gate_id, gate.kind.value, policy.interactive)
    else:
        raw = (interrupt_fn or interrupt)(payload)

    # ---- region 3: apply (all side effects live here) -------------------------
    outcome, instruction, coercion_note = coerce_resume(raw, kind=request.kind,
                                                        options=request.options)
    human = _resume_is_human(raw)
    decided_by = str(_resume_decider(raw)) if human else AUTO_DECIDER
    if coercion_note:
        instruction = " ".join(x for x in (instruction, coercion_note) if x).strip()

    decided_at = utcnow()
    decision = HumanDecision(
        gate_id=gate.gate_id,
        kind=gate.kind,
        outcome=outcome,
        decided_by=decided_by,
        instruction=instruction,
        at=decided_at,
    )
    approval = Approval(
        approval_id=new_id("apr"),
        gate_id=gate.gate_id,
        kind=gate.kind,
        outcome=outcome,
        decided_by=decided_by,
        action_taken=request.action_taken or f"{request.kind.value} gate answered",
        at=decided_at,
    )

    gate_entry = rt.board.post(
        ZONE_GATES, KIND_GATE, request.raised_by, gate.model_dump(mode="json"),
        confidence=1.0, source=DecisionSource.RULES,
    )
    decision_entry = rt.board.post(
        ZONE_GATES, KIND_DECISION, request.raised_by, decision.model_dump(mode="json"),
        refs=[gate_entry.entry_id],
        confidence=1.0 if human else 0.0,
        source=DecisionSource.RULES,
    )
    rt.board.post(
        ZONE_GATES, KIND_APPROVAL, request.raised_by, approval.model_dump(mode="json"),
        refs=[gate_entry.entry_id, decision_entry.entry_id],
        confidence=1.0, source=DecisionSource.RULES,
    )
    _trace_gate(rt, gate, decision, approval)

    log.info("gate %s (%s) decided by %s: %s%s", gate.gate_id, gate.kind.value,
             decided_by, outcome.value, f" - {instruction}" if instruction else "")
    return GateResult(gate=gate, decision=decision, approval=approval, human=human,
                      resume_value=raw)


def _resume_is_human(raw: Any) -> bool:
    """True when the resume value names a real person.

    Anything carrying ``decided_by="auto"`` (or an explicit ``auto`` marker) is
    not a human decision, no matter how the value arrived. An ``auto`` that
    arrived over the wire is still ``auto`` — that is the honest reading.
    """
    if isinstance(raw, dict):
        marker = str(raw.get("decided_by") or raw.get("by") or "").strip().lower()
        return bool(marker) and marker != AUTO_DECIDER
    return True


def _resume_decider(raw: Any) -> str:
    if isinstance(raw, dict):
        for key in ("decided_by", "by", "user", "operator"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "human"
    return "human"


def _trace_gate(rt: GraphRuntime, gate: HumanGate, decision: HumanDecision,
                approval: Approval) -> None:
    """Emit the gate to the trace. Guarded: a tracer fault must not lose a decision.

    The HumanDecision and Approval are already on the blackboard by this point,
    so the durable record exists even if the tracer refuses the span.
    """
    from core import TraceKind

    try:
        rt.tracer.event(
            TraceKind.HUMAN, f"gate.{gate.kind.value}",
            agent=None,
            gate_id=gate.gate_id,
            outcome=decision.outcome.value,
            decided_by=decision.decided_by,
            human=decided_by_is_human(decision),
            question=gate.question,
            preview=gate.payload_preview,
            action=approval.action_taken,
        )
    except Exception as exc:  # noqa: BLE001 - durable record already written
        log.warning("tracer rejected the gate event for %s: %s", gate.gate_id, exc)


def decided_by_is_human(decision: HumanDecision) -> bool:
    """Public predicate so callers do not re-derive the ``"auto"`` convention."""
    return str(decision.decided_by).strip().lower() != AUTO_DECIDER


# ------------------------------------------------------------------------ shorthands
def gate_send(rt: GraphRuntime, state: dict, *, question: str, preview: str,
              raised_by: AgentId = AgentId.A3_OUTREACH,
              context: dict[str, Any] | None = None,
              scripted_resume: dict[str, Any] | None = None) -> GateResult:
    """Before any outbound mail."""
    return run_gate(rt, state, GateRequest(
        kind=GateKind.SEND,
        question=question,
        payload_preview=preview,
        action_taken=f"send outreach mail: {preview}",
        raised_by=raised_by,
        context=context or {},
        scripted_resume=scripted_resume,
    ))


def gate_counter(rt: GraphRuntime, state: dict, *, question: str, preview: str,
                 raised_by: AgentId = AgentId.A3_OUTREACH,
                 context: dict[str, Any] | None = None,
                 scripted_resume: dict[str, Any] | None = None) -> GateResult:
    """Before accepting or answering a counter-offer."""
    return run_gate(rt, state, GateRequest(
        kind=GateKind.COUNTER,
        question=question,
        payload_preview=preview,
        action_taken=f"answer counter-offer: {preview}",
        raised_by=raised_by,
        context=context or {},
        scripted_resume=scripted_resume,
    ))


def gate_mou(rt: GraphRuntime, state: dict, *, question: str, preview: str,
             raised_by: AgentId = AgentId.A4_CONTRACT,
             context: dict[str, Any] | None = None,
             scripted_resume: dict[str, Any] | None = None) -> GateResult:
    """Before releasing an MoU."""
    return run_gate(rt, state, GateRequest(
        kind=GateKind.MOU,
        question=question,
        payload_preview=preview,
        action_taken=f"release MoU: {preview}",
        raised_by=raised_by,
        context=context or {},
        scripted_resume=scripted_resume,
    ))


def gate_escalation(rt: GraphRuntime, state: dict, *, question: str, preview: str,
                    raised_by: AgentId = AgentId.A7_ARBITER,
                    context: dict[str, Any] | None = None,
                    scripted_resume: dict[str, Any] | None = None) -> GateResult:
    """When the arbiter could not resolve a dispute."""
    return run_gate(rt, state, GateRequest(
        kind=GateKind.ESCALATION,
        question=question,
        payload_preview=preview,
        action_taken=f"escalated decision: {preview}",
        raised_by=raised_by,
        context=context or {},
        scripted_resume=scripted_resume,
    ))


# --------------------------------------------------------------------------- summary
def gate_summary(approvals: Sequence[Any]) -> dict[str, Any]:
    """A demo-ready summary of every gate answered.

    ``human_count`` and ``auto_count`` are separate fields rather than one
    ``count`` because conflating them is exactly how an unattended run gets
    presented as an attended one. A viewer showing this summary can always see
    how much of the run a person actually authorised.
    """
    rows: list[dict[str, Any]] = []
    by_kind: dict[str, dict[str, int]] = {}
    human = auto = 0
    for item in approvals or []:
        data = coerce_approval(item)
        if not data:
            continue
        kind = str(data.get("kind") or "unknown").rsplit(".", 1)[-1]
        outcome = str(data.get("outcome") or "unknown").rsplit(".", 1)[-1]
        decider = str(data.get("decided_by") or AUTO_DECIDER)
        is_auto = decider.strip().lower() == AUTO_DECIDER
        auto += int(is_auto)
        human += int(not is_auto)
        bucket = by_kind.setdefault(kind, {})
        bucket[outcome] = bucket.get(outcome, 0) + 1
        rows.append({
            "approval_id": data.get("approval_id"),
            "gate_id": data.get("gate_id"),
            "kind": kind,
            "outcome": outcome,
            "decided_by": decider,
            "human": not is_auto,
            "action_taken": data.get("action_taken", ""),
            "at": data.get("at"),
        })
    return {
        "total": len(rows),
        "human_count": human,
        "auto_count": auto,
        "all_human": bool(rows) and auto == 0,
        "by_kind": by_kind,
        "approvals": rows,
    }


def coerce_approval(item: Any) -> dict | None:
    """Normalise an :class:`Approval` or its JSON dict into a plain dict."""
    if item is None:
        return None
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    return None
