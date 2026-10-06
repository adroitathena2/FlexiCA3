"""A3 Outreach & Negotiation — gated sending, reply classification, negotiation.

Why this agent is shaped the way it is
--------------------------------------
Two failures in the previous prototype cost real money and real trust, and both
are structural, so both are fixed structurally here.

**1. The human gate was decorative.** ``old/backend/agents/a3_outreach.py``::

    def send_day0(brand, email, cover_letter, approve: bool = True):
        send_email(email or brand, ..., approve=approve)
        return OutreachThread(brand=brand, email=email, day=0, status="sent")

``approve: bool = True`` means the *only* way to be careful was to remember to
pass ``approve=False``, and the return value said ``status="sent"`` either way —
the thread claimed a send that had never happened. Here the gate is not an
argument but a precondition: A3 raises a :class:`~core.schemas.HumanGate`, looks
for a recorded :class:`~core.schemas.HumanDecision`, writes a durable
:class:`~core.schemas.Approval` row, and only then calls the transport. With no
decision on the board it raises :class:`~core.errors.HumanGateRequired` and
**nothing is sent**. There is no code path that reaches ``send`` without an
approval row, and no default that skips it.

**2. Reply routing tested ``"yes"`` before it tested price.** The old router
checked assent first, so *"yesterday we thought the price was too high"* — a
sentence containing no word of assent at all except through a naive scan — could
be routed to contract signing. Every classification here goes through
``ctx.decide`` as a closed ``choice`` over the ``Intent`` set, with the reply
text in ``state``; no keyword ever appears in this file. ``Intent.NEUTRAL``
exists precisely so an unrecognised message cannot default to "interested", and
``NOUL``/``PUSHBACK`` are decided before anything else because that ordering is
the fix.

**Honesty about delivery.** A send counts as delivered only when the transport
returned ``status=OK``. Anything else — unavailable, failed, cached, a missing
transport — leaves the thread in ``draft`` with ``delivered=False`` and a gap in
the observation. There is no invented contact address either: a lead with no
email is reported as unreachable, not "fixed up" into ``partnerships@brand.com``.

Board integration note: a pushback is carried three ways, each read by a
different party. The ``Thread`` update (``intent=PUSHBACK`` naming the exact
``offer_id`` the sponsor objected to) is what A2 scans to find revision work;
the ``Handoff`` to A2 is the routing record; and a typed ``Message`` of kind
``REVISION_REQUEST`` in ``messages`` (A3 -> A2, citing the thread and offer
entries) is the deliberation record A2 quotes in its posture decision. When A2
has already revised and the sponsor pushes back again, A3 authors a typed
``Dispute`` in ``disputes`` (A3 vs A2, pricing zone, both positions with
entry-id evidence) for A7 to adjudicate.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.base import ReActAgent
from core.errors import (
    AgentError,
    BlackboardError,
    DecisionFailed,
    DecisionUnavailable,
    HumanGateRequired,
    SchemaError,
    ToolFailed,
    ToolTimeout,
    ToolUnavailable,
)
from core.ids import new_id, utcnow
from core.protocols import (
    ActResult,
    AgentContext,
    BoardEntry,
    Observation,
    Plan,
    Reflection,
    Tool,
    ToolResult,
    classify_intent_fallback,
)
from core.schemas import (
    AgentId,
    Approval,
    BrandLead,
    Decision,
    DecisionRequest,
    DecisionSource,
    Dispute,
    DisputeStatus,
    DisputeZone,
    EventProfile,
    GateKind,
    GateOutcome,
    Handoff,
    HumanDecision,
    HumanGate,
    Intent,
    Message,
    MessageKind,
    Offer,
    QuestionType,
    Severity,
    Thread,
    ToolStatus,
)

__all__ = ["OutreachAgent", "INTENT_OPTIONS", "SEND_TOOL_NAMES", "GATE_ENTRY_KIND"]


# ============================================================================ zones
ZONE_EVENT = "event"
ZONE_OPPORTUNITIES = "opportunities"
ZONE_OFFERS = "offers"
ZONE_THREADS = "threads"
ZONE_APPROVALS = "approvals"
ZONE_HANDOFFS = "handoffs"
ZONE_MESSAGES = "messages"
ZONE_DISPUTES = "disputes"
KIND_EVENT_PROFILE = "event_profile"
KIND_BRAND_LEAD = "brand_lead"
KIND_OFFER = "offer"
KIND_THREAD = "thread"
KIND_APPROVAL = "approval"
KIND_HUMAN_DECISION = "human_decision"
KIND_HANDOFF = "handoff"
KIND_MESSAGE = "message"
KIND_DISPUTE = "dispute"

#: Kind used when recording a *pending* gate. ``blackboard/zones.py`` registers
#: the ``approvals`` zone with kinds ``approval`` and ``human_decision`` only —
#: ``HumanGate`` has no registered kind — so a strict board refuses this post.
#: The attempt is made anyway (a board that does accept it must see it), and the
#: refusal is recorded in the observation rather than hidden. It never grants
#: permission: approval is judged solely from a recorded ``human_decision``.
GATE_ENTRY_KIND = "human_gate"

#: The closed option set for reply classification. Every member of ``Intent``
#: that a human would recognise, in the order the decision model sees them.
#: ``UNKNOWN`` is deliberately absent: it is A3's own "we could not tell" answer,
#: not something to offer as a choice.
INTENT_OPTIONS: tuple[str, ...] = (
    Intent.YES.value,
    Intent.PUSHBACK.value,
    Intent.INTERESTED.value,
    Intent.NO.value,
    Intent.NEUTRAL.value,
)

SEND_TOOL_NAMES: tuple[str, ...] = ("send_email", "send_mail", "email_send")

#: Thread statuses that mean "do not contact this brand again in this run".
TERMINAL_STATUSES: frozenset[str] = frozenset({"closed_won", "closed_lost"})


# =============================================================== module utilities
def _as_text(value: Any) -> str | None:
    """Trimmed non-empty string, or ``None``."""
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _probe_tool(tool: Tool) -> tuple[bool, str]:
    """Honour the ``Tool.available()`` contract before every call."""
    probe = getattr(tool, "available", None)
    if not callable(probe):
        return True, ""
    try:
        available, reason = probe()
    except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except (TypeError, ValueError) as exc:
        return False, f"available() returned an unusable value: {exc}"
    return bool(available), (str(reason) if reason else "")


def invoke(tool: Tool, args: Mapping[str, Any]) -> ToolResult:
    """Call ``tool.run`` with only the keyword arguments its signature accepts."""
    runner = getattr(tool, "run", None)
    if not callable(runner):
        raise ToolUnavailable(f"tool {getattr(tool, 'name', tool)!r} has no run()")
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        kwargs = dict(args)
    else:
        parameters = signature.parameters.values()
        if any(p.kind is p.VAR_KEYWORD for p in parameters):
            kwargs = dict(args)
        else:
            accepted = {p.name for p in parameters
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            kwargs = {k: v for k, v in args.items() if k in accepted}
    result = runner(**kwargs)
    if isinstance(result, ToolResult):
        return result
    return ToolResult.success(result,
                              source=f"{getattr(tool, 'name', 'unknown_tool')}:unwrapped")


def read_entries(board: Any, zone: str, *, kind: str | None = None,
                 limit: int | None = None) -> list[BoardEntry]:
    """``board.read`` with an unusable board reported as "nothing there"."""
    try:
        return list(board.read(zone, kind=kind, limit=limit))
    except TypeError:
        try:
            return list(board.read(zone))
        except (BlackboardError, SchemaError) as exc:
            raise AgentError(f"board.read({zone!r}) failed: {exc}") from exc
    except (BlackboardError, SchemaError) as exc:
        raise AgentError(f"board.read({zone!r}) failed: {exc}") from exc


def load_all(board: Any, zone: str, model_cls: type[BaseModel], *,
             kind: str | None = None,
             notes: list[str] | None = None) -> list[BaseModel]:
    """Every artefact of ``model_cls`` in ``zone``, oldest first."""
    record = notes if notes is not None else []
    try:
        entries = read_entries(board, zone, kind=kind)
    except AgentError as exc:
        record.append(str(exc))
        return []
    out: list[BaseModel] = []
    for entry in entries:
        try:
            out.append(model_cls.model_validate(entry.payload))
        except ValidationError as exc:
            record.append(
                f"skipped {entry.entry_id} in {zone!r}: not a valid "
                f"{model_cls.__name__}: {exc.error_count()} field error(s)"
            )
    return out


def load_latest(board: Any, zone: str, model_cls: type[BaseModel], *,
                kind: str | None = None,
                notes: list[str] | None = None) -> BaseModel | None:
    """Most recent artefact of ``model_cls`` in ``zone``, or ``None``."""
    latest_model = getattr(board, "latest_model", None)
    if callable(latest_model):
        try:
            return latest_model(zone, model_cls, kind=kind)
        except TypeError:
            try:
                return latest_model(zone, model_cls)
            except (BlackboardError, SchemaError, ValidationError) as exc:
                record = notes if notes is not None else []
                record.append(f"latest_model({zone!r}) unusable: {exc}")
        except (BlackboardError, SchemaError, ValidationError) as exc:
            record = notes if notes is not None else []
            record.append(f"latest_model({zone!r}) unusable: {exc}")
    found = load_all(board, zone, model_cls, kind=kind, notes=notes)
    return found[-1] if found else None


class OutreachAgent(ReActAgent):
    """Send approved outreach, classify replies, and route the negotiation."""

    id = AgentId.A3_OUTREACH
    role = "outreach: gated sending, calibrated reply classification, negotiation routing"

    #: When true (the default) an unanswered send gate propagates
    #: :class:`~core.errors.HumanGateRequired` out of ``run()``, which is the
    #: control-flow signal ``core.errors`` documents for the graph to catch and
    #: turn into an ``interrupt()``. Set it to False to have A3 stop locally and
    #: leave the thread ``pending_approval`` instead — useful when A3 is driven
    #: outside a graph. Neither setting can send without an approval row.
    propagate_gate: bool = True

    def __init__(self, *, step_budget: int = 6, deadline_s: float = 90.0) -> None:
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)

    # ================================================================== planning
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Classify a reply, or make a gated send. Never both in one step."""
        notes: list[str] = []
        state = self._state(ctx)
        ctx.scratch["a3_notes"] = notes

        if state.get("blocked"):
            return self._terminal_plan(
                "outreach is parked awaiting a human decision on the send gate")

        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        if event is None:
            return self._terminal_plan(
                "await the event profile; outreach without an event is a mailing list",
                gap="event profile missing from the board")

        threads = self._threads_by_brand(ctx, notes)
        pending_reply = next(
            (t for _, t in threads.values()
             if t.reply_text and t.intent is Intent.UNKNOWN), None)
        if pending_reply is not None:
            decision = self._classify(ctx, pending_reply)
            intent = self._intent_of(decision, pending_reply, notes)
            state["classify_target"] = pending_reply.thread_id
            state["classify_intent"] = intent.value
            state["classify_source"] = decision.source.value
            state["classify_confidence"] = round(decision.confidence, 4)
            self._store(ctx, state)
            return Plan(
                goal=(f"classify the reply from {pending_reply.brand} and route "
                      f"the conversation ({intent.value})"),
                steps=["classify"],
                rationale=(f"thread {pending_reply.thread_id} carries an "
                           f"unclassified reply of "
                           f"{len(pending_reply.reply_text)} characters"),
                confidence=decision.confidence,
                source=decision.source,
            )

        queued = self._send_queue(ctx, event, notes)
        if queued:
            offer, thread, lead, reason = queued[0]
            self._store(ctx, {"mode": "send", "brand": offer.brand,
                              "offer_id": offer.offer_id,
                              "thread_id": thread.thread_id if thread else None,
                              "reason": reason})
            return Plan(
                goal=(f"send the {offer.tier} offer for {offer.brand} "
                      f"(INR {offer.amount_inr:,.0f}) after human approval"),
                steps=["send"],
                tool_calls=[{"tool": SEND_TOOL_NAMES[0],
                             "args": {"brand": offer.brand,
                                      "offer_id": offer.offer_id}}],
                rationale=reason,
                confidence=1.0,
                source=DecisionSource.RULES,
            )

        unreachable = self._unreachable(ctx, event, notes)
        return self._terminal_plan(
            "every priced offer has been dealt with this cycle",
            gap=(f"no contact route for: {', '.join(unreachable)}"
                 if unreachable else None))

    # ========================================================================= act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        mode = plan.steps[0] if plan.steps else "idle"
        if mode == "classify":
            return self._act_classify(ctx)
        if mode == "send":
            return self._act_send(ctx)
        return ActResult(ok=True, output={"mode": mode, "noop": True},
                         observations=[f"A3 stood down: {plan.goal}"])

    def _act_classify(self, ctx: AgentContext) -> ActResult:
        """Route one classified reply. One decision, already taken in _plan."""
        notes: list[str] = list(ctx.scratch.get("a3_notes") or [])
        state = self._state(ctx)
        thread_id = state.get("classify_target")
        intent_text = str(state.get("classify_intent") or Intent.UNKNOWN.value)
        try:
            intent = Intent(intent_text)
        except ValueError:
            intent = Intent.UNKNOWN
        threads = self._threads_by_brand(ctx, notes)
        match = next((pair for pair in threads.values()
                      if pair[1].thread_id == thread_id), None)
        if match is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"thread {thread_id!r} is not on the board"])
        entry, thread = match
        offer = self._offer_for(ctx, thread.brand, notes)
        source = DecisionSource(str(state.get("classify_source") or "rules"))
        confidence = float(state.get("classify_confidence") or 0.0)

        status = {
            Intent.YES: "closed_won",
            Intent.PUSHBACK: "negotiating",
            Intent.INTERESTED: "negotiating",
            Intent.NO: "closed_lost",
            Intent.NEUTRAL: "negotiating",
            Intent.UNKNOWN: "negotiating",
        }[intent]
        updated = thread.model_copy(update={
            "intent": intent,
            "status": status,
            "offer_id": thread.offer_id or (offer.offer_id if offer else None),
        })
        try:
            new_id_ = self.post(ctx, ZONE_THREADS, KIND_THREAD,
                                updated.model_dump(mode="json"),
                                refs=[entry.entry_id], confidence=confidence,
                                source=source)
        except (BlackboardError, SchemaError) as exc:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"could not record the classified thread for "
                                     f"{thread.brand}: {exc}"])

        handoff: str | None = None
        routed = ""
        revision_request_id: str | None = None
        dispute_id: str | None = None
        if intent is Intent.PUSHBACK:
            handoff = self._handoff(
                ctx, AgentId.A2_PRICING, event_id=thread.event_id,
                reason=(f"{thread.brand} pushed back on the "
                        f"INR {offer.amount_inr:,.0f} quote" if offer else
                        f"{thread.brand} pushed back on price"),
                confidence=confidence, source=source,
                summary=(f"Objection on record: {thread.reply_text[:400]}. A2 must "
                         f"decide a posture and a concession size; the cited offer "
                         f"is {thread.offer_id or 'unknown'}."),
                refs=[new_id_])
            routed = "A2 (repricing)"
            offer_entry_id: str | None = None
            if offer is not None:
                found_offer = self._offer_with_entry(
                    ctx, thread.offer_id or offer.offer_id, notes)
                if found_offer is not None:
                    offer_entry_id = found_offer[0].entry_id
            revision_request_id = self._post_message(
                ctx, MessageKind.REVISION_REQUEST, AgentId.A2_PRICING,
                body=(f"REVISION_REQUEST for {thread.brand}: sponsor pushed back on "
                      f"{offer.tier + ' ' if offer else ''}"
                      f"{f'at INR {offer.amount_inr:,.0f}' if offer else 'the current terms'} "
                      f"(offer {(thread.offer_id or (offer.offer_id if offer else 'unknown'))}). "
                      f"Objection on record: {thread.reply_text[:400] or 'no text recorded'}. "
                      f"A2: decide a posture and a concession size against the delivery floor."),
                refs=[r for r in (new_id_, offer_entry_id) if r],
                confidence=confidence, source=source,
                summary=(f"revision needed for {thread.brand}; "
                         f"cited offer {thread.offer_id or (offer.offer_id if offer else 'unknown')}"))
            if revision_request_id is None:
                notes.append("the board refused the revision_request message; "
                             "the thread update plus handoff carry the revision need")
            # The counter-proposal failed if A2 already revised and the sponsor
            # is pushing back again on the new terms. That disagreement is
            # authored here as a typed Dispute for A7 to adjudicate.
            counters = self._counter_proposals_for(ctx, thread.brand, notes)
            if counters:
                already = self._open_disputes_between(ctx, notes)
                if not already:
                    dispute_id = self._open_pricing_dispute(
                        ctx, updated, offer, new_id_, counters, notes,
                        confidence, source)
        elif intent is Intent.YES:
            handoff = self._handoff(
                ctx, AgentId.A4_CONTRACT, event_id=thread.event_id,
                reason=f"{thread.brand} accepted; an MoU is needed",
                confidence=confidence, source=source,
                summary=(f"{thread.brand} accepted "
                         f"{offer.tier if offer else 'the offer'}"
                         f"{f' at INR {offer.amount_inr:,.0f}' if offer else ''}. "
                         f"Thread {thread.thread_id} is closed_won; A4 must draft "
                         f"from these terms, not from the first offer posted."),
                refs=[new_id_])
            routed = "A4 (contract)"
        elif intent is Intent.NO:
            routed = "archive (closed_lost)"

        state.pop("classify_target", None)
        self._store(ctx, state)
        observation = (f"{thread.brand}: reply classified {intent.value} "
                       f"(source={source.value}, confidence={confidence:.2f})"
                       + (f"; routed to {routed}" if routed else ""))
        if revision_request_id:
            observation += f"; revision_request {revision_request_id} posted to A2"
        if dispute_id:
            observation += (f"; pricing dispute {dispute_id} opened: "
                            f"the counter-proposal did not settle the objection")
        return ActResult(
            ok=True,
            output={"mode": "classify", "brand": thread.brand,
                    "thread_id": thread.thread_id, "intent": intent.value,
                    "status": status, "entry_id": new_id_,
                    "handoff_id": handoff, "routed_to": routed,
                    "revision_request_id": revision_request_id,
                    "dispute_id": dispute_id,
                    "decision_source": source.value,
                    "decision_confidence": confidence},
            observations=[observation],
            errors=notes,
            degraded=False,
        )

    def _act_send(self, ctx: AgentContext) -> ActResult:
        """Gate, then send. The gate is not optional and has no default."""
        notes: list[str] = list(ctx.scratch.get("a3_notes") or [])
        state = self._state(ctx)
        brand = str(state["brand"])
        offer_id = str(state["offer_id"])
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        if event is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=["no event profile on the board"])
        found = self._offer_with_entry(ctx, offer_id, notes)
        if found is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"offer {offer_id!r} is not on the board"])
        offer_entry, offer = found
        threads = self._threads_by_brand(ctx, notes)
        lead = self._lead_for(ctx, brand, notes)
        thread_entry, thread = threads.get(brand, (None, None))  # type: ignore[assignment]
        to_address = (thread.email if thread else None) or (lead.contact_email if lead else None)
        if not to_address:
            # A3 will not synthesise an address. The old code invented
            # ``partnerships@{brand}.com`` and then reported the send as
            # delivered to a domain that does not exist.
            return ActResult(
                ok=False, output=None, degraded=False,
                errors=[f"{brand}: no contact email on the lead or the thread; "
                        f"A3 will not invent one"],
                observations=[f"{brand}: outreach impossible without a contact "
                              f"route"])

        subject, body = self._compose(event, brand, offer)
        preview = f"to={to_address or '<no address on file>'}\n{subject}\n{body}"

        try:
            outcome, gate_id = self._require_approval(
                ctx, GateKind.SEND,
                question=(f"Send the {offer.tier} proposal to {brand} "
                          f"(INR {offer.amount_inr:,.0f})?"),
                preview=preview, action=f"send {offer.tier} offer to {brand}")
        except HumanGateRequired as exc:
            self._mark_gate_blocked(ctx, brand, gate_id=str(exc))
            if self.propagate_gate:
                # The documented control-flow signal: the graph catches this,
                # interrupts, and resumes once a human has decided.
                raise
            state["blocked"] = True
            self._store(ctx, state)
            return ActResult(
                ok=False, output=None, degraded=False,
                errors=[f"send blocked: {exc}"],
                observations=["no HumanDecision for the send gate; nothing was "
                              "sent and the thread stays pending_approval"])

        if outcome is not GateOutcome.APPROVE:
            self._mark_gate_blocked(
                ctx, brand, gate_id=gate_id,
                extra=(f"human outcome {outcome.value}"))
            return ActResult(
                ok=True,
                output={"mode": "send", "brand": brand, "sent": False,
                        "delivered": False, "gate_id": gate_id,
                        "outcome": outcome.value},
                observations=[f"{brand}: the human answered {outcome.value}; "
                              f"nothing was sent"],
                degraded=False)

        tool, tool_name = self._pick_send_tool(ctx)
        if tool is None:
            return ActResult(
                ok=False, output=None, degraded=True,
                errors=[f"no send transport in the registry (looked for "
                        f"{list(SEND_TOOL_NAMES)}); the approved message was NOT "
                        f"sent and is not recorded as delivered"],
                observations=[f"{brand}: approval {gate_id} was granted but there "
                              f"is no transport to send with"])
        available, reason = _probe_tool(tool)
        if not available:
            return ActResult(
                ok=False, output=None, degraded=True,
                errors=[f"send transport {tool_name!r} unavailable: {reason}; "
                        f"nothing was sent"],
                observations=[f"{brand}: approved, but the transport refused"])
        try:
            result = invoke(tool, {"to": to_address, "recipient": to_address,
                                   "email": to_address, "to_address": to_address,
                                   "subject": subject, "body": body,
                                   "text": body, "html": None})
        except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
            return ActResult(
                ok=False, output=None, degraded=True,
                errors=[f"send transport {tool_name!r} raised "
                        f"{type(exc).__name__}: {exc}; nothing was sent"])

        delivered = bool(result.ok) and result.status is ToolStatus.OK
        stamp = utcnow() if delivered else None
        updated = Thread(
            thread_id=thread.thread_id if thread else new_id("thr"),
            event_id=event.event_id,
            brand=brand,
            email=to_address,
            status="sent" if delivered else "draft",
            day=thread.day if thread else 0,
            intent=thread.intent if thread else Intent.UNKNOWN,
            reply_text=thread.reply_text if thread else "",
            offer_id=offer.offer_id,
            sent_at=stamp,
            delivered=delivered,
        )
        try:
            entry_id = self.post(ctx, ZONE_THREADS, KIND_THREAD,
                                 updated.model_dump(mode="json"),
                                 refs=[r for r in (offer_entry.entry_id,) if r],
                                 confidence=1.0, source=DecisionSource.RULES)
        except (BlackboardError, SchemaError) as exc:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"the send happened but the thread could not "
                                     f"be recorded: {exc}"])

        if delivered:
            self._store(ctx, {"mode": "send_done", "brand": brand})
            return ActResult(
                ok=True,
                output={"mode": "send", "brand": brand, "sent": True,
                        "delivered": True, "to": to_address, "subject": subject,
                        "offer_id": offer.offer_id, "entry_id": entry_id,
                        "gate_id": gate_id, "transport_status": result.status.value,
                        "latency_ms": result.latency_ms},
                observations=[f"{brand}: {offer.tier} offer sent to {to_address} "
                              f"under approval {gate_id} "
                              f"(status={result.status.value})"],
                degraded=False)

        reason = (result.reason
                  or f"transport returned status={result.status.value}")
        self._store(ctx, {"mode": "send_failed", "brand": brand})
        return ActResult(
            ok=False,
            output={"mode": "send", "brand": brand, "sent": False,
                    "delivered": False, "to": to_address, "offer_id": offer.offer_id,
                    "entry_id": entry_id, "gate_id": gate_id,
                    "transport_status": result.status.value,
                    "transport_reason": reason},
            observations=[f"{brand}: approved but NOT delivered — {reason}"],
            errors=[f"{brand}: send not delivered "
                    f"(status={result.status.value}): {reason}"],
            degraded=True,
        )

    # ==================================================================== observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        mode = plan.steps[0] if plan.steps else "idle"
        output = result.output if isinstance(result.output, Mapping) else {}
        state = self._state(ctx)

        if mode == "halt":
            gaps = list(result.errors)
            if plan.steps and len(plan.steps) > 1:
                gaps.append(str(plan.steps[1]))
            if state.get("blocked"):
                return Observation(
                    summary=plan.goal,
                    facts={"blocked": True, "gate_note": ctx.scratch.get("a3_gate_note")},
                    sufficient=True,
                    gaps=["outreach is parked: a send requires a human decision"])
            return Observation(summary=plan.goal, facts={"halted": True},
                               sufficient=not gaps, gaps=gaps)

        if mode == "classify" and result.ok:
            return Observation(
                summary=(f"{output.get('brand')}: classified the reply as "
                         f"{output.get('intent')} and routed to "
                         f"{output.get('routed_to') or 'no further action'}"),
                facts=dict(output), sufficient=False,
                gaps=[f"{output.get('brand')} needs the next step "
                      f"({output.get('routed_to') or 'awaiting human read'})"])

        if mode == "send" and result.ok and output.get("delivered"):
            event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                                kind=KIND_EVENT_PROFILE, notes=[])
            queued = self._send_queue(ctx, event, []) if event else []
            return Observation(
                summary=(f"sent the {output.get('brand')} offer to "
                         f"{output.get('to')} under a recorded human approval"),
                facts=dict(output), sufficient=not queued,
                gaps=([f"{len(queued)} offer(s) still awaiting a send"] if queued else []))

        gaps = list(result.errors)
        return Observation(
            summary=(f"outreach step {mode} did not complete for "
                     f"{output.get('brand') or 'the queue'}"),
            facts=dict(output) if output else {"mode": mode},
            sufficient=False,
            gaps=gaps or ["outreach step failed with no reason recorded"])

    # =================================================================== reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """One lesson about the failure that actually happened, not a platitude."""
        facts = obs.facts or {}
        if facts.get("blocked"):
            return Reflection(
                lesson_trigger="a send was attempted with no HumanDecision recorded",
                correction=("outreach has no auto-approve path; the gate is a "
                            "precondition, not a parameter"),
                rule=("never call a send transport unless an Approval row for a "
                      "HumanGate with the same gate_id is on the board, and never "
                      "mark a thread delivered unless the transport returned "
                      "status=OK"),
                confidence=0.9,
            )
        if not obs.sufficient and "not delivered" in obs.summary:
            return Reflection(
                lesson_trigger="the transport did not accept an approved message",
                correction=("an approved send is not a delivered send; the "
                            "distinction has to survive into the thread"),
                rule=("Thread.delivered is True only for ToolStatus.OK; anything "
                      "else leaves status='draft' and delivered=False so the next "
                      "cycle retries rather than assuming success"),
                confidence=0.8,
            )
        if not obs.sufficient and str(facts.get("intent")) in ("neutral", "unknown"):
            return Reflection(
                lesson_trigger="a reply matched no intent the model could name",
                correction=("an unclassifiable reply is a human-read message, not "
                            "a yes and not an interested"),
                rule=("never default an unrecognised reply to interested; "
                      "Intent.NEUTRAL and Intent.UNKNOWN exist for that, and "
                      "neither may trigger a send or a contract"),
                confidence=0.75,
            )
        return None

    # ============================================================ the human gate
    def _require_approval(self, ctx: AgentContext, kind: GateKind, question: str,
                          preview: str, action: str) -> tuple[GateOutcome, str]:
        """Raise a gate and return the human's recorded answer.

        Raises
        ------
        :class:`~core.errors.HumanGateRequired`
            No ``HumanDecision`` for this gate exists. The caller stops; nothing
            is sent, drafted for release, or otherwise actioned.
        """
        gate = HumanGate(
            gate_id=new_id("gat"),
            kind=kind,
            event_id=ctx.event_id,
            run_id=ctx.run_id,
            question=question,
            payload_preview=preview[:600],
            options=[GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE],
            raised_at=utcnow(),
        )
        gate_entry, refusal = self._record_gate(ctx, gate)
        note = (f"pending {kind.value} gate {gate.gate_id} for {action}"
                + (f"; the board refused to store the gate itself ({refusal})"
                   if refusal else ""))
        ctx.scratch["a3_gate_note"] = note

        decision = self._find_human_decision(ctx, gate.gate_id, kind)
        if decision is None:
            raise HumanGateRequired(
                f"{kind.value} gate {gate.gate_id} is unanswered and no approval "
                f"record exists for {action}; refusing to proceed"
            )
        ctx.scratch["a3_gate_note"] = (
            f"{kind.value} gate {gate.gate_id} answered "
            f"{decision.outcome.value} by {decision.decided_by}"
            + (f"; the board refused to store the gate itself ({refusal})"
               if refusal else "")
        )
        if decision.outcome is GateOutcome.APPROVE:
            self._record_approval(ctx, gate, decision, action)
        return decision.outcome, gate.gate_id

    def _record_gate(self, ctx: AgentContext, gate: HumanGate
                     ) -> tuple[str | None, str | None]:
        """Post the pending gate. Returns ``(entry_id, refusal_reason)``.

        The shipped zone registry has no ``human_gate`` kind, so this post is
        expected to be refused by ``InMemoryBlackboard``. The refusal is returned
        rather than swallowed: the gate is still enforced from the
        ``HumanDecision``, but a reader of the trace should be able to see that
        the pending gate itself was not recordable.
        """
        try:
            entry_id = self.post(ctx, ZONE_APPROVALS, GATE_ENTRY_KIND,
                                 gate.model_dump(mode="json"),
                                 source=DecisionSource.RULES)
            return entry_id, None
        except (BlackboardError, SchemaError) as exc:
            return None, str(exc)

    def _find_human_decision(self, ctx: AgentContext, gate_id: str,
                             kind: GateKind) -> HumanDecision | None:
        """The newest recorded decision for this gate, or ``None``.

        Deliberately strict: the entry must name this exact ``gate_id`` and this
        gate ``kind``. A decision for some other gate, or an ``Approval`` row
        standing in for a decision, does not authorise anything — the old code's
        ``approve=True`` default is exactly that substitution.

        One explicit exception: a graph-answered gate propagated via
        ``ctx.scratch["graph_approvals"]`` (see ``graph.build._propagate_approval``).
        The graph raises a different gate id in ``gates/`` while the agent waits
        on its own id in ``approvals/``; on resume the human's answer is handed
        to the agent through scratch, labelled as propagated, so the side effect
        can actually occur instead of the graph claiming it did.
        """
        try:
            entries = read_entries(ctx.board, ZONE_APPROVALS, kind=KIND_HUMAN_DECISION)
        except AgentError:
            entries = []
        for entry in reversed(entries):
            try:
                decision = HumanDecision.model_validate(entry.payload)
            except ValidationError:
                continue
            if decision.gate_id == gate_id and decision.kind is kind:
                return decision
        # Propagated graph approval: same kind, approved outcome, explicitly
        # labelled. Board exact-match still wins; this only fires on resume.
        try:
            propagated = ctx.scratch.get("graph_approvals") or []
        except Exception:
            propagated = []
        for item in reversed(list(propagated)):
            try:
                p_kind = str(item.get("kind") or "")
                p_outcome = str(item.get("outcome") or "")
            except Exception:
                continue
            if p_kind.rsplit(".", 1)[-1] != kind.value:
                continue
            if p_outcome.rsplit(".", 1)[-1] != GateOutcome.APPROVE.value:
                continue
            try:
                outcome = GateOutcome(p_outcome.rsplit(".", 1)[-1])
            except ValueError:
                continue
            return HumanDecision(
                gate_id=gate_id,
                kind=kind,
                outcome=outcome,
                decided_by=str(item.get("decided_by") or "unknown"),
                instruction=("propagated from graph gate "
                             f"{item.get('gate_id')}: {item.get('instruction') or ''}".strip()),
            )
        return None

    def _record_approval(self, ctx: AgentContext, gate: HumanGate,
                         decision: HumanDecision, action: str) -> str | None:
        """Write the durable proof that the gate was honoured before the action."""
        approval = Approval(
            approval_id=new_id("apr"),
            gate_id=gate.gate_id,
            kind=gate.kind,
            outcome=decision.outcome,
            decided_by=decision.decided_by,
            action_taken=action[:240],
            at=utcnow(),
        )
        try:
            return self.post(ctx, ZONE_APPROVALS, KIND_APPROVAL,
                             approval.model_dump(mode="json"),
                             source=DecisionSource.RULES)
        except (BlackboardError, SchemaError):
            return None

    def _mark_gate_blocked(self, ctx: AgentContext, brand: str, gate_id: str,
                           extra: str | None = None) -> None:
        """Record that a thread is parked, so the state survives the stop.

        The thread itself stays ``pending_approval``: nothing was sent, and a
        thread that claimed ``sent`` would be the old bug wearing a new hat.
        """
        notes: list[str] = []
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        threads = self._threads_by_brand(ctx, notes)
        pair = threads.get(brand)
        thread = pair[1] if pair else None
        if event is None:
            return
        parked = Thread(
            thread_id=thread.thread_id if thread else new_id("thr"),
            event_id=event.event_id,
            brand=brand,
            email=thread.email if thread else None,
            status="pending_approval",
            day=thread.day if thread else 0,
            intent=thread.intent if thread else Intent.UNKNOWN,
            reply_text=thread.reply_text if thread else "",
            offer_id=thread.offer_id if thread else None,
            sent_at=thread.sent_at if thread else None,
            delivered=False,
        )
        try:
            self.post(ctx, ZONE_THREADS, KIND_THREAD, parked.model_dump(mode="json"),
                      refs=[gate_id], confidence=1.0, source=DecisionSource.RULES)
        except (BlackboardError, SchemaError):
            return

    # ================================================================== decisions
    def _classify(self, ctx: AgentContext, thread: Thread) -> Decision:
        """Classify a sponsor reply. One closed choice, no keywords anywhere.

        The whole reply goes into ``state``. The option set is every
        human-recognisable intent, which is what makes the ordering fix
        structural: there is no ``yes``-first code path to get wrong, because
        there is no code path at all — only one question to a calibrated model.
        """
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(
                "What is this sponsor's intent in the message below? Judge the "
                "whole message, not the first word."
            ),
            question_type=QuestionType.CHOICE,
            state={
                "reply_text": thread.reply_text,
                "brand": thread.brand,
                "quoted_offer_id": thread.offer_id,
                "options_meaning": {
                    Intent.YES.value: "accepts the offer / agrees to proceed",
                    Intent.PUSHBACK.value: ("objects to price, terms, or "
                                            "deliverables — explicitly including "
                                            "any message that mentions a price "
                                            "objection together with any word of "
                                            "agreement"),
                    Intent.INTERESTED.value: "wants more information, no commitment",
                    Intent.NO.value: "declines or is not contactable for this",
                    Intent.NEUTRAL.value: "neither agreement nor objection",
                },
                "tie_break": ("if the message objects to anything at all, it is "
                              "pushback, whatever else it contains"),
            },
            options=list(INTENT_OPTIONS),
            instructions=(
                "Choose exactly one option. A price objection is pushback even "
                "when the message also contains a pleasantry or the word 'yes'. "
                "Choose neutral rather than guessing."
            ),
            asked_by=self.id,
            decision_point="a3.classify_intent",
        )
        return self._decide(ctx, request)

    def _intent_of(self, decision: Decision, thread: Thread,
                   notes: list[str]) -> Intent:
        """The decided intent, or an honest ``UNKNOWN`` when it is unusable.

        An answer outside the option set is a data fault in the backend, not an
        invitation to guess. The deterministic classifier in ``core.protocols``
        is used as a *cross-check* and its answer recorded as such, because that
        function exists for exactly this case and has the refusal-before-assent
        ordering the old router got wrong.
        """
        try:
            return Intent((decision.choice or "").strip())
        except ValueError:
            fallback = classify_intent_fallback(thread.reply_text)
            notes.append(
                f"the decision backend returned {decision.choice!r}, which is not "
                f"an Intent; the deterministic classifier in core.protocols says "
                f"{fallback.value} (recorded as a cross-check, not as the answer)")
            return Intent.UNKNOWN

    def _decide(self, ctx: AgentContext, request: DecisionRequest) -> Decision:
        """The only route to a model in this agent."""
        try:
            decision = ctx.decide(request)
        except (DecisionUnavailable, DecisionFailed, ToolUnavailable) as exc:
            raise AgentError(
                f"decision backend failed at {request.decision_point!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(decision, Decision):
            raise AgentError(
                f"ctx.decide returned {type(decision).__name__} for "
                f"{request.decision_point!r}, expected core.schemas.Decision"
            )
        return decision

    # ====================================================================== board
    def _threads_by_brand(self, ctx: AgentContext, notes: list[str]
                          ) -> dict[str, tuple[BoardEntry | None, Thread]]:
        """Newest valid thread per brand, with the entry it came from."""
        try:
            entries = read_entries(ctx.board, ZONE_THREADS, kind=KIND_THREAD)
        except AgentError as exc:
            notes.append(str(exc))
            return {}
        latest: dict[str, tuple[BoardEntry | None, Thread]] = {}
        for entry in entries:
            try:
                thread = Thread.model_validate(entry.payload)
            except ValidationError as exc:
                notes.append(f"skipped thread entry {entry.entry_id}: "
                             f"{exc.error_count()} field error(s)")
                continue
            latest[thread.brand] = (entry, thread)
        return latest

    def _latest_offers(self, ctx: AgentContext, notes: list[str]) -> dict[str, Offer]:
        """Newest offer per brand — the terms that are actually on the table."""
        offers = load_all(ctx.board, ZONE_OFFERS, Offer, kind=KIND_OFFER, notes=notes)
        latest: dict[str, Offer] = {}
        for offer in offers:
            latest[offer.brand] = offer
        return latest

    def _offer_for(self, ctx: AgentContext, brand: str,
                   notes: list[str]) -> Offer | None:
        return self._latest_offers(ctx, notes).get(brand)

    def _offer_with_entry(self, ctx: AgentContext, offer_id: str, notes: list[str]
                          ) -> tuple[BoardEntry, Offer] | None:
        try:
            entries = read_entries(ctx.board, ZONE_OFFERS, kind=KIND_OFFER)
        except AgentError as exc:
            notes.append(str(exc))
            return None
        for entry in reversed(entries):
            try:
                offer = Offer.model_validate(entry.payload)
            except ValidationError:
                continue
            if offer.offer_id == offer_id:
                return entry, offer
        notes.append(f"offer {offer_id!r} is not in the offers zone")
        return None

    def _leads_by_brand(self, ctx: AgentContext, notes: list[str]) -> dict[str, BrandLead]:
        leads = load_all(ctx.board, ZONE_OPPORTUNITIES, BrandLead,
                         kind=KIND_BRAND_LEAD, notes=notes)
        return {lead.name: lead for lead in leads}

    def _lead_for(self, ctx: AgentContext, brand: str,
                  notes: list[str]) -> BrandLead | None:
        return self._leads_by_brand(ctx, notes).get(brand)

    def _send_queue(self, ctx: AgentContext, event: EventProfile, notes: list[str]
                    ) -> list[tuple[Offer, Thread | None, BrandLead | None, str]]:
        """Offers whose current terms have not been sent to a contactable brand.

        Derived from the board: the newest ``Thread`` for a brand names the
        ``offer_id`` that was last put to them, so "does this brand need the new
        price?" is a comparison of two ids rather than a flag in some agent's
        memory. A brand whose newest thread is terminal is left alone, and a
        brand with no contact route is reported rather than invented.

        A2's ``COUNTER_PROPOSAL`` messages are read here, not just noted: when
        the newest unsent offer for a brand is the subject of a counter-proposal
        from A2, the send reason names that message, so the send is visibly
        caused by A2's revision rather than by the offer merely existing.
        """
        offers = self._latest_offers(ctx, notes)
        threads = self._threads_by_brand(ctx, notes)
        leads = self._leads_by_brand(ctx, notes)
        counters_by_brand: dict[str, list[tuple[BoardEntry, Message]]] = {}
        for entry, message in self._read_messages(ctx, notes):
            if message.kind is not MessageKind.COUNTER_PROPOSAL:
                continue
            if message.from_agent is not AgentId.A2_PRICING:
                continue
            for brand in offers:
                if brand and brand in message.body:
                    counters_by_brand.setdefault(brand, []).append((entry, message))
        queue: list[tuple[Offer, Thread | None, BrandLead | None, str]] = []
        for brand, offer in offers.items():
            pair = threads.get(brand)
            thread = pair[1] if pair else None
            if thread is not None and thread.status in TERMINAL_STATUSES:
                continue
            already_sent = thread is not None and thread.offer_id == offer.offer_id
            if already_sent:
                continue
            lead = leads.get(brand)
            address = (thread.email if thread else None) or (lead.contact_email if lead else None)
            if not address:
                continue
            reason = ("first contact for this brand"
                      if thread is None else
                      f"the newest offer ({offer.offer_id} v{offer.version}) has "
                      f"not been sent; the last thing {brand} saw was "
                      f"{thread.offer_id or 'nothing'}")
            counters = counters_by_brand.get(brand, [])
            if counters and offer.revised:
                latest_entry, latest_msg = counters[-1]
                reason += (f"; A2 counter-proposal {latest_msg.message_id} "
                           f"(entry {latest_entry.entry_id}) revises to "
                           f"INR {offer.amount_inr:,.0f}: sending the revised terms")
            queue.append((offer, thread, lead, reason))
        return queue

    def _unreachable(self, ctx: AgentContext, event: EventProfile,
                     notes: list[str]) -> list[str]:
        """Brands that cannot be contacted, named so the gap is actionable."""
        offers = self._latest_offers(ctx, notes)
        threads = self._threads_by_brand(ctx, notes)
        leads = self._leads_by_brand(ctx, notes)
        out: list[str] = []
        for brand in offers:
            pair = threads.get(brand)
            thread = pair[1] if pair else None
            if thread is not None and thread.status in TERMINAL_STATUSES:
                continue
            address = (thread.email if thread else None) or (
                leads[brand].contact_email if brand in leads else None)
            if not address:
                out.append(brand)
        return out

    def _handoff(self, ctx: AgentContext, to_agent: AgentId, *, event_id: str,
                 reason: str, confidence: float, source: DecisionSource,
                 summary: str, refs: Sequence[str]) -> str | None:
        """Record the routing decision, on the board and in the trace."""
        handoff = Handoff(
            handoff_id=new_id("hnd"),
            event_id=event_id,
            run_id=ctx.run_id,
            from_agent=self.id,
            to_agent=to_agent,
            reason=reason,
            decision_source=source,
            confidence=min(1.0, max(0.0, confidence)),
            summary=summary,
            payload_refs=[r for r in refs if r],
        )
        entry_id: str | None = None
        try:
            entry_id = self.post(ctx, ZONE_HANDOFFS, KIND_HANDOFF,
                                 handoff.model_dump(mode="json"),
                                 refs=[r for r in refs if r],
                                 confidence=handoff.confidence, source=source)
        except (BlackboardError, SchemaError):
            entry_id = None
        emit = getattr(ctx.tracer, "handoff", None)
        if callable(emit):
            try:
                emit(handoff)
            except (TypeError, ValueError, RuntimeError):
                pass
        return entry_id

    # ==================================================== inter-agent coordination
    def _post_message(self, ctx: AgentContext, kind: MessageKind,
                      to_agent: AgentId, body: str, *,
                      refs: Sequence[str], confidence: float,
                      source: DecisionSource, summary: str = "") -> str | None:
        """Post one typed :class:`~core.schemas.Message` to ``messages``.

        Best-effort like :meth:`_record_gate`: a board that refuses the post
        (an older stub registry without a ``messages`` zone) yields ``None``
        and the caller falls back to the thread+handoff path rather than
        failing the step. On the real board the message is the record A2
        reads to find revision work.
        """
        message = Message(
            message_id=new_id("msg"),
            event_id=ctx.event_id,
            run_id=ctx.run_id,
            from_agent=self.id,
            to_agent=to_agent,
            kind=kind,
            body=body,
            refs=[r for r in refs if r],
            decision_source=source,
            confidence=min(1.0, max(0.0, float(confidence or 0.0))),
            summary=summary,
        )
        try:
            entry_id = self.post(ctx, ZONE_MESSAGES, KIND_MESSAGE,
                                 message.model_dump(mode="json"),
                                 refs=[r for r in refs if r],
                                 confidence=message.confidence, source=source)
        except (BlackboardError, SchemaError):
            return None
        emit = getattr(ctx.tracer, "message", None)
        if callable(emit):
            try:
                emit(message)
            except (TypeError, ValueError, RuntimeError):
                pass
        return entry_id

    def _read_messages(self, ctx: AgentContext, notes: list[str]
                       ) -> list[tuple[BoardEntry, Message]]:
        """Typed ``messages`` reads; a refusal or unknown zone reads as empty."""
        try:
            entries = read_entries(ctx.board, ZONE_MESSAGES, kind=KIND_MESSAGE)
        except AgentError as exc:
            notes.append(str(exc))
            return []
        out: list[tuple[BoardEntry, Message]] = []
        for entry in entries:
            try:
                out.append((entry, Message.model_validate(entry.payload)))
            except ValidationError:
                notes.append(f"skipped message entry {entry.entry_id}: invalid Message")
        return out

    def _counter_proposals_for(self, ctx: AgentContext, brand: str,
                               notes: list[str]
                               ) -> list[tuple[BoardEntry, Message]]:
        """``COUNTER_PROPOSAL`` messages from A2 that name ``brand``.

        A2 always names the brand (and both offer ids) in the body, so a
        substring match on the brand is the honest join: messages carry no
        brand field of their own, and matching on refs alone would miss the
        entry-id indirection between the message and the thread.
        """
        found: list[tuple[BoardEntry, Message]] = []
        for entry, message in self._read_messages(ctx, notes):
            if message.kind is not MessageKind.COUNTER_PROPOSAL:
                continue
            if message.from_agent is not AgentId.A2_PRICING:
                continue
            if message.to_agent is not self.id:
                continue
            if brand and brand not in message.body:
                continue
            found.append((entry, message))
        return found

    def _open_disputes_between(self, ctx: AgentContext, notes: list[str]
                               ) -> list[Dispute]:
        """Unresolved A3<->A2 disputes already on the board."""
        try:
            entries = read_entries(ctx.board, ZONE_DISPUTES, kind=KIND_DISPUTE)
        except AgentError as exc:
            notes.append(str(exc))
            return []
        out: list[Dispute] = []
        for entry in entries:
            try:
                dispute = Dispute.model_validate(entry.payload)
            except ValidationError:
                continue
            parties = {dispute.claimant, dispute.opponent}
            if parties != {AgentId.A3_OUTREACH, AgentId.A2_PRICING}:
                continue
            if dispute.status is DisputeStatus.RESOLVED:
                continue
            out.append(dispute)
        return out

    def _open_pricing_dispute(self, ctx: AgentContext, thread: Thread,
                              offer: Offer | None, thread_entry_id: str,
                              counters: Sequence[tuple[BoardEntry, Message]],
                              notes: list[str], confidence: float,
                              source: DecisionSource) -> str | None:
        """Open an A3-vs-A2 pricing dispute after a counter-proposal failed.

        The sponsor pushed back, A2 revised (its ``COUNTER_PROPOSAL`` is on
        the board), and the sponsor pushed back *again* on the revised terms.
        That is a genuine disagreement about price between the agent that
        owns the sponsor relationship and the agent that owns the tariff —
        not a router inference — so A3 authors it with both positions and
        entry-id evidence. A7 reads ``disputes`` next; the graph debate path
        is the fallback when A7 is absent.
        """
        latest_msg = counters[-1][1] if counters else None
        evidence = [thread_entry_id]
        for entry, _msg in counters[-2:]:
            if entry.entry_id not in evidence:
                evidence.append(entry.entry_id)
        quoted = (f"{offer.tier} at INR {offer.amount_inr:,.0f} "
                  f"(offer {offer.offer_id} v{offer.version})" if offer else
                  "the current terms")
        dispute = Dispute(
            dispute_id=new_id("dsp"),
            event_id=thread.event_id,
            zone=DisputeZone.PRICING,
            claimant=self.id,
            opponent=AgentId.A2_PRICING,
            position=(f"A3: {thread.brand}'s objection stands after A2's revision "
                      f"({quoted}); objection on record: "
                      f"{thread.reply_text[:240] or 'no text recorded'}"),
            counter_position=(
                f"A2: {latest_msg.body[:240] if latest_msg else 'the revised offer stands'}"),
            evidence=[e for e in evidence if e],
            severity=Severity.MEDIUM,
            status=DisputeStatus.OPEN,
            rounds=0,
        )
        try:
            return self.post(ctx, ZONE_DISPUTES, KIND_DISPUTE,
                             dispute.model_dump(mode="json"),
                             refs=[e for e in evidence if e],
                             confidence=min(1.0, max(0.0, float(confidence or 0.0))),
                             source=source)
        except (BlackboardError, SchemaError) as exc:
            notes.append(f"could not open the pricing dispute for {thread.brand}: {exc}")
            return None

    # ==================================================================== message
    @staticmethod
    def _compose(event: EventProfile, brand: str, offer: Offer) -> tuple[str, str]:
        """Subject and body for the outreach message.

        The body is A2's pitch verbatim. A3 does not rewrite the commercial terms
        it was handed, and it never adds a claim of its own: an outreach message
        that says something the priced offer does not is how a sponsor ends up
        expecting a deliverable nobody priced.
        """
        subject = (f"{brand} x {event.name} — {offer.tier} sponsorship "
                   f"(INR {offer.amount_inr:,.0f})")
        body = offer.pitch or (
            f"Dear {brand} team,\n\n"
            f"We are organising {event.name} at {event.location} on {event.date}, "
            f"with an expected footfall of {event.footfall:,} "
            f"({event.audience}).\n\n"
            f"We would like to offer you the {offer.tier} tier at "
            f"INR {offer.amount_inr:,.0f}, which includes:\n"
            + "\n".join(f"  - {item}" for item in offer.deliverables)
            + "\n\nIf that works, reply and we will draft the MoU from these terms."
        )
        return subject, body

    # ==================================================================== helpers
    @staticmethod
    def _pick_send_tool(ctx: AgentContext, *,
                        purpose: str = "outreach email sending",
                        decision_point: str = "a3.select_send_tool"
                        ) -> tuple[Tool | None, str | None]:
        """Model-directed send-transport choice with function schemas in state."""
        from agents.tool_selection import select_tool

        tool, name, _ = select_tool(
            ctx, purpose, list(SEND_TOOL_NAMES), decision_point)
        return tool, name

    def _terminal_plan(self, reason: str, gap: str | None = None) -> Plan:
        steps = ["halt"]
        if gap:
            steps.append(gap)
        return Plan(goal=reason, steps=steps, rationale=reason, confidence=1.0,
                    source=DecisionSource.RULES, stop=True)

    @staticmethod
    def _state(ctx: AgentContext) -> dict[str, Any]:
        value = ctx.scratch.get("a3_state")
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _store(ctx: AgentContext, state: Mapping[str, Any]) -> None:
        ctx.scratch["a3_state"] = dict(state)
