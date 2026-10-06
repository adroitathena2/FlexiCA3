"""

A router in this package returns a **string label** that
``StateGraph.add_conditional_edges`` maps to a node. That is not a limitation
adopted for convenience: a ``Command`` return value is unusable here because
LangGraph indexes the ``path_map`` dict with the router's return value, and a
``Command`` is unhashable (verified on langgraph 1.2.12). ``Command(goto=...)`` is
still used — by the nodes that resume from an ``interrupt``, which is where the
deprecation notes actually point.

The rule every router obeys
---------------------------
**A router that routes without recording why did not decide; it looked up an
answer.** So each router here:

1. builds a :class:`~core.schemas.DecisionRequest`,
2. calls ``ctx.decide`` (via :func:`graph.contract_net.ask_decision`, which
   converts a backend outage into a ``DecisionSource.RULES`` answer and *says
   so* rather than pretending a model spoke),
3. writes a :class:`~core.schemas.Handoff` carrying ``from_agent``,
   ``to_agent``, ``reason``, ``decision_source`` and ``confidence``,
4. posts that handoff to the blackboard and the tracer, and
5. returns the label.

The ``Handoff`` is queued on the runtime and drained into state by the *next*
node, because a router cannot write state. Every router's target is therefore a
node — never a bare ``END`` — so a handoff cannot be dropped between a routing
decision and the next persisted write. ``finalize`` is the terminal node for
exactly this reason.

The routers
-----------
``route_after_discovery``  too few viable leads -> replan with a wider radius
``route_after_proposal``  proceed to outreach, or escalate to debate on a dispute
``route_after_reply``      LLM classification of the sponsor's intent
``route_after_compliance`` no blocking flags -> audit; blocking -> arbiter
``route_progress``         Magentic-One: complete / progressing / stalled

plus three that the topology needs: ``route_after_auction``,
``route_after_adjudication``, and ``route_gate_outcome``.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from core import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    Handoff,
    Intent,
    QuestionType,
    new_id,
)

from .contract_net import ask_decision
from .state import (
    GraphRuntime,
    blocking_flags,
    latest_offer_for,
    ledger_is_complete,
    ledger_is_progressing,
    read_progress_ledger,
    read_task_ledger,
    runtime_context,
    stall_reason,
)

__all__ = [
    "route_after_discovery", "route_after_proposal", "route_after_reply",
    "route_after_compliance", "route_progress", "route_after_auction",
    "route_after_adjudication", "route_gate_outcome",
    "record_handoff", "flush_handoffs", "_local_decision",
    "record_message", "flush_messages", "flush_pending_disputes",
    "NODE", "LABEL", "PATH_MAPS", "CONDITIONAL_SOURCES", "static_predecessor",
    "MIN_VIABLE_LEADS", "VIABLE_FIT_FLOOR", "MAX_OUTREACH_RUNS", "RADIUS_STEPS",
    "MESSAGE_KINDS", "ZONE_MESSAGES", "ZONE_MESSAGES_FALLBACK",
]

log = logging.getLogger("paytriq.graph.edges")


class NODE:
    """Canonical node names. ``build`` wires these; routers return ``LABEL``s."""

    START = "__start__"
    END = "__end__"
    DISCOVER = "discover"
    REPLAN = "replan"
    PRICE = "price"
    AUCTION = "auction"
    OUTREACH = "outreach"
    REPLY = "reply"
    REVISE = "revise"
    ARCHIVE = "archive"
    DEBATE = "debate"
    ADJUDICATE = "adjudicate"
    ESCALATE = "escalate"
    CONTRACT = "contract"
    COMPLIANCE = "compliance"
    AUDIT = "audit"
    FINALIZE = "finalize"


class LABEL:
    """Router return labels. Distinct from node names on purpose.

    If labels and node names were the same string, a ``path_map`` bug would be
    invisible: the router would still resolve. Keeping them separate means a
    label that no node answers is a loud failure.
    """

    REPLAN = "replan_for_radius"
    PROCEED = "proceed"
    DISPUTE = "escalate_dispute"
    CONTRACT = "go_contract"
    REPRICE = "go_reprice"
    FOLLOWUP = "followup"
    ARCHIVE = "archive_thread"
    NEGOTIATION_EXHAUSTED = "negotiation_exhausted_dispute"
    CLEAR = "compliance_clear_for_audit"
    BLOCKING = "arbitrate"
    COMPLETE = "complete"
    PROGRESS = "progress"
    STALLED = "replan_progress"
    ROI_REVISE = "roi_critique_revise"
    GIVE_UP = "give_up"
    AWARDED = "execute_award"
    UNCONTRACTED = "self_execute"
    RESOLVED = "audit_after_resolution"
    NEEDS_ESCALATION = "escalate_to_human"
    APPROVED = "gate_approved_proceed"
    REJECTED = "gate_rejected_report"
    REVISE = "gate_revise_replan"


#: How many viable leads (fit_score >= 40 and a contact route) discovery must
#: return before pricing is worth doing. A constant, not a decision: the
#: *threshold* is policy, the *question of whether it is met* is a decision, and
#: only the latter goes to ``ctx.decide``.
MIN_VIABLE_LEADS = 3

#: Minimum ``fit_score`` for a lead to count as viable. Separate from
#: ``MIN_VIABLE_LEADS`` (the count) so the two policy knobs can move
#: independently.
VIABLE_FIT_FLOOR = 40.0

#: How many times ``outreach`` may run for one event. Three is the number of real
#: touchpoints (day 0, day 3, day 7); a fourth is nagging, not outreach. Without
#: this bound, an ``interested`` reply is an infinite loop, because each follow-up
#: can produce another ``interested`` reply.
MAX_OUTREACH_RUNS = 3

#: Radius multipliers applied by successive replans. Widening the net is the
#: concrete response to "not enough leads"; it is data in the state so the
#: replan is visible in the trace rather than implied.
RADIUS_STEPS = (2.0, 4.0, 8.0)


#: Message kinds for forced disagreement (Track 4). ``core.schemas.Message``
#: is owned by a parallel track, so every helper here duck-types it: a real
#: ``Message`` model is used when importable, otherwise a plain JSON dict with
#: the same fields. Never import ``Message`` at module top: it may not exist yet.
MESSAGE_KINDS: tuple[str, ...] = (
    "proposal", "critique", "counter_proposal", "revision_request", "verdict",
)

#: Primary board zone for messages. Not in ``blackboard/zones.py`` (frozen for
#: this track): the real board rejects it, which is why every post below tries
#: this zone first and falls back without ever breaking routing.
ZONE_MESSAGES = "messages"
ZONE_MESSAGES_FALLBACK = "handoffs"


# --------------------------------------------------------------------------- helpers
def record_handoff(rt: GraphRuntime, state: dict, *, to_agent: AgentId, reason: str,
                   decision: Decision, summary: str = "",
                   payload_refs: Sequence[str] | None = None,
                   from_agent: AgentId | None = None,
                   message_id: str | None = None) -> Handoff:
    """Write the coordination record for one routing decision.

    ``from_agent`` defaults to whichever agent produced the state the router read
    (``state["agent_id"]``), which is the honest answer: control was in that
    agent's hands when the decision was taken.

    ``message_id``, when given, links the forcing ``Message`` that caused this
    routing: it is appended to ``payload_refs`` and named in ``summary`` so the
    disagreement is traceable from the handoff alone.
    """
    context = runtime_context(rt)
    actor = from_agent or _actor_of(state, fallback=to_agent)
    if actor is to_agent:
        # ``Handoff`` forbids a self-handoff. A router that "hands to itself" has
        # not handed off anything; recording it as a handoff would inflate the
        # coordination record. Fall back to the run's supervisor so the decision
        # still has an author and a recipient.
        actor = AgentId.A7_ARBITER if to_agent is not AgentId.A7_ARBITER else AgentId.A6_AUDIT

    handoff = Handoff(
        handoff_id=new_id("hnd"),
        event_id=str(state.get("event_id") or context.event_id or ""),
        run_id=str(state.get("run_id") or context.run_id or ""),
        from_agent=actor,
        to_agent=to_agent,
        reason=reason,
        decision_source=decision.source,
        confidence=round(float(decision.confidence), 4),
        summary=(f"{summary} [message:{message_id}]" if message_id and summary
                 else (f"message:{message_id}" if message_id else summary)),
        payload_refs=([*list(payload_refs or []), str(message_id)]
                      if message_id and str(message_id) not in list(payload_refs or [])
                      else list(payload_refs or [])),
    )
    rt.board.post("handoffs", "handoff", actor, handoff.model_dump(mode="json"),
                  confidence=handoff.confidence, source=handoff.decision_source)
    rt.tracer.handoff(handoff)
    # Trace the routing decision itself, not just the handoff. Without this a
    # run routed entirely by clef shows decision_counts={} and llm_call_count=0
    # despite clef handoffs (the golden-trace bug). Best effort: a tracer fault
    # must not break routing. OtelTracer.decision is a context manager, so it
    # is entered; NullTracer.decision is a plain call.
    try:
        _dec = rt.tracer.decision(actor, decision,
                                  decision_point="route.handoff",
                                  handoff_id=handoff.handoff_id,
                                  to_agent=to_agent.value)
        if hasattr(_dec, "__enter__"):
            with _dec:
                pass
    except Exception:
        pass
    context.scratch.queue_handoff(handoff)
    log.info("handoff %s: %s -> %s via %s (confidence %.2f) because %s",
             handoff.handoff_id, actor.value, to_agent.value, decision.source.value,
             handoff.confidence, reason)
    return handoff


def flush_handoffs(rt: GraphRuntime) -> dict[str, Any]:
    """Drain queued handoffs into a state delta. Called by every node.

    Routers cannot write state, so they queue; nodes drain. The queue is on the
    runtime rather than in the state channel because a node's state writes are
    discarded when that node raises ``interrupt`` — a queued handoff is already
    durable on the blackboard and the tracer either way.
    """
    return runtime_context(rt).take_handoffs()


def _coerce_message_agent(value: Any) -> AgentId | None:
    """Best-effort ``AgentId`` coercion for message endpoints (duck-typed)."""
    if isinstance(value, AgentId):
        return value
    try:
        return AgentId(str(value).rsplit(".", 1)[-1])
    except (ValueError, AttributeError, TypeError):
        return None


def _normalise_message_kind(kind: Any) -> str:
    text = str(kind or "critique").strip().lower()
    if text in MESSAGE_KINDS:
        return text
    log.warning("unknown message kind %r; coercing to 'critique'", kind)
    return "critique"


def _build_message_payload(*, kind: str, from_agent: AgentId, to_agent: AgentId,
                           body: str, topic: str = "",
                           refs: Sequence[str] | None = None,
                           confidence: float = 0.7,
                           source: Any = None,
                           event_id: str = "", run_id: str = "",
                           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON message payload, using ``core.Message`` when importable.

    ``core/schemas.py Message`` is owned by a parallel track and may not exist
    yet, so this never imports at module top and tries several field aliases
    (``body``/``content``/``text``). Any failure falls back to the plain dict,
    which carries the same fields and is what the board stores either way.
    """
    message_id = new_id("msg")
    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.7
    try:
        src_value = source.value if hasattr(source, "value") else str(source or "rules")
    except Exception:
        src_value = "rules"
    base: dict[str, Any] = {
        "message_id": message_id,
        "kind": kind,
        "from_agent": from_agent.value,
        "to_agent": to_agent.value,
        "body": str(body or ""),
        "topic": str(topic or ""),
        "refs": [str(r) for r in (refs or []) if str(r or "").strip()],
        "confidence": conf,
        "source": src_value,
        "event_id": str(event_id or ""),
        "run_id": str(run_id or ""),
    }
    if extra:
        for k, v in extra.items():
            base.setdefault(k, v)
    try:
        import core as _core  # local: Message may not exist yet
        msg_cls = getattr(_core, "Message", None)
    except Exception:
        return base
    if msg_cls is None or not isinstance(msg_cls, type):
        return base
    attempts: list[dict[str, Any]] = [
        {"message_id": message_id, "kind": kind, "from_agent": from_agent,
         "to_agent": to_agent, "body": str(body or ""),
         "refs": list(base["refs"]), "confidence": conf},
        {"message_id": message_id, "kind": kind, "from_agent": from_agent,
         "to_agent": to_agent, "content": str(body or ""),
         "refs": list(base["refs"]), "confidence": conf},
        {"message_id": message_id, "kind": kind, "from_agent": from_agent,
         "to_agent": to_agent, "text": str(body or "")},
        {"kind": kind, "from_agent": from_agent, "to_agent": to_agent,
         "body": str(body or "")},
    ]
    for kwargs in attempts:
        try:
            obj = msg_cls(**kwargs)  # type: ignore[call-arg]
            dump = getattr(obj, "model_dump", None)
            if callable(dump):
                data = dict(dump(mode="json"))
            elif isinstance(obj, dict):
                data = dict(obj)
            else:
                data = dict(base)
            data.setdefault("message_id", message_id)
            data.setdefault("kind", kind)
            return data
        except Exception:
            continue
    return base


def record_message(rt: GraphRuntime, state: dict, *, kind: str,
                   from_agent: AgentId | str, to_agent: AgentId | str,
                   body: str, topic: str = "",
                   refs: Sequence[str] | None = None,
                   confidence: float = 0.7, source: Any = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Post one forcing ``Message`` (proposal/critique/...) without breaking routing.

    Mirrors :func:`record_handoff`: board post, tracer event, queue for flush.
    Every step is best-effort — a board refusal (the ``messages`` zone does not
    exist in the frozen registry) or a tracer fault is logged and routing
    continues. Returns the JSON payload (with ``message_id``) or ``None`` when
    even the payload could not be built.
    """
    try:
        context = runtime_context(rt)
    except Exception:
        context = rt  # type: ignore[assignment]
    try:
        src = _coerce_message_agent(from_agent)
        dst = _coerce_message_agent(to_agent)
        if src is None or dst is None:
            log.warning("record_message dropped: unrecognised agents %r -> %r",
                        from_agent, to_agent)
            return None
        norm_kind = _normalise_message_kind(kind)
        src_value = source
        if src_value is None:
            try:
                src_value = DecisionSource.RULES
            except Exception:
                src_value = "rules"
        payload = _build_message_payload(
            kind=norm_kind, from_agent=src, to_agent=dst, body=body, topic=topic,
            refs=refs, confidence=confidence, source=src_value,
            event_id=str(state.get("event_id") or getattr(context, "event_id", "") or ""),
            run_id=str(state.get("run_id") or getattr(context, "run_id", "") or ""),
            extra=extra,
        )
    except Exception as exc:  # noqa: BLE001 - message construction must not break routing
        log.warning("record_message could not build payload: %s: %s",
                    type(exc).__name__, exc)
        return None

    try:
        board = getattr(context, "board", getattr(rt, "board", None))
    except Exception:
        board = None
    if board is not None:
        candidates = [
            (ZONE_MESSAGES, "message"),
            (ZONE_MESSAGES_FALLBACK, "handoff"),
            ("debate", "rebuttal"),
        ]
        for zone, zone_kind in candidates:
            try:
                board.post(zone, zone_kind, src, dict(payload),
                           refs=list(payload.get("refs") or []),
                           confidence=float(payload.get("confidence") or 0.7),
                           source=(src_value if hasattr(src_value, "value") else DecisionSource.RULES))
                break
            except Exception as exc:  # noqa: BLE001 - try the next fallback zone
                log.debug("record_message post to %s/%s failed: %s: %s",
                          zone, zone_kind, type(exc).__name__, exc)
                continue
    try:
        tracer = getattr(context, "tracer", getattr(rt, "tracer", None))
        if tracer is not None and hasattr(tracer, "event"):
            try:
                from core import TraceKind as _TK
                _kind = _TK.HUMAN if hasattr(_TK, "HUMAN") else list(_TK)[0]
            except Exception:
                _kind = "message"  # type: ignore[assignment]
            try:
                tracer.event(_kind, f"message:{payload.get('kind')}",  # type: ignore[arg-type]
                             agent=src, **{k: v for k, v in payload.items()
                                           if k in ("message_id", "kind", "body", "topic",
                                                    "refs", "confidence", "source")})
            except Exception as exc:  # noqa: BLE001 - tracing is best-effort
                log.debug("record_message tracer event failed: %s: %s",
                          type(exc).__name__, exc)
    except Exception:
        pass
    try:
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            extras.setdefault("pending_messages", []).append(dict(payload))
    except Exception:
        pass
    try:
        log.info("message %s: %s -> %s kind=%s %.120s",
                 payload.get("message_id"), src.value, dst.value,
                 payload.get("kind"), str(body or ""))
    except Exception:
        pass
    return dict(payload)


def flush_messages(rt: GraphRuntime) -> dict[str, Any]:
    """Drain queued messages into ``run_events`` (no new state channel).

    ``PaytriqState`` has no ``messages`` channel, so messages travel as
    protocol-level ``run_events`` entries. Called by ``build._prep`` alongside
    :func:`flush_handoffs`; safe to call even when nothing is queued.
    """
    try:
        context = runtime_context(rt)
        pending = context.extras.pop("pending_messages", [])
    except Exception:
        return {}
    if not pending:
        return {}
    return {"run_events": [dict(m) if isinstance(m, dict) else {"value": str(m)}
                           for m in pending]}


def flush_pending_disputes(rt: GraphRuntime) -> dict[str, Any]:
    """Drain router-filed disputes queued in ``extras`` into ``disputes``."""
    try:
        context = runtime_context(rt)
        pending = context.extras.pop("pending_disputes", [])
    except Exception:
        return {}
    if not pending:
        return {}
    return {"disputes": [dict(d) if isinstance(d, dict) else {"value": str(d)}
                         for d in pending]}


def _ensure_blocking_dispute(rt: GraphRuntime, state: dict) -> dict[str, Any] | None:
    """File the A5 veto dispute when blocking flags exist but no dispute covers them.

    Best-effort and never fatal: any failure is logged and ``None`` returned,
    while routing still proceeds to adjudication. The dispute is posted to the
    board (``disputes`` then ``debate`` zones) and queued in
    ``extras["pending_disputes"]`` so the next node (``adjudicate`` via
    ``build._prep``) carries it into state.
    """
    try:
        blocking = blocking_flags(state)
    except Exception:
        return None
    if not blocking:
        return None
    try:
        for existing in reversed(list(state.get("disputes") or [])):
            if not isinstance(existing, dict):
                continue
            status = str(existing.get("status") or "").rsplit(".", 1)[-1].lower()
            if status in ("resolved", "resolved."):
                continue
            zone = str(existing.get("zone") or "").rsplit(".", 1)[-1].lower()
            severity = str(existing.get("severity") or "").rsplit(".", 1)[-1].lower()
            if zone == "compliance" or severity == "blocking":
                return dict(existing)
    except Exception:
        pass
    try:
        context = runtime_context(rt)
        event_id = str(state.get("event_id") or getattr(context, "event_id", "") or "")
        codes = ", ".join(str(f.get("code")) for f in blocking[:4])
        evidence = [str(f.get("flag_id") or f.get("code")) for f in blocking[:3]
                    if str(f.get("flag_id") or f.get("code") or "").strip()]
        position = (f"A5 blocks: {len(blocking)} blocking flag(s) ({codes}); "
                    f"no promise may be signed without evidence")
        counter = "the pipeline holds priced proposals that A5 has not cleared"
        payload: dict[str, Any] | None = None
        try:
            from core import Dispute as _Dispute
            from core import DisputeStatus as _DS
            from core import DisputeZone as _DZ
            from core import Severity as _Sev
            opponent = AgentId.A4_CONTRACT if (state.get("mous") or []) else AgentId.A2_PRICING
            dispute_obj = _Dispute(
                dispute_id=new_id("dsp"),
                event_id=event_id,
                zone=_DZ.COMPLIANCE,
                claimant=AgentId.A5_COMPLIANCE,
                opponent=opponent,
                position=position,
                counter_position=counter,
                evidence=list(evidence),
                severity=_Sev.BLOCKING,
                status=_DS.OPEN,
                rounds=0,
            )
            dump = getattr(dispute_obj, "model_dump", None)
            payload = dict(dump(mode="json")) if callable(dump) else dict(position=position)
        except Exception as exc:  # noqa: BLE001 - dict fallback keeps the veto visible
            log.debug("Dispute model unavailable, using dict fallback: %s", exc)
            payload = {
                "dispute_id": new_id("dsp"),
                "event_id": event_id,
                "zone": "compliance",
                "claimant": AgentId.A5_COMPLIANCE.value,
                "opponent": AgentId.A4_CONTRACT.value,
                "position": position,
                "counter_position": counter,
                "evidence": list(evidence),
                "severity": "blocking",
                "status": "open",
                "rounds": 0,
            }
        assert payload is not None
        try:
            board = getattr(context, "board", None)
            if board is not None:
                for zone, zone_kind in (("disputes", "dispute"), ("debate", "dispute")):
                    try:
                        board.post(zone, zone_kind, AgentId.A5_COMPLIANCE, dict(payload),
                                   refs=list(evidence),
                                   confidence=0.9, source=DecisionSource.RULES)
                    except Exception as exc:  # noqa: BLE001 - one zone may reject
                        log.debug("blocking dispute post to %s failed: %s", zone, exc)
                        continue
        except Exception:
            pass
        try:
            extras = getattr(context, "extras", None)
            if isinstance(extras, dict):
                extras.setdefault("pending_disputes", []).append(dict(payload))
        except Exception:
            pass
        log.info("blocking dispute %s filed by A5 (%d flag(s): %s)",
                 payload.get("dispute_id"), len(blocking), codes)
        return dict(payload)
    except Exception as exc:  # noqa: BLE001 - filing must not break routing
        log.warning("could not file blocking dispute: %s: %s",
                    type(exc).__name__, exc)
        return None


def _roi_revision_needed(state: dict) -> tuple[bool, str]:
    """True when A6's ROI report needs an A2 revision (Track 4 critique).

    Narrow by design: ``None`` (A6 never reported) is *not* a critique — it is
    "not measured", and routing on it would divert every clean run that has not
    reached audit yet. Only a *present but baseless or incoherent* report forces
    a revision: missing/empty ``assumptions`` (the schema requires non-empty, so
    an empty list here means the audit produced numbers with no stated basis) or
    a sponsored total with a zero multiple (figures that do not cohere).
    """
    roi = state.get("roi_report")
    if not roi or not isinstance(roi, dict):
        return False, ""
    assumptions = roi.get("assumptions")
    if not assumptions:
        return True, ("ROI report carries no assumptions; every derived figure "
                      "needs a stated basis — requesting an A2 revision")
    try:
        total = float(roi.get("total_sponsored_inr") or 0.0)
        multiple = float(roi.get("roi_multiple") or 0.0)
    except (TypeError, ValueError):
        return True, "ROI figures are not numeric; requesting an A2 revision"
    if total > 0 and multiple == 0:
        return True, (f"ROI multiple is 0 despite {total:.0f} INR sponsored; "
                      "assumptions do not support the figures — requesting an A2 revision")
    return False, ""


def _revision_budget(rt: GraphRuntime) -> int:
    """Budget for ROI-critique revisions: the tighter of the two loop caps."""
    try:
        settings = runtime_context(rt).settings
        return max(1, min(int(settings.max_replans), int(settings.max_debate_rounds)))
    except Exception:
        return 2


def _actor_of(state: dict, *, fallback: AgentId) -> AgentId:
    raw = state.get("agent_id")
    if not raw:
        return fallback
    try:
        return AgentId(str(raw).rsplit(".", 1)[-1])
    except ValueError:
        return fallback


def _decide(rt: GraphRuntime, request: DecisionRequest) -> Decision:
    """One entry point for every routing decision, so degradation is uniform."""
    return ask_decision(rt, request)


def _matches(choice: str | None, *labels: str) -> bool:
    text = str(choice or "").strip().lower()
    return any(text.startswith(label) for label in labels)


def _is_viable_lead(lead: dict) -> bool:
    """A lead is viable if it is worth contacting and there is a way to contact it.

    Both halves matter. A high-fit brand with no email is a nice logo in a
    report; a brand with an email and a fit score of zero is spam.
    """
    try:
        fit = float(lead.get("fit_score") or 0.0)
    except (TypeError, ValueError):
        return False
    reachable = bool(lead.get("contact_email") or lead.get("phone"))
    return fit >= VIABLE_FIT_FLOOR and reachable


# ============================================================== route_after_discovery
def route_after_discovery(state: dict, runtime: Any) -> str:
    """Too few viable leads -> replan with a wider radius. Otherwise price.

    This is a genuine conditional edge: with a stub decision layer that answers
    "proceed" it goes to pricing, and with one that answers "replan" it walks
    ``replan -> discover`` until ``max_replans`` is spent. It is not a fixed next
    step.
    """
    rt = runtime_context(runtime)
    leads = list(state.get("brands") or [])
    viable = [lead for lead in leads if _is_viable_lead(lead)]
    radius = float(state.get("discovery_radius_km") or 0.0)
    replan_count = int(state.get("replan_count") or 0)
    max_replans = int(runtime_context(rt).settings.max_replans)
    budget_spent = replan_count >= max_replans

    decision = _decide(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"{len(viable)} of {len(leads)} discovered leads are viable "
                  f"(fit >= {VIABLE_FIT_FLOOR}, contactable), within "
                  f"{radius:.1f} km. Is the pipeline rich enough to price "
                  f"sponsorships now, or must discovery widen its search?"),
        question_type=QuestionType.CHOICE,
        options=["proceed", "replan"],
        rubric=["proceed: enough viable leads exist to price and contact now",
                "replan: too few viable leads; widen the search radius and re-run discovery"],
        state={
            "viable_leads": len(viable),
            "total_leads": len(leads),
            "viable_names": [str(lead.get("name")) for lead in viable[:8]],
            "radius_km": radius,
            "required": MIN_VIABLE_LEADS,
            "replans_used": replan_count,
            "replan_budget": max_replans,
            "replan_budget_spent": budget_spent,
            "stall_count": int(state.get("stall_count") or 0),
        },
        instructions=("Judge only the size of the viable pipeline. The replan budget "
                      "is already spent, so a further replan cannot happen; answer "
                      "proceed if anything at all is workable."),
        asked_by=AgentId.A1_DISCOVERY,
        decision_point="route.after_discovery",
    ))

    ledger = read_task_ledger(state)
    if _matches(decision.choice, "replan") and not budget_spent:
        reason = (f"only {len(viable)} viable lead(s) at {radius:.1f} km "
                  f"(needs {MIN_VIABLE_LEADS}); decision={decision.choice}")
        record_handoff(rt, state, to_agent=AgentId.A1_DISCOVERY, reason=reason,
                       decision=decision, from_agent=AgentId.A1_DISCOVERY,
                       summary="discovery will re-run with a wider radius",
                       payload_refs=[str(lead.get("lead_id")) for lead in leads[:5]])
        return LABEL.REPLAN

    if _matches(decision.choice, "replan"):
        # The decision layer asked to replan but the budget is gone. Recording the
        # override rather than silently flipping the label is the point: a reader
        # can see that the reasoning wanted something else and the cost control
        # stopped it.
        note = (f"decision wanted a replan but the budget of {max_replans} is spent; "
                f"proceeding with {len(viable)} viable lead(s)")
        rt.note(note, "route.after_discovery")
        decision.raw = {**(decision.raw or {}), "budget_override": note}

    record_handoff(rt, state, to_agent=AgentId.A2_PRICING,
                   reason=(f"{len(viable)} viable lead(s) at {radius:.1f} km "
                           f"meets the threshold of {MIN_VIABLE_LEADS}"),
                   decision=decision, from_agent=AgentId.A1_DISCOVERY,
                   summary=f"task ledger revision {ledger.get('revision', 0)} priced",
                   payload_refs=[str(lead.get("lead_id")) for lead in viable[:5]])
    return LABEL.PROCEED


# ============================================================== route_after_proposal
def route_after_proposal(state: dict, runtime: Any) -> str:
    """Price something and either start outreach, or open a pricing dispute.

    A proposal that A2 cannot price cleanly — an amount below the floor implied
    by footfall, a deliverable it cannot honour, a dispute already open in the
    pricing zone — goes to debate rather than straight to the sponsor's inbox.
    Sending a number you already doubt is how the previous prototype lost a
    sponsor's trust.
    """
    rt = runtime_context(runtime)
    offers = list(state.get("offers") or [])
    open_disputes = [d for d in (state.get("disputes") or [])
                     if str(d.get("status")) not in ("resolved", "resolved.")]

    decision = _decide(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"{len(offers)} proposal(s) drafted, {len(open_disputes)} pricing "
                  f"dispute(s) open. Can outreach proceed, or must the pricing "
                  f"position be settled by debate first?"),
        question_type=QuestionType.CHOICE,
        options=["outreach", "dispute"],
        rubric=["outreach: the proposals are priced defensibly and ready to show a sponsor",
                "dispute: a number or a deliverable is not defensible yet"],
        state={
            "offers": [{"brand": o.get("brand"), "amount_inr": o.get("amount_inr"),
                        "version": o.get("version"), "revised": o.get("revised")}
                       for o in offers[:8]],
            "open_disputes": [{"dispute_id": d.get("dispute_id"), "zone": d.get("zone")}
                              for d in open_disputes[:5]],
            "compliance_score": state.get("compliance_score"),
            "blocking_flags": len(blocking_flags(state)),
        },
        instructions="If any amount is indefensible or a dispute is open, choose dispute.",
        asked_by=AgentId.A2_PRICING,
        decision_point="route.after_proposal",
    ))

    if _matches(decision.choice, "dispute") or (
            not _matches(decision.choice, "outreach") and open_disputes):
        record_handoff(rt, state, to_agent=AgentId.A7_ARBITER,
                       reason=(f"pricing dispute opened before outreach "
                               f"({len(open_disputes)} open, {len(offers)} proposals)"),
                       decision=decision, from_agent=AgentId.A2_PRICING,
                       summary="debate will contest the pricing position")
        return LABEL.DISPUTE

    record_handoff(rt, state, to_agent=AgentId.A3_OUTREACH,
                   reason=f"{len(offers)} proposal(s) priced and defensible",
                   decision=decision, from_agent=AgentId.A2_PRICING,
                   summary="outreach may contact sponsors",
                   payload_refs=[str(o.get("offer_id")) for o in offers[:5]])
    return LABEL.PROCEED


# ==================================================================== route_after_reply
_INTENT_FOR_LABEL = {
    LABEL.CONTRACT: Intent.YES,
    LABEL.REPRICE: Intent.PUSHBACK,
    LABEL.FOLLOWUP: Intent.INTERESTED,
    LABEL.ARCHIVE: Intent.NO,
}

_INTENT_TARGET = {
    Intent.YES: (AgentId.A4_CONTRACT, "sponsor accepted; an MoU can be drafted"),
    Intent.PUSHBACK: (AgentId.A2_PRICING, "sponsor pushed back on price; the offer needs revision"),
    Intent.INTERESTED: (AgentId.A3_OUTREACH, "sponsor is interested but undecided; keep the thread warm"),
    # ARCHIVE is owned by A3 (NODE_OWNERS[ARCHIVE] is A3_OUTREACH): a decline is
    # closed by the outreach owner, not by the supervisor. Naming A7 here while
    # PATH_MAPS routes ARCHIVE to the archive node (A3) made the handoff lie
    # about who acts next.
    Intent.NO: (AgentId.A3_OUTREACH, "sponsor declined; outreach closes the thread"),
    Intent.NEUTRAL: (AgentId.A3_OUTREACH, "reply was unclassifiable; continue nurturing"),
    Intent.UNKNOWN: (AgentId.A3_OUTREACH, "reply could not be classified; outreach holds for review"),
}


def route_after_reply(state: dict, runtime: Any) -> str:
    """Classify the sponsor's reply and route on it.

    The classification is a real ``DecisionRequest`` to the decision layer, whose
    answer names the intent. The keyword classifier in
    :func:`core.protocols.classify_intent_fallback` is used only when the
    decision layer cannot answer, and when it is used the returned
    :class:`Decision` carries ``DecisionSource.RULES`` and ``degraded=True``, so
    the trace shows a fallback rather than a model call.
    """
    rt = runtime_context(runtime)
    reply = _latest_reply(state)
    intent, decision, note = _classify(rt, state, reply)

    label = _LABEL_FOR_INTENT(intent)
    target, reason = _INTENT_TARGET[intent]

    # Two loops in the negotiation graph are unbounded by construction
    # (pushback -> revise -> outreach -> reply, and interested -> outreach ->
    # reply). Both are given a budget here, and both overrides are recorded on the
    # handoff so the reader can see the decision that was overruled.
    override = ""
    if label == LABEL.REPRICE:
        rounds = _negotiation_rounds(state)
        budget = int(runtime_context(rt).settings.max_replans)
        if rounds >= budget:
            label, target, reason = (LABEL.NEGOTIATION_EXHAUSTED, AgentId.A7_ARBITER,
                                     f"{rounds} pricing revision(s) already made and "
                                     f"pushback continues")
            override = (f"pushback would be revision {rounds + 1} but only {budget} "
                        f"are budgeted; escalating instead of looping")
    elif label == LABEL.FOLLOWUP:
        runs = list(state.get("steps_done") or []).count(NODE.OUTREACH)
        if runs >= MAX_OUTREACH_RUNS:
            label, target, reason = (LABEL.ARCHIVE, AgentId.A3_OUTREACH,
                                     f"outreach has run {runs} time(s) without a "
                                     f"decision; closing the thread")
            override = (f"follow-up {runs + 1} exceeds the {MAX_OUTREACH_RUNS} "
                        f"touchpoints an event gets")
    if override:
        rt.note(override, "route.after_reply")
        decision.raw = {**(decision.raw or {}), "budget_override": override}

    handoff = record_handoff(rt, state, to_agent=target, reason=reason, decision=decision,
                   from_agent=AgentId.A3_OUTREACH,
                   summary=f"intent={intent.value}" + (f"; {note}" if note else "")
                           + (f"; {override}" if override else ""),
                   payload_refs=[str(t.get("thread_id")) for t in
                                 (state.get("threads") or [])[-3:]])
    # Persist the classification where A2 actually looks: A2 revises only threads
    # whose intent is PUSHBACK (agents/a2_pricing.py). The handoff alone is not
    # enough — without a thread intent update the pushback is invisible to the
    # pricer and the loop never revises. Best effort; a board refusal must not
    # break routing.
    try:
        _mirror_classification(rt, state, intent, decision, handoff)
    except Exception:
        pass
    return label


def _mirror_classification(rt: GraphRuntime, state: dict, intent: Intent,
                           decision: Decision, handoff: Handoff) -> None:
    """Project the router's classification onto the board where A2 reads it.

    A2 finds pushback by scanning ``threads`` for ``intent==PUSHBACK``; the
    handoff alone never reaches it. The newest thread carrying reply text is
    reposted with the classified intent and a ref to the handoff, so the
    decision source/confidence survive via the handoff chain. Best effort: any
    failure is logged and routing still returns its label.
    """
    threads = list(state.get("threads") or [])
    target = None
    for thread in reversed(threads):
        if thread.get("reply_text"):
            target = thread
            break
    if target is None:
        return
    try:
        updated = dict(target)
        updated["intent"] = intent.value
        # Keep status consistent: a classified pushback/interest stays
        # negotiable; a yes/no closes appropriately. Never invent delivery.
        if intent is Intent.YES:
            updated["status"] = "closed_won"
        elif intent is Intent.NO:
            updated["status"] = "closed_lost"
        elif updated.get("status") not in ("sent", "replied", "negotiating",
                                           "closed_won", "closed_lost"):
            updated["status"] = "negotiating"
        rt.board.post("threads", "thread", AgentId.A3_OUTREACH, updated,
                      refs=[handoff.handoff_id],
                      confidence=float(decision.confidence),
                      source=decision.source)
    except Exception as exc:  # noqa: BLE001 - routing must survive a board fault
        log.warning("could not mirror reply classification to threads: %s: %s",
                    type(exc).__name__, exc)


def _negotiation_rounds(state: dict) -> int:
    """How many counter-offer revisions this run has already granted.

    Read from the explicit ``negotiation_rounds`` counter that the ``revise`` node
    increments, *not* inferred from ``Offer.version``. A version only advances when
    A2 actually rewrites the offer, so inferring from it would make the loop bound
    depend on whether A2 is registered — and an unbounded pushback loop would hang
    a graph that happens to be running without a pricing agent.
    """
    return int(state.get("negotiation_rounds") or 0)


def _latest_reply(state: dict) -> str:
    for thread in reversed(list(state.get("threads") or [])):
        text = thread.get("reply_text")
        if text:
            return str(text)
    return ""


def _classify(rt: GraphRuntime, state: dict, reply: str) -> tuple[Intent, Decision, str]:
    """Ask the decision layer to classify the reply.

    Returns ``(intent, decision, note)``. The :class:`Decision` is returned
    rather than rebuilt so the ``Handoff`` records the *same* decision that
    produced the routing — its source, confidence and degradation all belong to
    the classification, and a router that fabricated a second, tidier decision
    would be laundering the evidence. ``note`` is non-empty when the answer came
    from a fallback and must therefore be labelled as such.
    """
    if not reply.strip():
        decision = _local_decision(
            question="Classify this sponsor reply.",
            choice=Intent.UNKNOWN.value,
            options=[i.value for i in Intent],
            decision_point="route.after_reply.intent",
            note="no reply text in state",
            degraded=True,
        )
        rt.tracer.decision(AgentId.A3_OUTREACH, decision,
                           decision_point="route.after_reply.intent")
        return Intent.UNKNOWN, decision, "no reply text in state"

    decision = _decide(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"A sponsor replied to a sponsorship proposal: \"{reply[:400]}\". "
                  f"What is their intent?"),
        question_type=QuestionType.CHOICE,
        options=[Intent.YES.value, Intent.PUSHBACK.value, Intent.INTERESTED.value,
                 Intent.NO.value, Intent.NEUTRAL.value],
        rubric=[
            "yes: an unambiguous acceptance; they have agreed to proceed",
            "pushback: they object to price, terms, or scope and want it changed",
            "interested: positive but not yet a commitment",
            "no: an unambiguous refusal or decline",
            "neutral: polite but content-free; do not guess from enthusiasm",
        ],
        instructions=("Refusal and pushback must be detected before assent. Never "
                      "read enthusiasm as acceptance."),
        state={"reply": reply[:400],
               "brand": _reply_brand(state),
               "offer_amount_inr": _reply_amount(state)},
        asked_by=AgentId.A3_OUTREACH,
        decision_point="route.after_reply.intent",
    ))
    rt.tracer.decision(AgentId.A3_OUTREACH, decision,
                       decision_point="route.after_reply.intent",
                       reply=reply[:200])

    choice = str(decision.choice or "").strip().lower()
    matched = next((i for i in Intent if choice.startswith(i.value)), None)
    if matched is None:
        return (Intent.UNKNOWN, decision,
                f"decision layer returned unrecognised choice {choice!r}")
    note = ""
    if decision.degraded:
        note = f"classified by {decision.source.value} fallback"
    return matched, decision, note


def _local_decision(*, question: str, choice: str, options: Sequence[str],
                    decision_point: str, note: str = "", degraded: bool = False,
                    confidence: float = 0.7) -> Decision:
    """A sourced, explicitly-labelled decision for a locally-computed routing fact.

    Used where the routing predicate is arithmetic over the state (a gate outcome,
    an award winner) rather than a question for the model. It is still a
    ``Decision`` — with a ``source`` and a ``raw`` note — so the ``Handoff``
    contract holds: no routing decision in this package is anonymous.
    """
    share = 1.0 / len(options) if options else 1.0
    probabilities = {opt: round(share, 4) for opt in options}
    total = sum(probabilities.values()) or 1.0
    probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}
    picked = probabilities.get(choice, round(confidence, 4))
    return Decision(
        request_id=new_id("dec"),
        question=question,
        choice=choice,
        probabilities=probabilities,
        confidence=min(picked, confidence),
        source=DecisionSource.RULES,
        model="graph.edges.local",
        degraded=degraded,
        raw={"note": note, "decision_point": decision_point} if note
            else {"decision_point": decision_point},
    )


def _reply_brand(state: dict) -> str:
    for thread in reversed(list(state.get("threads") or [])):
        if thread.get("reply_text"):
            return str(thread.get("brand") or "")
    return ""


def _reply_amount(state: dict) -> float:
    brand = _reply_brand(state)
    offer = latest_offer_for(state, brand) if brand else None
    if offer:
        try:
            return float(offer.get("amount_inr") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _LABEL_FOR_INTENT(intent: Intent) -> str:
    return {
        Intent.YES: LABEL.CONTRACT,
        Intent.PUSHBACK: LABEL.REPRICE,
        Intent.INTERESTED: LABEL.FOLLOWUP,
        Intent.NO: LABEL.ARCHIVE,
        Intent.NEUTRAL: LABEL.FOLLOWUP,
        Intent.UNKNOWN: LABEL.ARCHIVE,
    }[intent]


# ========================================================== route_after_compliance
def route_after_compliance(state: dict, runtime: Any) -> str:
    """A5's verdict: clear -> audit, blocking -> arbiter.

    A5 holds veto authority, and this edge is where that authority is exercised.
    A non-blocking flag is recorded and the run continues to audit; a blocking
    flag routes to adjudication instead, because signing an MoU whose promises
    cannot be evidenced is the failure the whole compliance stage exists to
    prevent.
    """
    rt = runtime_context(runtime)
    flags = list(state.get("risk_flags") or [])
    blocking = blocking_flags(state)
    raw_score = state.get("compliance_score")
    # ``None`` means A5 never reported one. Coercing that to 0.0 would put
    # "compliance score 0.0" into a handoff, which a reviewer would reasonably
    # read as "compliance ran and found nothing verifiable" — the exact opposite
    # of "compliance did not run".
    score_text = ("not measured" if raw_score is None
                  else f"{float(raw_score):.1f}")

    decision = _decide(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"Compliance score {score_text} with {len(flags)} flag(s), "
                  f"{len(blocking)} of them BLOCKING. May the run proceed to audit "
                  f"and reporting?"),
        question_type=QuestionType.CHOICE,
        options=["audit", "arbitrate"],
        rubric=["audit: no promise is unverifiable; findings can be reported honestly",
                "arbitrate: at least one promise cannot be evidenced and must be contested"],
        state={
            "compliance_score": raw_score,
            "flags": [{"code": f.get("code"), "severity": f.get("severity"),
                       "brand": f.get("brand"), "message": f.get("message")}
                      for f in flags[:8]],
            "blocking_codes": [str(f.get("code")) for f in blocking],
            "mous": [{"mou_id": m.get("mou_id"), "status": m.get("status"),
                      "brand": m.get("brand")} for m in (state.get("mous") or [])[:5]],
        },
        instructions="A blocking flag always means arbitrate. Do not average a veto away.",
        asked_by=AgentId.A5_COMPLIANCE,
        decision_point="route.after_compliance",
    ))

    if blocking or _matches(decision.choice, "arbitrate"):
        # Track 4: a blocking veto must leave a filed Dispute, not just a label.
        # The router cannot write state, so the dispute is posted to the board
        # and queued in extras for the next node (adjudicate via build._prep).
        # Best-effort: filing must never break routing.
        dispute: dict[str, Any] | None = None
        try:
            dispute = _ensure_blocking_dispute(rt, state)
        except Exception as exc:  # noqa: BLE001 - routing survives a filing fault
            log.warning("blocking dispute filing failed: %s: %s",
                        type(exc).__name__, exc)
            dispute = None
        message_id: str | None = None
        try:
            msg = record_message(
                rt, state, kind="critique",
                from_agent=AgentId.A5_COMPLIANCE, to_agent=AgentId.A7_ARBITER,
                body=(f"A5 veto: {len(blocking)} blocking flag(s) "
                      f"({', '.join(str(f.get('code')) for f in blocking[:3])}); "
                      f"dispute {dispute.get('dispute_id') if dispute else 'pending'} "
                      f"requires adjudication"),
                topic="compliance veto",
                refs=([str(dispute.get("dispute_id")) for dispute in [dispute]
                       if dispute and dispute.get("dispute_id")]
                      + [str(f.get("flag_id")) for f in blocking[:3]]),
                confidence=float(decision.confidence),
                source=decision.source,
            )
            if msg:
                message_id = str(msg.get("message_id") or "")
        except Exception as exc:  # noqa: BLE001 - emission never breaks routing
            log.debug("compliance critique message failed: %s: %s",
                      type(exc).__name__, exc)
        refs = [str(f.get("flag_id")) for f in blocking[:5]]
        if dispute and dispute.get("dispute_id"):
            refs.append(str(dispute["dispute_id"]))
        record_handoff(rt, state, to_agent=AgentId.A7_ARBITER,
                       reason=(f"{len(blocking)} blocking compliance flag(s): "
                               + ", ".join(str(f.get("code")) for f in blocking[:4])),
                       decision=decision, from_agent=AgentId.A5_COMPLIANCE,
                       summary="A5 exercised its veto; the dispute goes to adjudication",
                       payload_refs=refs, message_id=message_id or None)
        return LABEL.BLOCKING

    record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                   reason=(f"compliance score {score_text}, no blocking flags "
                           f"({len(flags)} non-blocking)"),
                   decision=decision, from_agent=AgentId.A5_COMPLIANCE,
                   summary="deliverable commitments may proceed to audit",
                   payload_refs=[str(f.get("flag_id")) for f in flags[:5]])
    return LABEL.CLEAR


# =============================================================== route_after_auction
def route_after_auction(state: dict, runtime: Any) -> str:
    """The contract-net award decides who is accountable for the outreach task.

    Single-executor design (documented, not theatre): only A3 holds a send
    transport, so the OUTREACH node always runs as A3 regardless of the award.
    The award is still load-bearing as evidence — it names the accountable
    contractor (or records manager self-execution per Smith when uncontracted),
    carries the winning utility on the handoff, and is projected into
    ``run_events`` by the auction node. Both branches target OUTREACH in
    ``PATH_MAPS`` because there is exactly one executor; the difference lives
    in the handoff reason/summary and the award payload, which is where a
    reviewer tells "A3 won" apart from "nobody bid".
    """
    rt = runtime_context(runtime)
    award_entry = _latest_entry(rt, "tasks", "award")
    winner = str((award_entry or {}).get("winner") or "")

    if winner:
        try:
            target = AgentId(winner)
        except ValueError:
            target = AgentId.A3_OUTREACH
        decision = _local_decision(
            question=f"Contract-net awarded the outreach task to {winner}; execute it?",
            choice="execute", options=["execute", "self_execute"],
            decision_point="route.after_auction",
            note=f"winner={winner} utility={award_entry.get('utility')}")
        record_handoff(rt, state, to_agent=target,
                       reason=f"contract-net award went to {winner}",
                       decision=decision, from_agent=AgentId.A2_PRICING,
                       summary=f"utility={award_entry.get('utility')}")
        return LABEL.AWARDED

    decision = _local_decision(
        question="Nobody bid on the outreach task; execute it as the manager?",
        choice="self_execute", options=["self_execute", "defer"],
        decision_point="route.after_auction",
        note="no eligible bidders submitted a bid", degraded=True)
    record_handoff(rt, state, to_agent=AgentId.A3_OUTREACH,
                   reason="no bids were submitted; the manager self-executes",
                   decision=decision, from_agent=AgentId.A2_PRICING,
                   summary="outreach proceeds without a contracted contractor")
    return LABEL.UNCONTRACTED


# ======================================================== route_after_adjudication
def route_after_adjudication(state: dict, runtime: Any) -> str:
    """After A7: resolved -> audit, unresolved -> the human escalation gate."""
    rt = runtime_context(runtime)
    disputes = list(state.get("disputes") or [])
    escalated = [d for d in disputes
                 if str(d.get("status")).rsplit(".", 1)[-1] == "escalated"]
    blocking = blocking_flags(state)

    decision = _decide(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"Adjudication finished with {len(escalated)} escalated dispute(s) "
                  f"and {len(blocking)} blocking flag(s). Can the run continue to "
                  f"audit, or does a human need to decide?"),
        question_type=QuestionType.CHOICE,
        options=["audit", "escalate"],
        rubric=["audit: every disputed area has a defensible resolution",
                "escalate: at least one area has no evidence-backed resolution"],
        state={
            "disputes": [{"dispute_id": d.get("dispute_id"), "zone": d.get("zone"),
                          "status": d.get("status"), "resolution": d.get("resolution")}
                         for d in disputes[-6:]],
            "escalated_ids": [str(d.get("dispute_id")) for d in escalated],
            "blocking_flags": [str(f.get("code")) for f in blocking],
        },
        instructions="Escalate rather than proceed on an unresolved blocking area.",
        asked_by=AgentId.A7_ARBITER,
        decision_point="route.after_adjudication",
    ))

    if escalated or _matches(decision.choice, "escalate"):
        record_handoff(rt, state, to_agent=AgentId.A7_ARBITER,
                       reason=(f"{len(escalated)} dispute(s) escalated with no "
                               f"evidence-backed resolution"),
                       decision=decision, from_agent=AgentId.A7_ARBITER,
                       summary="the escalation gate will ask a person",
                       payload_refs=[str(d.get("dispute_id")) for d in escalated[:5]])
        return LABEL.NEEDS_ESCALATION

    record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                   reason=f"all {len(disputes)} dispute(s) resolved by adjudication",
                   decision=decision, from_agent=AgentId.A7_ARBITER,
                   summary="the run may proceed to audit and reporting")
    return LABEL.RESOLVED


# ================================================================ route_gate_outcome
def route_gate_outcome(state: dict, runtime: Any) -> str:
    """Turn an answered gate into a next step.

    Three outcomes, three destinations, and ``REVISE`` deliberately routes back
    into ``replan`` rather than into the step that raised the gate: a reviewer
    saying "revise" is saying the *plan* was wrong, and re-running the same step
    with the same inputs would reproduce the thing they rejected.
    """
    rt = runtime_context(runtime)
    pending = state.get("pending_gate") or {}
    outcome = str(pending.get("outcome") or "reject").rsplit(".", 1)[-1]
    kind = str(pending.get("kind") or "escalation")
    decided_by = str(pending.get("decided_by") or "unknown")

    choice = {"approve": "approved", "revise": "revise"}.get(outcome, "rejected")
    decision = _local_decision(
        question=f"The {kind} gate was answered '{outcome}' by {decided_by}. Next step?",
        choice=choice, options=["approved", "rejected", "revise"],
        decision_point="route.gate_outcome",
        note=f"gate_outcome={outcome} kind={kind} decided_by={decided_by}")

    if outcome == "approve":
        record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                       reason=f"{kind} gate approved by {decided_by}",
                       decision=decision, from_agent=AgentId.A7_ARBITER,
                       summary="the approved side effect may now be performed")
        return LABEL.APPROVED
    if outcome == "revise":
        record_handoff(rt, state, to_agent=AgentId.A1_DISCOVERY,
                       reason=f"{kind} gate returned REVISE by {decided_by}; "
                              f"the plan itself must change",
                       decision=decision, from_agent=AgentId.A7_ARBITER,
                       summary="replan from the task ledger")
        return LABEL.REVISE

    record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                   reason=f"{kind} gate rejected by {decided_by}; "
                          f"record the refusal and report honestly",
                   decision=decision, from_agent=AgentId.A7_ARBITER,
                   summary="the side effect did not happen")
    return LABEL.REJECTED


# =================================================================== route_progress
def route_progress(state: dict, runtime: Any) -> str:
    """Magentic-One's progress router: complete, progressing, or stalled.

    ``progress_ledger`` in the state is the arbiter's or the node's claim; this
    router's own job is to (a) ask whether the claim holds, and (b) count the
    stalls honestly.

    The counter is incremented **here** rather than inside a node because a
    router is the only place that observes "we came back to the same place
    again". ``stall_count`` is carried in the ledger's ``changed`` field by the
    node and compared against ``settings.max_replans`` here, so exceeding the
    budget ends the run with a stated reason rather than looping until LangGraph's
    ``recursion_limit`` trips and reports a stack overflow as the error.
    """
    rt = runtime_context(runtime)
    settings = runtime_context(rt).settings
    max_replans = int(settings.max_replans)
    stall_count = int(state.get("stall_count") or 0)
    replan_count = int(state.get("replan_count") or 0)
    ledger = read_progress_ledger(state)

    decision = _decide(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=("Progress check: is the sponsorship objective met, is the run "
                  "still making progress, or is it repeating itself?"),
        question_type=QuestionType.CHOICE,
        options=["complete", "progress", "stalled"],
        rubric=[
            "complete: the objective is met and no further work adds value",
            "progress: the last step changed the world state; keep going",
            "stalled: the last step produced no new evidence and would repeat itself",
        ],
        state={
            "claimed_complete": ledger_is_complete(state),
            "claimed_progress": ledger_is_progressing(state),
            "progress_reason": ledger.get("progress_reason", ""),
            "changed": ledger.get("changed", ""),
            "stall_count": stall_count,
            "replan_count": replan_count,
            "brands": len(state.get("brands") or []),
            "offers": len(state.get("offers") or []),
            "threads": len(state.get("threads") or []),
            "signed_mous": len([m for m in (state.get("mous") or [])
                                if str(m.get("status")) == "signed"]),
            "roi": (state.get("roi_report") or {}).get("roi_multiple"),
            "compliance_score": state.get("compliance_score"),
            "blocking_flags": len(blocking_flags(state)),
            "status": state.get("status"),
        },
        instructions=("Stalled is not 'hard'. It is 'the last step produced no new "
                      "evidence'. If anything changed, say progress."),
        asked_by=AgentId.A7_ARBITER,
        decision_point="route.progress",
    ))

    choice = str(decision.choice or "").strip().lower()
    complete = choice.startswith("complete") or ledger_is_complete(state)
    # The decision is authoritative on stalling. The ledger's claim is passed into
    # the request, so a conflict between "the plan says nothing changed" and "we are
    # still progressing" is information the decision layer can weigh. Overriding the
    # decision with a local predicate here would be exactly the fixed ``if`` chain
    # pretending to be agentic reasoning that this module exists to avoid.
    stalled = choice.startswith("stalled")

    if not complete and choice.startswith("progress") and not ledger_is_progressing(state):
        conflict = (f"the progress ledger claims no movement ({stall_reason(state)}) "
                    f"while the decision reports progress; recorded as a conflict "
                    f"rather than silently overridden")
        rt.note(conflict, "route.progress")
        decision.raw = {**(decision.raw or {}), "ledger_conflict": conflict}

    # Track 4: A6 -> A2 critique -> revise. A present-but-baseless ROI report
    # (empty assumptions, or incoherent figures) is a forcing disagreement, not
    # progress: A6 emits a revision_request Message and the run revisits REVISE
    # with a budgeted round. Budget reuses max_replans/max_debate_rounds via the
    # shared negotiation_rounds counter, so the loop cannot run forever.
    try:
        roi_needed, roi_reason = _roi_revision_needed(state)
    except Exception:
        roi_needed, roi_reason = False, ""
    if roi_needed:
        budget = _revision_budget(rt)
        rounds = _negotiation_rounds(state)
        if rounds < budget:
            message_id = None
            try:
                msg = record_message(
                    rt, state, kind="revision_request",
                    from_agent=AgentId.A6_AUDIT, to_agent=AgentId.A2_PRICING,
                    body=(f"A6 critiques the ROI report: {roi_reason} "
                          f"(revision {rounds + 1}/{budget})"),
                    topic="roi assumptions",
                    refs=[str((state.get('roi_report') or {}).get('report_id') or "")],
                    confidence=float(decision.confidence),
                    source=decision.source,
                )
                if msg:
                    message_id = str(msg.get("message_id") or "")
            except Exception as exc:  # noqa: BLE001 - emission never breaks routing
                log.debug("roi revision_request message failed: %s: %s",
                          type(exc).__name__, exc)
            record_handoff(rt, state, to_agent=AgentId.A2_PRICING,
                           reason=f"{roi_reason} (revision {rounds + 1}/{budget})",
                           decision=decision, from_agent=AgentId.A6_AUDIT,
                           summary="A6 critique sends the offer back for revision",
                           payload_refs=([message_id] if message_id else []),
                           message_id=message_id or None)
            return LABEL.ROI_REVISE
        override = (f"ROI critique would revise again but {rounds} revision(s) "
                    f"already used of {budget} budgeted; proceeding without looping")
        rt.note(override, "route.progress")
        try:
            decision.raw = {**(decision.raw or {}), "budget_override": override}
        except Exception:
            pass

    if complete:
        record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                       reason=f"objective met: {ledger.get('is_complete_reason') or 'decision says complete'}",
                       decision=decision, from_agent=AgentId.A7_ARBITER,
                       summary="run will finalise and close")
        return LABEL.COMPLETE

    if stalled:
        if stall_count >= max_replans:
            record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                           reason=(f"stalled {stall_count} time(s) and the budget of "
                                   f"{max_replans} replan(s) is spent: {stall_reason(state)}"),
                           decision=decision, from_agent=AgentId.A7_ARBITER,
                           summary="giving up honestly rather than looping")
            log.warning("giving up after %d stall(s) (max_replans=%d): %s",
                        stall_count, max_replans, stall_reason(state))
            return LABEL.GIVE_UP
        record_handoff(rt, state, to_agent=AgentId.A1_DISCOVERY,
                       reason=(f"stalled ({stall_reason(state)}); replan "
                               f"{stall_count + 1}/{max_replans} rewrites the task ledger"),
                       decision=decision, from_agent=AgentId.A7_ARBITER,
                       summary="the task ledger will be rewritten and discovery re-run")
        return LABEL.STALLED

    record_handoff(rt, state, to_agent=AgentId.A6_AUDIT,
                   reason=f"progressing: {ledger.get('progress_reason') or 'new evidence'}",
                   decision=decision, from_agent=AgentId.A7_ARBITER,
                   summary="finalising with what has been established")
    return LABEL.PROGRESS


# ------------------------------------------------------------------------- utilities
def _latest_entry(rt: GraphRuntime, zone: str, kind: str | None = None) -> dict | None:
    """The most recent board entry in a zone as a plain dict, or ``None``."""
    try:
        entry = rt.board.latest(zone, kind)
    except TypeError:
        # A board whose ``latest`` does not accept ``kind``.
        try:
            entry = rt.board.latest(zone)
        except Exception as exc:  # noqa: BLE001 - reported, treated as "none"
            log.warning("board.latest(%r) failed: %s", zone, exc)
            return None
    except Exception as exc:  # noqa: BLE001 - reported, treated as "none"
        log.warning("board.latest(%r) failed: %s", zone, exc)
        return None
    if entry is None:
        return None
    payload = getattr(entry, "payload", None)
    if isinstance(payload, dict):
        return payload
    return None


#: node -> the node it is entered from by a *fixed* edge, or ``None`` when it is
#: entered from ``START``. Mirrors the ``add_edge`` calls in ``graph.build``; a
#: duplicate or missing entry is a wiring bug the replay tests surface.
_STATIC_PREDECESSORS: dict[str, str | None] = {
    NODE.REPLAN: NODE.DISCOVER,      # replan -> discover
    NODE.OUTREACH: NODE.AUCTION,     # auction -> outreach
    NODE.REPLY: NODE.OUTREACH,       # outreach -> reply
    NODE.ADJUDICATE: NODE.DEBATE,    # debate -> adjudicate
    NODE.COMPLIANCE: NODE.CONTRACT,  # contract -> compliance
    NODE.CONTRACT: NODE.REPLY,       # reply -> contract (labelled go_contract)
    NODE.AUDIT: NODE.ARCHIVE,        # archive -> audit (one of several entries)
    NODE.FINALIZE: NODE.AUDIT,       # audit -> finalize (one of several)
    NODE.PRICE: NODE.DISCOVER,       # discover -> price (labelled proceed)
    NODE.AUCTION: NODE.PRICE,        # price -> auction (labelled proceed)
    NODE.DEBATE: NODE.PRICE,         # price -> debate (labelled escalate_dispute)
    NODE.ESCALATE: NODE.ADJUDICATE,  # adjudicate -> escalate
}

#: The ``path_map`` for every conditional edge in the graph, declared in one place
#: so ``build`` cannot wire a label to the wrong node and ``draw_mermaid`` shows
#: what actually runs.
PATH_MAPS: dict[str, dict[str, str]] = {
    NODE.DISCOVER: {
        LABEL.REPLAN: NODE.REPLAN,
        LABEL.PROCEED: NODE.PRICE,
    },
    NODE.PRICE: {
        LABEL.PROCEED: NODE.AUCTION,
        LABEL.DISPUTE: NODE.DEBATE,
    },
    NODE.AUCTION: {
        # Single-executor design: only A3 holds a send transport, so both the
        # awarded and the uncontracted paths execute via OUTREACH (A3). The award
        # remains load-bearing as the accountability record — winner + utility on
        # the handoff and in run_events — not as an executor switch. A separate
        # executor per winner would imply every agent can send mail, which is
        # false and would be the real theatre.
        LABEL.AWARDED: NODE.OUTREACH,
        LABEL.UNCONTRACTED: NODE.OUTREACH,
    },
    NODE.REPLY: {
        LABEL.CONTRACT: NODE.CONTRACT,
        LABEL.REPRICE: NODE.REVISE,
        LABEL.FOLLOWUP: NODE.OUTREACH,
        LABEL.ARCHIVE: NODE.ARCHIVE,
        LABEL.NEGOTIATION_EXHAUSTED: NODE.DEBATE,
    },
    NODE.COMPLIANCE: {
        LABEL.CLEAR: NODE.AUDIT,
        LABEL.BLOCKING: NODE.ADJUDICATE,
    },
    NODE.ADJUDICATE: {
        LABEL.RESOLVED: NODE.AUDIT,
        LABEL.NEEDS_ESCALATION: NODE.ESCALATE,
    },
    NODE.ESCALATE: {
        LABEL.APPROVED: NODE.AUDIT,
        LABEL.REJECTED: NODE.AUDIT,
        LABEL.REVISE: NODE.REPLAN,
    },
    NODE.AUDIT: {
        LABEL.COMPLETE: NODE.FINALIZE,
        LABEL.PROGRESS: NODE.FINALIZE,
        LABEL.STALLED: NODE.REPLAN,
        LABEL.GIVE_UP: NODE.FINALIZE,
        # Track 4: A6 critique of a baseless ROI report revisits REVISE (A2)
        # with a budgeted round. REVISE has a fixed edge to OUTREACH, so this
        # is a second entry into an existing node, not a new loop.
        LABEL.ROI_REVISE: NODE.REVISE,
    },
}

#: Nodes whose outgoing edges are chosen by a router rather than fixed. Used by
#: :mod:`graph.replay`: ``update_state(as_node=...)`` cannot evaluate a conditional
#: edge, because it has no runtime context to hand the router, so a fork has to be
#: seeded at the last *static*-edge node before the pending one.
CONDITIONAL_SOURCES: frozenset[str] = frozenset(PATH_MAPS)


def static_predecessor(node: str) -> str | None:
    """The nearest node before ``node`` that reaches it by a fixed edge.

    Walks back through ``_STATIC_PREDECESSORS`` while the current node is a
    conditional-edge source. Returns ``None`` when the walk runs off the top
    (``discover`` is entered from ``START``, for instance), which the caller must
    treat as "cannot rewind further" rather than as an error to paper over.
    """
    seen = {node}
    cursor = node
    while cursor in CONDITIONAL_SOURCES:
        cursor = _STATIC_PREDECESSORS.get(cursor)
        if cursor is None or cursor in seen:
            return None
        seen.add(cursor)
    return cursor
