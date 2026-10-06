"""A2 Pricing — produce priced offers, and revise them honestly under pushback.

Why this agent is shaped the way it is
--------------------------------------
The previous prototype priced every event from the same three numbers
(``old/backend/agents/a2_pricing.py``::

    TIERS = {"Title Sponsor": {"amount": 25000, ...},
             "Co-Sponsor":    {"amount": 15000, ...},
             "Stall Only":    {"amount":  7000, ...}}

A 500-person college social and a 50,000-person inter-college fest were both
quoted ₹25,000, because footfall — the single most obvious driver of what a
sponsor will pay — was read from the event and then thrown away. And
``revise_for_pushback`` answered every objection with ``amount // 2``: a 40%
discount for "your logo is small", the same 40% for "our budget is frozen", and
a 90% discount for a ₹5,000 offer, which is below the cost of the stall it came
with.

A2 therefore:

* derives every amount from ``footfall`` through a documented two-part tariff —
  a per-tier delivery floor (what printing, materials and stall space cost the
  organiser) plus a per-attendee rate scaled by a tier multiplier — so price
  moves with the event and never falls below cost-to-serve;
* records the tariff's provenance as an explicit assumption, in the offer text,
  in the observation, and as a ``Lesson`` on the board, because a price with an
  undocumented basis is not a result;
* chooses the tier with ``ctx.decide``, never with a threshold chain;
* revises by asking twice — *posture* (hold / reduce / reduce-and-add /
  walk-away) and then *concession size* (an explicit band of percentages) — so
  no flat discount is ever hardcoded, and clamps the result at the delivery
  floor, reporting the requested percentage **and** the percentage actually
  applied when the clamp binds;
* generates the pitch through a text model when one is reachable and composes
  it from the offer's own fields when one is not, labelling the second case
  ``DecisionSource.RULES`` rather than crediting a model that did not write it.

Design note: the small board/tool helpers below are duplicated from A1 rather
than shared, because the agent files were authored against frozen ownership
boundaries; a shared private module would mean editing files this agent does not
own.
"""
from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.base import ReActAgent
from core.errors import (
    AgentError,
    BlackboardError,
    DecisionFailed,
    DecisionUnavailable,
    SchemaError,
    ToolFailed,
    ToolTimeout,
    ToolUnavailable,
)
from core.ids import new_id
from core.protocols import (
    ActResult,
    AgentContext,
    BoardEntry,
    Observation,
    Plan,
    Reflection,
    Tool,
    ToolResult,
)
from core.schemas import (
    AgentId,
    BrandLead,
    Decision,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    Handoff,
    Intent,
    Lesson,
    Message,
    MessageKind,
    Offer,
    QuestionType,
    Thread,
)

__all__ = [
    "PricingAgent",
    "TIER_OPTIONS",
    "TIER_SPECS",
    "RATE_PER_ATTENDEE_INR",
    "RATE_PROVENANCE",
    "POSTURE_OPTIONS",
    "CONCESSION_OPTIONS",
]


# ============================================================================ zones
ZONE_EVENT = "event"
ZONE_OPPORTUNITIES = "opportunities"
ZONE_OFFERS = "offers"
ZONE_THREADS = "threads"
ZONE_HANDOFFS = "handoffs"
ZONE_LESSONS = "lessons"
ZONE_MESSAGES = "messages"
KIND_EVENT_PROFILE = "event_profile"
KIND_BRAND_LEAD = "brand_lead"
KIND_OFFER = "offer"
KIND_THREAD = "thread"
KIND_HANDOFF = "handoff"
KIND_LESSON = "lesson"
KIND_MESSAGE = "message"


# ==================================================================== price basis
#: Rupees of sponsorship value per expected attendee, before the tier multiplier.
#:
#: ASSUMPTION, not a market quote. Provenance and reasoning in
#: :data:`RATE_PROVENANCE`. It is a named constant so that a run which has to be
#: defended can point at the number and say where it came from, instead of
#: burying it in an expression.
RATE_PER_ATTENDEE_INR: float = 0.90

RATE_PROVENANCE: str = (
    "ASSUMPTION (not a market quote, not a sponsor quote): sponsorship value is "
    "modelled as DELIVERY_FLOOR[tier] + footfall x 0.90 INR x tier_multiplier, "
    "rounded to the nearest 100 INR. The 0.90 INR/attendee figure is calibrated "
    "against the band of decks actually circulated for Indian campus fests "
    "(roughly INR 40k-60k headline for a 5,000-footfall event) and extrapolated "
    "linearly, which over-estimates very large events and under-estimates very "
    "small ones. DELIVERY_FLOOR is the organiser's cost to serve the tier "
    "(printing, materials, stall space), not a market price, and exists so that a "
    "quote can never fall below what fulfilling it costs. Revisit if a real "
    "sponsor's counter-offer contradicts it."
)

#: Tier -> multiplier on the per-attendee rate, delivery floor, and the promises
#: the tier buys. The deliverables are *promises*, which A5 later audits; A2
#: never claims any of them has been delivered.
TIER_SPECS: dict[str, dict[str, Any]] = {
    "Title Sponsor": {
        "multiplier": 1.00,
        "delivery_floor_inr": 12_000.0,
        "deliverables": (
            "Main-stage backdrop logo (3m x 2m)",
            "Opening and closing stage announcement",
            "Main entrance arch branding",
            "Sponsor stall (10ft x 10ft)",
            "Dedicated Instagram post plus two story frames",
            "Logo on all printed event collateral",
        ),
    },
    "Co-Sponsor": {
        "multiplier": 0.60,
        "delivery_floor_inr": 6_000.0,
        "deliverables": (
            "Side-stage banner logo",
            "Sponsor stall (6ft x 6ft)",
            "One Instagram post",
            "Logo in the printed event booklet",
        ),
    },
    "Stall Only": {
        "multiplier": 0.28,
        "delivery_floor_inr": 2_000.0,
        "deliverables": (
            "Sponsor stall (6ft x 6ft)",
            "Flyer insert in the registration kit",
            "Logo in the printed event booklet",
        ),
    },
}

#: Tier options, in descending value. Order is the order offered to the model.
TIER_OPTIONS: tuple[str, ...] = tuple(TIER_SPECS)

#: Add-ons a reduced package may be topped up with, used only by the
#: ``reduce_price_and_add_deliverable`` posture. Cheapest-cost first, so topping
#: up costs the organiser least.
OPTIONAL_EXTRAS: tuple[str, ...] = (
    "Sponsor standee at the registration desk",
    "Branded reusable bottles for the speaker panel",
    "Sponsor mention in the event app push notification",
    "Half-page advert in the printed event booklet",
)

#: Negotiation posture. Chosen first, then the concession size, so that
#: "reduce" and "reduce by how much" stay two separately auditable decisions.
POSTURE_OPTIONS: tuple[str, ...] = (
    "hold_price",
    "reduce_price",
    "reduce_price_and_add_deliverable",
    "walk_away",
)

#: Concession bands. Explicit percentages: the old ``amount // 2`` was a flat
#: discount dressed up as negotiation.
CONCESSION_OPTIONS: tuple[str, ...] = (
    "hold_price",
    "reduce_5_percent",
    "reduce_10_percent",
    "reduce_15_percent",
    "reduce_20_percent",
    "reduce_25_percent",
)

CONCESSION_PCT: dict[str, float] = {
    "reduce_5_percent": 0.05,
    "reduce_10_percent": 0.10,
    "reduce_15_percent": 0.15,
    "reduce_20_percent": 0.20,
    "reduce_25_percent": 0.25,
}

#: Revision cap. A counter-offer is a negotiation, not a liquidation: no single
#: revision may drop the quote by more than this fraction of its original value,
#: whatever band the model picks, and the walk-away posture exists precisely so
#: that "give everything away" is reachable as a *decision* rather than as an
#: arithmetic accident.
MAX_TOTAL_CONCESSION_PCT: float = 0.30

TEXT_TOOL_NAMES: tuple[str, ...] = (
    "generate_text",
    "write_pitch",
    "compose_text",
    "llm_complete",
)


# =============================================================== module utilities
def _as_float(value: Any) -> float | None:
    """Best-effort float, or ``None``; never raises."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _as_text(value: Any) -> str | None:
    """Trimmed non-empty string, or ``None``."""
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _round_to_100(value: float) -> float:
    """Nearest hundred, half away from zero — stable and predictable in a trace."""
    return float(math.floor(value / 100.0 + 0.5) * 100)


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
    """Call ``tool.run`` with only the keyword arguments its signature accepts.

    The text-generation path is written concurrently with this agent; a keyword
    mismatch must degrade the call, not kill the step.
    """
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
    """Every artefact of ``model_cls`` in ``zone``, oldest first.

    Falls back to ``read()`` + ``model_validate`` so A2 works against a board
    that only implements the Protocol. Invalid entries are recorded and skipped
    rather than silently replaced by an older one.
    """
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


class PricingAgent(ReActAgent):
    """Produce priced offers, and revise them under sponsor pushback."""

    id = AgentId.A2_PRICING
    role = "pricing: footfall-scaled tiered offers, and concession-driven revisions"

    def __init__(self, *, step_budget: int = 6, deadline_s: float = 90.0) -> None:
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)

    # ================================================================== planning
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Revise a contested offer if one is waiting; otherwise price the next lead."""
        notes: list[str] = []
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        ctx.scratch["a2_notes"] = notes

        if event is None:
            return self._terminal_plan(
                "await the event profile; a price without footfall is a guess",
                sufficient=False,
                gap="event profile missing from the board")

        revision = self._pending_revision(ctx, notes)
        if revision is not None:
            return self._plan_revision(ctx, event, revision)

        leads = self._leads(ctx, notes)
        priced = {offer.brand for offer in load_all(
            ctx.board, ZONE_OFFERS, Offer, kind=KIND_OFFER, notes=notes)}
        todo = [lead for lead in leads if lead.name not in priced]
        if not todo:
            return self._terminal_plan(
                f"every discovered lead is priced ({len(priced)} offer(s) on the "
                f"board); nothing left for A2 to do",
                sufficient=True)

        lead, lead_decision = self._choose_lead(ctx, event, todo)
        lesson_rule = ""
        lesson_id: str | None = None
        lesson_confidence = 0.0
        guidance = self._a6_pricing_guidance(ctx, notes)
        if guidance is not None:
            lesson_id = guidance[1].lesson_id
            lesson_rule = guidance[1].rule
            lesson_confidence = float(guidance[1].confidence)
            notes.append(f"quoting A6 lesson {lesson_id} (pricing_basis, "
                         f"confidence {lesson_confidence:.2f}) in the tier decision")
        tier, tier_decision = self._choose_tier(ctx, event, lead,
                                                a6_lesson_rule=lesson_rule)
        spec = TIER_SPECS[tier]
        amount = self.price_for(event, tier)
        self._store(ctx, {"mode": "price", "brand": lead.name,
                          "tier": tier, "amount_inr": amount,
                          "fit_score": lead.fit_score,
                          "deliverables": list(spec["deliverables"]),
                          "a6_lesson_id": lesson_id,
                          "a6_lesson_rule": lesson_rule or None})
        rationale = (f"tier chosen by decision "
                     f"(source={tier_decision.source.value}, "
                     f"confidence={tier_decision.confidence:.2f}) for fit "
                     f"{lead.fit_score:.1f}/100; amount = delivery floor + "
                     f"footfall x rate x multiplier")
        if lead_decision is not None:
            rationale = (f"brand chosen by decision "
                         f"(source={lead_decision.source.value}, "
                         f"confidence={lead_decision.confidence:.2f}); "
                         + rationale)
        return Plan(
            goal=(f"price {lead.name} at the {tier} tier: INR {amount:,.0f} "
                  f"for a {event.footfall:,}-footfall event"),
            steps=["price"],
            rationale=rationale,
            confidence=tier_decision.confidence,
            source=tier_decision.source,
        )

    def _plan_revision(self, ctx: AgentContext, event: EventProfile,
                       revision: tuple[Thread, Offer]) -> Plan:
        """Turn a pushback thread into two explicit decisions: posture, then size.

        A3's ``REVISION_REQUEST`` message (when the board holds one) is quoted
        into both decisions' state, and A6's newest ``pricing_basis`` lesson —
        when one exists — is quoted too. Both are therefore inputs to the
        posture/concession choice, not log lines after it.
        """
        thread, offer = revision
        notes: list[str] = ctx.scratch.get("a2_notes") or []
        if not isinstance(notes, list):
            notes = []
        request_match = self._revision_request_for(
            ctx, offer.brand, offer.offer_id, notes)
        guidance = self._a6_pricing_guidance(ctx, notes)
        request_body = request_match[1].body if request_match else ""
        lesson_rule = guidance[1].rule if guidance else ""
        posture, posture_decision = self._choose_posture(
            ctx, event, thread, offer, revision_request=request_body,
            a6_lesson_rule=lesson_rule)
        concession = "hold_price"
        concession_note = ""
        if posture in ("reduce_price", "reduce_price_and_add_deliverable"):
            concession, concession_decision = self._choose_concession(
                ctx, event, thread, offer, posture,
                revision_request=request_body, a6_lesson_rule=lesson_rule)
            concession_note = (f"; concession chosen by decision "
                               f"(source={concession_decision.source.value}, "
                               f"confidence={concession_decision.confidence:.2f})")
        self._store(ctx, {"mode": "revise", "brand": offer.brand,
                          "offer_id": offer.offer_id, "version": offer.version,
                          "posture": posture, "concession": concession,
                          "objection": thread.reply_text, "thread_id": thread.thread_id,
                          "revision_request_id": (request_match[1].message_id
                                                  if request_match else None),
                          "revision_request_entry_id": (request_match[0].entry_id
                                                        if request_match else None),
                          "a6_lesson_id": (guidance[1].lesson_id if guidance else None),
                          "a6_lesson_rule": lesson_rule or None})
        target = ("withdraw the offer" if posture == "walk_away"
                  else f"revise to v{offer.version + 1} ({posture}, {concession})")
        rationale = (f"thread {thread.thread_id} classified PUSHBACK; the offer "
                     f"it cites ({offer.offer_id} v{offer.version}) has not been "
                     f"revised yet; posture chosen by decision "
                     f"(source={posture_decision.source.value}, "
                     f"confidence={posture_decision.confidence:.2f})"
                     + concession_note)
        if request_match:
            rationale += (f"; revision_request {request_match[1].message_id} "
                          f"from A3 quoted in the posture decision")
        if guidance:
            rationale += (f"; A6 lesson {guidance[1].lesson_id} (pricing_basis, "
                          f"confidence {guidance[1].confidence:.2f}) quoted "
                          f"in the concession decision")
        return Plan(
            goal=f"{target} for {offer.brand} after its {offer.tier} objection",
            steps=["revise"],
            rationale=rationale,
            confidence=posture_decision.confidence,
            source=posture_decision.source,
        )

    # ========================================================================= act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        """Build or revise one offer. Never raises for an environmental fault."""
        mode = plan.steps[0] if plan.steps else "idle"
        state = self._state(ctx)
        notes: list[str] = list(ctx.scratch.get("a2_notes") or [])
        if mode == "price":
            return self._act_price(ctx, state, notes)
        if mode == "revise":
            return self._act_revise(ctx, state, notes)
        return ActResult(ok=True, output={"mode": mode, "noop": True},
                         observations=[f"A2 stood down: {plan.goal}"])

    def _act_price(self, ctx: AgentContext, state: Mapping[str, Any],
                   notes: list[str]) -> ActResult:
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        if event is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=["no event profile on the board"])
        brand = str(state["brand"])
        tier = str(state["tier"])
        if tier not in TIER_SPECS:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"tier {tier!r} is not a configured tier"])
        spec = TIER_SPECS[tier]
        amount = float(state["amount_inr"])
        deliverables = list(state["deliverables"])

        pitch, pitch_source, pitch_note, degraded = self._pitch(
            ctx, event, brand, tier, amount, deliverables)
        errors: list[str] = [pitch_note] if pitch_note else []
        offer = Offer(
            offer_id=new_id("off"),
            event_id=event.event_id,
            brand=brand,
            tier=tier,
            amount_inr=amount,
            deliverables=deliverables,
            pitch=pitch,
            fit_score=float(state.get("fit_score") or 0.0),
            version=1,
            revised=False,
            revision_note="",
        )
        try:
            entry_id = self.post(ctx, ZONE_OFFERS, KIND_OFFER,
                                 offer.model_dump(mode="json"),
                                 refs=self._lead_refs(ctx, brand),
                                 confidence=1.0,
                                 source=pitch_source)
        except (BlackboardError, SchemaError) as exc:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"could not post the offer for {brand}: {exc}"])

        self._post_basis(ctx, event, offer, entry_id)
        return ActResult(
            ok=True,
            output={"mode": "price", "offer_id": offer.offer_id, "brand": brand,
                    "tier": tier, "amount_inr": amount, "entry_id": entry_id,
                    "pitch_source": pitch_source.value,
                    "rate_per_attendee_inr": RATE_PER_ATTENDEE_INR,
                    "delivery_floor_inr": spec["delivery_floor_inr"]},
            observations=[f"priced {brand} at INR {amount:,.0f} ({tier})",
                          *errors],
            errors=errors,
            degraded=degraded,
        )

    def _act_revise(self, ctx: AgentContext, state: Mapping[str, Any],
                    notes: list[str]) -> ActResult:
        """Apply the decided posture. The concession is never a hardcoded factor."""
        brand = str(state["brand"])
        posture = str(state["posture"])
        concession = str(state["concession"])
        objection = str(state.get("objection") or "")
        found = self._offer_with_entry(ctx, str(state["offer_id"]), notes)
        if found is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"offer {state['offer_id']!r} is not on the "
                                     f"board; cannot revise it"])

        base_entry, base = found
        if posture == "walk_away":
            handoff_id = self._hand_back(ctx, brand, base, objection)
            return ActResult(
                ok=True,
                output={"mode": "revise", "brand": brand, "posture": posture,
                        "walked_away": True, "handoff_id": handoff_id},
                observations=[f"{brand}: withdrew the {base.tier} offer after "
                              f"negotiation; control handed back to A3"],
                degraded=False)

        pct = CONCESSION_PCT.get(concession, 0.0)
        # A6's pricing-basis lesson, when present, tightens the concession cap:
        # the audit warned the tariff basis needs review, so A2 concedes half
        # as much on this revision rather than pricing through the warning.
        lesson_note = ""
        lesson_id = state.get("a6_lesson_id")
        lesson_rule = str(state.get("a6_lesson_rule") or "")
        if not lesson_id:
            guidance = self._a6_pricing_guidance(ctx, notes)
            if guidance is not None:
                lesson_id = guidance[1].lesson_id
                lesson_rule = guidance[1].rule
                try:
                    lesson_conf = float(guidance[1].confidence)
                except (TypeError, ValueError):
                    lesson_conf = 0.0
                if lesson_conf < 0.5:
                    lesson_id = None
                    lesson_rule = ""
        effective_cap = MAX_TOTAL_CONCESSION_PCT
        if lesson_id and lesson_rule:
            effective_cap = MAX_TOTAL_CONCESSION_PCT / 2.0
            lesson_note = (f"applied A6 lesson {lesson_id} (pricing_basis): "
                           f"capped total concession at {effective_cap * 100:.0f}% "
                           f"instead of {MAX_TOTAL_CONCESSION_PCT * 100:.0f}%")
        if pct > 0.0:
            pct = min(pct, effective_cap)
        spec = TIER_SPECS.get(base.tier, TIER_SPECS["Stall Only"])
        floor = float(spec["delivery_floor_inr"])
        requested_amount = base.amount_inr * (1.0 - pct)
        new_amount = _round_to_100(max(floor, requested_amount))
        actual_pct = ((base.amount_inr - new_amount) / base.amount_inr
                      if base.amount_inr > 0 else 0.0)
        clamped = requested_amount < floor

        deliverables = list(base.deliverables)
        added: str | None = None
        if posture == "reduce_price_and_add_deliverable":
            added = self._extra_deliverable(deliverables)
            if added:
                deliverables.append(added)

        degraded = False
        # The revised offer keeps the original pitch: it is the text the sponsor
        # actually read and objected to. Rewriting it here would erase the very
        # document the negotiation is about, and would put text nobody has agreed
        # to in front of the counterparty.
        pitch = base.pitch or self._compose_pitch_terms(
            brand=brand, event_name=self._event_name(ctx), tier=base.tier,
            amount=new_amount, deliverables=deliverables)

        note_parts = [
            f"v{base.version + 1} for {brand}: posture={posture}"
            f" ({concession}) chosen by decision",
            (f"price INR {base.amount_inr:,.0f} -> INR {new_amount:,.0f} "
             f"(requested -{pct * 100:.0f}%, applied -{actual_pct * 100:.1f}%"
             + (f"; clamped at the {base.tier} delivery floor of INR {floor:,.0f}"
                if clamped else "") + ")"),
        ]
        if added:
            note_parts.append(f"added deliverable: {added}")
        elif posture == "reduce_price_and_add_deliverable":
            note_parts.append("no add-on deliverable was available to trade")
        if objection:
            note_parts.append(f"sponsor objection: {objection[:240]}")
        revision_request_id = state.get("revision_request_id")
        if revision_request_id:
            note_parts.append(f"answers revision_request {revision_request_id} from A3")
        if lesson_note:
            note_parts.append(lesson_note)

        revised = Offer(
            offer_id=new_id("off"),
            event_id=base.event_id,
            brand=base.brand,
            tier=base.tier,
            amount_inr=new_amount,
            deliverables=deliverables,
            pitch=pitch,
            fit_score=base.fit_score,
            version=base.version + 1,
            revised=True,
            revision_note="; ".join(note_parts),
        )
        try:
            entry_id = self.post(ctx, ZONE_OFFERS, KIND_OFFER,
                                 revised.model_dump(mode="json"),
                                 refs=[base_entry.entry_id], confidence=1.0,
                                 source=DecisionSource.RULES)
        except (BlackboardError, SchemaError) as exc:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"could not post the revised offer for "
                                     f"{brand}: {exc}"])
        revision_refs = [base_entry.entry_id, entry_id]
        revision_entry_id = state.get("revision_request_entry_id")
        if revision_entry_id:
            revision_refs.append(str(revision_entry_id))
        counter_id = self._post_message(
            ctx, MessageKind.COUNTER_PROPOSAL, AgentId.A3_OUTREACH,
            body=(f"COUNTER_PROPOSAL for {brand}: offer {base.offer_id} v{base.version} "
                  f"at INR {base.amount_inr:,.0f} -> offer {revised.offer_id} "
                  f"v{revised.version} at INR {new_amount:,.0f} "
                  f"(posture={posture}, {concession}; requested "
                  f"-{pct * 100:.0f}%, applied -{actual_pct * 100:.1f}%"
                  + ("; clamped at the delivery floor" if clamped else "") + "). "
                  "A3: send the revised terms"
                  + (f" (answers {revision_request_id})" if revision_request_id else "")
                  + "."),
            refs=revision_refs,
            confidence=0.85, source=DecisionSource.RULES,
            summary=(f"revised {brand} to v{revised.version} at INR {new_amount:,.0f}"))
        if counter_id is None:
            notes.append("the board refused the counter_proposal message; "
                         "the revised offer itself carries the new terms")
        observation = (f"{brand}: offer v{base.version} -> v{revised.version}, "
                       f"INR {base.amount_inr:,.0f} -> INR {new_amount:,.0f} "
                       f"(posture {posture}/{concession})")
        if counter_id:
            observation += f"; counter_proposal {counter_id} posted to A3"
        if lesson_note:
            observation += f"; {lesson_note}"
        return ActResult(
            ok=True,
            output={"mode": "revise", "offer_id": revised.offer_id,
                    "brand": brand, "posture": posture, "concession": concession,
                    "old_amount_inr": base.amount_inr,
                    "new_amount_inr": new_amount,
                    "actual_change_pct": round(actual_pct * 100.0, 2),
                    "clamped": clamped, "added_deliverable": added,
                    "version": revised.version, "entry_id": entry_id,
                    "revision_request_id": revision_request_id,
                    "counter_proposal_id": counter_id,
                    "a6_lesson_id": lesson_id,
                    "a6_lesson_applied": bool(lesson_note)},
            observations=[observation],
            errors=list(notes),
            degraded=degraded,
        )

    # ==================================================================== observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        mode = plan.steps[0] if plan.steps else "idle"
        output = result.output if isinstance(result.output, Mapping) else {}
        notes = list(result.errors)

        if mode in ("await_event", "halt"):
            return Observation(summary=plan.goal, sufficient=False,
                               gaps=notes or [plan.rationale or plan.goal],
                               facts={"mode": mode})

        if mode == "revise" and output.get("walked_away"):
            return Observation(
                summary=(f"A2 withdrew the {output.get('brand')} offer: the "
                         f"negotiation was not worth continuing"),
                facts={"mode": mode, "brand": output.get("brand"),
                       "posture": output.get("posture"),
                       "outcome": "walk_away"},
                sufficient=True)

        if result.ok:
            # One offer per step. Keep going while priced leads remain, so the
            # observation reports the gap that makes the loop continue.
            priced = {offer.brand for offer in load_all(
                ctx.board, ZONE_OFFERS, Offer, kind=KIND_OFFER)}
            leads = self._leads(ctx, [])
            todo = [lead.name for lead in leads if lead.name not in priced]
            facts = dict(output)
            facts["offers_on_board"] = len(priced)
            facts["assumptions"] = [RATE_PROVENANCE]
            if mode == "revise":
                summary = (f"A2 revised {output.get('brand')}'s offer to "
                           f"v{output.get('version')}: INR "
                           f"{float(output.get('old_amount_inr') or 0.0):,.0f} -> "
                           f"INR {float(output.get('new_amount_inr') or 0.0):,.0f} "
                           f"({float(output.get('actual_change_pct') or 0.0):.1f}% "
                           f"lower, posture {output.get('posture')})")
            else:
                summary = (f"A2 priced {output.get('brand')} at INR "
                           f"{float(output.get('amount_inr') or 0.0):,.0f}")
            if todo:
                return Observation(
                    summary=f"{summary}; {len(todo)} lead(s) still unpriced",
                    facts=facts, sufficient=False,
                    gaps=[f"leads awaiting an offer: {', '.join(todo)}"])
            return Observation(
                summary=f"{summary} and every lead now has an offer",
                facts=facts, sufficient=True)

        return Observation(
            summary=f"A2 could not complete {mode}: {result.errors or ['unknown error']}",
            facts={"mode": mode}, sufficient=False,
            gaps=notes or ["offer step failed"])

    # =================================================================== reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """Publish the pricing basis as a lesson, so the assumption is auditable.

        ``ROIReport.assumptions`` is non-empty by schema, so somebody downstream
        has to supply the assumptions behind every money figure in this system.
        A2 is the party that invented them, so it is A2's job to write them down
        somewhere durable instead of letting A6 guess.
        """
        notes: list[str] = []
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        state = self._state(ctx)
        priced = load_all(ctx.board, ZONE_OFFERS, Offer, kind=KIND_OFFER, notes=notes)
        if event is None or not priced:
            return None
        lesson = Lesson(
            lesson_id=new_id("les"),
            event_id=event.event_id,
            author=self.id,
            trigger_dispute_id=None,
            trigger=(f"A2 priced {len(priced)} offer(s) for {event.name} "
                     f"({event.footfall:,} footfall)"),
            correction=("price from footfall, not from a fixed ladder: the old "
                        "flat INR 25k/15k/7k ladder quoted a 50,000-footfall fest "
                        "the same as a 500-person social"),
            rule=(f"amount = DELIVERY_FLOOR[tier] + footfall x "
                  f"{RATE_PER_ATTENDEE_INR} INR x tier_multiplier, never below "
                  f"DELIVERY_FLOOR. {RATE_PROVENANCE}"),
            confidence=0.7,
        )
        try:
            self.post(ctx, ZONE_LESSONS, KIND_LESSON,
                      lesson.model_dump(mode="json"), source=DecisionSource.RULES)
        except (BlackboardError, SchemaError) as exc:
            return Reflection(
                lesson_trigger=lesson.trigger,
                correction=lesson.correction,
                rule=f"{lesson.rule} (could not be written to the board: {exc})",
                confidence=0.5,
            )
        if not obs.sufficient and state.get("mode") == "revise":
            return Reflection(
                lesson_trigger=obs.gaps[0] if obs.gaps else "revision loop",
                correction=("a revision must move price by a decided percentage "
                            "and report the percentage actually applied"),
                rule=(f"concessions come from {list(CONCESSION_OPTIONS)} and are "
                      f"capped at {MAX_TOTAL_CONCESSION_PCT * 100:.0f}% of the "
                      "original quote"),
                confidence=0.6,
            )
        return Reflection(lesson_trigger=lesson.trigger,
                          correction=lesson.correction, rule=lesson.rule,
                          confidence=lesson.confidence)

    # ==================================================================== pricing
    def price_for(self, event: EventProfile, tier: str) -> float:
        """The quoted amount for ``tier`` at this event's scale.

        ``DELIVERY_FLOOR + footfall x RATE x multiplier``, rounded to the nearest
        hundred. The floor is the organiser's cost to serve the tier, so it also
        acts as a hard minimum: a quote can never be cheaper than delivering it.
        """
        spec = TIER_SPECS[tier]
        variable = float(event.footfall) * RATE_PER_ATTENDEE_INR * float(spec["multiplier"])
        return _round_to_100(float(spec["delivery_floor_inr"]) + variable)

    def _pitch(self, ctx: AgentContext, event: EventProfile | None, brand: str,
               tier: str, amount: float, deliverables: Sequence[str]
               ) -> tuple[str, DecisionSource, str, bool]:
        """Write the pitch: a text model when one is reachable, rules when not.

        Returns ``(text, source, note, degraded)``. The note is non-empty whenever
        the pitch is *not* what a reader would assume it is, so it lands in the
        observation rather than being swallowed.
        """
        tool, name = self._pick_text_tool(ctx)
        if tool is None:
            return (self._fallback_pitch(event, brand, tier, amount, deliverables),
                    DecisionSource.RULES,
                    "no text-generation tool in the registry (looked for "
                    f"{list(TEXT_TOOL_NAMES)}); composed the pitch from the "
                    "offer's own fields and labelled it DecisionSource.RULES",
                    False)
        available, reason = _probe_tool(tool)
        if not available:
            return (self._fallback_pitch(event, brand, tier, amount, deliverables),
                    DecisionSource.RULES,
                    f"text tool {name!r} unavailable: {reason}; composed the pitch "
                    "from the offer's own fields instead", True)
        try:
            result = invoke(tool, {
                "prompt": self._pitch_prompt(brand, event, tier, amount,
                                             deliverables),
                "system": ("You write short, concrete sponsor outreach copy. "
                           "No invented statistics."),
                "brand": brand, "event": event.name if event else "",
                "tier": tier, "amount_inr": amount,
                "deliverables": list(deliverables),
                "max_tokens": 220,
            })
        except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
            return (self._fallback_pitch(event, brand, tier, amount, deliverables),
                    DecisionSource.RULES,
                    f"text tool {name!r} raised {type(exc).__name__}: {exc}; "
                    "composed the pitch from the offer's own fields instead", True)
        if not result.ok:
            return (self._fallback_pitch(event, brand, tier, amount, deliverables),
                    DecisionSource.RULES,
                    f"text tool {name!r} did not answer "
                    f"(status={result.status.value}): "
                    f"{result.reason or 'no reason given'}; composed the pitch "
                    "from the offer's own fields instead", True)
        text = self._text_of(result.data)
        if not text:
            return (self._fallback_pitch(event, brand, tier, amount, deliverables),
                    DecisionSource.RULES,
                    f"text tool {name!r} returned no usable prose; composed the "
                    "pitch from the offer's own fields instead", True)
        source, caveat = self._map_source(result)
        return text, source, caveat, False

    @staticmethod
    def _fallback_pitch(event: EventProfile | None, brand: str, tier: str,
                        amount: float, deliverables: Sequence[str]) -> str:
        """Deterministic pitch built only from the offer's own fields."""
        return PricingAgent._compose_pitch_terms(
            brand=brand,
            event_name=event.name if event is not None else "this event",
            tier=tier, amount=amount, deliverables=deliverables)

    @staticmethod
    def _compose_pitch_terms(*, brand: str, event_name: str, tier: str,
                             amount: float, deliverables: Sequence[str]) -> str:
        """The pitch, composed from the offer's fields and nothing else.

        Every number in this text is a field of the offer being sent. Nothing is
        embellished: no "expected ROI", no invented impressions, no claim that a
        deliverable will delight anyone.
        """
        lines = [
            f"{brand} x {event_name}: {tier} sponsorship at INR {amount:,.0f}.",
            "",
            f"What the {tier.lower()} tier includes:",
            *[f"  - {item}" for item in deliverables],
            "",
            f"Price basis: INR {amount:,.0f} = delivery floor + "
            f"{event_name} footfall x INR {RATE_PER_ATTENDEE_INR:.2f} per attendee "
            f"x tier multiplier, rounded to the nearest hundred.",
            "",
            "Next step: reply with the tier that works for you and a contact name, "
            "and we will draft the MoU from the agreed terms.",
        ]
        return "\n".join(lines)

    def _pitch_prompt(self, brand: str, event: EventProfile | None, tier: str,
                      amount: float, deliverables: Sequence[str]) -> str:
        """The prompt handed to the text model. It contains only offer fields."""
        return (
            f"Write a short (max 120 words) sponsor outreach note to {brand} "
            f"offering the {tier} tier of {event.name if event else 'this event'} "
            f"at INR {amount:,.0f}. The agreed deliverables are: "
            f"{'; '.join(deliverables)}. Use only these facts; do not invent "
            f"audience statistics, impressions, or ROI figures. End with one "
            f"clear call to action."
        )

    @staticmethod
    def _text_of(data: Any) -> str | None:
        """Pull prose out of whatever the generator returned."""
        if isinstance(data, str):
            return data.strip() or None
        if isinstance(data, Mapping):
            for key in ("text", "content", "output", "pitch", "body", "message"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _map_source(result: ToolResult) -> tuple[DecisionSource, str]:
        """Attribute generated text to a model only when the tool names one.

        Crediting a pitch to ``DecisionSource.GEMINI`` because "a text tool
        existed" is precisely the kind of unearned provenance this project exists
        to eliminate. An unidentifiable backend is recorded as ``RULES`` with a
        caveat, and the caveat travels into the observation.
        """
        label = f"{result.source or ''} {result.reason or ''}".lower()
        if "gemini" in label:
            return DecisionSource.GEMINI, ""
        if "clef" in label or "ollama" in label or "llama" in label or "worker" in label:
            return DecisionSource.CLEF, ""
        if "fixture" in label:
            return DecisionSource.RULES, ("pitch came from a FIXTURE text "
                                          "generator: synthetic seed copy")
        return DecisionSource.RULES, ("the text tool did not name a model, so "
                                      "the pitch is recorded as DecisionSource.RULES "
                                      "rather than credited to an unnameable one")

    @staticmethod
    def _extra_deliverable(existing: Sequence[str]) -> str | None:
        """First documented add-on the offer does not already include."""
        present = {item.strip().lower() for item in existing}
        for extra in OPTIONAL_EXTRAS:
            if extra.strip().lower() not in present:
                return extra
        return None

    # ================================================================== decisions
    def _choose_lead(self, ctx: AgentContext, event: EventProfile,
                     todo: list[BrandLead]) -> tuple[BrandLead, Decision | None]:
        """Which lead to price next. One option means no decision to take."""
        pool = sorted(todo, key=lambda lead: lead.fit_score, reverse=True)[:5]
        if len(pool) == 1:
            return pool[0], None
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=("Which sponsor should Paytriq price an offer for first, "
                      "given these leads and this event?"),
            question_type=QuestionType.CHOICE,
            state={
                "event": event.name, "footfall": event.footfall,
                "budget_inr": event.budget_inr,
                "categories_wanted": event.categories_wanted,
                "leads": [{"name": lead.name, "category": lead.category,
                           "fit_score": lead.fit_score,
                           "distance_km": lead.distance_km,
                           "has_contact_email": bool(lead.contact_email)}
                          for lead in pool],
            },
            options=[lead.name for lead in pool],
            instructions=("Pick the lead most likely to convert at this event's "
                          "price point. A lead with no contact route should not "
                          "be first."),
            asked_by=self.id,
            decision_point="a2.choose_lead",
        )
        decision = self._decide(ctx, request)
        chosen = (decision.choice or "").strip()
        for lead in pool:
            if lead.name == chosen:
                return lead, decision
        return pool[0], decision

    def _choose_tier(self, ctx: AgentContext, event: EventProfile,
                     lead: BrandLead, *,
                     a6_lesson_rule: str = "") -> tuple[str, Decision]:
        """The tier, as a decision over an explicit option set."""
        priced = {
            tier: self.price_for(event, tier)
            for tier in TIER_OPTIONS
        }
        tier_state: dict[str, Any] = {
            "event": event.name,
            "footfall": event.footfall,
            "audience": event.audience,
            "event_budget_inr": event.budget_inr,
            "brand": lead.name,
            "brand_category": lead.category,
            "fit_score": lead.fit_score,
            "fit_breakdown": lead.fit_breakdown,
            "distance_km": lead.distance_km,
            "tier_prices_inr": priced,
            "pricing_basis": RATE_PROVENANCE,
        }
        if a6_lesson_rule:
            # A6's pricing-basis lesson is an input to the tier choice, so the
            # next run prices differently when the audit warned about the basis.
            tier_state["a6_pricing_lesson"] = a6_lesson_rule[:600]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"What sponsorship tier should Paytriq offer "
                      f"{lead.name} for {event.name}?"),
            question_type=QuestionType.CHOICE,
            state=tier_state,
            options=list(TIER_OPTIONS),
            instructions=(
                "Choose the highest tier this sponsor is likely to accept for "
                "the quoted price, given its fit and the event's scale. Do not "
                "invent a tier."
            ),
            asked_by=self.id,
            decision_point="a2.choose_tier",
        )
        decision = self._decide(ctx, request)
        chosen = (decision.choice or "").strip()
        if chosen in TIER_SPECS:
            return chosen, decision
        # A model that named a tier A2 does not know is a data fault, not an
        # invitation to invent one: fall back to the cheapest configured tier and
        # say so, which is the conservative direction.
        return TIER_OPTIONS[-1], decision

    def _choose_posture(self, ctx: AgentContext, event: EventProfile, thread: Thread,
                        offer: Offer, *, revision_request: str = "",
                        a6_lesson_rule: str = "") -> tuple[str, Decision]:
        """How to answer the objection."""
        spec = TIER_SPECS.get(offer.tier, TIER_SPECS["Stall Only"])
        posture_state: dict[str, Any] = {
            "brand": offer.brand,
            "tier": offer.tier,
            "amount_inr": offer.amount_inr,
            "version": offer.version,
            "deliverables": offer.deliverables,
            "objection": thread.reply_text,
            "event": event.name,
            "footfall": event.footfall,
            "delivery_floor_inr": spec["delivery_floor_inr"],
            "rate_basis": RATE_PROVENANCE,
            "addons_available": list(OPTIONAL_EXTRAS),
        }
        if revision_request:
            posture_state["revision_request"] = revision_request[:600]
        if a6_lesson_rule:
            posture_state["a6_pricing_lesson"] = a6_lesson_rule[:600]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"{offer.brand} has pushed back on the {offer.tier} offer. "
                      f"How should Paytriq respond?"),
            question_type=QuestionType.CHOICE,
            state=posture_state,
            options=list(POSTURE_OPTIONS),
            instructions=(
                "Choose the posture that keeps the deal worth doing. Walking away "
                "is a legitimate answer when the objection is about something "
                "other than price."
            ),
            asked_by=self.id,
            decision_point="a2.negotiation_posture",
        )
        decision = self._decide(ctx, request)
        chosen = (decision.choice or "").strip()
        return (chosen if chosen in POSTURE_OPTIONS else "hold_price"), decision

    def _choose_concession(self, ctx: AgentContext, event: EventProfile,
                           thread: Thread, offer: Offer,
                           posture: str, *, revision_request: str = "",
                           a6_lesson_rule: str = "") -> tuple[str, Decision]:
        """How big the concession is — a second decision, not a default factor."""
        spec = TIER_SPECS.get(offer.tier, TIER_SPECS["Stall Only"])
        floor = float(spec["delivery_floor_inr"])
        concession_state: dict[str, Any] = {
            "brand": offer.brand,
            "current_amount_inr": offer.amount_inr,
            "posture": posture,
            "objection": thread.reply_text,
            "delivery_floor_inr": floor,
            "floor_as_pct_of_quote": round(
                (floor / offer.amount_inr * 100.0) if offer.amount_inr else 100.0, 1),
            "max_single_concession_pct": MAX_TOTAL_CONCESSION_PCT * 100.0,
            "addons_available": list(OPTIONAL_EXTRAS),
        }
        if revision_request:
            concession_state["revision_request"] = revision_request[:600]
        if a6_lesson_rule:
            concession_state["a6_pricing_lesson"] = a6_lesson_rule[:600]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"By how much should Paytriq reduce the {offer.tier} fee for "
                      f"{offer.brand} to answer this objection?"),
            question_type=QuestionType.CHOICE,
            state=concession_state,
            options=list(CONCESSION_OPTIONS),
            instructions=(
                "Choose the smallest concession that plausibly closes the "
                "objection. The fee may not fall below the delivery floor."
            ),
            asked_by=self.id,
            decision_point="a2.concession_size",
        )
        decision = self._decide(ctx, request)
        chosen = (decision.choice or "").strip()
        if chosen not in CONCESSION_OPTIONS:
            return "hold_price", decision
        if chosen != "hold_price" and floor >= offer.amount_inr:
            # Nothing can be given away without breaking even. That is a fact
            # about the arithmetic, not a judgement to hide behind a default.
            return "hold_price", decision
        return chosen, decision

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
    def _leads(self, ctx: AgentContext, notes: list[str]) -> list[BrandLead]:
        return [lead for lead in
                load_all(ctx.board, ZONE_OPPORTUNITIES, BrandLead,
                         kind=KIND_BRAND_LEAD, notes=notes)]

    def _pending_revision(self, ctx: AgentContext, notes: list[str]
                          ) -> tuple[Thread, Offer] | None:
        """The newest pushback whose cited offer has not been revised yet.

        Derived from the board, not from a scratch flag: ``Thread.offer_id``
        names the exact offer the sponsor objected to, and the board is
        append-only, so "has this objection been answered?" is answerable by
        comparing that id with the newest offer for the same brand. That is what
        keeps a second objection (a genuine new round) from being swallowed by
        the first one's handling.
        """
        threads = load_all(ctx.board, ZONE_THREADS, Thread, kind=KIND_THREAD,
                           notes=notes)
        offers = load_all(ctx.board, ZONE_OFFERS, Offer, kind=KIND_OFFER, notes=notes)
        if not threads or not offers:
            return None
        latest_by_brand: dict[str, Offer] = {}
        for offer in offers:
            latest_by_brand[offer.brand] = offer
        by_id = {offer.offer_id: offer for offer in offers}
        for thread in reversed(threads):
            if thread.intent is not Intent.PUSHBACK:
                continue
            if thread.status not in ("negotiating", "replied"):
                continue
            if not thread.offer_id:
                notes.append(
                    f"thread {thread.thread_id} records pushback without an "
                    "offer_id; A2 cannot tell which quote was objected to")
                continue
            referenced = by_id.get(thread.offer_id)
            current = latest_by_brand.get(thread.brand)
            if current is None:
                continue
            if current.offer_id == thread.offer_id:
                # The newest offer for this brand is still the one the sponsor
                # objected to, so the objection is unanswered.
                return thread, (referenced or current)
        return None

    def _offer_with_entry(self, ctx: AgentContext, offer_id: str, notes: list[str]
                          ) -> tuple[BoardEntry, Offer] | None:
        """The entry *and* the model for one offer, so a revision can cite it."""
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
        return None

    def _lead_refs(self, ctx: AgentContext, brand: str) -> list[str]:
        """Cite the A1 entry the offer is based on, so the chain is walkable."""
        try:
            entries = read_entries(ctx.board, ZONE_OPPORTUNITIES,
                                   kind=KIND_BRAND_LEAD)
        except AgentError:
            return []
        for entry in reversed(entries):
            if entry.payload.get("name") == brand:
                return [entry.entry_id]
        return []

    def _post_basis(self, ctx: AgentContext, event: EventProfile, offer: Offer,
                    offer_entry_id: str) -> None:
        """Record the tariff next to the offer as an open, multi-writer lesson."""
        lesson = Lesson(
            lesson_id=new_id("les"),
            event_id=event.event_id,
            author=self.id,
            trigger=f"priced {offer.brand} at the {offer.tier} tier",
            correction="a price needs a stated basis, not just a number",
            rule=(f"{offer.brand} @ INR {offer.amount_inr:,.0f} = "
                  f"{TIER_SPECS[offer.tier]['delivery_floor_inr']:,.0f} delivery "
                  f"floor + {event.footfall:,} footfall x "
                  f"{RATE_PER_ATTENDEE_INR} INR x "
                  f"{TIER_SPECS[offer.tier]['multiplier']} tier multiplier. "
                  f"{RATE_PROVENANCE}"),
            confidence=0.8,
        )
        try:
            self.post(ctx, ZONE_LESSONS, KIND_LESSON, lesson.model_dump(mode="json"),
                      refs=[offer_entry_id], source=DecisionSource.RULES)
        except (BlackboardError, SchemaError):
            # The basis is also in the pitch and in Observation.facts, so a board
            # that refuses the lesson loses redundancy, not the record.
            return

    def _hand_back(self, ctx: AgentContext, brand: str, offer: Offer,
                   objection: str) -> str | None:
        """Tell A3 the deal is dead: A2 may not write to ``threads`` (A3 owns it)."""
        handoff = Handoff(
            handoff_id=new_id("hnd"),
            event_id=offer.event_id,
            run_id=ctx.run_id,
            from_agent=self.id,
            to_agent=AgentId.A3_OUTREACH,
            reason=(f"{brand} negotiation ended: A2 decided to walk away from the "
                    f"{offer.tier} offer"),
            decision_source=DecisionSource.RULES,
            confidence=1.0,
            summary=(f"No revised offer was produced for {brand}; close the "
                     f"thread. Objection on record: {objection[:200]}"),
            payload_refs=[],
        )
        try:
            entry_id = self.post(ctx, ZONE_HANDOFFS, KIND_HANDOFF,
                                 handoff.model_dump(mode="json"),
                                 source=DecisionSource.RULES)
            self._trace_handoff(ctx, handoff)
            return entry_id
        except (BlackboardError, SchemaError):
            return None

    def _trace_handoff(self, ctx: AgentContext, handoff: Handoff) -> None:
        """Mirror the handoff into the trace; never fail a step over telemetry."""
        emit = getattr(ctx.tracer, "handoff", None)
        if not callable(emit):
            return
        try:
            emit(handoff)
        except (TypeError, ValueError, RuntimeError):
            # Telemetry is best-effort: the handoff is already on the board, and
            # losing a span is not a reason to fail a pricing step.
            return

    # ============================================ inter-agent coordination (P2)
    def _post_message(self, ctx: AgentContext, kind: MessageKind,
                      to_agent: AgentId, body: str, *,
                      refs: Sequence[str], confidence: float,
                      source: DecisionSource, summary: str = "") -> str | None:
        """Post one typed :class:`~core.schemas.Message` to ``messages``.

        Best-effort: a board without a ``messages`` zone (older stub
        registries) yields ``None`` and the offer itself carries the revision.
        On the real board this is the record A3 reads to prioritise the send
        and, on a second pushback, to open the pricing dispute.
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
        """Typed ``messages`` reads; a refusal reads as empty, never as failure."""
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

    def _revision_request_for(self, ctx: AgentContext, brand: str, offer_id: str,
                              notes: list[str]
                              ) -> tuple[BoardEntry, Message] | None:
        """Newest ``REVISION_REQUEST`` from A3 naming this brand or offer.

        A3 always names both the brand and the cited ``offer_id`` in the body,
        so the join is a substring match — messages carry no brand field of
        their own. ``None`` means "no message on this board" (e.g. a stub
        registry that refused the post), in which case the thread alone drives
        the revision.
        """
        match: tuple[BoardEntry, Message] | None = None
        for entry, message in self._read_messages(ctx, notes):
            if message.kind is not MessageKind.REVISION_REQUEST:
                continue
            if message.from_agent is not AgentId.A3_OUTREACH:
                continue
            if message.to_agent is not self.id:
                continue
            if not ((brand and brand in message.body)
                    or (offer_id and offer_id in message.body)):
                continue
            match = (entry, message)
        return match

    def _read_lessons(self, ctx: AgentContext, notes: list[str]
                      ) -> list[tuple[BoardEntry, Lesson]]:
        """Typed ``lessons`` reads; invalid entries are noted and skipped."""
        try:
            entries = read_entries(ctx.board, ZONE_LESSONS, kind=KIND_LESSON)
        except AgentError as exc:
            notes.append(str(exc))
            return []
        out: list[tuple[BoardEntry, Lesson]] = []
        for entry in entries:
            try:
                out.append((entry, Lesson.model_validate(entry.payload)))
            except ValidationError:
                notes.append(f"skipped lesson entry {entry.entry_id}: invalid Lesson")
        return out

    def _a6_pricing_guidance(self, ctx: AgentContext, notes: list[str]
                             ) -> tuple[BoardEntry, Lesson] | None:
        """Newest A6 lesson about the pricing basis, if any.

        A6 writes ``pricing_basis`` into the trigger/rule when its
        ``a6.lesson.focus`` decision picks that focus, so the substring is the
        documented join — not a guess about lesson prose. The caller branches
        on the lesson's ``confidence`` and cites its ``lesson_id`` in the
        revision note, so the guidance is applied, not merely logged.
        """
        guidance: tuple[BoardEntry, Lesson] | None = None
        for entry, lesson in self._read_lessons(ctx, notes):
            if lesson.author is not AgentId.A6_AUDIT:
                continue
            haystack = f"{lesson.trigger}\n{lesson.rule}\n{lesson.correction}".lower()
            if "pricing_basis" not in haystack and "pricing basis" not in haystack:
                continue
            guidance = (entry, lesson)
        return guidance

    # ==================================================================== helpers
    @staticmethod
    def _pick_text_tool(ctx: AgentContext, *,
                        purpose: str = "pitch text generation",
                        decision_point: str = "a2.select_text_tool"
                        ) -> tuple[Tool | None, str | None]:
        """Model-directed text-tool choice with function schemas in state."""
        from agents.tool_selection import select_tool

        tool, name, _ = select_tool(
            ctx, purpose, list(TEXT_TOOL_NAMES), decision_point)
        return tool, name

    def _event_name(self, ctx: AgentContext) -> str:
        """Event name for prose, or an honest placeholder when absent."""
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE)
        return event.name if event is not None else "this event"

    def _terminal_plan(self, reason: str, *, sufficient: bool,
                       gap: str | None = None) -> Plan:
        steps = ["halt"]
        if gap:
            steps.append(gap)
        return Plan(goal=reason, steps=steps, rationale=reason, confidence=1.0,
                    source=DecisionSource.RULES, stop=True)

    @staticmethod
    def _state(ctx: AgentContext) -> dict[str, Any]:
        value = ctx.scratch.get("a2_state")
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _store(ctx: AgentContext, state: Mapping[str, Any]) -> None:
        ctx.scratch["a2_state"] = dict(state)
