"""The simulated sponsor: an environment, not an agent.

Read this before counting the agents
------------------------------------
:class:`SponsorEnvironment` stands in for the human counterparty on the other side
of the negotiation — the person who receives the outreach email and decides
whether to say yes. It is part of the **test environment**, exactly like the
simulated opponents in SOTOPIA (Zhou et al., ICLR 2024) or the simulated
counterparties in NegotiationArena (Bianchi et al., ICML 2024). Those papers
demonstrated that an LLM's negotiation ability cannot be measured against a
script: if the counterpart always agrees, "it negotiated" is unfalsifiable. The
same reasoning applies here.

Three properties follow, and each is load-bearing for the academic claim:

1. **It is not an agent.** ``id`` is :attr:`AgentId.ENVIRONMENT`, which
   :attr:`AgentId.is_reasoning_agent` returns ``False`` for and which
   ``REASONING_AGENTS`` excludes. Paytriq has **seven** reasoning agents, A1–A7.
   Counting the sponsor would inflate that number to eight and misrepresent the
   architecture being assessed.
2. **It carries private constraints the agents cannot read.** Each sponsor has a
   budget ceiling, must-have deliverables, walk-away conditions, patience, and a
   hidden reservation value. These live in :attr:`SponsorEnvironment._profiles`
   and are never posted to the blackboard, never included in a decision state the
   agents can observe, and never surfaced through any board entry. They are the
   reason the negotiation is a problem to be solved rather than a script to be
   replayed: A2 cannot price its way out of a constraint it has never been told
   about. The reservation value in particular must stay hidden — a sponsor that
   reveals its walk-away price has revealed its answer.
3. **It can say no.** :meth:`SponsorEnvironment.receive` returns
   :attr:`SponsorReply` objects across ``yes``, ``pushback``, ``interested``,
   ``no`` and ``neutral``. A counterparty that always agrees demonstrates
   nothing, and every number the pipeline reports after it would be meaningless.

Every reply is composed from the *actual* offer — its tier, its amount, the
specific deliverable in question — and from the constraint that binds, so two
offers produce two different replies. The previous demo cycled through three
hardcoded sentences in a ``for`` loop regardless of what it was replying to.

Test helper
-----------
:meth:`SponsorEnvironment.constraints_for` exists so a unit test can assert that
the hidden constraints drove the reply. **No agent may call it.** It is marked
``TEST HELPER`` in the name, the docstring, and the trace, and
``tests/unit/test_agents_governance.py`` asserts that no agent module references
it. The agents' counterparty must be as opaque to them as a real one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings, get_settings
from core.ids import new_id, utcnow
from core.protocols import AgentContext, Plan
from core.schemas import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    Intent,
    Offer,
    QuestionType,
    Thread,
)

__all__ = [
    "SponsorEnvironment",
    "SponsorProfile",
    "SponsorReply",
    "SponsorConstraints",
    "REPLY_CHOICES",
    "AGENT_ID",
]

#: This class is the environment. It is not, and must never be counted as, an agent.
AGENT_ID = AgentId.ENVIRONMENT

#: The closed option set every reply decision is drawn from.
REPLY_CHOICES: tuple[str, ...] = (
    Intent.YES.value,
    Intent.PUSHBACK.value,
    Intent.INTERESTED.value,
    Intent.NO.value,
    Intent.NEUTRAL.value,
)


# ================================================================== private state
@dataclass(slots=True)
class SponsorProfile:
    """What the sponsor knows and wants. **Never leaves this object.**

    Nothing here is ever posted to the blackboard or placed in a decision state
    that an agent can read. ``reservation_value_inr`` is the strictest secret:
    revealing it hands the negotiation to A2.
    """

    brand: str
    budget_ceiling_inr: float
    #: Deliverables without which the deal is not worth doing.
    must_have: list[str] = field(default_factory=list)
    #: Conditions that end the conversation outright.
    walk_away_conditions: list[str] = field(default_factory=list)
    #: How many negotiation rounds before the sponsor disengages.
    patience: int = 2
    #: The price above which the sponsor walks. Strictly private.
    reservation_value_inr: float = 0.0
    #: Preferred wording style, so replies do not all read identically.
    tone: str = "neutral"

    def public_facts(self) -> dict[str, Any]:
        """The subset the sponsor is willing to have *it* reason over.

        Used to build the decision state. The reservation value is deliberately
        absent: the model that plays the sponsor must be able to choose ``no`` for
        reasons it was told about, not because it was handed the answer.
        """
        return {
            "brand": self.brand,
            "budget_ceiling_inr": self.budget_ceiling_inr,
            "must_have": list(self.must_have),
            "walk_away_conditions": list(self.walk_away_conditions),
            "patience": self.patience,
            "tone": self.tone,
        }


@dataclass(slots=True)
class SponsorConstraints:
    """A read-only snapshot of the private constraints. **TEST HELPER ONLY.**

    Returned by :meth:`SponsorEnvironment.constraints_for`. Present so a test can
    assert *why* a reply came out the way it did. No agent may call it; if one
    does, the hidden-information requirement of the experiment is void.
    """

    brand: str
    budget_ceiling_inr: float
    must_have: list[str]
    walk_away_conditions: list[str]
    patience: int
    reservation_value_inr: float
    tone: str


@dataclass(slots=True)
class SponsorReply:
    """One inbound message from the counterparty."""

    thread_id: str
    brand: str
    offer_id: str
    intent: Intent
    body: str
    #: Set when the sponsor names a price of its own.
    counter_amount_inr: float | None = None
    #: Which deliverable the reply is about, quoted from the offer.
    about_deliverable: str | None = None
    #: Why this reply, referencing the private constraint that bound. Diagnostic
    #: only — never posted to a zone the agents read.
    rationale: str = ""
    confidence: float = 0.0
    source: DecisionSource = DecisionSource.RULES
    round_index: int = 0
    decided_at: str = ""

    @property
    def is_terminal(self) -> bool:
        """True when the sponsor has ended the conversation."""
        return self.intent is Intent.NO

    def as_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "brand": self.brand,
            "offer_id": self.offer_id,
            "intent": self.intent.value,
            "body": self.body,
            "counter_amount_inr": self.counter_amount_inr,
            "about_deliverable": self.about_deliverable,
            "confidence": round(self.confidence, 6),
            "source": self.source.value,
            "round_index": self.round_index,
            "decided_at": self.decided_at,
        }


# ==================================================================== environment
class SponsorEnvironment:
    """The counterparty. Simulates the human on the other side of the email.

    Implements the ``Agent`` shape (``id``, ``role``, ``plan``, ``act``,
    ``observe``) so the graph can hold it in the same registry slot as an agent,
    but it is *not* one: :attr:`id` is ``AgentId.ENVIRONMENT``, which
    ``REASONING_AGENTS`` excludes, and :attr:`is_agent` is ``False``. Its decisions
    do not count toward the system's agent count, its handoffs are not agent
    handoffs, and its trace entries say ``ENV`` rather than ``A1``–``A7``.
    """

    id = AGENT_ID
    role = "Simulated sponsor counterparty (test environment, not a reasoning agent)"
    is_agent = False

    def __init__(self, profiles: Mapping[str, SponsorProfile] | None = None,
                 *, settings: Settings | None = None) -> None:
        self._profiles: dict[str, SponsorProfile] = dict(profiles or {})
        self.settings = settings or get_settings()
        #: Rounds seen per thread, so patience actually depletes across a run.
        self._round_counts: dict[str, int] = {}

    # ------------------------------------------------------------- registration
    def register(self, profile: SponsorProfile) -> None:
        """Add a sponsor's private constraints."""
        self._profiles[profile.brand] = profile

    def known_brands(self) -> list[str]:
        return sorted(self._profiles)

    def profile_for(self, brand: str) -> SponsorProfile:
        """The private profile for ``brand``.

        ``ValueError`` for an unregistered sponsor rather than a synthesised
        default: a sponsor with no constraints is not a counterparty, it is a
        rubber stamp, and pretending otherwise would hide that the run had no
        real adversary.
        """
        try:
            return self._profiles[brand]
        except KeyError:
            raise ValueError(
                f"no private SponsorProfile registered for {brand!r}; the "
                f"environment cannot simulate a counterparty it knows nothing "
                f"about. Registered: {sorted(self._profiles)}"
            ) from None

    # ------------------------------------------------------ TEST HELPER ONLY
    def constraints_for(self, brand: str) -> SponsorConstraints:
        """**TEST HELPER — no agent may call this.**

        Returns the private constraints so a unit test can assert that a reply was
        driven by a hidden constraint rather than by luck. Its existence is a
        deliberate hole in the information barrier, and it is only ever opened
        from a test.
        """
        profile = self.profile_for(brand)
        return SponsorConstraints(
            brand=profile.brand,
            budget_ceiling_inr=profile.budget_ceiling_inr,
            must_have=list(profile.must_have),
            walk_away_conditions=list(profile.walk_away_conditions),
            patience=profile.patience,
            reservation_value_inr=profile.reservation_value_inr,
            tone=profile.tone,
        )

    # ================================================================== reply
    def receive(self, offer: Offer, thread: Thread, ctx: AgentContext) -> SponsorReply:
        """Decide how this sponsor answers ``offer``, and say why.

        The decision goes through ``ctx.decide`` over the five intents. The reply
        text is composed afterwards from the *actual* offer — its tier, its
        amount, the specific deliverable the reply concerns — plus the constraint
        that bound, so no two offers of the same tier get the same sentence.
        """
        profile = self.profile_for(offer.brand)
        round_index = self._round_counts.get(thread.thread_id, 0) + 1
        self._round_counts[thread.thread_id] = round_index

        over_budget = offer.amount_inr > profile.budget_ceiling_inr
        missing = [d for d in profile.must_have if not _offer_covers(offer, d)]
        about = missing[0] if missing else _first_deliverable(offer)
        patience_exhausted = round_index > profile.patience

        state = {
            "offer_id": offer.offer_id,
            "offer_version": offer.version,
            "tier": offer.tier,
            "amount_inr": offer.amount_inr,
            "deliverables": list(offer.deliverables),
            "brand": profile.brand,
            "round_index": round_index,
            "patience": profile.patience,
            "over_budget": over_budget,
            "budget_ceiling_inr": profile.budget_ceiling_inr,
            "missing_must_have": missing,
            "walk_away_conditions": list(profile.walk_away_conditions),
            "tone": profile.tone,
            # NOTE: reservation_value_inr is deliberately absent. The sponsor's
            # true ceiling is not an input to its own decision-making *as stated*;
            # it is the ground truth a test compares against.
            "prior_reply_intent": thread.intent.value,
            "prior_reply_text": thread.reply_text,
        }
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"As {offer.brand}, how do you respond to the {offer.tier} tier "
                      f"proposal worth Rs {offer.amount_inr:,.0f} for round "
                      f"{round_index}?"),
            question_type=QuestionType.CHOICE,
            state=state,
            options=list(REPLY_CHOICES),
            rubric=[
                "yes: accept the tier and its amount as offered",
                "pushback: the price or scope must change before proceeding",
                "interested: want more detail before committing either way",
                "no: decline; the deal is off",
                "neutral: acknowledge without a position",
            ],
            instructions=(
                "Answer no if the amount exceeds what you can justify or a must-have "
                "deliverable is absent. Answer pushback if the amount is above your "
                "ceiling but a smaller scope could work. You are a busy "
                "counterparty: a refusal is a valid and common outcome."
            ),
            asked_by=AGENT_ID,
            decision_point="environment.sponsor.reply",
        )
        decision: Decision = ctx.decide(request)
        intent_value = _coerce_choice(decision.choice, thread.intent.value)
        intent = Intent(intent_value)
        floor_applied = ""
        if patience_exhausted and intent is not Intent.NO:
            # The patience floor. A decision model asked the same question three
            # times may well keep answering "pushback"; no real sponsor negotiates
            # indefinitely, and a counterparty that never terminates makes the
            # whole run hang. The decision is still asked, recorded and reported —
            # the override is disclosed in the reply's rationale rather than
            # silently replacing the model's answer.
            floor_applied = (f"patience floor: round {round_index} exceeds patience "
                             f"{profile.patience}, so the counterparty withdraws "
                             f"regardless of the model's {intent_value!r} answer")
            intent = Intent.NO

        body, rationale = self._compose(
            profile=profile, offer=offer, intent=intent, about=about,
            missing=missing, over_budget=over_budget,
            patience_exhausted=patience_exhausted, round_index=round_index,
        )
        if floor_applied:
            rationale = f"{rationale}; {floor_applied}"
        counter = None
        if intent is Intent.PUSHBACK and over_budget:
            # A counter anchored on the sponsor's own ceiling, not on the offer.
            counter = float(profile.budget_ceiling_inr)

        return SponsorReply(
            thread_id=thread.thread_id,
            brand=offer.brand,
            offer_id=offer.offer_id,
            intent=intent,
            body=body,
            counter_amount_inr=counter,
            about_deliverable=about,
            rationale=rationale,
            confidence=round(float(decision.confidence), 6),
            source=decision.source,
            round_index=round_index,
            decided_at=utcnow().isoformat(),
        )

    # ============================================================== composition
    def _compose(self, *, profile: SponsorProfile, offer: Offer, intent: Intent,
                 about: str | None, missing: Sequence[str], over_budget: bool,
                 patience_exhausted: bool, round_index: int) -> tuple[str, str]:
        """Build the reply body and its rationale.

        The body quotes the offer's own tier and amount and names the deliverable
        the answer concerns, so it is about *this* proposal rather than a generic
        template. The rationale names the constraint that bound; it is diagnostic
        and is never posted where an agent could read it.
        """
        tier = offer.tier
        amount = f"Rs {offer.amount_inr:,.0f}"
        ceiling = f"Rs {profile.budget_ceiling_inr:,.0f}"
        deliverable_clause = f' on "{about}"' if about else ""
        binding = _binding_constraint(profile, over_budget, missing, patience_exhausted)

        if intent is Intent.YES:
            body = (f"{profile.brand} confirms the {tier} tier at {amount}. "
                    f"Please proceed{deliverable_clause} and send the MoU for signature.")
            rationale = (f"offer at {amount} is within the {ceiling} ceiling and covers "
                         f"every must-have")
        elif intent is Intent.PUSHBACK:
            if over_budget:
                body = (f"{profile.brand} is interested in the {tier} tier but {amount} "
                        f"is above what we can sign off this quarter{deliverable_clause}. "
                        f"Can you come back at {ceiling}, or narrow the scope to the "
                        f"essentials?")
            else:
                want = about or "the deliverables"
                body = (f"{profile.brand} can support the {tier} tier, but {amount} is "
                        f"hard to justify internally without \"{want}\" tightened. "
                        f"What would the scope look like at a lower number?")
            rationale = (f"amount {amount} exceeds the {ceiling} ceiling"
                         if over_budget else
                         f"amount {amount} is within the {ceiling} ceiling but the "
                         f"scope must change before the sponsor can commit")
        elif intent is Intent.INTERESTED:
            body = (f"{profile.brand} is interested in the {tier} tier at {amount}. "
                    f"Before we commit{deliverable_clause}, we would like audience "
                    f"numbers and the fulfilment timeline.")
            rationale = "amount is affordable but the sponsor wants more detail first"
        elif intent is Intent.NO:
            if patience_exhausted:
                body = (f"{profile.brand} is withdrawing from the {tier} discussion. "
                        f"After {round_index} rounds at {amount} we have not closed the "
                        f"gap, so we are passing{deliverable_clause}.")
            elif missing:
                body = (f"{profile.brand} must decline the {tier} tier at {amount}. "
                        f"Without \"{missing[0]}\" the arrangement does not work for us.")
            else:
                body = (f"{profile.brand} is declining the {tier} tier at {amount}. It "
                        f"is not the right fit for us this cycle{deliverable_clause}.")
            rationale = f"declined: {binding}"
        else:  # Intent.NEUTRAL and Intent.UNKNOWN both land here
            body = (f"{profile.brand} acknowledges receipt of the {tier} proposal at "
                    f"{amount}. We are still reviewing it internally and will revert.")
            rationale = "no position taken yet; sponsor is still reviewing"

        if profile.tone == "blunt" and intent in (Intent.NO, Intent.PUSHBACK):
            body = _sentence_end(body) + " No room to move on this one."
        elif profile.tone == "warm" and intent in (Intent.YES, Intent.INTERESTED):
            body = _sentence_end(body) + " Looking forward to it."
        return body, rationale

    # ================================================================= Agent API
    def plan(self, ctx: AgentContext, obs: Any = None) -> Plan:
        """Environment-shaped planning. Always the same plan; nothing to decide."""
        del obs
        return Plan(
            goal=f"decide the {ctx.event_id} counterparty response",
            steps=["resolve the sponsor's private constraints",
                   "decide the reply intent through ctx.decide",
                   "compose a reply about the actual offer"],
            rationale="the counterparty has one job and no branching beyond the reply",
            confidence=1.0,
            source=DecisionSource.RULES,
            stop=False,
        )

    def act(self, ctx: AgentContext, plan: Plan) -> Any:  # pragma: no cover - env glue
        """The graph supplies the offer/thread; see :meth:`receive`."""
        del ctx, plan
        raise NotImplementedError(
            "SponsorEnvironment.receive(offer, thread, ctx) is the environment's "
            "entry point; it is not driven through the agent plan/act loop"
        )

    def observe(self, ctx: AgentContext, plan: Plan, result: Any) -> Any:  # pragma: no cover
        del ctx, plan, result
        raise NotImplementedError(
            "SponsorEnvironment has no observation phase; it returns a SponsorReply"
        )

    def reflect(self, ctx: AgentContext, obs: Any) -> None:  # pragma: no cover
        del ctx, obs
        return

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<SponsorEnvironment id={self.id.value} "
                f"is_agent={self.is_agent} sponsors={sorted(self._profiles)}>")


# ======================================================================= helpers
def _sentence_end(body: str) -> str:
    """Terminate a body with a full stop unless it already ends in punctuation.

    Tone flourishes are appended to composed text; without this a pushback that
    ends in a question picks up ``"?. No room to move."``
    """
    stripped = body.rstrip()
    return stripped if stripped.endswith((".", "!", "?")) else stripped + "."


def _offer_covers(offer: Offer, must_have: str) -> bool:
    """Whether the offer already promises ``must_have``.

    A structured comparison of the offer's deliverable list, not sentiment
    analysis of a pitch. Case-insensitive because deliverable names are
    user-authored and casing is not meaning.
    """
    wanted = " ".join(str(must_have).split()).casefold()
    for deliverable in offer.deliverables:
        if wanted in " ".join(str(deliverable).split()).casefold():
            return True
    return False


def _first_deliverable(offer: Offer) -> str | None:
    """The deliverable a reply should be *about*, or ``None`` for a bare offer."""
    return str(offer.deliverables[0]) if offer.deliverables else None


def _binding_constraint(profile: SponsorProfile, over_budget: bool,
                       missing: Sequence[str], patience_exhausted: bool) -> str:
    """Which private constraint actually bound. Diagnostic text.

    Patience is checked first: when it is exhausted it is the *proximate* reason
    the sponsor is withdrawing now, whereas an over-budget offer is the standing
    reason the rounds were hard in the first place. Naming the standing reason
    when the sponsor actually walked would misdescribe the decision.
    """
    if patience_exhausted:
        return f"patience of {profile.patience} round(s) exhausted"
    if missing:
        return f"must-have deliverable absent: {missing[0]}"
    if over_budget:
        return f"offer exceeds budget ceiling {profile.budget_ceiling_inr:,.0f}"
    return f"walk-away condition met: {', '.join(profile.walk_away_conditions) or 'n/a'}"


def _coerce_choice(choice: str | None, fallback: str) -> str:
    """Map a decision's answer onto :data:`REPLY_CHOICES`.

    An off-menu answer becomes the thread's prior intent if that is on-menu, and
    ``neutral`` otherwise. ``NEUTRAL`` rather than ``YES``: the previous router
    defaulted every unrecognised message to ``interested``, which is how
    "yesterday we thought the price was too high" reached contract signing.
    """
    value = str(choice or "").strip().lower()
    if value in REPLY_CHOICES:
        return value
    if fallback in REPLY_CHOICES:
        return fallback
    return Intent.NEUTRAL.value
