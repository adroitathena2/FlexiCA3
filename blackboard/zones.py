"""Zone registry for the Paytriq blackboard.

A *zone* is a named, typed partition of the shared workspace. Hayes-Roth's
original blackboard was explicitly partitioned into regions -- a knowledge area,
a task area, a management area -- and Nii's survey formalised the requirement
that such a structure be subdivided into typed areas with their own acceptance
conditions. Paytriq follows that: an entry is not a free-floating blob, it
belongs to exactly one zone and to one *kind* within that zone.

Why zones at all, when a list of dicts would be simpler?
--------------------------------------------------------
The failure mode of an untyped shared list is not usually a crash, it is two
agents quietly disagreeing about what "an offer" contains, with nothing in the
system able to say which one is real. Binding every artefact-shaped kind to one
``core.schemas`` model makes that impossible: there is one definition of an
offer, it validates before it is accepted, and a revision is a new immutable
entry rather than an overwrite. The record of how the price moved from v1 to v3
survives the run.

Zones additionally give each region a declared owner (:attr:`Zone.authoritative_agent`)
and a closed set of accepted kinds, so both an unexpected author and an
unexpected kind are *visible* in ``stats()`` or loud, respectively.

The registry is a module-level singleton. Reads are lock-free (a dict lookup of a
key that already exists cannot tear, and the definitions are frozen), while
registration takes a lock because the API layer may add a zone from a request
handler.

Import rule: ``blackboard`` depends on ``core`` only.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from pydantic import BaseModel

from core.errors import BlackboardError, ZoneNotFound
from core.schemas import (
    AgentId,
    Approval,
    AuditFinding,
    Bid,
    BrandLead,
    Dispute,
    EventProfile,
    Handoff,
    HumanDecision,
    HumanGate,
    Lesson,
    MoU,
    Offer,
    RiskFlag,
    ROIReport,
    Thread,
)

try:
    from core.schemas import Message, MessageKind
except ImportError:  # Track 1 (core.schemas Message) has not landed yet.
    # Temporary placeholder so the blackboard/state plumbing (Track 2) can
    # land and be tested before the schema itself does. It mirrors the
    # specified Message fields (message_id, event_id, run_id, from_agent,
    # to_agent, kind, body, refs, decision_source, confidence, summary, at).
    # Once core.schemas provides the real models the try succeeds, this block
    # never runs, and it can be deleted.
    from datetime import datetime
    from enum import StrEnum

    from pydantic import ConfigDict, Field

    from core.ids import utcnow
    from core.schemas import DecisionSource

    class MessageKind(StrEnum):
        """Placeholder kind enum until core.schemas provides the real one."""

        NOTE = "note"

    class Message(BaseModel):
        """Placeholder message until core.schemas provides the real model."""

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            str_strip_whitespace=True,
            use_enum_values=False,
        )

        message_id: str
        event_id: str
        run_id: str
        from_agent: AgentId
        to_agent: AgentId
        kind: str = MessageKind.NOTE.value
        body: str = ""
        refs: list[str] = Field(default_factory=list)
        decision_source: DecisionSource = DecisionSource.RULES
        confidence: float = Field(default=1.0, ge=0, le=1)
        summary: str = ""
        at: datetime = Field(default_factory=utcnow)

__all__ = [
    "Zone",
    "ZONES",
    "ZONE_NAMES",
    "REGISTERED_ZONES",
    "KIND_MODELS",
    "MODEL_KINDS",
    "UNTYPED_KINDS",
    "AUTHOR_FIELD",
    "CONFIDENCE_FIELD",
    "SOURCE_FIELD",
    "zone",
    "zones_all",
    "register_zone",
    "validate_post",
]


@dataclass(frozen=True, slots=True)
class Zone:
    """One typed region of the blackboard.

    ``authoritative_agent`` names the agent that *owns* the zone: the one whose
    judgement the zone records. It is used as the default author when a typed
    model is posted without one, and it is reported by
    :meth:`~blackboard.board.InMemoryBlackboard.stats` whenever some other agent
    posts here.

    Ownership is **declared and reported, not enforced**, because the frozen core
    schemas fix no author per artefact: ``RiskFlag.raised_by``, ``Bid.agent``,
    ``Lesson.author`` and ``Handoff.from_agent`` are all caller-supplied, and
    cross-zone work is real (A5 writes its own ``audit_finding`` entries into the
    audit zone; A6 raises a ``risk_flag`` when it catches a compliance-summary
    mismatch). Rejecting those would break legitimate work, and it would protect
    against nothing: an artefact with two authors is still one typed,
    append-only, schema-validated record, and the two versions remain separately
    readable. What genuinely prevents "two incompatible schemas" is the
    kind-to-model binding in :data:`KIND_MODELS` plus append-only versioning,
    both of which are hard guarantees.

    Owner declared per zone: ``opportunities``/A1, ``offers``/A2, ``threads``/A3,
    ``contracts``/A4, ``risk_flags``/A5, ``audit``/A6, ``decisions``/A7. Left
    ``None`` where no single owner exists: ``event``, ``disputes``, ``bids``,
    ``lessons``, ``approvals``, ``handoffs``, ``messages``.
    """

    name: str
    purpose: str
    allowed_kinds: tuple[str, ...]
    authoritative_agent: AgentId | None = None

    def allows(self, kind: str) -> bool:
        """True when ``kind`` is one of this zone's accepted entry kinds."""
        return _normalise(kind) in self.allowed_kinds

    def is_owned_by(self, author: AgentId) -> bool:
        """True when ``author`` is this zone's declared owner (or it has none)."""
        return self.authoritative_agent is None or author is self.authoritative_agent


#: ``kind`` -> the ``core.schemas`` model that kind is supposed to carry.
#:
#: This is the mapping that lets the board store *typed* domain objects instead
#: of untyped dicts. It is also the guard rail against "two implementations with
#: incompatible schemas": every artefact-shaped kind names one canonical model,
#: so "an offer" has exactly one definition no matter which agent is holding it.
KIND_MODELS: dict[str, type[BaseModel]] = {
    "event_profile": EventProfile,
    "brand_lead": BrandLead,
    "offer": Offer,
    "thread": Thread,
    "mou": MoU,
    "risk_flag": RiskFlag,
    "roi_report": ROIReport,
    "audit_finding": AuditFinding,
    "dispute": Dispute,
    "bid": Bid,
    "lesson": Lesson,
    "handoff": Handoff,
    "human_gate": HumanGate,
    "approval": Approval,
    "human_decision": HumanDecision,
    "message": Message,
}

#: Reverse index of :data:`KIND_MODELS`. Built once, never mutated.
MODEL_KINDS: dict[type[BaseModel], str] = {
    model: kind for kind, model in KIND_MODELS.items()
}

#: Kinds that are deliberately *not* typed, and why.
#:
#: An arbitration ruling and a compliance summary have no canonical model because
#: their content is free text plus the ids of the artefacts it rests on -- there
#: is no schema that could be stricter without inventing fields the author never
#: wrote. Naming them here keeps the registry strict (a typo'd kind still fails at
#: registration time) without pretending they have a shape.
UNTYPED_KINDS: frozenset[str] = frozenset({
    "arbitration_decision",
    "escalation",
    "compliance_summary",
    # Contract Net lifecycle markers, as graph.contract_net spells them.
    # The *bids* are typed (core.schemas.Bid); the announcement, the award
    # decision and the progress report are supervisor-side bookkeeping whose
    # payload is whatever the calling module put there. Typing them would mean
    # inventing fields nobody wrote, so they are declared untyped -- still strict
    # about the name, still validated for a real author and zone.
    "task_announcement",
    "announcement",
    "award",
    "expedite",
    # Debate transcript markers. The ``Dispute`` itself is typed; a rebuttal, a
    # concession and the summary are turn-by-turn prose written by graph.debate,
    # so typing them would mean inventing a schema the debate does not have.
    "rebuttal",
    "concession",
    "adjudication",
    "debate_summary",
    # A run summary is a report, not a domain artefact.
    "run_summary",
})

#: Model fields that can supply the entry's ``author`` when the caller omits it.
#: Consulted before the zone's declared owner. A disagreement between two agents
#: has no single author, so this lookup is a convenience and never a guess about
#: attribution.
AUTHOR_FIELD: dict[type[BaseModel], str] = {
    model: field
    for model, field in (
        (Lesson, "author"),
        (Handoff, "from_agent"),
        (RiskFlag, "raised_by"),
        (Bid, "agent"),
        # A dispute has two parties; the claimant is the one who opened it, so
        # the claimant authors the board entry. The opponent is named in the
        # payload, which is where both parties always live.
        (Dispute, "claimant"),
        # A message is authored by its sender, mirroring Handoff.from_agent.
        (Message, "from_agent"),
    )
}

#: Model fields that can supply the entry's ``confidence`` when omitted.
CONFIDENCE_FIELD: dict[type[BaseModel], str] = {
    model: field
    for model, field in (
        (Lesson, "confidence"),
        (Handoff, "confidence"),
        (Message, "confidence"),
    )
}

#: Model fields that can supply the entry's ``DecisionSource`` when omitted.
SOURCE_FIELD: dict[type[BaseModel], str] = {
    model: field
    for model, field in (
        (Handoff, "decision_source"),
        (Message, "decision_source"),
    )
}


def _normalise(token: str) -> str:
    """Lower-case, strip, collapse a zone or kind identifier.

    Forgiving lookup on *identifiers* only. A typo in an internal string must not
    abort a run mid-pipeline, and both zone and kind names are matched against a
    closed registry anyway, so no invalid identifier can slip through.
    """
    return (token or "").strip().lower()


#: Read/write lock protecting registry mutation. Zone *lookups* are lock-free:
#: a plain dict ``get`` on a key that already exists is atomic under the GIL,
#: and reading an immutable frozen dataclass cannot tear.
_LOCK = threading.RLock()

#: name -> Zone. The single source of truth; mutate only via
#: :func:`register_zone`.
ZONES: dict[str, Zone] = {}

#: Ordered zone names, kept in step with :data:`ZONES` by :func:`register_zone`.
ZONE_NAMES: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Registry population
# --------------------------------------------------------------------------- #
def register_zone(definition: Zone) -> Zone:
    """Add (or reject) a zone definition.

    Registration is explicit and validated so a malformed definition fails at
    wiring time rather than at the first ``post``. Duplicate names are refused:
    silently replacing a zone would invalidate every entry already written into
    it, because entries store the zone name they were validated against.
    """
    global ZONE_NAMES

    name = _normalise(definition.name)
    if not name:
        raise BlackboardError("a zone name must be a non-empty string")
    if not definition.allowed_kinds:
        raise BlackboardError(f"zone {name!r} must allow at least one kind")
    kinds = tuple(dict.fromkeys(_normalise(k) for k in definition.allowed_kinds))
    if any(not k for k in kinds):
        raise BlackboardError(f"zone {name!r} declares an empty kind")
    unknown = [
        k for k in kinds if k not in KIND_MODELS and k not in UNTYPED_KINDS
    ]
    if unknown:
        raise BlackboardError(
            f"zone {name!r} declares kinds with no canonical core model: "
            f"{unknown}; add a KIND_MODELS entry, list them in UNTYPED_KINDS, "
            "or drop them"
        )
    if definition.authoritative_agent is not None and not isinstance(
        definition.authoritative_agent, AgentId
    ):
        raise BlackboardError(
            f"zone {name!r} authoritative_agent must be an AgentId, got "
            f"{type(definition.authoritative_agent).__name__}"
        )
    zone_def = Zone(
        name=name,
        purpose=definition.purpose,
        allowed_kinds=kinds,
        authoritative_agent=definition.authoritative_agent,
    )
    with _LOCK:
        existing = ZONES.get(name)
        if existing is not None:
            raise BlackboardError(
                f"zone {name!r} is already registered with kinds "
                f"{list(existing.allowed_kinds)}; zones are immutable once "
                "entries may reference them"
            )
        ZONES[name] = zone_def
        ZONE_NAMES = tuple(ZONES)
    return zone_def


#: The built-in zone set. Registered at import time, in this order, which is
#: also the pipeline order a reader of a trace will see.
_BUILT_IN_ZONES: tuple[Zone, ...] = (
    Zone(
        name="event",
        purpose=(
            "The event profile every other entry is keyed by. Exactly one "
            "event_profile entry defines the event_id; downstream entries cite "
            "it so an audit can always answer 'which event was this run about'."
        ),
        allowed_kinds=("event_profile",),
        authoritative_agent=None,
    ),
    Zone(
        name="opportunities",
        purpose=(
            "A1's evidence-backed sponsor candidates. Owned by A1: only the agent "
            "that actually ran discovery is expected to assert that a lead exists."
        ),
        allowed_kinds=("brand_lead",),
        authoritative_agent=AgentId.A1_DISCOVERY,
    ),
    Zone(
        name="offers",
        purpose=(
            "A2's priced proposals. Owned by A2. Every revision is a new entry, "
            "never an overwrite, so the price history a sponsor negotiated "
            "against survives the run."
        ),
        allowed_kinds=("offer",),
        authoritative_agent=AgentId.A2_PRICING,
    ),
    Zone(
        name="threads",
        purpose=(
            "A3's outreach conversations with sponsors. Owned by A3, because the "
            "thread is the record of what was actually said, and two entries "
            "claiming different messages to the same brand is a defect."
        ),
        allowed_kinds=("thread",),
        authoritative_agent=AgentId.A3_OUTREACH,
    ),
    Zone(
        name="contracts",
        purpose=(
            "A4's memoranda of understanding. Owned by A4. Contract terms are a "
            "fact of record, so they get one owner and an immutable version chain."
        ),
        allowed_kinds=("mou",),
        authoritative_agent=AgentId.A4_CONTRACT,
    ),
    Zone(
        name="risk_flags",
        purpose=(
            "Compliance objections, with A5 as the declared owner. A BLOCKING "
            "flag carries veto authority over the pipeline, so A5's flags are the "
            "ones that bind. Other agents may raise flags too -- A6 does when it "
            "catches a compliance-summary mismatch -- and any such post is "
            "reported by stats() as a foreign author rather than being hidden."
        ),
        allowed_kinds=("risk_flag",),
        authoritative_agent=AgentId.A5_COMPLIANCE,
    ),
    Zone(
        name="audit",
        purpose=(
            "Post-event audit: A6's ROI report, plus the per-promise findings and "
            "the compliance summary that feed it. Owned by A6; A5 writes its own "
            "findings here because a compliance verdict is audit evidence, and "
            "those posts are reported by stats() too."
        ),
        allowed_kinds=("roi_report", "audit_finding", "compliance_summary"),
        authoritative_agent=AgentId.A6_AUDIT,
    ),
    Zone(
        name="disputes",
        purpose=(
            "Schema-validated disagreements between two reasoning agents. Open to "
            "every agent by design: a dispute is authored by the disagreeing "
            "parties, so no single owner could hold it. The Arbitration procedure "
            "of Hayes-Roth (1985) begins here."
        ),
        allowed_kinds=("dispute",),
        authoritative_agent=None,
    ),
    Zone(
        name="gates",
        purpose=(
            "Human-in-the-loop gates raised by the orchestration layer: the gate "
            "itself, the human's answer, and the approval row that authorises one "
            "specific side effect. Deliberately a SEPARATE zone from 'approvals', "
            "which is where the revenue agents (A3/A4) record the approvals they "
            "consume. Keeping the two apart is what makes it possible to ask 'was "
            "this action approved, and by whom?' without conflating a request for "
            "approval with a grant of it -- and an approval names exactly one gate "
            "kind, so a send approval can never authorise an MoU."
        ),
        allowed_kinds=("human_gate", "human_decision", "approval"),
        authoritative_agent=None,
    ),
    Zone(
        name="debate",
        purpose=(
            "Multi-agent debate transcript (Du et al., ICML 2024): the dispute "
            "each side opens, its rebuttal, any concession, and the arbitration "
            "outcome. A transcript rather than a verdict, because the point of the "
            "mechanism is that the reasoning survives."
        ),
        allowed_kinds=("dispute", "rebuttal", "concession", "adjudication",
                       "debate_summary"),
        authoritative_agent=None,
    ),
    Zone(
        name="runs",
        purpose=(
            "One entry per completed run: the summary a reader would want first "
            "-- outcome, counts, and which backends actually answered. Written by "
            "the graph at finalisation, so a run is inspectable without replaying "
            "its trace."
        ),
        allowed_kinds=("run_summary",),
        authoritative_agent=None,
    ),
    Zone(
        name="tasks",
        purpose=(
            "Contract Net Protocol task lifecycle (Smith 1980): announcement -> "
            "bidding -> awarding -> expediting. Owned by the orchestrator, not by "
            "any worker, because the announcement and the award are supervisor "
            "acts while the bid is the agent's. Separate from 'bids' -- which "
            "holds the raw bid records -- so the negotiation transcript stays "
            "readable as a lifecycle rather than a flat pile of bids."
        ),
        # ``task_announcement`` is the name graph.contract_net posts under. The
        # bare ``announcement`` spelling is accepted as an alias so a zone table
        # written from the Smith lifecycle diagram also works; the prefixed name
        # is the one the code uses.
        allowed_kinds=("task_announcement", "announcement", "bid", "award", "expedite"),
        authoritative_agent=None,
    ),
    Zone(
        name="bids",
        purpose=(
            "Contract Net (Smith 1980) bids against an announced task, plus the "
            "costs a bidder estimates. Open to all agents: bidding is competitive "
            "by construction and a single owner would defeat it."
        ),
        allowed_kinds=("bid",),
        authoritative_agent=None,
    ),
    Zone(
        name="decisions",
        purpose=(
            "A7's arbitration rulings and escalations. Owned by A7: the arbiter's "
            "outcome is the resolution of record, and its justification is the "
            "reference chain pointing back at the dispute."
        ),
        allowed_kinds=("arbitration_decision", "escalation"),
        authoritative_agent=AgentId.A7_ARBITER,
    ),
    Zone(
        name="lessons",
        purpose=(
            "Reflexion lessons (Shinn et al. 2023) written after a dispute "
            "resolves or an agent fails. Open to all: any agent may learn from "
            "any outcome, and no weights change -- the lesson is retrieved as "
            "context on the next cycle."
        ),
        allowed_kinds=("lesson",),
        authoritative_agent=None,
    ),
    Zone(
        name="approvals",
        purpose=(
            "Human gate decisions: the gate A7 raised, the answer a human gave, "
            "and the durable Approval row proving the gate was honoured before a "
            "side effect. Open: the author is the agent that obtained the "
            "approval, and an auto-resolved gate is still recorded so it can "
            "never be mistaken for a human decision."
        ),
        allowed_kinds=("human_gate", "approval", "human_decision"),
        authoritative_agent=None,
    ),
    Zone(
        name="handoffs",
        purpose=(
            "The routing record: who handed control to whom, why, with what "
            "decision source and confidence, and which board entries the "
            "receiving agent should read. Open: a handoff is authored by the "
            "agent handing over."
        ),
        allowed_kinds=("handoff",),
        authoritative_agent=None,
    ),
    Zone(
        name="messages",
        purpose=(
            "Agent-to-agent messages: who sent what to whom, with what "
            "decision source and confidence, and which board entries the "
            "recipient should read. Open: a message is authored by the "
            "agent sending it."
        ),
        allowed_kinds=("message",),
        authoritative_agent=None,
    ),
)

#: The built-in zones as registered, in registration order.
REGISTERED_ZONES: tuple[Zone, ...] = tuple(
    register_zone(_definition) for _definition in _BUILT_IN_ZONES
)


def zone(name: str) -> Zone:
    """Look up a zone by name, or raise :class:`~core.errors.ZoneNotFound`.

    ``ZoneNotFound`` rather than ``KeyError`` because an unknown zone is a
    programming fault in the agent that named it, and it should read as one.
    """
    key = _normalise(name)
    found = ZONES.get(key)
    if found is None:
        raise ZoneNotFound(
            f"unknown blackboard zone {name!r}; known zones: {list(ZONE_NAMES)}"
        )
    return found


def zones_all() -> list[Zone]:
    """Every registered :class:`Zone` definition, in registration order.

    Includes zones added after import via :func:`register_zone`, not just the
    built-in set in :data:`REGISTERED_ZONES`.
    """
    return [ZONES[name] for name in ZONE_NAMES]


def validate_post(zone_name: str, kind: str, author: AgentId) -> Zone:
    """Check that ``kind`` is postable into ``zone_name``; return the zone.

    Ownership is *not* enforced here -- see :class:`Zone` for why. The author is
    checked only for type, because ``BoardEntry.author`` is declared as an
    :class:`~core.schemas.AgentId` and anything else would put a non-agent in
    the audit trail.

    Raises
    ------
    :class:`~core.errors.ZoneNotFound`
        the zone does not exist.
    :class:`~core.errors.BlackboardError`
        the kind is empty or is not one the zone accepts, or the author is not
        an :class:`~core.schemas.AgentId`.
    """
    zone_def = zone(zone_name)
    normalised_kind = _normalise(kind)
    if not normalised_kind:
        raise BlackboardError(
            f"zone {zone_def.name!r} requires a non-empty kind; "
            f"allowed kinds: {list(zone_def.allowed_kinds)}"
        )
    if normalised_kind not in zone_def.allowed_kinds:
        raise BlackboardError(
            f"zone {zone_def.name!r} does not accept kind {kind!r}; allowed "
            f"kinds: {list(zone_def.allowed_kinds)}"
        )
    if not isinstance(author, AgentId):
        raise BlackboardError(
            f"author must be a core.schemas.AgentId, got "
            f"{type(author).__name__}: {author!r}"
        )
    return zone_def
