"""Typed domain artefacts for Paytriq.

Every object that crosses a module boundary is defined here. Nothing in this file
performs I/O, imports an SDK, or knows which model is behind it — that is what
lets eight modules be written in parallel against frozen contracts, and what makes
``core`` trivially testable.

The artefacts fall into six families:

1. **Domain**      — ``EventProfile``, ``BrandLead``, ``Offer``, ``Thread``, ``MoU``,
                     ``Deliverable``, ``ROIReport``
2. **Coordination** — ``Dispute``, ``Bid``, ``Lesson``, ``Handoff``, ``RiskFlag``
3. **Decision**    — ``DecisionRequest``, ``Decision`` (calibrated, sourced)
4. **Observability** — ``TraceEvent``, ``SpanRecord``
5. **Control**     — ``HumanGate``, ``HumanDecision``, ``Approval``
6. **Enums**       — ``AgentId``, ``Intent``, ``Severity``, ``RunMode``, ...

Two invariants are enforced here rather than trusted to callers:

* A ``Decision`` always names its ``source``. There is no way to represent a
  decision that does not say whether a model or a rule produced it.
* A ``TraceEvent`` is append-only and serialisable to one JSON object per line.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ids import utcnow

__all__ = [
    # enums
    "AgentId", "REASONING_AGENTS", "Intent", "DisputeZone", "Severity",
    "DisputeStatus", "DecisionSource", "RunMode", "QuestionType", "TraceKind",
    "SpanStatus", "GateKind", "GateOutcome", "ToolStatus", "MessageKind",
    # domain
    "EventProfile", "BrandLead", "Offer", "Thread", "MoU", "Deliverable",
    "ROIReport", "AuditFinding",
    # coordination
    "Dispute", "Bid", "Lesson", "Handoff", "RiskFlag", "Message",
    # decision
    "DecisionRequest", "Decision",
    # observability
    "TraceEvent", "SpanRecord", "TraceSummary",
    # control
    "HumanGate", "HumanDecision", "Approval",
]


# ============================================================================ enums
class AgentId(StrEnum):
    """The seven reasoning agents. The sponsor counterparty is NOT an agent.

    ``A7`` is the arbiter. ``ENVIRONMENT`` is reserved for the simulated
    counterparty so that trace records never present it as a reasoning agent.
    """

    A1_DISCOVERY = "A1"
    A2_PRICING = "A2"
    A3_OUTREACH = "A3"
    A4_CONTRACT = "A4"
    A5_COMPLIANCE = "A5"
    A6_AUDIT = "A6"
    A7_ARBITER = "A7"
    ENVIRONMENT = "ENV"          # simulated sponsor; never counted as an agent

    @property
    def is_reasoning_agent(self) -> bool:
        return self is not AgentId.ENVIRONMENT

    @property
    def label(self) -> str:
        return {
            AgentId.A1_DISCOVERY: "Discovery",
            AgentId.A2_PRICING: "Pricing",
            AgentId.A3_OUTREACH: "Outreach",
            AgentId.A4_CONTRACT: "Contract",
            AgentId.A5_COMPLIANCE: "Compliance",
            AgentId.A6_AUDIT: "Audit",
            AgentId.A7_ARBITER: "Arbiter",
            AgentId.ENVIRONMENT: "Sponsor (simulated)",
        }[self]


#: The seven reasoning agents, in pipeline order. Used for registration checks.
REASONING_AGENTS: tuple[AgentId, ...] = (
    AgentId.A1_DISCOVERY,
    AgentId.A2_PRICING,
    AgentId.A3_OUTREACH,
    AgentId.A4_CONTRACT,
    AgentId.A5_COMPLIANCE,
    AgentId.A6_AUDIT,
    AgentId.A7_ARBITER,
)


class Intent(StrEnum):
    """Classified intent of an inbound sponsor reply.

    Replaces the keyword-substring routers of the previous prototype. Note
    ``NEUTRAL`` exists deliberately: the old router defaulted every unrecognised
    message to ``interested``, which is how "yesterday we thought the price was
    too high" got routed to contract signing.
    """

    YES = "yes"
    PUSHBACK = "pushback"
    INTERESTED = "interested"
    NO = "no"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class DisputeZone(StrEnum):
    """Which part of the deal a dispute is about."""

    DISCOVERY = "discovery"
    PRICING = "pricing"
    OUTREACH = "outreach"
    CONTRACT = "contract"
    COMPLIANCE = "compliance"
    ROI = "roi"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    #: A5 compliance holds veto authority; a blocking flag stops the pipeline.
    BLOCKING = "blocking"


class DisputeStatus(StrEnum):
    OPEN = "open"
    DEBATED = "debated"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class DecisionSource(StrEnum):
    """Which subsystem produced a decision.

    Recorded on **every** decision without exception. ``RULES`` means a
    deterministic fallback ran; ``REPLAY`` means a recorded output was replayed
    because no model was reachable. Both are legitimate in a demo; what is not
    legitimate is a decision whose source cannot be named.
    """

    CLEF = "clef"          # decision model (Ollama local or Workers AI)
    GEMINI = "gemini"      # generative model, structured output
    RULES = "rules"        # deterministic fallback
    REPLAY = "replay"      # recorded output replayed offline


class RunMode(StrEnum):
    """How the current process is producing results."""

    LIVE = "live"          # real model calls
    REPLAY = "replay"      # recorded transcript, no network
    OFFLINE = "offline"    # rules only, no models at all


class QuestionType(StrEnum):
    """Decision-model question types (Clef / Jev System One vocabulary)."""

    NOUL = "noul"          # yes/no with calibrated probability
    CHOICE = "choice"      # pick one of a closed option set
    SCORE = "score"        # ordered rubric


class TraceKind(StrEnum):
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    DECISION = "decision"
    HANDOFF = "handoff"
    MESSAGE = "message"
    HUMAN = "human"
    ERROR = "error"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    FALLBACK = "fallback"   # a real failure was caught and a fallback used


class ToolStatus(StrEnum):
    OK = "ok"
    CACHED = "cached"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class GateKind(StrEnum):
    SEND = "send"            # before any outbound email
    COUNTER = "counter"      # before accepting/replying to a counter-offer
    MOU = "mou"              # before releasing an MoU
    ESCALATION = "escalation"  # arbiter could not resolve


class GateOutcome(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"


class MessageKind(StrEnum):
    """Kind of a first-class inter-agent message."""

    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    COUNTER_PROPOSAL = "counter_proposal"
    REVISION_REQUEST = "revision_request"
    VERDICT = "verdict"


# =========================================================================== domain
class _Base(BaseModel):
    """Shared config: strict-ish, immutable-by-convention, no silent coercion."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class EventProfile(_Base):
    """The student event being sponsored. The system's real input."""

    event_id: str
    name: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=200)
    footfall: int = Field(ge=0, le=10_000_000)
    date: str = Field(min_length=4, max_length=40)
    audience: str = Field(min_length=1, max_length=500)
    budget_inr: float | None = Field(default=None, ge=0)
    categories_wanted: list[str] = Field(default_factory=list)
    deliverables_offered: list[str] = Field(default_factory=list)
    contact_email: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("categories_wanted", "deliverables_offered", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """Accept ``"a,b,c"`` as well as ``["a","b","c"]`` from a UI form."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("contact_email")
    @classmethod
    def _email_shape(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError(f"not a plausible email address: {v!r}")
        return v


class BrandLead(_Base):
    """A candidate sponsor discovered by A1. Evidence-backed, not invented."""

    lead_id: str
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    distance_km: float = Field(ge=0)
    contact_email: str | None = None
    phone: str | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    fit_score: float = Field(default=0.0, ge=0, le=100)
    fit_breakdown: dict[str, float] = Field(default_factory=dict)
    fit_rationale: str = ""
    #: Where this lead actually came from. ``rules`` means a seeded fixture.
    source: str = "unknown"
    source_url: str | None = None
    discovered_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _breakdown_sums(self) -> BrandLead:
        if self.fit_breakdown:
            total = round(sum(self.fit_breakdown.values()), 1)
            if abs(total - self.fit_score) > 1.0:
                # Not fatal, but it is the exact inconsistency that made the
                # previous prototype's artifacts unverifiable. Surface it.
                object.__setattr__(self, "fit_rationale",
                                   (self.fit_rationale + f" [breakdown sum={total} "
                                    f"vs fit_score={self.fit_score}]").strip())
        return self


class Offer(_Base):
    """A priced proposal from A2. ``version`` increments on every revision."""

    offer_id: str
    event_id: str
    brand: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    amount_inr: float = Field(ge=0)
    deliverables: list[str] = Field(min_length=1)
    pitch: str = ""
    fit_score: float = Field(default=0.0, ge=0, le=100)
    version: int = Field(default=1, ge=1)
    revised: bool = False
    revision_note: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class Thread(_Base):
    """One outreach conversation with one sponsor."""

    thread_id: str
    event_id: str
    brand: str
    email: str | None = None
    status: Literal["draft", "pending_approval", "sent", "replied",
                    "negotiating", "closed_won", "closed_lost"] = "draft"
    day: int = Field(default=0, ge=0)
    intent: Intent = Intent.UNKNOWN
    reply_text: str = ""
    offer_id: str | None = None
    sent_at: datetime | None = None
    #: True only when a real transport accepted the message.
    delivered: bool = False


class Deliverable(_Base):
    """A promise in the MoU plus its fulfilment evidence."""

    promise: str = Field(min_length=1)
    status: Literal["pending", "fulfilled", "waived", "breached"] = "pending"
    evidence_url: str | None = None
    evidence_note: str = ""
    verified_by: AgentId | None = None
    verified_at: datetime | None = None


class MoU(_Base):
    """Memorandum of understanding produced by A4."""

    mou_id: str
    event_id: str
    brand: str
    amount_inr: float = Field(ge=0)
    terms: str = Field(min_length=1)
    deliverables: list[str] = Field(default_factory=list)
    status: Literal["draft", "pending_approval", "approved", "signed", "rejected"] = "draft"
    document_path: str | None = None
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)


class AuditFinding(_Base):
    """One compliance observation from A5."""

    promise: str
    fulfilled: bool
    evidence_url: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    source: DecisionSource = DecisionSource.RULES
    note: str = ""


class ROIReport(_Base):
    """A6's audit. Every figure carries its assumption."""

    report_id: str
    event_id: str
    total_sponsored_inr: float = Field(default=0.0, ge=0)
    footfall: int = Field(default=0, ge=0)
    estimated_leads: int = Field(default=0, ge=0)
    spend_inr: float = Field(default=0.0, ge=0)
    pipeline_value_inr: float = Field(default=0.0, ge=0)
    roi_multiple: float = Field(default=0.0)
    assumptions: list[str] = Field(default_factory=list, validate_default=True)
    findings: list[AuditFinding] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=utcnow)

    @field_validator("assumptions")
    @classmethod
    def _must_document(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "ROIReport.assumptions must be non-empty: every derived figure "
                "needs a stated basis. An undocumented ROI is not a result."
            )
        return v


# ====================================================================== coordination
class RiskFlag(_Base):
    """A5's objection. A BLOCKING flag holds veto authority."""

    flag_id: str
    event_id: str
    brand: str = Field(min_length=1)
    severity: Severity = Severity.MEDIUM
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    raised_by: AgentId = AgentId.A5_COMPLIANCE
    raised_at: datetime = Field(default_factory=utcnow)
    resolved: bool = False
    resolution: str = ""


class Dispute(_Base):
    """A first-class, schema-validated disagreement between two agents.

    This is the central artefact of the coordination layer. Disagreement is
    *authored by the agents* and carries evidence references — it is never
    inferred from string matching on free text.
    """

    dispute_id: str
    event_id: str
    zone: DisputeZone
    claimant: AgentId
    opponent: AgentId
    position: str = Field(min_length=1)
    counter_position: str = Field(min_length=1)
    #: References to blackboard entries or risk-flag ids. Not free-form opinion.
    evidence: list[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    status: DisputeStatus = DisputeStatus.OPEN
    rounds: int = Field(default=0, ge=0)
    resolution: str = ""
    resolved_by: AgentId | None = None
    created_at: datetime = Field(default_factory=utcnow)
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def _distinct_parties(self) -> Dispute:
        if self.claimant == self.opponent:
            raise ValueError("a dispute requires two distinct agents")
        if not self.claimant.is_reasoning_agent or not self.opponent.is_reasoning_agent:
            raise ValueError("disputes are between reasoning agents only")
        return self

    @property
    def is_blocking(self) -> bool:
        return self.severity is Severity.BLOCKING and self.status is not DisputeStatus.RESOLVED


class Bid(_Base):
    """A Contract Net bid (Smith 1980, announcement -> bid -> award)."""

    bid_id: str
    task_id: str
    agent: AgentId
    feasibility: float = Field(ge=0, le=1)
    expected_value: float
    effort: int = Field(ge=1)
    rationale: str = Field(min_length=1)
    submitted_at: datetime = Field(default_factory=utcnow)

    @property
    def utility(self) -> float:
        """Award score. Feasibility-weighted value per unit of effort."""
        if self.effort <= 0:
            return 0.0
        return round(self.expected_value * self.feasibility / self.effort, 6)


class Lesson(_Base):
    """A Reflexion-style verbal lesson (Shinn et al. 2023).

    Stored on the blackboard after a dispute resolves. No weights change; the
    next cycle retrieves the lesson as context, so behaviour changes observably
    between runs on identical input.
    """

    lesson_id: str
    event_id: str
    author: AgentId
    trigger_dispute_id: str | None = None
    trigger: str = Field(min_length=1)
    correction: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    created_at: datetime = Field(default_factory=utcnow)


class Handoff(_Base):
    """One agent transferring control to another. The core coordination record.

    ``decision_source`` and ``confidence`` are mandatory: every routing decision
    names what made it. That single field is what turns "the agent decided" from
    an assertion into evidence.
    """

    handoff_id: str
    event_id: str
    run_id: str
    from_agent: AgentId
    to_agent: AgentId
    reason: str = Field(min_length=1)
    decision_source: DecisionSource
    confidence: float = Field(ge=0, le=1)
    summary: str = ""
    #: Blackboard zone ids the receiving agent should read.
    payload_refs: list[str] = Field(default_factory=list)
    at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _not_self(self) -> Handoff:
        if self.from_agent == self.to_agent:
            raise ValueError("a handoff must change the active agent")
        return self


class Message(_Base):
    """One agent addressing another. The core deliberation record.

    Mirrors :class:`Handoff`: ``decision_source`` and ``confidence`` are
    mandatory so every message names what produced it.
    """

    message_id: str
    event_id: str
    run_id: str
    from_agent: AgentId
    to_agent: AgentId
    kind: MessageKind
    body: str = Field(min_length=1)
    refs: list[str] = Field(default_factory=list)
    decision_source: DecisionSource
    confidence: float = Field(ge=0, le=1)
    summary: str = ""
    at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _not_self(self) -> Message:
        if self.from_agent == self.to_agent:
            raise ValueError("a message must change the active agent")
        return self


# ======================================================================== decision
class DecisionRequest(_Base):
    """A question put to the decision layer.

    Mirrors the Clef / Jev System One shape so the same object can be sent to a
    local llama.cpp server, Cloudflare Workers AI, or a Gemini structured-output
    call without translation at the call site.

    ``options`` accepts either form:

    * a plain list — ``["yes", "no"]``
    * a criteria map — ``{"yes": "accepts the offer", "no": "declines"}``

    System One's ``choice`` type takes the *map*, because the model needs to know
    what each option means. Passing the map populates ``option_criteria`` too, so
    a caller can always recover the descriptions.
    """

    request_id: str
    question: str = Field(min_length=1)
    question_type: QuestionType = QuestionType.CHOICE
    state: dict[str, Any] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list)
    #: option -> description, as System One's ``criteria`` map expects.
    option_criteria: dict[str, str] = Field(default_factory=dict)
    rubric: list[str] = Field(default_factory=list)
    instructions: str = ""
    asked_by: AgentId | None = None
    #: Where this decision steers the run. Recorded in the trace for the ablation.
    decision_point: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_criteria_map(cls, data: Any) -> Any:
        """Allow ``options={"yes": "accepts", ...}`` as well as ``["yes", ...]``."""
        if not isinstance(data, dict):
            return data
        opts = data.get("options")
        if isinstance(opts, dict):
            data = dict(data)
            data["option_criteria"] = {str(k): str(v) for k, v in opts.items()}
            data["options"] = [str(k) for k in opts]
        return data

    @property
    def option_map(self) -> dict[str, str]:
        """The System One ``criteria`` map, descriptions defaulting to the label."""
        return {o: self.option_criteria.get(o, o) for o in self.options}

    @model_validator(mode="after")
    def _type_needs_shape(self) -> DecisionRequest:
        if self.question_type is QuestionType.CHOICE and len(self.options) < 2:
            raise ValueError("a choice question needs at least two options")
        if self.question_type is QuestionType.SCORE and len(self.rubric) < 2:
            raise ValueError("a score question needs a rubric of at least two levels")
        stray = set(self.option_criteria) - set(self.options)
        if stray:
            raise ValueError(
                f"option_criteria has keys absent from options: {sorted(stray)}"
            )
        return self


class Decision(_Base):
    """A calibrated decision. Always sourced. Never anonymous."""

    request_id: str
    question: str
    choice: str | None = None
    #: Normalised, sums to 1.0 within tolerance.
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    source: DecisionSource
    model: str = "unknown"
    latency_ms: float = Field(default=0.0, ge=0)
    #: False when a backend failed and a fallback answered instead.
    degraded: bool = False
    raw: dict[str, Any] | None = None
    at: datetime = Field(default_factory=utcnow)

    @field_validator("probabilities")
    @classmethod
    def _unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for k, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probability for {k!r} out of range: {p}")
        if v and abs(sum(v.values()) - 1.0) > 0.02:
            raise ValueError(f"probabilities must sum to 1.0, got {sum(v.values()):.4f}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> Decision:
        if self.choice is not None and self.probabilities:
            if self.choice not in self.probabilities:
                raise ValueError(
                    f"choice {self.choice!r} is absent from probabilities "
                    f"{sorted(self.probabilities)}"
                )
            if self.probabilities[self.choice] + 1e-6 < self.confidence:
                raise ValueError(
                    "confidence exceeds the chosen option's probability; "
                    "confidence must be the probability of the choice taken"
                )
        return self

    @property
    def needs_escalation(self) -> bool:
        return self.degraded or self.confidence < 0.5


# ==================================================================== observability
class TraceEvent(_Base):
    """One line of the execution trace. Exactly one JSON object per event.

    The trace is the primary evidence for architecture marks, so it carries the
    full decision context: who acted, what decided it, how confident the model
    was, and how long it took.
    """

    seq: int = Field(ge=0)
    run_id: str
    event_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: TraceKind
    name: str = Field(min_length=1)
    agent: AgentId | None = None
    ts: datetime = Field(default_factory=utcnow)
    duration_ms: float | None = Field(default=None, ge=0)
    status: SpanStatus = SpanStatus.OK
    #: Free-form but JSON-safe: model ids, probabilities, URLs, counts.
    attributes: dict[str, Any] = Field(default_factory=dict)

    def to_jsonl(self) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)


class SpanRecord(_Base):
    """A parent/child span, mirroring the OpenTelemetry GenAI span model."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: TraceKind
    agent: AgentId | None = None
    start_unix_nano: int
    end_unix_nano: int
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round((self.end_unix_nano - self.start_unix_nano) / 1e6, 3)


class TraceSummary(_Base):
    """Derived statistics proving a trace is a genuine capture.

    ``distinct_gap_values`` is the important one. A real run has irregular
    timing; a hand-written trace has a constant interval and therefore exactly
    one distinct gap value. Publishing this number makes fabrication detectable
    by anyone who can read the file.
    """

    run_id: str
    trace_file: str
    git_commit: str = "unknown"
    code_sha256: str = "unknown"
    event_count: int = 0
    span_count: int = 0
    root_span_count: int = 0
    handoff_count: int = 0
    message_count: int = 0
    decision_counts: dict[str, int] = Field(default_factory=dict)
    degraded_decision_count: int = 0
    wall_clock_ms: float = 0.0
    consecutive_start_gaps_ms: list[float] = Field(default_factory=list)
    distinct_gap_values: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_status_counts: dict[str, int] = Field(default_factory=dict)
    human_gate_count: int = 0
    generated_at: datetime = Field(default_factory=utcnow)


# =========================================================================== control
class HumanGate(_Base):
    """A side-effecting action awaiting approval.

    Gates exist for exactly three actions: sending mail, replying to a
    counter-offer, and releasing an MoU. There is no code path that performs
    any of these without passing a gate first.
    """

    gate_id: str
    kind: GateKind
    event_id: str
    run_id: str
    question: str = Field(min_length=1)
    payload_preview: str = ""
    options: list[GateOutcome] = Field(
        default_factory=lambda: [GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE]
    )
    raised_at: datetime = Field(default_factory=utcnow)


class HumanDecision(_Base):
    """A human's answer to a gate. Recorded permanently in the trace."""

    gate_id: str
    kind: GateKind
    outcome: GateOutcome
    decided_by: str = Field(min_length=1)
    instruction: str = ""
    at: datetime = Field(default_factory=utcnow)


class Approval(_Base):
    """Durable audit row proving a gate was honoured before a side effect."""

    approval_id: str
    gate_id: str
    kind: GateKind
    outcome: GateOutcome
    decided_by: str
    action_taken: str
    at: datetime = Field(default_factory=utcnow)
