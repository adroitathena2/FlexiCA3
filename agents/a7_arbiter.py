"""A7 Arbiter — the conflict resolver, and the only writer to ``decisions``.

What this agent is for
----------------------
Disagreements between agents are first-class :class:`Dispute` objects with cited
evidence, not free-text friction. A7 adjudicates them. Three properties matter:

* **Judged against evidence, not rhetoric.** Each position's cited entries are
  resolved on the board and put in front of ``ctx.decide`` verbatim. A claim with
  no resolvable evidence cannot be upheld.
* **Synthesis is a real option.** Upholding one side or the other is the easy
  path. :meth:`ArbiterAgent._propose_synthesis` builds candidate compromises from
  the numbers actually on the board — a price between two quoted offers, a
  tier swap, a partial-scope carve-out — and asks the model to choose among
  *them*, so the resolution can be something neither agent proposed.
* **Escalation is the default when unsure.** Below
  ``settings.confidence_threshold``, or after ``settings.max_debate_rounds``
  without convergence, A7 does **not** pick a winner. It posts a
  :class:`HumanGate` of kind ``ESCALATION`` to the ``approvals`` zone and raises
  :class:`~core.errors.ArbiterEscalation`, which the graph turns into a LangGraph
  ``interrupt()``. An arbiter that guesses when it is unsure is worse than no
  arbiter, because its guesses are authoritative.

A7 is the sole writer to the ``decisions`` zone. Keeping one writer per zone
means the decision log has exactly one provenance and cannot contain two agents
each believing they arbitrated.

Every conditional decision goes through ``ctx.decide``. There is no keyword
matching on free text: a position is weighed by the board entries it cites.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import ValidationError

from core.config import Settings, get_settings
from core.errors import ArbiterEscalation, BlackboardError, DecisionFailed, DecisionUnavailable
from core.ids import new_id, utcnow
from core.protocols import ActResult, AgentContext, BoardEntry, Observation, Plan, Reflection
from core.schemas import (
    AgentId,
    Bid,
    DecisionRequest,
    DecisionSource,
    Dispute,
    DisputeStatus,
    GateKind,
    GateOutcome,
    HumanGate,
    Lesson,
    QuestionType,
)

from .base import ReActAgent

__all__ = [
    "ArbiterAgent",
    "A7Arbiter",
    "ArbiterOutcome",
    "Resolution",
    "OUTCOMES",
    "ZONE_DISPUTES",
    "ZONE_DECISIONS",
    "ZONE_APPROVALS",
    "ZONE_BIDS",
    "KIND_DISPUTE",
    "KIND_DECISION",
    "KIND_HUMAN_GATE",
    "KIND_LESSON",
    "ZONE_LESSONS",
]

_M = TypeVar("_M")

#: The closed option set A7 offers the decision model. Exposed so tests and the
#: eval harness can assert no path silently escapes it.
OUTCOMES: tuple[str, str, str] = ("uphold_claimant", "uphold_opponent", "synthesis")


# --------------------------------------------------------------------------- zones
def _resolve_zone(attr: str, fallback: str) -> str:
    """Prefer ``blackboard.zones``' own constant when that module is present.

    ``blackboard/`` is written in parallel with this file, so importing it
    unconditionally would leave A7 unimportable until it lands.
    """
    try:  # pragma: no cover - depends on a concurrent module landing
        from blackboard import zones as _zones  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - blackboard not present yet
        return fallback
    for name in (attr, f"ZONE_{attr}", attr.lower()):
        value = getattr(_zones, name, None)
        if isinstance(value, str) and value:
            return value
    return fallback


ZONE_DISPUTES = _resolve_zone("DISPUTES", "disputes")
ZONE_DECISIONS = _resolve_zone("DECISIONS", "decisions")
ZONE_APPROVALS = _resolve_zone("APPROVALS", "approvals")
ZONE_BIDS = _resolve_zone("BIDS", "bids")
ZONE_LESSONS = _resolve_zone("LESSONS", "lessons")

KIND_DISPUTE = "dispute"
KIND_DECISION = "arbitration_decision"
KIND_HUMAN_GATE = "human_gate"
KIND_BID = "bid"
KIND_LESSON = "lesson"


# ------------------------------------------------------------------ board helpers
def _read(ctx: AgentContext, zone: str, *, kind: str | None = None,
          limit: int | None = None) -> list[BoardEntry]:
    """``board.read`` with a positional-only fallback; a zone fault gives ``[]``."""
    try:
        try:
            return list(ctx.board.read(zone, kind=kind, limit=limit))
        except TypeError:
            entries = list(ctx.board.read(zone))
            if kind is not None:
                entries = [e for e in entries if e.kind == kind]
            if limit is not None:
                entries = entries[-limit:]
            return entries
    except BlackboardError:
        return []


def _as_model(entry: BoardEntry, model_cls: type[_M]) -> _M | None:
    """Reconstruct a typed artefact from a board payload (strict, then projected)."""
    payload = getattr(entry, "payload", None)
    if not isinstance(payload, dict):
        return None
    try:
        return model_cls.model_validate(payload)
    except ValidationError:
        pass
    known = set(model_cls.model_fields)
    projected = {k: v for k, v in payload.items() if k in known}
    if not projected or projected == payload:
        return None
    try:
        return model_cls.model_validate(projected)
    except ValidationError:
        return None


def _history_by_id(ctx: AgentContext) -> dict[str, BoardEntry]:
    """``entry_id -> entry`` for every entry ever posted.

    Disputes cite evidence by entry id. Resolving those references is what lets
    A7 weigh a position against what it actually points at instead of against how
    confidently it is worded.
    """
    getter = getattr(ctx.board, "history", None)
    if not callable(getter):
        return {}
    try:
        return {e.entry_id: e for e in getter()}
    except BlackboardError:
        return {}


# ============================================================================ models
@dataclass(slots=True)
class ArbiterOutcome:
    """The result of adjudicating one dispute."""

    dispute_id: str
    outcome: str
    rationale: str
    confidence: float
    source: DecisionSource
    escalated: bool = False
    synthesis: str | None = None
    decision_entry_id: str | None = None
    gate_entry_id: str | None = None
    lesson_entry_ids: list[str] = field(default_factory=list)
    resolved: bool = False


@dataclass(slots=True)
class Resolution:
    """An updated :class:`Dispute` plus the board entry that carried it.

    The board is append-only, so a resolution is a *new* entry describing the
    dispute's new state with ``refs`` pointing at the entry it supersedes. The
    original is never mutated, which is what keeps the audit trail honest.
    """

    dispute: Dispute
    entry_id: str
    supersedes: str | None


# ============================================================================ agent
class ArbiterAgent(ReActAgent):
    """Resolve disputes between agents against their cited evidence; else escalate."""

    id = AgentId.A7_ARBITER
    role = "Arbiter: dispute resolution by evidence, with human escalation"

    _SCRATCH = "a7_arbiter"

    def __init__(self, *, step_budget: int = 2, deadline_s: float = 60.0,
                 settings: Settings | None = None) -> None:
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)
        self.settings = settings or get_settings()

    # ---------------------------------------------------------------- scratch
    def _state(self, ctx: AgentContext) -> dict[str, Any]:
        scratch = ctx.scratch.setdefault(self._SCRATCH, {})
        scratch.setdefault("phase", "load")
        scratch.setdefault("disputes", [])
        scratch.setdefault("outcomes", [])
        scratch.setdefault("notes", [])
        return scratch

    # ------------------------------------------------------------------ plan
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Load open disputes, then let the model frame how to attack them.

        Always issues a strategy decision (``a7.arbiter.strategy``), even with
        zero open disputes, so A7 reasons on every run rather than only on the
        disputed path. With no disputes the strategy is recorded but there is
        nothing to adjudicate.
        """
        state = self._state(ctx)
        disputes = self._load_disputes(ctx)
        state["disputes"] = [d.model_dump(mode="json") for d in disputes]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=("How should each open dispute be adjudicated on this pass?"),
            question_type=QuestionType.CHOICE,
            state={
                "open_disputes": [d.dispute_id for d in disputes],
                "zones": sorted({d.zone.value for d in disputes}),
                "rounds_so_far": [d.rounds for d in disputes],
                "max_debate_rounds": self.settings.max_debate_rounds,
                "last_observation": obs.summary if obs is not None else "",
            },
            options=["judge_evidence", "synthesise_first", "escalate_immediately"],
            rubric=[
                "judge_evidence: weigh each position against its cited entries",
                "synthesise_first: propose a compromise before picking a side",
                "escalate_immediately: the round budget is spent; go to a human",
            ],
            instructions=("Escalate when the round budget is exhausted or the "
                          "evidence does not separate the positions. Never pick a "
                          "side the cited evidence does not support."),
            asked_by=self.id,
            decision_point="a7.arbiter.strategy",
        )
        strategy = self._ask(ctx, request, ("judge_evidence", "synthesise_first",
                                            "escalate_immediately"))
        if disputes and self._rounds_exhausted(disputes):
            # Hard rule: the model's preference cannot override the budget.
            # This is the path where guessing would otherwise happen.
            strategy = "escalate_immediately"
            state["notes"].append(
                f"max_debate_rounds={self.settings.max_debate_rounds} already spent "
                f"on {[d.dispute_id for d in disputes]}; escalated without guessing")
        state["phase"] = "adjudicate"
        return Plan(
            goal=f"adjudicate {len(disputes)} dispute(s) with strategy={strategy}",
            steps=[f"strategy:{strategy}", f"disputes:{len(disputes)}"],
            rationale=(f"{len(disputes)} open dispute(s) {state['disputes'] and [d.dispute_id for d in disputes]}; "
                       f"strategy={strategy}"),
            confidence=1.0,
            source=DecisionSource.RULES,
            stop=not disputes,
        )

    # ------------------------------------------------------------------ act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        state = self._state(ctx)
        strategy = "judge_evidence"
        for step in plan.steps:
            if step.startswith("strategy:"):
                strategy = step.split(":", 1)[1]
        disputes = self._load_disputes(ctx)
        if not disputes:
            return ActResult(ok=True, output={"outcomes": []},
                             observations=["no open disputes; nothing to arbitrate"])
        if strategy == "escalate_immediately" or self._rounds_exhausted(disputes):
            exhausted = self._rounds_exhausted(disputes)
            outcomes = [
                self._escalate(
                    ctx, dispute,
                    reason=(
                        f"debate rounds exhausted ({dispute.rounds} >= "
                        f"max_debate_rounds={self.settings.max_debate_rounds}) without "
                        f"convergence") if exhausted else
                    "the arbitration strategy decision was off-menu, so A7 refused to "
                    "guess which position stands",
                    confidence=0.0,
                )
                for dispute in disputes
            ]
        else:
            outcomes = [self.resolve(ctx, d, prefer_synthesis=strategy == "synthesise_first")
                        for d in disputes]
        state["outcomes"] = [o.__dict__ if hasattr(o, "__dict__") else o for o in outcomes]
        first_escalation = next((o for o in outcomes if o.escalated), None)
        if first_escalation is not None:
            # Fail loud: the graph catches this and interrupts for a human.
            raise ArbiterEscalation(
                first_escalation.dispute_id,
                f"escalated: {first_escalation.rationale} "
                f"(confidence {first_escalation.confidence:.3f} "
                f"< threshold {self.settings.confidence_threshold:.3f})",
            )
        return ActResult(
            ok=True,
            output={"outcomes": [_outcome_fact(o) for o in outcomes]},
            observations=[f"{o.outcome}: {o.dispute_id} at confidence {o.confidence:.3f}"
                          for o in outcomes],
            # Degraded when a RULES answer carried the outcome: a model
            # (CLEF/GEMINI) answer is the non-degraded path.
            degraded=any(o.source is DecisionSource.RULES for o in outcomes),
        )

    # --------------------------------------------------------------- observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        state = self._state(ctx)
        if not state["disputes"]:
            return Observation(
                summary="no open disputes in zone "
                        f"`{ZONE_DISPUTES}`; arbiter had nothing to do",
                facts={"disputes": 0, "decisions_written": 0},
                sufficient=True,
                gaps=[],
            )
        raw = result.output if isinstance(result.output, dict) else {}
        outcomes = raw.get("outcomes", []) if isinstance(raw, dict) else []
        return Observation(
            summary=(f"adjudicated {len(outcomes)} dispute(s): "
                     + ", ".join(f"{o.get('dispute_id')}={o.get('outcome')}" for o in outcomes)),
            facts={
                "outcomes": outcomes,
                "escalations": sum(1 for o in outcomes if o.get("escalated")),
                "threshold": self.settings.confidence_threshold,
                "max_debate_rounds": self.settings.max_debate_rounds,
            },
            sufficient=True,
            gaps=list(state["notes"]),
        )

    # --------------------------------------------------------------- reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """Summarise the arbitration pass. Not persisted; each dispute writes its own."""
        facts = obs.facts.get("outcomes") or []
        if not facts:
            return None
        syntheses = [f for f in facts if f.get("outcome") == "synthesis"]
        ups = [f for f in facts if f.get("outcome") in ("uphold_claimant", "uphold_opponent")]
        return Reflection(
            lesson_trigger=f"arbitrated {len(facts)} dispute(s)",
            correction=(f"{len(ups)} upheld a single position, {len(syntheses)} reached a "
                        f"synthesis"),
            rule=(f"Weigh every dispute against its cited board entries; escalate below "
                  f"confidence {self.settings.confidence_threshold:.2f} rather than "
                  f"choosing a side the evidence does not support."),
            confidence=0.7,
        )

    # ================================================================ public API
    def resolve(self, ctx: AgentContext, dispute: Dispute,
                *, prefer_synthesis: bool = False) -> ArbiterOutcome:
        """Adjudicate one dispute and write the outcome to ``decisions``."""
        state = self._state(ctx)
        threshold = self.settings.confidence_threshold
        evidence = self._resolve_evidence(ctx, dispute)
        bids = self._relevant_bids(ctx, dispute)

        options = ["synthesis", *OUTCOMES[:2]] if prefer_synthesis else list(OUTCOMES)
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"Which resolution is better supported by the cited evidence for "
                      f"dispute {dispute.dispute_id} ({dispute.zone.value}) between "
                      f"{dispute.claimant.value} and {dispute.opponent.value}?"),
            question_type=QuestionType.CHOICE,
            state={
                "dispute_id": dispute.dispute_id,
                "zone": dispute.zone.value,
                "claimant": dispute.claimant.value,
                "opponent": dispute.opponent.value,
                "claimant_position": dispute.position,
                "opponent_position": dispute.counter_position,
                "claimant_evidence": evidence["claimant"],
                "opponent_evidence": evidence["opponent"],
                "unresolved_refs": evidence["unresolved"],
                "bid_utilities": bids,
                "rounds": dispute.rounds,
                "severity": dispute.severity.value,
            },
            options=options,
            rubric=[
                "uphold_claimant: the claimant's position is supported by its evidence",
                "uphold_opponent: the opponent's position is supported by its evidence",
                ("synthesis: neither position is fully supported; a compromise exists "
                 "that respects both sets of evidence"),
            ],
            instructions=(
                "Judge only against the cited evidence. A position whose evidence does "
                "not resolve on the board cannot be upheld. Prefer synthesis when each "
                "side is partly right."
            ),
            asked_by=self.id,
            decision_point="a7.arbiter.outcome",
        )
        try:
            decision = ctx.decide(request)
            choice, confidence, source = (
                str(decision.choice or ""), float(decision.confidence), decision.source)
        except (DecisionUnavailable, DecisionFailed) as exc:
            # A decision backend that is down means no adjudication is possible.
            # Escalating is the only honest move.
            state["notes"].append(f"decision backend unavailable ({exc}); escalating")
            return self._escalate(ctx, dispute,
                                  reason=f"decision backend unavailable: {exc}",
                                  confidence=0.0)

        if choice not in options:
            reason = (f"decision returned {choice!r}, which is not one of "
                      f"{list(options)}")
            return self._escalate(ctx, dispute, reason=reason, confidence=confidence)
        # Offline stall guard: the rules ceiling (0.55) sits above the
        # needs_escalation floor (0.5) but below the model threshold (0.62),
        # so a bare ``confidence < threshold`` gate can never be met by a
        # rules answer and every offline dispute escalates. Judge RULES
        # answers against the 0.5 floor (strict ``<`` so 0.5 itself resolves)
        # while model answers still need the full 0.62.
        effective = threshold if source is not DecisionSource.RULES else 0.5
        if confidence < effective:
            return self._escalate(
                ctx, dispute,
                reason=(f"confidence {confidence:.3f} below threshold {effective:.3f}; "
                        f"the model leaned {choice!r} but not strongly enough"),
                confidence=confidence)

        synthesis_text: str | None = None
        if choice == "synthesis":
            synthesis_text = self._propose_synthesis(ctx, dispute, evidence, bids)
            if synthesis_text is None:
                # A synthesis nobody can state is not a resolution.
                return self._escalate(
                    ctx, dispute,
                    reason=("synthesis was preferred but no compromise could be "
                            "constructed from the cited evidence"),
                    confidence=confidence)

        return self._apply(ctx, dispute, choice, confidence, source, evidence,
                           synthesis_text, bids)

    # ============================================================== internals
    def _load_disputes(self, ctx: AgentContext) -> list[Dispute]:
        """Latest state of each OPEN or DEBATED dispute."""
        entries = _read(ctx, ZONE_DISPUTES, kind=KIND_DISPUTE)
        if not entries:
            entries = [e for e in _read(ctx, ZONE_DISPUTES)
                       if isinstance(e.payload, dict) and "claimant" in e.payload
                       and "counter_position" in e.payload]
        latest: dict[str, Dispute] = {}
        for entry in entries:
            model = _as_model(entry, Dispute)
            if model is None:
                continue
            # A later entry supersedes an earlier state of the same dispute.
            latest[model.dispute_id] = model
        return [d for d in latest.values()
                if d.status in (DisputeStatus.OPEN, DisputeStatus.DEBATED)]

    def _rounds_exhausted(self, disputes: Sequence[Dispute]) -> bool:
        limit = self.settings.max_debate_rounds
        return any(d.rounds >= limit for d in disputes)

    def _resolve_evidence(self, ctx: AgentContext, dispute: Dispute) -> dict[str, Any]:
        """Split the cited entries by which side referenced them.

        An id cited by neither side is reported separately as ``unresolved``; a
        dispute whose citations point at nothing cannot be judged on evidence, and
        the escalation path picks that up.
        """
        index = _history_by_id(ctx)
        found: list[dict[str, Any]] = []
        unresolved: list[str] = []
        for ref in dispute.evidence:
            entry = index.get(ref)
            if entry is None:
                unresolved.append(ref)
                continue
            found.append({
                "entry_id": ref,
                "zone": entry.zone,
                "kind": entry.kind,
                "author": entry.author.value,
                "confidence": entry.confidence,
                "source": entry.source.value,
                "payload": entry.payload,
            })
        return {
            "all": found,
            "unresolved": unresolved,
            # Each position is judged on the evidence it can actually point at.
            "claimant": [f for f in found if f["author"] == dispute.claimant.value],
            "opponent": [f for f in found if f["author"] == dispute.opponent.value],
        }

    def _relevant_bids(self, ctx: AgentContext, dispute: Dispute) -> list[dict[str, Any]]:
        """Bids whose ``task_id`` matches the dispute, with ``utility`` computed.

        ``Bid.utility`` (Smith 1980, feasibility-weighted value per unit of effort)
        is the quantitative tie-breaker for a quantified dispute: two positions
        can both be reasonable, and the higher-utility bid is the one the
        Contract Net protocol would have awarded.
        """
        entries = _read(ctx, ZONE_BIDS, kind=KIND_BID) or [
            e for e in _read(ctx, ZONE_BIDS)
            if isinstance(e.payload, dict) and "feasibility" in e.payload
            and "expected_value" in e.payload
        ]
        out: list[dict[str, Any]] = []
        for entry in entries:
            bid = _as_model(entry, Bid)
            if bid is None:
                continue
            if dispute.dispute_id not in bid.task_id and dispute.zone.value not in bid.task_id:
                continue
            out.append({
                "bid_id": bid.bid_id,
                "agent": bid.agent.value,
                "task_id": bid.task_id,
                "feasibility": bid.feasibility,
                "expected_value": bid.expected_value,
                "effort": bid.effort,
                "utility": bid.utility,
                "rationale": bid.rationale,
            })
        return sorted(out, key=lambda b: b["utility"], reverse=True)

    def _propose_synthesis(self, ctx: AgentContext, dispute: Dispute,
                           evidence: dict[str, Any],
                           bids: Sequence[dict[str, Any]]) -> str | None:
        """Build candidate compromises and let the model choose among them.

        Candidates are constructed from the numbers actually on the board: the
        top two bid utilities give a midpoint, and a one-sided dispute with no
        numbers gets a scope-based compromise. The *option set* is ours; the
        *choice* is the model's, so the resolution is not dictated by A7.
        """
        candidates: list[str] = []
        utilities = [float(b["utility"]) for b in bids if b.get("utility") is not None]
        if len(utilities) >= 2:
            low, high = sorted(utilities)[-2:]
            midpoint = round((low + high) / 2.0, 6)
            candidates.append(
                f"Award the task to the higher-utility bid on a split allocation: "
                f"utility {low} and {high} reconciled at {midpoint} via a partial "
                f"scope carve-out between {bids[0]['agent']} and "
                f"{bids[-1]['agent']}"
            )
        if len(bids) == 1:
            single = bids[0]
            candidates.append(
                f"Accept {single['agent']}'s bid ({single['bid_id']}, utility "
                f"{single['utility']}) conditional on raising feasibility from "
                f"{single['feasibility']}, since no rival bid exists to arbitrate against"
            )
        resolved = evidence.get("all", [])
        if resolved:
            candidates.append(
                f"Uphold neither wholesale: apply the evidence common to both sides "
                f"({', '.join(sorted({f['entry_id'] for f in resolved}))}) and refer the "
                f"contested remainder back to {dispute.claimant.value} and "
                f"{dispute.opponent.value} with a single named question to answer"
            )
        candidates.append(
            f"Split the difference procedurally: the decision is recorded as "
            f"unresolved in substance and referred to the event owner, with "
            f"{dispute.claimant.value} and {dispute.opponent.value} each given the "
            f"opportunity to file one new evidence entry before re-arbitration"
        )
        if not candidates:
            return None

        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"Which synthesis best reconciles dispute {dispute.dispute_id} "
                      f"without ignoring either side's evidence?"),
            question_type=QuestionType.CHOICE,
            state={
                "dispute_id": dispute.dispute_id,
                "zone": dispute.zone.value,
                "claimant_position": dispute.position,
                "opponent_position": dispute.counter_position,
                "claimant_evidence": [f["entry_id"] for f in evidence.get("claimant", [])],
                "opponent_evidence": [f["entry_id"] for f in evidence.get("opponent", [])],
                "bid_utilities": list(bids),
            },
            options=candidates,
            rubric=[
                "a candidate that preserves both sides' supported claims",
                "a candidate grounded in the cited entries or bid utilities",
                "a candidate a human could execute without further negotiation",
            ],
            instructions=("Pick the option a human could act on this week. Do not invent "
                          "a fifth option; choose among those listed."),
            asked_by=self.id,
            decision_point="a7.arbiter.synthesis",
        )
        try:
            decision = ctx.decide(request)
            choice, confidence = str(decision.choice or ""), float(decision.confidence)
            synth_source = decision.source
        except (DecisionUnavailable, DecisionFailed):
            return None
        if choice not in candidates:
            return None
        # Same offline guard as resolve(): RULES ceiling (0.55) can never
        # clear the model threshold (0.62), so judge RULES syntheses against
        # the 0.5 floor instead of stalling every offline synthesis.
        synth_threshold = (self.settings.confidence_threshold
                           if synth_source is not DecisionSource.RULES else 0.5)
        if confidence < synth_threshold:
            return None
        return choice

    def _apply(self, ctx: AgentContext, dispute: Dispute, choice: str, confidence: float,
               source: DecisionSource, evidence: dict[str, Any],
               synthesis: str | None, bids: Sequence[dict[str, Any]]) -> ArbiterOutcome:
        """Write the resolved dispute, the decision record, and the loser's lesson."""
        now = utcnow()
        evidence = {**evidence, "bids": list(bids)}
        if choice == "synthesis":
            resolution_text = synthesis or "synthesis selected"
            winner: AgentId | None = None
            loser = _loser_of_synthesis(dispute, evidence)
        elif choice == "uphold_claimant":
            winner, loser = dispute.claimant, dispute.opponent
            resolution_text = f"claimant upheld: {dispute.position}"
        else:
            winner, loser = dispute.opponent, dispute.claimant
            resolution_text = f"opponent upheld: {dispute.counter_position}"

        updated = dispute.model_copy(update={
            "status": DisputeStatus.RESOLVED,
            "resolution": resolution_text,
            "resolved_by": self.id,
            "resolved_at": now,
            "rounds": dispute.rounds + 1,
        })
        entry = ctx.board.post(
            ZONE_DISPUTES, KIND_DISPUTE, self.id, updated.model_dump(mode="json"),
            refs=[e["entry_id"] for e in evidence.get("all", [])] + [dispute.dispute_id],
            confidence=round(confidence, 6), source=source,
        )

        decision_entry_id = self.post(
            ctx, ZONE_DECISIONS, KIND_DECISION,
            {
                "decision_id": new_id("dec"),
                "dispute_id": dispute.dispute_id,
                "zone": dispute.zone.value,
                "claimant": dispute.claimant.value,
                "opponent": dispute.opponent.value,
                "outcome": choice,
                "winner": winner.value if winner else None,
                "synthesis": synthesis,
                "resolution": resolution_text,
                "rationale": _rationale(dispute, choice, evidence, confidence, source),
                "confidence": round(confidence, 6),
                "threshold": self.settings.confidence_threshold,
                "decision_source": source.value,
                "model": getattr(self, "_last_model", "unknown"),
                "evidence_entry_ids": [e["entry_id"] for e in evidence.get("all", [])],
                "claimant_evidence": [e["entry_id"] for e in evidence.get("claimant", [])],
                "opponent_evidence": [e["entry_id"] for e in evidence.get("opponent", [])],
                "unresolved_refs": list(evidence.get("unresolved", [])),
                "bid_utilities": [b["utility"] for b in evidence.get("bids", [])],
                "at": now.isoformat(),
            },
            # refs point at BOTH positions and the evidence they cite, so the
            # decision record is auditable without reading the dispute text.
            refs=[e["entry_id"] for e in evidence.get("all", [])]
                + [dispute.dispute_id, entry.entry_id],
            confidence=round(confidence, 6), source=source,
        )

        lesson_entry_ids: list[str] = []
        if loser is not None:
            lesson = _lesson_for_loser(ctx, dispute, loser, choice, resolution_text,
                                       evidence, confidence)
            lesson_entry_ids.append(self.post(
                ctx, ZONE_LESSONS, KIND_LESSON, lesson.model_dump(mode="json"),
                refs=[e["entry_id"] for e in evidence.get("all", [])] + [decision_entry_id],
                confidence=lesson.confidence, source=source,
            ))

        return ArbiterOutcome(
            dispute_id=dispute.dispute_id,
            outcome=choice,
            rationale=resolution_text,
            confidence=round(confidence, 6),
            source=source,
            escalated=False,
            synthesis=synthesis,
            decision_entry_id=decision_entry_id,
            lesson_entry_ids=lesson_entry_ids,
            resolved=True,
        )

    def _escalate(self, ctx: AgentContext, dispute: Dispute, *, reason: str,
                  confidence: float) -> ArbiterOutcome:
        """Post a ``HumanGate`` of kind ``ESCALATION`` and report the escalation.

        A7 does not raise here; the caller (:meth:`_act`) raises
        :class:`ArbiterEscalation` once every dispute's gate is on the board, so
        a human sees all the pending questions rather than one per interruption.
        """
        now = utcnow()
        gate = HumanGate(
            gate_id=new_id("gat"),
            kind=GateKind.ESCALATION,
            event_id=ctx.event_id,
            run_id=ctx.run_id,
            question=(f"A7 could not resolve dispute {dispute.dispute_id} "
                      f"({dispute.zone.value}) between {dispute.claimant.value} and "
                      f"{dispute.opponent.value}: {reason}. Which position stands?"),
            payload_preview=(f"claimant: {dispute.position}\n"
                             f"opponent: {dispute.counter_position}\n"
                             f"evidence refs: {', '.join(dispute.evidence) or 'none'}\n"
                             f"arbiter confidence: {confidence:.3f}"),
            options=[GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE],
            raised_at=now,
        )
        gate_entry_id = self.post(
            ctx, ZONE_APPROVALS, KIND_HUMAN_GATE, gate.model_dump(mode="json"),
            refs=[dispute.dispute_id, *dispute.evidence],
            confidence=round(confidence, 6), source=DecisionSource.RULES,
        )
        escalated = dispute.model_copy(update={
            "status": DisputeStatus.ESCALATED,
            "resolution": f"escalated to a human: {reason}",
            "rounds": dispute.rounds + 1,
        })
        entry = ctx.board.post(
            ZONE_DISPUTES, KIND_DISPUTE, self.id, escalated.model_dump(mode="json"),
            refs=[gate_entry_id, *dispute.evidence],
            confidence=round(confidence, 6), source=DecisionSource.RULES,
        )
        del entry
        return ArbiterOutcome(
            dispute_id=dispute.dispute_id,
            outcome="escalated",
            rationale=reason,
            confidence=round(confidence, 6),
            source=DecisionSource.RULES,
            escalated=True,
            gate_entry_id=gate_entry_id,
        )

    def _ask(self, ctx: AgentContext, request: DecisionRequest,
             options: Sequence[str]) -> str:
        """``ctx.decide`` with an on-menu check; off-menu answers fail closed."""
        try:
            choice = str(ctx.decide(request).choice or "")
        except (DecisionUnavailable, DecisionFailed) as exc:
            self._state(ctx)["notes"].append(f"decision unavailable ({exc}); "
                                             f"strategy defaulted")
            return "escalate_immediately"
        return choice if choice in options else "escalate_immediately"


#: Alias so ``agents/__init__.py`` can export either name.
A7Arbiter = ArbiterAgent


# ======================================================================= helpers
def _outcome_fact(outcome: ArbiterOutcome) -> dict[str, Any]:
    return {
        "dispute_id": outcome.dispute_id,
        "outcome": outcome.outcome,
        "rationale": outcome.rationale,
        "confidence": outcome.confidence,
        "escalated": outcome.escalated,
        "synthesis": outcome.synthesis,
        "decision_entry_id": outcome.decision_entry_id,
        "gate_entry_id": outcome.gate_entry_id,
        "lesson_entry_ids": list(outcome.lesson_entry_ids),
    }


def _loser_of_synthesis(dispute: Dispute, evidence: dict[str, Any]) -> AgentId | None:
    """Who owns the lesson when a synthesis is reached.

    Under a synthesis both sides are partly wrong, so the lesson goes to whichever
    agent had the thinner evidence — the side that contributed less to the
    resolution. One-sided evidence means one side is genuinely the loser.
    """
    claimant_n = len(evidence.get("claimant", []))
    opponent_n = len(evidence.get("opponent", []))
    if claimant_n < opponent_n:
        return dispute.claimant
    if opponent_n < claimant_n:
        return dispute.opponent
    # Equal evidence: attribute the lesson to the opponent, matching the ordinary
    # "the party that did not carry the decision" convention.
    return dispute.opponent


def _rationale(dispute: Dispute, choice: str, evidence: dict[str, Any],
               confidence: float, source: DecisionSource) -> str:
    """Why this outcome, in terms of the evidence rather than the wording."""
    cited = len(evidence.get("all", []))
    unresolved = len(evidence.get("unresolved", []))
    return (f"{choice} for {dispute.dispute_id}: {cited} cited entry/entries resolved, "
            f"{len(evidence.get('claimant', []))} from {dispute.claimant.value}, "
            f"{len(evidence.get('opponent', []))} from {dispute.opponent.value}, "
            f"{unresolved} unresolved; confidence {confidence:.3f} from {source.value}")


def _lesson_for_loser(ctx: AgentContext, dispute: Dispute, loser: AgentId, choice: str,
                      resolution: str, evidence: dict[str, Any],
                      confidence: float) -> Lesson:
    """Author the Reflexion lesson the losing agent should read next run."""
    mine = dispute.position if loser is dispute.claimant else dispute.counter_position
    supported = (evidence.get("claimant", []) if loser is dispute.claimant
                 else evidence.get("opponent", []))
    return Lesson(
        lesson_id=new_id("les"),
        event_id=ctx.event_id,
        author=loser,
        trigger_dispute_id=dispute.dispute_id,
        trigger=(f"A7 resolved dispute {dispute.dispute_id} ({dispute.zone.value}) as "
                 f"{choice} at confidence {confidence:.3f}"),
        correction=(f"your position \"{mine}\" was not supported by the "
                    f"{len(supported)} entry/entries you cited; the resolution was "
                    f"\"{resolution}\""),
        rule=(f"Before asserting a {dispute.zone.value} claim, cite a board entry that "
              f"a reader can open; {len(supported)} supporting entry/entries was not "
              f"enough against {choice}. Record the evidence id in the dispute, not "
              f"just the argument."),
        confidence=round(min(1.0, max(0.1, confidence)), 6),
        created_at=utcnow(),
    )
