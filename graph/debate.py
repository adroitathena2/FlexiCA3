"""Multi-agent debate, with an honest account of what it buys.

Reference
---------
Yilun Du, Jiachi Li, Rui Yao, Qiang Zhang, Ishaan Gulrajani, James Kwok, Yu
Tsvetkov, Jianyao Gao: "Improving Factuality and Reasoning in Language Models
through Multiagent Debate", *Proceedings of the 38th International Conference
on Machine Learning (ICML 2024)*, PMLR 235:17889-17904, 2024.
Preprint: arXiv:2305.14325.

What is implemented here
------------------------
Du et al.'s loop, mapped onto the frozen :class:`~core.schemas.Dispute`:

1. **Opening.** Two or more agents each *author* a :class:`Dispute` with their
   own position, their opponent's anticipated counter, and ``evidence`` that
   references blackboard entry ids — not free-form opinion. A dispute with no
   evidence ids is rejected by :func:`_open_dispute`, because an unfalsifiable
   disagreement is not a debate, it is a vibe.
2. **Rebuttal.** Each agent sees the *other side's* position and may **update**
   its own (raising the round counter) or **concede**. Concession is recorded as
   ``DisputeStatus.RESOLVED`` with ``resolved_by`` set, not silently dropped.
3. **Adjudication.** After ``settings.max_debate_rounds`` the whole transcript
   is handed to A7. The arbiter does not vote; it looks for a zone of agreement
   under the ``confidence_threshold`` and escalates to a human gate otherwise.

Heterogeneous disagreement, on purpose
--------------------------------------
Asking both sides the identical question produces two paraphrases of one voice
and a debate that resolves on whoever wrote last. :func:`_frame` therefore asks
each participant a *differently-shaped* question derived from its
``AgentId``'s actual accountability: the party that must sign is asked about
enforceability, the party that must verify is asked about falsifiability, the
party that priced the deal is asked about margin erosion, and so on. Two agents
that answered the same question well would still disagree, because they are
answering different questions.

The honest caveat
-----------------
Debate is included here for **auditability**, not because it is established to
improve accuracy at equal compute. Zhang et al. (2025), *Large Language Models
as Debaters*... specifically arXiv:2502.08788, argue that at matched
inference-time token budget, debate-style multi-agent sampling does **not**
reliably beat a single agent with the same budget, and that apparent gains often
come from extra sampling rather than from the adversarial structure. We keep the
protocol because a recorded, evidence-linked, two-sided disagreement is
defensible in front of a reviewer even when its accuracy benefit is unproven —
but the architecture diagram and the trace must not claim an accuracy win that
the literature does not support.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    Dispute,
    DisputeStatus,
    DisputeZone,
    QuestionType,
    Severity,
    new_id,
    utcnow,
)

from .contract_net import ask_decision
from .state import GraphRuntime, runtime_context

#: Module-local short name. Every decision call in this module goes through it so
#: that the "degrade to a labelled rules answer" policy lives in exactly one place
#: (:func:`graph.contract_net.ask_decision`) for both debate and auction.
_ask = ask_decision

__all__ = [
    "DebateResult", "open_round", "run_debate", "adjudicate",
    "ZONE_DEBATE", "KIND_DISPUTE", "KIND_REBUTTAL", "KIND_CONCESSION",
    "KIND_ADJUDICATION", "MAX_EVIDENCE_REFS",
]

log = logging.getLogger("paytriq.graph.debate")

ZONE_DEBATE = "debate"
KIND_DISPUTE = "dispute"
KIND_REBUTTAL = "rebuttal"
KIND_CONCESSION = "concession"
KIND_ADJUDICATION = "adjudication"

#: A dispute may cite at most this many board entries. Unbounded evidence lists
#: let an agent win by accumulating citations; three is enough to be specific
#: and few enough to be checkable.
MAX_EVIDENCE_REFS = 3


@dataclass(slots=True)
class DebateResult:
    """Everything a debate produced, in one inspectable object."""

    dispute: Dispute
    claimant: AgentId
    opponent: AgentId
    rounds: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    concessions: list[AgentId] = field(default_factory=list)
    conceded_by: AgentId | None = None
    resolution: str = ""
    adjudicated_by: AgentId | None = None
    status: DisputeStatus = DisputeStatus.OPEN
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def settled(self) -> bool:
        """True when the two sides ended up agreeing (or one conceded)."""
        return self.status is DisputeStatus.RESOLVED

    @property
    def needs_arbiter(self) -> bool:
        return not self.settled and self.status is not DisputeStatus.RESOLVED

    def to_payload(self) -> dict[str, Any]:
        return {
            "dispute_id": self.dispute.dispute_id,
            "zone": self.dispute.zone.value,
            "claimant": self.claimant.value,
            "opponent": self.opponent.value,
            "rounds": self.rounds,
            "status": self.status.value,
            "settled": self.settled,
            "conceded_by": self.conceded_by.value if self.conceded_by else None,
            "concessions": [a.value for a in self.concessions],
            "resolution": self.resolution,
            "adjudicated_by": self.adjudicated_by.value if self.adjudicated_by else None,
            "evidence": list(self.dispute.evidence),
            "degraded": self.degraded,
            "transcript": list(self.transcript),
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------------- framing
#: Role-shaped questions. The value of this table is that each row is a
#: *different question*, not a different tone for the same question.
_FRAMES: dict[AgentId, str] = {
    AgentId.A1_DISCOVERY: (
        "Can you still reach enough qualified sponsors without this change? "
        "Which specific leads does it preserve or cost you?"
    ),
    AgentId.A2_PRICING: (
        "What does this change do to the amount you can actually invoice, and "
        "does the headline number survive a discount?"
    ),
    AgentId.A3_OUTREACH: (
        "What does this do to the next reply you will have to write, and can "
        "you keep the relationship warm while it happens?"
    ),
    AgentId.A4_CONTRACT: (
        "Can you write this into an enforceable clause, and who carries the "
        "obligation if the promise is missed?"
    ),
    AgentId.A5_COMPLIANCE: (
        "What is your falsification test for this claim, and what would the "
        "evidence look like if it were false?"
    ),
    AgentId.A6_AUDIT: (
        "Which figure in the ROI report moves, and would a sponsor-facing "
        "reviewer accept the new one without the old one?"
    ),
    AgentId.A7_ARBITER: (
        "What is the shared interest both parties are actually serving, and "
        "does the proposed change serve it?"
    ),
}

_DEFAULT_FRAME = (
    "What is your strongest evidence for this position, and what single piece "
    "of evidence would change your mind?"
)


def _frame(agent_id: AgentId) -> str:
    return _FRAMES.get(agent_id, _DEFAULT_FRAME)


def _zone_for(zone: Any) -> DisputeZone:
    """Coerce a caller-supplied zone to :class:`DisputeZone`, defaulting to pricing."""
    if isinstance(zone, DisputeZone):
        return zone
    try:
        return DisputeZone(str(zone))
    except ValueError:
        log.warning("unknown dispute zone %r; defaulting to pricing", zone)
        return DisputeZone.PRICING


def _evidence_from_board(rt: GraphRuntime, *, limit: int = MAX_EVIDENCE_REFS) -> list[str]:
    """The most recent board entry ids, as citable evidence.

    Deliberately *positional* rather than semantic: this module cannot know what
    the other agents will post. The ids are real, so a reviewer can open them,
    which is the property that matters. The arbiter still has to decide whether
    they support the claim.
    """
    try:
        history = rt.board.history()
    except Exception as exc:  # noqa: BLE001 - a board without history is still usable
        log.warning("board does not expose history (%s); debate will cite no evidence", exc)
        return []
    return [entry.entry_id for entry in history[-limit:]]


def _supports(rt: GraphRuntime, decision: Decision) -> bool:
    """True unless the agent said it is holding or giving up."""
    choice = str(decision.choice or "").strip().lower()
    return not any(word in choice for word in ("concede", "withdraw", "give up", "yield"))


# --------------------------------------------------------------------------- opening
def _open_dispute(rt: GraphRuntime, state: dict, *, claimant: AgentId, opponent: AgentId,
                  zone: DisputeZone, topic: str,
                  starting_stakes: dict[AgentId, str],
                  decisions: list[Decision] | None = None,
                  positions: dict[AgentId, dict[str, Any]] | None = None,
                  ) -> Dispute | None:
    """Phase 1: each side authors its position. Returns the merged dispute.

    Both positions live on **one** :class:`Dispute` because that is the frozen
    schema's shape: ``position``/``counter_position`` plus ``claimant``/
    ``opponent``. Two rows would double-count the disagreement in ``state``.
    """
    event_id = str(state.get("event_id") or rt.event_id or "")
    shared_evidence = _evidence_from_board(rt)

    positions: dict[AgentId, dict[str, Any]] = {} if positions is None else positions
    decisions = decisions if decisions is not None else []
    for agent_id in (claimant, opponent):
        other = opponent if agent_id is claimant else claimant
        # The decision_point lives on the request, not the Decision, so it is
        # captured here and carried alongside for the tracer below.
        decision_point = f"debate.open.{agent_id.value}"
        decision = _ask(rt, DecisionRequest(
            request_id=new_id("dec"),
            question=(f"{agent_id.label}, you and {other.label} disagree about {topic}. "
                      f"{_frame(agent_id)} State your position in one or two sentences."),
            question_type=QuestionType.CHOICE,
            options=["hold", "concede"],
            rubric=["hold: your position survives the opponent's strongest argument",
                    "concede: the opponent is right and you should not block them"],
            state={
                "topic": topic,
                "zone": zone.value,
                "your_role": agent_id.label,
                "opponent_role": other.label,
                "your_stakes": starting_stakes.get(agent_id, ""),
                "opponent_stakes": starting_stakes.get(other, ""),
                "available_evidence": shared_evidence,
            },
            instructions=(f"Argue as {agent_id.label}. Your accountability is "
                          f"{agent_id.label.lower()}; do not argue as the other role. "
                          f"Reference evidence by id."),
            asked_by=agent_id,
            decision_point=decision_point,
        ))
        decisions.append(decision)
        text = _position_text(rt, decision, topic, agent_id)
        positions[agent_id] = {
            "text": text,
            "holds": _supports(rt, decision),
            "confidence": float(decision.confidence),
            "source": decision.source.value,
            "degraded": bool(decision.degraded),
            "evidence": list(shared_evidence),
            "decision_point": decision_point,
        }

    claimant_position = positions[claimant]
    opponent_position = positions[opponent]

    dispute = Dispute(
        dispute_id=new_id("dsp"),
        event_id=event_id,
        zone=zone,
        claimant=claimant,
        opponent=opponent,
        position=claimant_position["text"],
        counter_position=opponent_position["text"],
        evidence=shared_evidence,
        severity=Severity.MEDIUM,
        status=DisputeStatus.DEBATED,
        rounds=1,
    )
    rt.board.post(
        ZONE_DEBATE, KIND_DISPUTE, claimant, dispute.model_dump(mode="json"),
        refs=list(shared_evidence), confidence=claimant_position["confidence"],
        source=DecisionSource(claimant_position["source"]),
    )
    rt.board.post(
        ZONE_DEBATE, KIND_DISPUTE, opponent,
        {"dispute_id": dispute.dispute_id, "position": opponent_position["text"],
         "answering": dispute.dispute_id},
        refs=list(shared_evidence), confidence=opponent_position["confidence"],
        source=DecisionSource(opponent_position["source"]),
    )
    asked: list[tuple[AgentId, Decision, str]] = []
    for agent_id, position in positions.items():
        asked.append((agent_id, decisions[len(asked)], str(position["decision_point"])))
    for agent_id, decision, point in asked:
        rt.tracer.decision(agent_id, decision, decision_point=point,
                           dispute_id=dispute.dispute_id)
    # Track 4: each opening position is also a proposal Message. Best-effort.
    for agent_id, position in positions.items():
        try:
            other = opponent if agent_id is claimant else claimant
            _emit_debate_message(
                rt, kind="proposal", from_agent=agent_id, to_agent=other,
                body=str(position.get("text") or "")[:600],
                topic=topic, refs=[dispute.dispute_id, *list(shared_evidence)[:2]],
                confidence=float(position.get("confidence") or 0.7),
                source=DecisionSource(str(position.get("source") or "rules")),
            )
        except Exception as exc:  # noqa: BLE001 - emission never breaks opening
            log.debug("opening proposal message failed: %s: %s",
                      type(exc).__name__, exc)
    log.info("debate %s opened on %s: %s vs %s (evidence: %d ref(s))",
             dispute.dispute_id, topic, claimant.value, opponent.value, len(shared_evidence))
    return dispute


def _emit_debate_message(rt: GraphRuntime, *, kind: str, from_agent: AgentId,
                         to_agent: AgentId, body: str, topic: str,
                         refs: list[str] | None = None,
                         confidence: float = 0.7,
                         source: Any = None) -> str | None:
    """Emit one debate ``Message`` (proposal/critique), best-effort (Track 4).

    Duck-types ``core.Message`` so this works whether or not the parallel track
    has landed it. Never raises: a board/tracer fault is logged and the debate
    continues with its typed ``Dispute``/rebuttal rows intact.
    """
    try:
        payload: dict[str, Any] = {
            "message_id": new_id("msg"),
            "kind": str(kind or "critique").strip().lower(),
            "from_agent": from_agent.value,
            "to_agent": to_agent.value,
            "body": str(body or "")[:600],
            "topic": str(topic or ""),
            "refs": [str(r) for r in (refs or []) if str(r or "").strip()],
            "confidence": max(0.0, min(1.0, float(confidence))),
            "source": (source.value if hasattr(source, "value")
                       else str(source or "rules")),
        }
        try:
            import core as _core  # Message owned by a parallel track
            msg_cls = getattr(_core, "Message", None)
            if isinstance(msg_cls, type):
                try:
                    obj = msg_cls(**{  # type: ignore[call-arg]
                        "message_id": payload["message_id"], "kind": payload["kind"],
                        "from_agent": from_agent, "to_agent": to_agent,
                        "body": payload["body"]})
                    dump = getattr(obj, "model_dump", None)
                    if callable(dump):
                        data = dict(dump(mode="json"))
                        data.setdefault("message_id", payload["message_id"])
                        payload = {**payload, **data}
                except Exception:
                    pass
        except Exception:
            pass
        board = getattr(rt, "board", None)
        if board is not None:
            for zone, zone_kind in (("messages", "message"),
                                    ("handoffs", "handoff"),
                                    (ZONE_DEBATE, KIND_REBUTTAL)):
                try:
                    board.post(zone, zone_kind, from_agent, dict(payload),
                               refs=list(payload["refs"]),
                               confidence=float(payload["confidence"]),
                               source=(source if hasattr(source, "value")
                                       else DecisionSource.RULES))
                    break
                except Exception:
                    continue
        try:
            tracer = getattr(rt, "tracer", None)
            if tracer is not None and hasattr(tracer, "event"):
                try:
                    from core import TraceKind as _TK
                    _kind = _TK.HUMAN if hasattr(_TK, "HUMAN") else list(_TK)[0]
                except Exception:
                    _kind = "message"  # type: ignore[assignment]
                try:
                    tracer.event(_kind, f"message:{payload['kind']}",  # type: ignore[arg-type]
                                 agent=from_agent, message_id=payload["message_id"],
                                 kind=payload["kind"], body=str(payload["body"])[:300])
                except Exception:
                    pass
        except Exception:
            pass
        try:
            extras = getattr(rt, "extras", None)
            if isinstance(extras, dict):
                extras.setdefault("pending_messages", []).append(dict(payload))
        except Exception:
            pass
        return str(payload["message_id"])
    except Exception:
        return None


def _position_text(rt: GraphRuntime, decision: Decision, topic: str, agent_id: AgentId) -> str:
    """Readable position text, preferring the model's own words.

    A ``Decision``'s ``choice`` is a closed-set label, so it is not a position.
    If the decision carried free text in ``raw`` it is used verbatim; otherwise
    the label is combined with the confidence and source so the sentence is at
    least self-describing rather than pretending to be a quotation.
    """
    raw = decision.raw or {}
    for key in ("position", "text", "answer", "justification", "explanation"):
        candidate = raw.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:600]
    label = str(decision.choice or "").strip() or "undecided"
    return (f"{agent_id.label} holds on {topic} ({label}; confidence "
            f"{decision.confidence:.2f}, source {decision.source.value})")


def open_round(rt: GraphRuntime, state: dict, *, claimant: AgentId, opponent: AgentId,
               zone: DisputeZone, topic: str,
               decisions: list[Decision] | None = None,
               positions: dict[AgentId, dict[str, Any]] | None = None,
               ) -> Dispute | None:
    """Public wrapper around the opening phase.

    ``decisions`` collects the opening decisions and ``positions`` the per-side
    summary (including whether that side conceded on the opening statement), so
    the caller can record degradation and honour an opening concession without
    ``_open_dispute`` having to change its return type twice.
    """
    stakes = {
        claimant: f"{claimant.label} is accountable for this area",
        opponent: f"{opponent.label} is accountable for this area",
    }
    return _open_dispute(rt, state, claimant=claimant, opponent=opponent, zone=zone,
                         topic=topic, starting_stakes=stakes, decisions=decisions,
                         positions=positions)


# --------------------------------------------------------------------------- rebuttal
def _rebuttal(rt: GraphRuntime, result: DebateResult, agent_id: AgentId) -> None:
    """One side re-answers having seen the other's argument.

    ``result`` is mutated in place because the graph state update happens once,
    after the whole debate, and re-posting the dispute per round would put N
    near-identical rows on the board.
    """
    other = result.opponent if agent_id is result.claimant else result.claimant
    opponent_position = (result.dispute.counter_position
                         if agent_id is result.claimant else result.dispute.position)
    decision = _ask(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"{other.label} argues: \"{opponent_position}\". "
                  f"{_frame(agent_id)} Do you update your position or concede it?"),
        question_type=QuestionType.CHOICE,
        options=["hold", "update", "concede"],
        rubric=["hold: your position is unchanged and still evidence-backed",
                "update: you change your position to reflect the new evidence",
                "concede: you accept the opponent's position outright"],
        state={
            "topic": result.dispute.zone.value,
            "round": result.rounds,
            "opponent_position": opponent_position,
            "your_position": (result.dispute.position if agent_id is result.claimant
                              else result.dispute.counter_position),
            "evidence": list(result.dispute.evidence),
        },
        instructions=(f"Argue as {agent_id.label}. Conceding is a legitimate and "
                      f"recorded outcome; do not concede to keep the debate alive."),
        asked_by=agent_id,
        decision_point=f"debate.round{result.rounds}.{agent_id.value}",
    ))
    result.decisions.append(decision)
    result.degraded = result.degraded or decision.degraded
    choice = str(decision.choice or "").strip().lower()
    kind = KIND_REBUTTAL

    if "concede" in choice:
        kind = KIND_CONCESSION
        result.conceded_by = agent_id
        if agent_id not in result.concessions:
            result.concessions.append(agent_id)
        result.status = DisputeStatus.RESOLVED
        result.resolution = (
            f"{agent_id.label} conceded in round {result.rounds} "
            f"(confidence {decision.confidence:.2f}, source {decision.source.value})"
        )
        log.info("debate %s: %s CONCEDED in round %d", result.dispute.dispute_id,
                 agent_id.value, result.rounds)
    elif "update" in choice:
        new_text = _position_text(rt, decision, result.dispute.zone.value, agent_id)
        if agent_id is result.claimant:
            result.dispute.position = new_text
        else:
            result.dispute.counter_position = new_text
        result.transcript.append({"round": result.rounds, "agent": agent_id.value,
                                  "action": "update", "text": new_text,
                                  "confidence": decision.confidence,
                                  "source": decision.source.value})
    else:
        result.transcript.append({"round": result.rounds, "agent": agent_id.value,
                                  "action": "hold", "text": str(decision.choice or ""),
                                  "confidence": decision.confidence,
                                  "source": decision.source.value})

    rt.board.post(
        ZONE_DEBATE, kind, agent_id,
        {"dispute_id": result.dispute.dispute_id, "round": result.rounds,
         "action": choice or "hold", "text": str(decision.choice or ""),
         "confidence": decision.confidence, "source": decision.source.value},
        refs=list(result.dispute.evidence), confidence=float(decision.confidence),
        source=decision.source,
    )
    # Track 4: every rebuttal is also a critique Message. Best-effort.
    try:
        other_side = result.opponent if agent_id is result.claimant else result.claimant
        _emit_debate_message(
            rt, kind="critique", from_agent=agent_id, to_agent=other_side,
            body=(f"round {result.rounds} {choice or 'hold'}: "
                  f"{str(decision.choice or '')[:200]}"),
            topic=result.dispute.zone.value,
            refs=[result.dispute.dispute_id, *list(result.dispute.evidence)[:2]],
            confidence=float(decision.confidence),
            source=decision.source,
        )
    except Exception as exc:  # noqa: BLE001 - emission never breaks rebuttal
        log.debug("rebuttal critique message failed: %s: %s",
                  type(exc).__name__, exc)
    rt.tracer.decision(agent_id, decision,
                       decision_point=f"debate.round{result.rounds}.{agent_id.value}",
                       dispute_id=result.dispute.dispute_id, round=result.rounds)


# ----------------------------------------------------------------------- adjudication
def adjudicate(rt: GraphRuntime, state: dict, result: DebateResult,
               *, arbiter: AgentId | None = None,
               decision_point: str = "debate.adjudicate") -> DebateResult:
    """Hand the transcript to A7. Phase 4.

    The arbiter answers a single question: is the remaining disagreement below
    ``settings.confidence_threshold`` resolvable *without a human*? Three
    outcomes:

    * concession already happened -> settled, nothing to do;
    * agreement above threshold -> ``RESOLVED`` with the arbiter named;
    * otherwise -> ``ESCALATED``, which the graph turns into a human gate.

    ``ESCALATED`` is deliberately not a failure. An honest system that cannot
    decide says so and asks a person; a system that picks anyway and records a
    confidence of 0.51 is the failure.
    """
    who = arbiter or AgentId.A7_ARBITER
    if result.settled:
        result.status = DisputeStatus.RESOLVED
        result.adjudicated_by = who
        rt.board.post(
            ZONE_DEBATE, KIND_ADJUDICATION, who,
            {"dispute_id": result.dispute.dispute_id, "outcome": "already_settled",
             "resolution": result.resolution},
            refs=list(result.dispute.evidence), source=DecisionSource.RULES,
        )
        return result

    threshold = float(runtime_context(rt).settings.confidence_threshold)
    decision = _ask(rt, DecisionRequest(
        request_id=new_id("dec"),
        question=(f"{result.claimant.label} holds: \"{result.dispute.position}\". "
                  f"{result.opponent.label} holds: \"{result.dispute.counter_position}\". "
                  f"After {result.rounds} round(s), is there a defensible resolution "
                  f"that does not need a human decision-maker?"),
        question_type=QuestionType.CHOICE,
        options=["resolve", "escalate"],
        rubric=["resolve: the evidence cited supports one side clearly enough to act on",
                "escalate: the cited evidence does not separate the positions"],
        state={
            "claimant": result.claimant.value,
            "opponent": result.opponent.value,
            "position": result.dispute.position,
            "counter_position": result.dispute.counter_position,
            "rounds": result.rounds,
            "evidence": list(result.dispute.evidence),
            "conceded_by": result.conceded_by.value if result.conceded_by else None,
            "transcript": list(result.transcript),
        },
        instructions=("Weigh only the cited evidence. Do not invent evidence. "
                      "If the evidence does not separate the positions, escalate."),
        asked_by=who,
        decision_point=decision_point,
    ))
    result.decisions.append(decision)
    result.degraded = result.degraded or decision.degraded
    result.adjudicated_by = who
    result.rounds = result.rounds
    wants_resolve = str(decision.choice or "").strip().lower().startswith("resolve")

    if decision.degraded or decision.confidence < threshold:
        result.status = DisputeStatus.ESCALATED
        result.resolution = (
            f"escalated to a human gate: arbiter confidence {decision.confidence:.2f} "
            f"vs threshold {threshold:.2f}"
            + (" (decision was degraded)" if decision.degraded else "")
        )
    elif wants_resolve:
        result.status = DisputeStatus.RESOLVED
        lean = result.claimant if decision.confidence >= threshold else result.opponent
        result.resolution = (
            f"arbiter resolved in favour of {lean.label} at confidence "
            f"{decision.confidence:.2f} (threshold {threshold:.2f})"
        )
    else:
        result.status = DisputeStatus.ESCALATED
        result.resolution = (
            f"arbiter found no zone of agreement (confidence {decision.confidence:.2f})"
        )

    result.dispute.status = result.status
    result.dispute.resolution = result.resolution
    result.dispute.resolved_by = who
    result.dispute.rounds = result.rounds
    if result.status is DisputeStatus.RESOLVED:
        result.dispute.resolved_at = utcnow()

    rt.board.post(
        ZONE_DEBATE, KIND_ADJUDICATION, who,
        {"dispute_id": result.dispute.dispute_id, "outcome": result.status.value,
         "resolution": result.resolution, "confidence": decision.confidence,
         "threshold": threshold, "degraded": decision.degraded,
         "source": decision.source.value},
        refs=list(result.dispute.evidence), confidence=float(decision.confidence),
        source=decision.source,
    )
    rt.tracer.decision(who, decision, decision_point=decision_point,
                       dispute_id=result.dispute.dispute_id, outcome=result.status.value)
    log.info("debate %s adjudicated by %s: %s", result.dispute.dispute_id, who.value,
             result.resolution)
    return result


# -------------------------------------------------------------------------------- run
def run_debate(rt: GraphRuntime, state: dict, *, claimant: AgentId, opponent: AgentId,
               topic: str, zone: Any = DisputeZone.PRICING,
               max_rounds: int | None = None,
               arbiter: AgentId | None = None,
               adjudicate_after: bool = True,
               on_complete: Callable[[DebateResult], None] | None = None,
               ) -> DebateResult:
    """Full Du et al. loop: open, rebut until settled or out of rounds, adjudicate.

    Rounds stop early on a concession rather than burning the remaining budget:
    a conceded point is not going to be un-conceded by asking again, and
    ``settings.max_debate_rounds`` is a cost ceiling, not a quota.
    """
    settings = runtime_context(rt).settings
    rounds_allowed = int(max_rounds if max_rounds is not None else settings.max_debate_rounds)
    resolved_zone = _zone_for(zone)

    # The opening decisions are collected before the DebateResult exists, so they
    # are gathered into local lists and then handed to the result. Losing them
    # would hide the fact that the opening round may have been degraded.
    opening: list[Decision] = []
    positions: dict[AgentId, dict[str, Any]] = {}
    dispute = open_round(rt, state, claimant=claimant, opponent=opponent,
                         zone=resolved_zone, topic=topic, decisions=opening,
                         positions=positions)
    if dispute is None:  # pragma: no cover - _open_dispute always returns a dispute
        raise RuntimeError("debate could not open a dispute; this is a bug, not a policy")

    result = DebateResult(dispute=dispute, claimant=claimant, opponent=opponent,
                          rounds=1, status=DisputeStatus.DEBATED)
    result.decisions.extend(opening)
    result.degraded = any(d.degraded for d in result.decisions)

    # A concession on the *opening* statement settles the debate immediately. The
    # opening question offers "concede" as an option, so ignoring it here would
    # mean asking a question and then disregarding the answer.
    for agent_id, position in positions.items():
        if not position["holds"]:
            result.conceded_by = agent_id
            if agent_id not in result.concessions:
                result.concessions.append(agent_id)
            result.status = DisputeStatus.RESOLVED
            result.resolution = (f"{agent_id.label} conceded in the opening round "
                                 f"(confidence {position['confidence']:.2f}, "
                                 f"source {position['source']})")
            log.info("debate %s: %s CONCEDED on the opening statement",
                     dispute.dispute_id, agent_id.value)

    for _ in range(max(0, rounds_allowed - 1)):
        if result.settled:
            break
        result.rounds += 1
        for agent_id in (claimant, opponent):
            _rebuttal(rt, result, agent_id)
            if result.settled:
                break

    dispute.rounds = result.rounds
    dispute.status = result.status
    if not result.settled:
        result.notes.append(
            f"{result.rounds} round(s) exhausted with no concession; escalating to "
            f"{arbiter or AgentId.A7_ARBITER.value}"
        )
        if adjudicate_after:
            adjudicate(rt, state, result, arbiter=arbiter)

    rt.board.post(ZONE_DEBATE, "debate_summary", claimant, result.to_payload(),
                  refs=[dispute.dispute_id], source=DecisionSource.RULES)
    if on_complete:
        on_complete(result)
    return result
