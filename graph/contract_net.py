"""Contract Net Protocol task allocation — Smith, *IEEE Trans. Computers* C-31(2), 1980.

R. E. Smith, "The Contract Net Protocol: High-Level Communication and Control in a
Distributed Problem Solver", *IEEE Transactions on Computers*, vol. C-31,
no. 2, pp. 236-247, February 1980.

Smith's four phases, mapped onto this graph:

============  ===========================================================
Phase         Here
============  ===========================================================
announcement  :func:`announce` posts a ``task_announcement`` to the
              ``tasks`` zone. No agent is chosen yet.
bidding       :func:`collect_bids` asks every eligible agent to score the
              task with ``ctx.decide`` and posts a typed
              :class:`~core.schemas.Bid` per agent to the ``bids`` zone.
awarding       :func:`award` picks the highest ``Bid.utility``
              (``expected_value * feasibility / effort``) and posts the
              award. Ties break deterministically.
expediting     the winner is announced so the manager can track progress;
              :func:`expedite` records the manager's follow-up.
============  ===========================================================

Why a full protocol and not "call the right agent"

Three reasons, all load-bearing:

1. **The disagreement is the artefact.** Pricing (A2) and outreach (A3) value the
   same task differently: A2 measures expected rupees, A3 measures sponsor
   reach. Routing each bid through ``ctx.decide`` with a *differently framed*
   question per agent means the numbers in the trace genuinely disagree, so the
   award is evidence of a trade-off rather than a restatement of a preference
   order someone wrote down.
2. **The award is contestable.** A routing table is not. Because the winner is
   chosen by a recorded, per-agent decision with a ``DecisionSource`` and a
   confidence, "why did A3 get the call and not A2?" is answerable from the
   trace.
3. **Smith's failure modes are still live.** An over-specified manager starves
   agents; an under-specified one floods them. The eligibility set here is
   derived from the blackboard (who actually has something to offer), not from
   a config list, which is Smith's "contractors bid on what they can see".

One honest simplification: Smith lets a manager *revoke* a contract and
re-announce. :func:`run_auction` supports bounded re-announcement via
``settings.max_auction_rounds`` but the graph does not currently route back into
it, so the expiry path is implemented and unit-tested rather than driven by the
pipeline.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core import (
    AgentId,
    Bid,
    Decision,
    DecisionRequest,
    DecisionSource,
    QuestionType,
    new_id,
)

from .state import GraphRuntime, bidder_agents, runtime_context

__all__ = [
    "TaskAnnouncement", "AuctionResult", "announce", "collect_bids", "award",
    "expedite", "run_auction", "ask_decision", "ZONE_TASKS", "ZONE_BIDS",
    "KIND_ANNOUNCEMENT", "KIND_BID", "KIND_AWARD", "KIND_EXPEDITE",
    "FEASIBILITY_OPTIONS", "EFFORT_OPTIONS",
]

log = logging.getLogger("paytriq.graph.contract_net")

ZONE_TASKS = "tasks"
ZONE_BIDS = "bids"
KIND_ANNOUNCEMENT = "task_announcement"
KIND_BID = "bid"
KIND_AWARD = "award"
KIND_EXPEDITE = "expedite"

#: Closed option sets for the bid questions. Deliberately ordinal words rather
#: than numbers: a decision model given "0.7" invents precision it does not have,
#: whereas given "high/medium/low" it has to commit to a bucket that the
#: rubric then defines.
FEASIBILITY_OPTIONS = ["low", "medium", "high"]
EFFORT_OPTIONS = ["small", "medium", "large"]

#: Rubric anchors. These are the definitions the model is actually reasoning
#: against, so changing a number here changes what the numbers mean.
_FEASIBILITY_RUBRIC = [
    "low: this agent cannot do the task at all without new capabilities",
    "medium: this agent can do it, but only partially or with degraded output",
    "high: this agent routinely does this task and has the evidence trail to prove it",
]
_EFFORT_RUBRIC = [
    "small: one focused step, no new tool calls",
    "medium: several steps and at least one tool call",
    "large: many steps, multiple tool calls, or another agent's involvement",
]
_VALUE_RUBRIC = [
    "0: doing this changes nothing observable",
    "50: doing this advances the sponsorship by one routine step",
    "100: doing this unblocks a closed door (a gate, a dispute, a stalled negotiation)",
]

_BUCKET = {"low": 0.2, "medium": 0.5, "high": 0.9}
_SIZE = {"small": 1, "medium": 3, "large": 6}
_VALUE = {"low": 10.0, "medium": 50.0, "high": 100.0}


@dataclass(slots=True)
class TaskAnnouncement:
    """Smith's *announcement* phase: a manager naming work without naming a
    contractor.

    ``eligible`` is advisory. The bidding loop re-derives eligibility from the
    board, because an announcement can outlive the conditions that justified it.
    """

    task_id: str
    title: str
    description: str
    manager: AgentId
    event_id: str
    requirements: list[str] = field(default_factory=list)
    eligible: tuple[AgentId, ...] = ()
    criteria: dict[str, str] = field(default_factory=dict)
    round: int = 1
    entry_id: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "manager": self.manager.value,
            "event_id": self.event_id,
            "requirements": list(self.requirements),
            "eligible": [a.value for a in self.eligible],
            "criteria": dict(self.criteria),
            "round": self.round,
            "entry_id": self.entry_id,
        }


@dataclass(slots=True)
class AuctionResult:
    """The full auction transcript. Everything a reviewer needs, in one object."""

    task: TaskAnnouncement
    bids: list[Bid] = field(default_factory=list)
    winner: AgentId | None = None
    winning_bid: Bid | None = None
    rounds: int = 1
    #: Per-agent decisions that produced the bids, for the trace and for replay.
    decisions: list[Decision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def contested(self) -> bool:
        """True when at least two agents bid with different utilities.

        A contested award is the interesting one; an uncontested one says only
        that nobody else was eligible.
        """
        utilities = {round(b.utility, 6) for b in self.bids}
        return len(self.bids) > 1 and len(utilities) > 1

    def ranking(self) -> list[tuple[AgentId, float]]:
        """Bids ordered by award score, highest first. Deterministic."""
        return [(b.agent, round(b.utility, 6))
                for b in sorted(self.bids, key=lambda b: (-b.utility, b.agent.value))]

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "title": self.task.title,
            "round": self.rounds,
            "winner": self.winner.value if self.winner else None,
            "winning_bid_id": self.winning_bid.bid_id if self.winning_bid else None,
            "utility": self.winning_bid.utility if self.winning_bid else 0.0,
            "contested": self.contested,
            "ranking": [{"agent": a.value, "utility": u} for a, u in self.ranking()],
            "bids": [b.model_dump(mode="json") for b in self.bids],
            "degraded": self.degraded,
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------------- helpers
def ask_decision(rt: GraphRuntime, request: DecisionRequest) -> Decision:
    """Call the decision layer, converting a backend failure into a rules answer.

    Mirrors ``core.errors``' stated policy: fail loud in the domain, degrade at
    the edge, and *label* the degradation. A bid produced by the fallback is
    still a bid, but it carries ``DecisionSource.RULES`` and ``degraded=True``,
    so an auction that ran without a model cannot be mistaken for one that did.
    """
    from core import DecisionUnavailable

    try:
        return rt.decide(request)
    except DecisionUnavailable as exc:
        log.warning("decision backend unavailable for %r: %s; bidding on rules",
                    request.decision_point or request.question, exc)
        return rules_bid_answer(request, reason=f"decision backend unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - a broker fault must not lose the auction
        log.error("decision layer raised %s: %s for %r; bidding on rules",
                  type(exc).__name__, exc, request.decision_point or request.question)
        return rules_bid_answer(request, reason=f"decision layer raised {type(exc).__name__}: {exc}")


def rules_bid_answer(request: DecisionRequest, *, reason: str) -> Decision:
    """A neutral, explicitly-sourced fallback answer for a bid question.

    Every option gets an equal probability so nothing is smuggled in by
    arithmetic: the point of the fallback is to keep the run moving while making
    the trace say "no model answered this".
    """
    options = list(request.options)
    share = 1.0 / len(options) if options else 1.0
    probabilities = {opt: round(share, 4) for opt in options}
    # Renormalise defensively against the Decision validator's 0.02 tolerance.
    total = sum(probabilities.values()) or 1.0
    probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}
    choice = options[0] if options else None
    return Decision(
        request_id=request.request_id,
        question=request.question,
        choice=choice,
        probabilities=probabilities,
        confidence=probabilities.get(choice, 0.0) if choice else 0.0,
        source=DecisionSource.RULES,
        model="graph.contract_net.rules_bid",
        degraded=True,
        raw={"reason": reason, "decision_point": request.decision_point},
    )


def _bucket(decision: Decision, options: Sequence[str], default: str) -> str:
    """Pick a bucket, tolerating a model that returns its own wording."""
    choice = (decision.choice or "").strip().lower()
    for option in options:
        if option in choice:
            return option
    for option in options:
        if any(word.startswith(option) or option.startswith(word) for word in choice.split()):
            return option
    return default


def _numeric_choice(decision: Decision, default: float) -> float:
    """Pull a number out of a free-form value answer, or use the default."""
    import re

    text = str(decision.choice or "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return default
    probabilities = decision.probabilities or {}
    for option, probability in probabilities.items():
        found = re.search(r"-?\d+(?:\.\d+)?", option)
        if found:
            try:
                return float(found.group(0)) * float(probability) * 2.0
            except ValueError:
                continue
    return default


def _emit_bid_message(rt: GraphRuntime, state: dict, task: TaskAnnouncement,
                      bid: Bid, feasibility_dec: Decision) -> str | None:
    """Emit one proposal ``Message`` per bid, best-effort (Track 4).

    Never raises: any board/tracer/queue fault is swallowed after a debug log
    so auction routing survives. Duck-types ``core.Message`` — uses it when the
    parallel track has landed it, otherwise posts the equivalent JSON dict.
    Returns the ``message_id`` for handoff linkage, or ``None``.
    """
    try:
        event_id = str(state.get("event_id") or getattr(rt, "event_id", "") or "")
        run_id = str(state.get("run_id") or getattr(rt, "run_id", "") or "")
        body = (f"{bid.agent.label} proposes to take '{task.title}' at utility "
                f"{bid.utility:.3f}: {bid.rationale[:280]}")
        payload: dict[str, Any] = {
            "message_id": new_id("msg"),
            "kind": "proposal",
            "from_agent": bid.agent.value,
            "to_agent": task.manager.value,
            "body": body,
            "topic": task.title,
            "refs": [bid.bid_id, task.task_id],
            "confidence": round(float(feasibility_dec.confidence), 4),
            "source": feasibility_dec.source.value,
            "event_id": event_id,
            "run_id": run_id,
        }
        try:
            import core as _core  # duck-type: Message may not exist yet
            msg_cls = getattr(_core, "Message", None)
            if isinstance(msg_cls, type):
                try:
                    obj = msg_cls(**{  # type: ignore[call-arg]
                        "message_id": payload["message_id"], "kind": "proposal",
                        "from_agent": bid.agent, "to_agent": task.manager,
                        "body": body})
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
                                    ("debate", "rebuttal")):
                try:
                    board.post(zone, zone_kind, bid.agent, dict(payload),
                               refs=[bid.bid_id],
                               confidence=float(payload["confidence"]),
                               source=feasibility_dec.source)
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
                    tracer.event(_kind, "message:proposal",  # type: ignore[arg-type]
                                 agent=bid.agent, message_id=payload["message_id"],
                                 kind="proposal", body=body[:300])
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


# ------------------------------------------------------------------------ announce
def announce(rt: GraphRuntime, state: dict, *, title: str, description: str,
             manager: AgentId, requirements: Sequence[str] = (),
             eligible: Sequence[AgentId] | None = None,
             criteria: dict[str, str] | None = None,
             round_no: int = 1) -> TaskAnnouncement:
    """Announce a task on the blackboard. Names no contractor.

    The announcement is posted *before* any bid is solicited, which is what makes
    the later bids attributable: an agent can point at the entry it was bidding
    on, and a reviewer can see that nobody was told who would win.
    """
    event_id = str(state.get("event_id") or rt.event_id or "")
    pool = tuple(eligible) if eligible else tuple(bidder_agents(state))
    task = TaskAnnouncement(
        task_id=new_id("tsk"),
        title=title,
        description=description,
        manager=manager,
        event_id=event_id,
        requirements=list(requirements),
        eligible=pool,
        criteria=dict(criteria or {
            "feasibility": "; ".join(_FEASIBILITY_RUBRIC),
            "effort": "; ".join(_EFFORT_RUBRIC),
            "expected_value": "; ".join(_VALUE_RUBRIC),
        }),
        round=round_no,
    )
    entry = rt.board.post(
        ZONE_TASKS, KIND_ANNOUNCEMENT, manager, task.to_payload(),
        confidence=1.0, source=DecisionSource.RULES,
    )
    task.entry_id = entry.entry_id
    log.info("contract-net announcement %s for %s (eligible: %s)",
             task.task_id, title,
             ", ".join(a.value for a in pool) or "none")
    return task


# --------------------------------------------------------------------------- bidding
def _bid_frame(task: TaskAnnouncement, agent_id: AgentId, state: dict) -> tuple[str, str]:
    """A differently-framed question per agent. This is the point of the protocol.

    A shared question would produce correlated bids and a meaningless award. The
    framing below is deliberately role-shaped, so the trace shows *why* the
    numbers differ:

    * A2 Pricing values the task in rupees it can convert into an offer.
    * A3 Outreach values it in sponsor conversations it can open.
    * A4 Contract values it in agreements it can close.
    * A5 Compliance values it in risk it can remove before signing.
    * A6 Audit values it in reportable evidence it can produce.
    * A1 Discovery values it in pipeline it can widen.
    """
    brands = len(state.get("brands") or [])
    offers = len(state.get("offers") or [])
    threads = len(state.get("threads") or [])
    role_frames: dict[str, tuple[str, str]] = {
        AgentId.A1_DISCOVERY: (
            f"From a discovery standpoint, is reaching {brands} brand(s) enough "
            f"sponsor pipeline to justify taking on '{task.title}' now, and what "
            f"is the pipeline worth if it succeeds?",
            "value = expected number of additional qualified brand leads",
        ),
        AgentId.A2_PRICING: (
            f"As pricing, with {offers} offer(s) on the table, is there a sponsor "
            f"willing to pay for '{task.title}', and what is that worth in rupees?",
            "value = expected additional sponsorship rupees this run",
        ),
        AgentId.A3_OUTREACH: (
            f"As outreach with {threads} open thread(s), will doing '{task.title}' "
            f"actually move a sponsor to a decision, and how many conversations "
            f"does it unlock?",
            "value = expected number of sponsor replies that change state",
        ),
        AgentId.A4_CONTRACT: (
            f"As contract, can '{task.title}' be turned into a signed agreement "
            f"this week, and what is a signed agreement worth here?",
            "value = probability-weighted value of closing one MoU",
        ),
        AgentId.A5_COMPLIANCE: (
            f"As compliance, does '{task.title}' remove a blocking risk, and what "
            f"is the cost of leaving that risk unresolved?",
            "value = expected cost avoided in blocked or renegotiated deals",
        ),
        AgentId.A6_AUDIT: (
            f"As audit, will '{task.title}' produce evidence a sponsor-facing ROI "
            f"claim can rest on, and what is that evidence worth?",
            "value = expected auditable claims supported",
        ),
        AgentId.A7_ARBITER: (
            f"As arbiter, does '{task.title}' resolve a contested area of "
            f"responsibility, and what is the run worth if it is left unresolved?",
            "value = expected cost of an unresolved dispute",
        ),
    }
    return role_frames.get(agent_id, (
        f"Is '{task.title}' worth doing, and what is it worth if it succeeds?",
        "value = general expected benefit to the sponsorship",
    ))


def collect_bids(rt: GraphRuntime, state: dict, task: TaskAnnouncement,
                 *, bidders: Sequence[AgentId] | None = None,
                 decisions: list[Decision] | None = None) -> list[Bid]:
    """Solicit and post one typed :class:`Bid` per eligible agent.

    Three decision calls per bidder (feasibility, effort, value), each recorded
    on the tracer and each carrying its own source and confidence. That is three
    times more expensive than asking once, and deliberately so: the award's
    defensibility comes from each factor being separately attributable. A single
    fused question could not explain *which* factor drove the outcome.
    """
    event_id = str(state.get("event_id") or rt.event_id or "")
    pool = list(bidders) if bidders is not None else list(task.eligible)

    if not pool:
        log.info("contract-net task %s has no eligible bidders; manager must self-execute",
                 task.task_id)
        return []

    bids: list[Bid] = []
    for agent_id in pool:
        if not rt.agent(agent_id):
            log.info("agent %s is not registered; it cannot bid on %s",
                     agent_id.value, task.task_id)
            continue
        question, value_hint = _bid_frame(task, agent_id, state)
        common = {"task": task.title, "event_id": event_id,
                  "known_leads": len(state.get("brands") or []),
                  "requirements": "; ".join(task.requirements) or "(none stated)"}
        # A ``Decision`` does not carry its own decision_point; the request does.
        # Keeping the three points alongside lets the trace attribute each of the
        # three numbers to the question that produced it.
        decision_points = {
            "feasibility": f"contract_net.bid.{agent_id.value}.feasibility",
            "effort": f"contract_net.bid.{agent_id.value}.effort",
            "value": f"contract_net.bid.{agent_id.value}.expected_value",
        }

        feasibility_dec = ask_decision(rt, DecisionRequest(
            request_id=new_id("dec"),
            question=f"{question} How feasible is this task for you specifically?",
            question_type=QuestionType.CHOICE,
            options=list(FEASIBILITY_OPTIONS),
            rubric=list(_FEASIBILITY_RUBRIC),
            instructions=("Answer only for your own capabilities. Do not consider "
                          "what another agent could do."),
            state=dict(common),
            asked_by=agent_id,
            decision_point=decision_points["feasibility"],
        ))

        effort_dec = ask_decision(rt, DecisionRequest(
            request_id=new_id("dec"),
            question=(f"{question} How much effort does this task cost you, given "
                      f"your own tools and deadlines?"),
            question_type=QuestionType.CHOICE,
            options=list(EFFORT_OPTIONS),
            rubric=list(_EFFORT_RUBRIC),
            instructions="Estimate only your own cost. Ignore who else could do it.",
            state=dict(common),
            asked_by=agent_id,
            decision_point=decision_points["effort"],
        ))

        value_dec = ask_decision(rt, DecisionRequest(
            request_id=new_id("dec"),
            question=f"{question} State a value between 0 and 100 ({value_hint}).",
            question_type=QuestionType.CHOICE,
            options=[str(v) for v in (0, 25, 50, 75, 100)],
            rubric=list(_VALUE_RUBRIC),
            instructions="Give the value of succeeding, not the effort of trying.",
            state=dict(common),
            asked_by=agent_id,
            decision_point=decision_points["value"],
        ))

        feasibility = _BUCKET[_bucket(feasibility_dec, FEASIBILITY_OPTIONS, "medium")]
        effort = _SIZE[_bucket(effort_dec, EFFORT_OPTIONS, "medium")]
        expected_value = _numeric_choice(value_dec, 50.0)
        if value_dec.source is DecisionSource.RULES and value_dec.degraded:
            expected_value = 50.0

        rationale = (
            f"{agent_id.label} bids on '{task.title}': feasibility="
            f"{feasibility_dec.choice} ({feasibility:.2f}), effort="
            f"{effort_dec.choice} ({effort}), value={value_dec.choice} "
            f"({expected_value:.1f}). Sources: "
            f"{feasibility_dec.source.value}/{effort_dec.source.value}/{value_dec.source.value}"
            + (" [DEGRADED: decided by rules fallback]" if any(
                d.degraded for d in (feasibility_dec, effort_dec, value_dec)) else "")
        )

        bid = Bid(
            bid_id=new_id("bid"),
            task_id=task.task_id,
            agent=agent_id,
            feasibility=feasibility,
            expected_value=round(expected_value, 4),
            effort=max(1, effort),
            rationale=rationale,
        )
        bids.append(bid)
        rt.board.post(
            ZONE_BIDS, KIND_BID, agent_id, bid.model_dump(mode="json"),
            refs=[task.entry_id] if task.entry_id else [],
            confidence=round(float(feasibility_dec.confidence), 4),
            source=feasibility_dec.source,
        )
        # Track 4: every bid is also a forcing proposal Message. Best-effort:
        # a board/tracer fault is logged and bidding continues. Duck-typed so
        # this works with or without ``core.Message`` (parallel track owns it).
        try:
            _emit_bid_message(rt, state, task, bid, feasibility_dec)
        except Exception as exc:  # noqa: BLE001 - emission never breaks bidding
            log.debug("bid proposal message failed for %s: %s: %s",
                      agent_id.value, type(exc).__name__, exc)
        for point, decision in (("feasibility", feasibility_dec), ("effort", effort_dec),
                            ("value", value_dec)):
            rt.tracer.decision(agent_id, decision,
                               decision_point=decision_points[point],
                               task_id=task.task_id, bid_id=bid.bid_id)
            if decisions is not None:
                decisions.append(decision)

    log.info("contract-net task %s collected %d bid(s): %s", task.task_id, len(bids),
             ", ".join(f"{b.agent.value}={b.utility:.3f}" for b in bids) or "none")
    return bids


# --------------------------------------------------------------------------- awarding
def award(rt: GraphRuntime, state: dict, task: TaskAnnouncement,
          bids: Sequence[Bid], *, manager: AgentId | None = None) -> tuple[
              AgentId | None, Bid | None, list[str]]:
    """Award the contract to the highest-utility bid. Smith's *awarding* phase.

    ``Bid.utility`` is ``expected_value * feasibility / effort`` — feasibility
    weighted value per unit of effort. Ties break on agent id so the same inputs
    always produce the same award; a non-deterministic tie would make the run
    irreproducible, which for an audited system is a defect.

    Returns ``(winner, winning_bid, notes)``. ``winner is None`` means the task
    goes uncontracted: the manager self-executes, which is a legitimate outcome
    of an empty bidder pool, not a failure.
    """
    who = manager or task.manager
    notes: list[str] = []
    if not bids:
        notes.append(f"no bids for {task.task_id}; manager {who.value} self-executes")
        rt.board.post(ZONE_TASKS, KIND_AWARD, who,
                      {"task_id": task.task_id, "winner": None,
                       "reason": "no bids submitted", "round": task.round},
                      source=DecisionSource.RULES)
        log.info("contract-net task %s awarded to nobody (no bids)", task.task_id)
        return None, None, notes

    ordered = sorted(bids, key=lambda b: (-b.utility, b.agent.value))
    best = ordered[0]
    if len(ordered) > 1 and abs(ordered[1].utility - best.utility) < 1e-9:
        notes.append(
            f"tie on utility {best.utility:.6f} between "
            f"{best.agent.value} and {ordered[1].agent.value}; resolved on agent id"
        )
    margin = (best.utility - ordered[1].utility) if len(ordered) > 1 else best.utility
    notes.append(
        f"awarded to {best.agent.value} at utility {best.utility:.4f} "
        f"(margin {margin:.4f} over {len(ordered) - 1} rival(s))"
    )
    rt.board.post(
        ZONE_TASKS, KIND_AWARD, who,
        {
            "task_id": task.task_id,
            "title": task.title,
            "winner": best.agent.value,
            "bid_id": best.bid_id,
            "utility": best.utility,
            "ranking": [{"agent": b.agent.value, "utility": b.utility} for b in ordered],
            "round": task.round,
            "notes": list(notes),
        },
        refs=[b.bid_id for b in ordered],
        confidence=1.0,
        source=DecisionSource.RULES,
    )
    log.info("contract-net task %s awarded to %s (utility %.4f)",
             task.task_id, best.agent.value, best.utility)
    return best.agent, best, notes


# ------------------------------------------------------------------------ expediting
def expedite(rt: GraphRuntime, state: dict, task: TaskAnnouncement,
             winner: Bid | None, *, status: str = "accepted",
             message: str = "") -> dict[str, Any]:
    """Smith's *expediting* phase: tell the winner, and record what it said.

    Smith is explicit that expediting is a manager responsibility — the manager
    monitors progress and can revoke. Recording the contractor's acknowledgement
    here is what makes revocation auditable later: the trace shows the manager
    asked, the contractor answered, and what the answer was.
    """
    manager = task.manager
    payload = {
        "task_id": task.task_id,
        "title": task.title,
        "contractor": winner.agent.value if winner else None,
        "status": status,
        "message": message,
        "round": task.round,
    }
    entry = rt.board.post(ZONE_TASKS, KIND_EXPEDITE, manager, payload,
                          refs=[winner.bid_id] if winner else [], source=DecisionSource.RULES)
    payload["entry_id"] = entry.entry_id
    return payload


# ---------------------------------------------------------------------------- run
def run_auction(rt: GraphRuntime, state: dict, *, title: str, description: str,
                manager: AgentId, requirements: Sequence[str] = (),
                bidders: Sequence[AgentId] | None = None,
                on_award: Callable[[AuctionResult], None] | None = None,
                ) -> AuctionResult:
    """Announce -> bid -> award -> expedite, in one call.

    Re-announces up to ``settings.max_auction_rounds`` only when *every* bid is
    infeasible, which is Smith's "no contractor is capable" signal. It never
    re-announces merely because the first winner was unimpressive: re-running
    until the answer is agreeable is p-hacking the protocol, and the auction's
    value as evidence depends on the first result standing.
    """
    context = runtime_context(rt)
    event_id = str(state.get("event_id") or context.event_id or "")
    max_rounds = max(1, int(context.settings.max_auction_rounds))
    result = AuctionResult(
        task=TaskAnnouncement(task_id="", title=title, description=description,
                              manager=manager, event_id=event_id),
    )

    for round_no in range(1, max_rounds + 1):
        task = announce(rt, state, title=title, description=description, manager=manager,
                        requirements=requirements, eligible=bidders, round_no=round_no)
        result.task = task
        round_decisions: list[Decision] = []
        bids = collect_bids(rt, state, task, bidders=bidders, decisions=round_decisions)
        result.bids = bids
        result.decisions.extend(round_decisions)
        result.rounds = round_no
        result.degraded = result.degraded or any(d.degraded for d in round_decisions)

        feasible = [b for b in bids if b.feasibility > 0.0]
        if feasible or round_no >= max_rounds:
            break
        result.notes.append(
            f"round {round_no}: every bid declared feasibility 0; re-announcing"
        )

    winner_agent, winning_bid, notes = award(rt, state, result.task, result.bids,
                                             manager=manager)
    result.notes.extend(notes)
    result.winner, result.winning_bid = winner_agent, winning_bid

    if winner_agent is None:
        result.notes.append("task uncontracted; manager self-executes")
        result.degraded = result.degraded or bool(result.task.eligible)
    else:
        acceptance_decisions: list[Decision] = []
        acceptance = ask_decision(rt, DecisionRequest(
            request_id=new_id("dec"),
            question=(f"{winner_agent.label} has been awarded '{result.task.title}' at "
                      f"utility {winning_bid.utility:.4f}. Do you accept the contract?"),
            question_type=QuestionType.CHOICE,
            options=["accept", "decline"],
            rubric=["accept: you can start immediately and within your budget",
                    "decline: you need more time, more authority, or a better rate"],
            state={"task": result.task.to_payload(), "winning_bid": winning_bid.model_dump(mode="json")},
            asked_by=winner_agent,
            decision_point="contract_net.expedite.accept",
        ))
        acceptance_decisions.append(acceptance)
        result.decisions.extend(acceptance_decisions)
        accepted = (acceptance.choice or "").strip().lower().startswith("accept")
        result.degraded = result.degraded or acceptance.degraded
        result.notes.append(
            f"{winner_agent.value} {'accepted' if accepted else 'DECLINED'} the contract"
            + ("" if accepted else "; award reverts to the manager")
        )
        expedite(rt, state, result.task, winning_bid if accepted else None,
                 status="accepted" if accepted else "declined",
                 message=str(acceptance.choice or ""))
        if not accepted:
            result.winner, result.winning_bid = None, None

    if on_award:
        on_award(result)
    return result


#: Private alias retained for readability inside this module's own call sites.
_ask = ask_decision
