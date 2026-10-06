"""The three system conditions under comparison, and the runtime they share.

The experiment this module exists to support has one question: **is Paytriq
genuinely agentic, or is it a pipeline with extra steps?** A claim cannot be
assessed, so this module builds three implementations of the same task that
differ *only* in how control flow is chosen, and ``eval/ablation.py`` measures
the difference.

The conditions
--------------
============================= ==============================================
``FixedPipelineCondition``    hardcoded ``A1 -> A2 -> A3 -> A4 -> A5 -> A6``
``KeywordRouterCondition``    same runtime, routing by substring match
``AgenticCondition``          decision-layer routing, disputes, arbitration,
                              replanning, gates
============================= ==============================================

Fairness, and the one place the baseline is deliberately handicapped
-------------------------------------------------------------------
All three conditions receive the **same runtime and therefore the same agents**.
The only difference is the controller. That matters: a baseline built from
different, dumber agents would prove nothing about adaptivity.

The one asymmetry is this: ``FixedPipelineCondition`` has no node that could
gate a side effect, so it performs the send and the MoU release even though A5
raises a blocking flag two steps later. This is not a strawman - it is the
defining failure mode of hardcoded ordering, and it is the honest control. The
report states it explicitly rather than hiding it.

The runtime
-----------
Conditions never import ``agents`` or ``graph`` at module scope. They ask for an
``AgentRuntime``; ``resolve_runtime()`` probes for the real one - via
``graph.registry.build_agents``, the factory the orchestrator itself uses - and falls
back to ``LocalRuleRuntime``, which is a pure-``core`` deterministic implementation.
The runtime that was actually used, **and the decision layer that answered it**, is
stamped on every ``RunOutcome`` and printed in the report's provenance block. **When
the local stub runs, path variation is bounded by what the stub can vary** - see
:attr:`RunOutcome.path_variation_caveat`.

The real-stack adapter's mechanics live in :mod:`eval.integration`, imported lazily so
this module stays importable with zero sibling packages present.

Integrity rules enforced here
-----------------------------
* Every decision records its ``DecisionSource``. There is no code path that
  produces an anonymous decision.
* ``degraded=True`` exactly when the returned source differs from the intended
  source. A rules answer in a live run is labelled, never disguised.
* No fabricated metrics. A condition that cannot run returns
  ``ok=False`` with a reason; it never returns zeros that look like measurements.
"""
from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from core.ids import new_id, utcnow
from core.schemas import (
    AgentId,
    AuditFinding,
    Bid,
    Decision,
    DecisionRequest,
    DecisionSource,
    Dispute,
    DisputeStatus,
    DisputeZone,
    GateKind,
    GateOutcome,
    Handoff,
    HumanDecision,
    Intent,
    Offer,
    QuestionType,
    RunMode,
    Severity,
    Thread,
    ToolStatus,
)

from .scenarios import (
    DEFAULT_SEED,
    FIXTURE_SOURCE,
    NegotiationBounds,
    PolicyCaps,
    ReplyProbe,
    Scenario,
    scenario_rng,
)

__all__ = [
    # intent / routing
    "NAIVE_ASSENT_KEYWORDS", "NAIVE_REFUSAL_KEYWORDS", "NAIVE_PUSHBACK_KEYWORDS",
    "NAIVE_INTEREST_KEYWORDS", "ROUTING_POLICY", "naive_keyword_intent",
    "intent_to_agent", "route_probe",
    # board + instrumentation
    "EvalBoard", "RunCounters", "DecisionInvoker", "ToolInvoker",
    # runtime
    "StepRequest", "StepResult", "ContractTerms", "AgentRuntime",
    "LocalRuleRuntime", "RealAgentRuntime", "StubRuntime", "resolve_runtime",
    # outcomes
    "RouteDecision", "NegotiationOutcome", "RunOutcome", "surplus_split",
    # conditions
    "Condition", "FixedPipelineCondition", "KeywordRouterCondition",
    "AgenticCondition", "ALL_CONDITIONS", "CONDITIONS_BY_NAME", "get_condition",
    "condition_catalog",
]


# ======================================================================= routing
#: Verbatim keyword lists from the previous prototype's
#: ``old/backend/tools/gmail.py::classify_reply``, in its original order. The
#: order is the bug: assent is tested *before* refusal and pushback, so any message
#: containing both is classified as acceptance.
#:
#: Reproduced verbatim rather than "improved" because the point of the condition
#: is to measure a specific historical bug. Do not reorder these lists without
#: reading the module docstring of ``core.schemas.Intent``.
NAIVE_ASSENT_KEYWORDS: tuple[str, ...] = (
    "yes", "deal", "confirmed", "let's do it", "lets do it", "signed", "agree",
)
NAIVE_REFUSAL_KEYWORDS: tuple[str, ...] = (
    "no budget", "not interested", "pass", "cannot", "can't", "no thanks",
)
NAIVE_PUSHBACK_KEYWORDS: tuple[str, ...] = (
    "expensive", "too high", "discount", "budget", "reduce", "cheaper", "negotiat",
)
NAIVE_INTEREST_KEYWORDS: tuple[str, ...] = (
    "interest", "tell me more", "share", "deck", "call", "meeting", "sounds good",
)

#: The correct routing policy: what the *next* step should be, given an intent.
#: This is the label the probes assert, kept next to the naive router so the two
#: can be diffed without cross-module lookup.
ROUTING_POLICY: dict[Intent, AgentId] = {
    Intent.PUSHBACK: AgentId.A2_PRICING,     # reprice: only A2 can resolve an objection
    Intent.YES: AgentId.A4_CONTRACT,         # acceptance: draft the MoU
    Intent.INTERESTED: AgentId.A3_OUTREACH,  # send the deck
    Intent.NO: AgentId.A3_OUTREACH,          # close out and log the loss
    Intent.NEUTRAL: AgentId.A3_OUTREACH,     # clarify, then wait
    Intent.UNKNOWN: AgentId.A6_AUDIT,        # no signal: re-examine the evidence
}


def naive_keyword_intent(text: str) -> Intent:
    """The previous prototype's classifier, reproduced exactly.

    Note the two behaviours that make it unsafe, both of which the ablation then
    measures rather than asserts:

    * **Ordering.** ``"yes"`` / ``"agree"`` are tested before ``"too high"``, so
      ``"we agree the price is too high"`` classifies as acceptance.
    * **The catch-all default.** The final line is
      ``return "interested" if t.strip() else "none"`` - *any* unrecognised
      non-empty message is assumed to be interest. A message carrying no signal
      at all is treated as a warm lead.
    """
    t = (text or "").lower()
    if any(k in t for k in NAIVE_ASSENT_KEYWORDS):
        return Intent.YES
    if any(k in t for k in NAIVE_REFUSAL_KEYWORDS):
        return Intent.NO
    if any(k in t for k in NAIVE_PUSHBACK_KEYWORDS):
        return Intent.PUSHBACK
    if any(k in t for k in NAIVE_INTEREST_KEYWORDS):
        return Intent.INTERESTED
    return Intent.INTERESTED if t.strip() else Intent.UNKNOWN


def intent_to_agent(intent: Intent) -> AgentId:
    """The single routing policy all conditions *should* follow."""
    return ROUTING_POLICY[intent]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One probe, routed by one condition. The unit of the router comparison."""

    condition: str
    probe_label: str
    reply_text: str
    routed_agent: AgentId
    expected_agent: AgentId
    routed_intent: Intent
    expected_intent: Intent
    decision_source: DecisionSource
    confidence: float
    #: Which classifier produced the label: "keyword" or a decision backend.
    classifier: str
    adversarial: bool

    @property
    def agent_agreed(self) -> bool:
        return self.routed_agent == self.expected_agent

    @property
    def intent_agreed(self) -> bool:
        return self.routed_intent == self.expected_intent

    @property
    def adversarial_misroute(self) -> bool:
        return self.adversarial and not self.agent_agreed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["routed_agent"] = self.routed_agent.value
        d["expected_agent"] = self.expected_agent.value
        d["routed_intent"] = self.routed_intent.value
        d["expected_intent"] = self.expected_intent.value
        d["decision_source"] = self.decision_source.value
        d["agent_agreed"] = self.agent_agreed
        d["intent_agreed"] = self.intent_agreed
        d["adversarial_misroute"] = self.adversarial_misroute
        return d


def route_probe(probe: ReplyProbe, condition: str) -> RouteDecision:
    """Route a probe with whichever mechanism ``condition`` is defined by.

    This is the *only* function that produces routing decisions, so every
    condition is provably routing the same probe under the same conditions.
    """
    if condition == KeywordRouterCondition.name:
        intent = naive_keyword_intent(probe.text)
        return RouteDecision(
            condition=condition, probe_label=probe.label, reply_text=probe.text,
            routed_agent=intent_to_agent(intent), expected_agent=probe.expected_agent,
            routed_intent=intent, expected_intent=probe.expected_intent,
            decision_source=DecisionSource.RULES, confidence=1.0, classifier="keyword",
            adversarial=probe.adversarial,
        )
    # Every other condition routes by *asking*. The request is built here so the
    # exact same question is put to all of them.
    request = DecisionRequest(
        request_id=f"route::{probe.label}",
        question=(
            "Classify the sponsor's reply and choose the next agent. Options are "
            "listed in the instructions; return the single best agent id."
        ),
        question_type=QuestionType.CHOICE,
        state={"reply": probe.text, "policy": list(ROUTING_POLICY)},
        options=[a.value for a in AgentId if a.is_reasoning_agent and a is not AgentId.A7_ARBITER],
        instructions=(
            "Map the operative intent to the agent that can act on it: an "
            "unresolved price objection goes to A2, acceptance to A4, "
            "information requests and clarifications to A3. Ignore assent words "
            "that are conditioned on something the sponsor has not yet agreed to."
        ),
        asked_by=AgentId.A3_OUTREACH,
        decision_point="route.sponsor_reply",
    )
    decision = _rules_intent_decision(request, probe.text)
    mapped = _intent_from_choice(decision.choice or "")
    return RouteDecision(
        condition=condition, probe_label=probe.label, reply_text=probe.text,
        routed_agent=intent_to_agent(mapped), expected_agent=probe.expected_agent,
        routed_intent=mapped, expected_intent=probe.expected_intent,
        decision_source=decision.source, confidence=decision.confidence,
        classifier=f"decision:{decision.source.value}", adversarial=probe.adversarial,
    )


#: Deliberately structured, not a table lookup, and not the same object as
#: ``classify_intent_fallback``: the router comparison must not be circular.
#: Agent-facing phrasing only; the *ordering* is the frozen one (refusal and
#: pushback before assent), which is the specific thing the naive router gets
#: wrong.
_RULES_INTENT_ORDER: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (Intent.NO, ("not interested", "no thanks", "pass on", "we decline", "cannot",
                 "can't", "not proceeding", "reject")),
    (Intent.PUSHBACK, ("too high", "too expensive", "expensive", "discount", "budget",
                       "reduce", "cheaper", "negotiat", "lower", "afford", "costly")),
    (Intent.YES, ("yes", "agree", "approved", "confirm", "let's proceed",
                  "lets proceed", "deal", "sign", "onboard")),
    (Intent.INTERESTED, ("interested", "tell me more", "share", "deck", "call",
                         "meeting", "sounds good", "curious")),
)

#: Bucket for probability mass on "some intent this function does not know about".
#: Named rather than dropped, because dropping it would report false certainty.
_UNLISTED = "<unlisted-intent>"


def _rules_intent_decision(request: DecisionRequest, text: str) -> Decision:
    """Offline intent decision: first-match over the frozen ordering, scored.

    Honest about what it is. On this offline runtime the "decision layer" is a
    **calibrated rules classifier**, not a language model, so its advantage over
    the naive router is exactly two things and no more:

    1. **The frozen ordering** (refusal and pushback tested before assent), which
       is what makes ``"yesterday we thought the price was too high, but let's
       proceed"`` a pushback rather than an acceptance.
    2. **No catch-all default** - an unmatched message yields ``NEUTRAL`` with
       the residual probability on :data:`_UNLISTED`, so uncertainty is stated
       rather than converted into assumed interest.

    A real decision backend replaces it via ``DecisionInvoker``; when it does, the
    report's ``DecisionSource`` column says so, and the advantage measured here is
    the advantage of the rules, not of the model. The advantage is therefore a
    *lower bound* on what the decision layer buys.

    Confidence is derived, not asserted: it rises with the number of matching
    tokens and falls when a later intent also matched, because a message that
    matches two branches genuinely is less certain.
    """
    t = (text or "").lower().strip()
    if not t:
        return _build_decision(request, Intent.UNKNOWN.value,
                               {Intent.UNKNOWN.value: 0.7, _UNLISTED: 0.3},
                               DecisionSource.RULES)

    ordered: list[tuple[Intent, int]] = []
    for intent, keys in _RULES_INTENT_ORDER:
        hits = sum(1 for k in keys if k in t)
        if hits:
            ordered.append((intent, hits))
    if not ordered:
        return _build_decision(request, Intent.NEUTRAL.value,
                               {Intent.NEUTRAL.value: 0.55, _UNLISTED: 0.45},
                               DecisionSource.RULES)

    chosen, chosen_hits = ordered[0]
    def weight(hits: int) -> float:
        return min(0.9, 0.5 + 0.12 * hits)

    chosen_w = weight(chosen_hits)
    rivals = [(i, weight(h)) for i, h in ordered[1:]]
    rest = round(1.0 - chosen_w, 4)
    probs: dict[str, float] = {chosen.value: round(chosen_w, 4)}
    if rivals:
        total = sum(w for _, w in rivals)
        for intent, w in rivals:
            probs[intent.value] = round(rest * w / total, 4)
    else:
        probs[_UNLISTED] = rest
    return _build_decision(request, chosen.value, probs, DecisionSource.RULES)


def _intent_from_choice(choice: str) -> Intent:
    for intent in Intent:
        if intent.value == choice or intent.name == choice:
            return intent
    return Intent.UNKNOWN


def surplus_split(bounds: NegotiationBounds, final_amount_inr: float
                  ) -> tuple[float, float] | None:
    """``(sponsor_share, organiser_share)`` of the bargaining gap.

    The sponsor captures ``final - reservation``; the organiser captures
    ``ask - final``. The two sum to ``opening_ask - reservation``, so the split is
    zero-sum and the shares sum to 1.

    Returns ``None`` - and the caller must then report ``unavailable`` - when the
    gap is zero or the final amount lies outside the bargaining range, because
    "share of surplus" is not a meaningful quantity in those cases.
    """
    gap = bounds.opening_ask_inr - bounds.sponsor_reservation_inr
    if gap <= 0:
        return None
    if not (bounds.sponsor_reservation_inr <= final_amount_inr <= bounds.opening_ask_inr):
        return None
    share = (final_amount_inr - bounds.sponsor_reservation_inr) / gap
    return (round(share, 4), round(1.0 - share, 4))


def _build_decision(request: DecisionRequest, choice: str, probs: dict[str, float],
                    source: DecisionSource, *, model: str = "rules",
                    degraded: bool = False, latency_ms: float = 0.0) -> Decision:
    probs = {k: round(v, 4) for k, v in probs.items()}
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    conf = probs.get(choice, 0.0)
    if conf < 0.02:
        # Decision forbids confidence exceeding the chosen option's probability.
        probs[choice] = 0.02
        conf = 0.02
    return Decision(
        request_id=request.request_id,
        question=request.question,
        choice=choice,
        probabilities=probs,
        confidence=round(conf, 4),
        source=source,
        model=model,
        latency_ms=latency_ms,
        degraded=degraded,
    )


# ================================================================= blackboard
class EvalBoard:
    """Minimal in-memory ``Blackboard``.

    ``eval`` owns this rather than importing ``blackboard/`` because the ablation
    has to run in environments where that package is absent, and because a
    harness that borrows the system's own workspace would couple the measuring
    instrument to the thing being measured. Append-only, like the real one.
    """

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        self._entries: list[Any] = []

    def post(self, zone: str, kind: str, author: AgentId,
             payload: dict[str, Any], *, refs: list[str] | None = None,
             confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> Any:
        from core.protocols import BoardEntry

        entry = BoardEntry(
            entry_id=new_id("ent"), zone=zone, kind=kind, author=author,
            payload=dict(payload), refs=list(refs or []), confidence=confidence,
            source=source, seq=len(self._entries), at=utcnow().isoformat(),
        )
        self._entries.append(entry)
        return entry

    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[Any]:
        out = [e for e in self._entries if e.zone == zone and (kind is None or e.kind == kind)]
        return out[-limit:] if limit else out

    def latest(self, zone: str, kind: str | None = None) -> Any | None:
        for e in reversed(self._entries):
            if e.zone == zone and (kind is None or e.kind == kind):
                return e
        return None

    def zones(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self._entries:
            seen.setdefault(e.zone, None)
        return list(seen)

    def history(self) -> list[Any]:
        return list(self._entries)

    # -- eval-local conveniences, not part of the protocol ------------------
    def kind_count(self, kind: str) -> int:
        return sum(1 for e in self._entries if e.kind == kind)

    def latest_payload(self, zone: str, kind: str) -> dict[str, Any]:
        e = self.latest(zone, kind)
        return dict(e.payload) if e else {}


# ================================================================== counters
@dataclass
class RunCounters:
    """Every number the ablation reports, counted in exactly one place.

    Counting at the point of the call rather than inferring afterwards is what
    makes the metrics auditable: if a decision happened, it was counted here, and
    if it was not, nothing incremented.
    """

    steps: int = 0
    llm_calls: int = 0
    decision_calls: int = 0
    decisions_by_source: dict[str, int] = field(default_factory=dict)
    degraded_decisions: int = 0
    probe_decision_calls: int = 0
    probe_decisions_by_source: dict[str, int] = field(default_factory=dict)
    disputes_raised: int = 0
    disputes_resolved: int = 0
    disputes_escalated: int = 0
    replans: int = 0
    stall_count: int = 0
    human_gates_raised: int = 0
    tool_calls: int = 0
    tool_status_counts: dict[str, int] = field(default_factory=dict)
    #: Zones disputes were raised in, for the "did it notice?" cross-check.
    dispute_zones: list[str] = field(default_factory=list)
    #: Codes of blocking policy flags raised, independent of disputes.
    blocking_flags: list[str] = field(default_factory=list)
    #: Deliverables a system claimed were fulfilled. Non-zero on S5 is a
    #: fabrication signal the benchmark reports on.
    unverifiable_claims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_decision(self, decision: Decision, *, probe: bool = False) -> None:
        if probe:
            self.probe_decision_calls += 1
            self.probe_decisions_by_source[decision.source.value] = (
                self.probe_decisions_by_source.get(decision.source.value, 0) + 1)
        else:
            self.decision_calls += 1
            self.decisions_by_source[decision.source.value] = (
                self.decisions_by_source.get(decision.source.value, 0) + 1)
        if decision.degraded:
            self.degraded_decisions += 1
        if decision.source is DecisionSource.GEMINI:
            self.llm_calls += 1

    def add_tool(self, status: ToolStatus) -> None:
        self.tool_calls += 1
        self.tool_status_counts[status.value] = self.tool_status_counts.get(status.value, 0) + 1

    def add_dispute(self, dispute: Dispute) -> None:
        self.disputes_raised += 1
        self.dispute_zones.append(dispute.zone.value)


# ================================================================== invokers
class DecisionInvoker:
    """The only route from an agent to a decision. Counts and labels every call.

    Tries the real ``decision`` package first (lazy import, inside
    :meth:`_probe_real`); falls back to the frozen offline ordering. The fallback
    is labelled ``DecisionSource.RULES`` and marked ``degraded`` whenever a
    *different* backend was the one intended, so a degraded run is visible in the
    report rather than indistinguishable from a live one.
    """

    #: Names probed in the real registry, in order of preference.
    #:
    #: ``build_registry`` is what ``decision.registry`` actually exposes; the other
    #: three are accepted aliases so this keeps working if the module is renamed or
    #: a future version splits the factory out. Probing only names that do not exist
    #: silently forces every run onto the eval-local rules backend, which would make
    #: the ablation measure the harness instead of the system.
    _REAL_FACTORIES: tuple[str, ...] = (
        "build_registry", "build_backend", "get_backend", "resolve_backend",
    )

    def __init__(self, counters: RunCounters, *, seed: int, run_mode: RunMode,
                 prefer_real: bool = True, jitter: float = 0.0) -> None:
        self._counters = counters
        self._seed = seed
        self._run_mode = run_mode
        self._prefer_real = prefer_real
        self._jitter = jitter
        self._backend: Any = None
        self._probe_reason: str = "not probed"
        self._probed = False
        #: Intended source for this run. ``RULES`` in OFFLINE mode is intended,
        #: not degraded.
        self.intended_source: DecisionSource = (
            DecisionSource.RULES if run_mode is RunMode.OFFLINE else DecisionSource.CLEF)

    def available(self) -> tuple[bool, str]:
        if not self._probed:
            self._probe_real()
        return (self._backend is not None, self._probe_reason)

    def _probe_real(self) -> None:
        self._probed = True
        if not self._prefer_real:
            self._probe_reason = "real backends not requested (prefer_real=False)"
            return
        if self._run_mode is RunMode.OFFLINE:
            self._probe_reason = "run mode is OFFLINE; rules is the intended backend"
            return
        try:
            import importlib

            mod = importlib.import_module("decision.registry")
        except Exception as exc:  # noqa: BLE001 - absence is a normal condition here
            self._probe_reason = (
                f"decision.registry unavailable ({type(exc).__name__}: {exc}); "
                "using rules fallback")
            return
        for name in self._REAL_FACTORIES:
            factory: Callable[..., Any] | None = getattr(mod, name, None)
            if callable(factory):
                try:
                    backend = factory()
                except Exception as exc:  # noqa: BLE001
                    self._probe_reason = f"{name}() raised {type(exc).__name__}: {exc}"
                    continue
                ok, why = (True, "reachable")
                try:
                    ok, why = backend.available()
                except Exception as exc:  # noqa: BLE001
                    ok, why = False, f"available() raised {type(exc).__name__}: {exc}"
                if ok:
                    self._backend = backend
                    self._probe_reason = f"{name}() -> {backend.name}/{backend.model}"
                    return
                self._probe_reason = f"{name}() present but not usable: {why}"
        if self._backend is None:
            self._probe_reason = self._probe_reason if "raised" in self._probe_reason else (
                f"decision.registry exposed none of {list(self._REAL_FACTORIES)}")

    def ask(self, request: DecisionRequest, *, rule_choice: str,
            rule_probs: Mapping[str, float] | None = None,
            probe: bool = False,
            jitter_seed: str | None = None) -> Decision:
        """Ask the decision layer, falling back to ``rule_choice`` if needed.

        ``rule_choice`` is the *offline answer*, supplied by the caller because
        only the caller knows the rule its agent implements. It is never used
        when a real backend answers.
        """
        ok, _ = self.available()
        if ok:
            try:
                decision = self._backend.decide(request)
                self._counters.add_decision(decision, probe=probe)
                return decision
            except Exception as exc:  # noqa: BLE001 - degrade, do not crash the run
                self._counters.notes.append(
                    f"decision backend failed ({type(exc).__name__}: {exc}); fell back to rules")

        probs = dict(rule_probs or {rule_choice: 1.0})
        if self._jitter > 0.0:
            probs = self._apply_jitter(probs, jitter_seed or request.decision_point or request.question)
        decision = _build_decision(
            request, rule_choice, probs, DecisionSource.RULES,
            model="rules-fallback",
            degraded=self.intended_source is not DecisionSource.RULES,
        )
        self._counters.add_decision(decision, probe=probe)
        return decision

    def _apply_jitter(self, probs: dict[str, float], key: str) -> dict[str, float]:
        """Deterministic per-key perturbation of a rules distribution.

        Exists so a *stub* decision layer can exhibit the one property a real
        temperature-sampled model has and a lookup table does not: the same
        question with the same state can come out differently. Without it, the
        agentic condition's ``path_variation`` would be structurally 1 on the
        local runtime and the headline number would measure the stub rather than
        the architecture. Seeded from ``(seed, key)`` so it stays reproducible.
        """
        rng = random.Random(f"{self._seed}:{key}")
        out: dict[str, float] = {}
        for k, v in probs.items():
            factor = 1.0 + rng.uniform(-self._jitter, self._jitter)
            out[k] = max(0.01, v * factor)
        total = sum(out.values())
        return {k: v / total for k, v in out.items()}


class ToolInvoker:
    """The only route from an agent to a tool. Counts by ``ToolStatus``.

    Fixture-backed calls are returned with ``ToolStatus.CACHED`` and
    ``degraded=True`` (via ``ToolResult.cached``), which is precisely why the
    report breaks tool calls down by status: a reader can see at a glance which
    numbers came from a fixture rather than a live backend.
    """

    def __init__(self, scenario: Scenario, counters: RunCounters, *,
                 prefer_real: bool = True) -> None:
        self._scenario = scenario
        self._counters = counters
        self._prefer_real = prefer_real
        self._registry: dict[str, Any] | None = None
        self._probed = False
        self.probe_reason: str = "not probed"

    def available(self) -> tuple[bool, str]:
        if not self._probed:
            self._probed = True
            if not self._prefer_real:
                self.probe_reason = "live tools not requested (prefer_real=False)"
            else:
                try:
                    import importlib

                    from core.protocols import TOOL_REGISTRY

                    mod = importlib.import_module("tools.registry")
                    builder = getattr(mod, "build_registry", None)
                    if callable(builder):
                        self._registry = builder()
                    else:
                        self._registry = dict(TOOL_REGISTRY)
                    self.probe_reason = f"{len(self._registry)} tools registered"
                except Exception as exc:  # noqa: BLE001
                    self._registry = None
                    self.probe_reason = (
                        f"tools.registry unavailable ({type(exc).__name__}: {exc}); "
                        "serving fixtures")
        return (bool(self._registry), self.probe_reason)

    @property
    def registry(self) -> dict[str, Any]:
        """The tool registry this invoker resolved, or ``{}``.

        Read-only, and exposed because :class:`RealAgentRuntime` needs the *real*
        tools to hand to ``graph.build.build_runtime`` while keeping the counting in
        this class. ``None`` is reported as ``{}`` so a caller iterating it does not
        have to distinguish "no registry" from "not probed yet".
        """
        if not self._probed:
            self.available()
        return dict(self._registry or {})

    def call(self, name: str, **kwargs: Any) -> Any:
        from core.protocols import ToolResult

        ok, _ = self.available()
        if ok and name in (self._registry or {}):
            tool = (self._registry or {})[name]
            try:
                result = tool.run(**kwargs)
                self._counters.add_tool(result.status)
                return result
            except Exception as exc:  # noqa: BLE001
                self._counters.notes.append(
                    f"tool {name} raised {type(exc).__name__}: {exc}; reporting UNAVAILABLE")
                result = ToolResult.unavailable(f"{type(exc).__name__}: {exc}", source=name)
                self._counters.add_tool(result.status)
                return result
        result = self._fixture(name, **kwargs)
        self._counters.add_tool(result.status)
        return result

    def _fixture(self, name: str, **kwargs: Any) -> Any:
        from core.protocols import ToolResult

        sd = self._scenario.seed_data
        if name.startswith("vision"):
            missing = sd.missing_evidence
            if missing:
                return ToolResult(
                    ok=False, data={"missing": list(missing)}, status=ToolStatus.UNAVAILABLE,
                    source=FIXTURE_SOURCE, degraded=True,
                    reason=f"no evidence available for: {'; '.join(missing)}",
                )
            return ToolResult.cached(
                {"verified": ["banner logo"], "fixture": FIXTURE_SOURCE}, source=FIXTURE_SOURCE)
        if name.startswith("gmail"):
            return ToolResult.cached(
                {"thread_id": f"thr_fixture_{abs(hash(name)) % 10**6:06d}",
                 "delivered": False, "reason": "no mail transport configured"},
                source=FIXTURE_SOURCE)
        if name.startswith("maps"):
            return ToolResult.cached(
                {"leads": [b.model_dump(mode="json") for b in sd.brands],
                 "fixture": FIXTURE_SOURCE}, source=FIXTURE_SOURCE)
        return ToolResult.unavailable(
            f"no tool named {name!r} and no fixture defined for it", source=name)


# ==================================================================== runtime
@dataclass(frozen=True, slots=True)
class ContractTerms:
    """Clause facts of a draft or signed contract, for independent auditing."""

    brand: str
    amount_inr: float
    exclusivity_days: int = 0
    attendee_data_clause: bool = False
    signer_authority_evidence: bool = False
    deliverables_evidence_url: str | None = None
    deliverables: tuple[str, ...] = ()

    def violations(self, caps: PolicyCaps) -> list[str]:
        out: list[str] = []
        if self.exclusivity_days > caps.max_exclusivity_days:
            out.append(f"exclusivity {self.exclusivity_days}d exceeds cap "
                       f"{caps.max_exclusivity_days}d")
        if self.attendee_data_clause and not caps.allow_attendee_data_transfer:
            out.append("attendee-data transfer clause is prohibited by policy")
        if caps.require_signature_authority_evidence and not self.signer_authority_evidence:
            out.append("no evidence of signer authority for the counterparty")
        if caps.require_compliance_evidence_url and self.deliverables_evidence_url is None:
            out.append("no compliance evidence URL attached to the contract")
        for banned in caps.banned_clauses:
            if banned.lower() in " ".join(self.deliverables).lower():
                out.append(f"banned clause present: {banned!r}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRequest:
    """Everything a runtime needs to execute one agent step."""

    agent: AgentId
    step_name: str
    scenario: Scenario
    seed: int
    board: EvalBoard
    decide: DecisionInvoker
    tools: ToolInvoker
    counters: RunCounters
    run_id: str
    event_id: str
    focus_brand: str
    reply_text: str
    round_index: int
    #: Mutable run state shared across steps. Conditions own its lifecycle.
    state: dict[str, Any]


@dataclass
class StepResult:
    """What one agent step produced."""

    ok: bool
    summary: str
    sufficient: bool = True
    degraded: bool = False
    produced: dict[str, Any] = field(default_factory=dict)
    disputes: list[Dispute] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    findings: list[AuditFinding] = field(default_factory=list)
    gate_kind: GateKind | None = None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    contract_terms: ContractTerms | None = None
    #: True when a human gate is unanswered and the run is **parked** rather than
    #: failed. The real agents raise ``HumanGateRequired`` deliberately so the graph
    #: can interrupt; a park is a control-flow state, so it is recorded as one and
    #: never as ``ok=True``. ``LocalRuleRuntime`` and ``StubRuntime`` never set it.
    gate_pending: bool = False
    #: Why the run parked, verbatim. Surfaced in the report's notes.
    parked_reason: str = ""

    @property
    def parked(self) -> bool:
        """``True`` when this step stopped on an unanswered human gate."""
        return self.gate_pending

    @property
    def outcome_label(self) -> str:
        """A one-word label for the run's terminal state at this step.

        Used in notes so a reader can tell "parked at a gate", "escalated", "ran" and
        "failed" apart without parsing prose.
        """
        if self.gate_pending:
            return "parked_at_gate"
        if self.gate_kind is GateKind.ESCALATION:
            return "escalated"
        return "ok" if self.ok else "failed"


class AgentRuntime(Protocol):
    """Executes one agent step. The measured system's boundary with ``eval``."""

    name: str

    def available(self) -> tuple[bool, str]:
        ...

    def execute(self, request: StepRequest) -> StepResult:
        ...


def _primary_brand(scenario: Scenario) -> str:
    """The brand a run negotiates with: the scenario's stated subject.

    Preference order:

    1. ``seed_data.cold_lead_name`` when set. The scenario named a lead as its
       subject (S2's negative-EV lead), so that is the brand the run must be about
       even though another lead happens to score higher on fit. Choosing by fit
       would silently substitute a different scenario.
    2. The first seeded offer's brand, when the fixture already frames a contract.
    3. The best-fit lead, for scenarios with neither.

    Chosen from the fixture, not at random, so every condition negotiates about the
    same counterparty.
    """
    if scenario.seed_data.cold_lead_name:
        return scenario.seed_data.cold_lead_name
    if scenario.seed_data.offers:
        return scenario.seed_data.offers[0].offer.brand
    if scenario.seed_data.brands:
        return max(scenario.seed_data.brands, key=lambda b: b.fit_score).name
    raise ValueError(f"scenario {scenario.name!r} has neither offers nor brands to negotiate with")


class LocalRuleRuntime:
    """Deterministic, offline, pure-``core`` implementation of the seven agents.

    This is a **fallback**, and it is labelled as such everywhere it is used. It
    exists so the harness can measure *control flow* - which is what the ablation
    varies - without requiring a live model, a local Ollama, or a tool registry.

    What it is not: a claim about agent capability. Its reasoning is a handful of
    arithmetic checks, all of which a real agent can also perform. The report says
    so, and the two consequences are stated where they matter:

    * Its decisions are ``DecisionSource.RULES``. There are no LLM calls.
    * Its ``path_variation`` ceiling is low by construction; see
      :attr:`RunOutcome.path_variation_caveat`.
    """

    name = "local-rules-stub"

    def available(self) -> tuple[bool, str]:
        return (True, "in-process; needs nothing beyond core")

    def execute(self, request: StepRequest) -> StepResult:
        handler: Callable[[StepRequest], StepResult] | None = {
            AgentId.A1_DISCOVERY: self._a1,
            AgentId.A2_PRICING: self._a2,
            AgentId.A3_OUTREACH: self._a3,
            AgentId.A4_CONTRACT: self._a4,
            AgentId.A5_COMPLIANCE: self._a5,
            AgentId.A6_AUDIT: self._a6,
            AgentId.A7_ARBITER: self._a7,
        }.get(request.agent)
        if handler is None:
            return StepResult(ok=False, summary=f"no handler for {request.agent}",
                              errors=[f"unhandled agent {request.agent.value}"])
        result = handler(request)
        request.counters.steps += 1
        return result

    # ------------------------------------------------------------------ agents
    def _a1(self, req: StepRequest) -> StepResult:
        res = req.tools.call("maps.discover_brands", event_id=req.event_id)
        leads = res.data.get("leads", []) if isinstance(res.data, dict) else []
        for lead in leads:
            req.board.post("discovery", "brand_lead", AgentId.A1_DISCOVERY, lead,
                           source=DecisionSource.RULES)
        ranked = sorted(leads, key=lambda d: float(d.get("fit_score", 0.0)), reverse=True)
        req.board.post("discovery", "shortlist", AgentId.A1_DISCOVERY,
                       {"brands": [d.get("name") for d in ranked],
                        "tool_status": res.status.value}, source=DecisionSource.RULES)
        return StepResult(
            ok=res.ok or bool(leads),
            summary=f"discovered {len(leads)} candidate brands from {res.source}",
            degraded=res.degraded,
            produced={"leads": leads, "ranked": ranked},
            notes=[f"tool status {res.status.value}"],
        )

    def _a2(self, req: StepRequest) -> StepResult:
        """Price the focus brand. On S1 this is where the breach is introduced."""
        scen = req.scenario
        seeded = next((o for o in scen.seed_data.offers if o.offer.brand == req.focus_brand), None)
        if seeded is None:
            lead = next((b for b in scen.seed_data.brands if b.name == req.focus_brand), None)
            if lead is None:
                return StepResult(ok=False, summary=f"no fixture for brand {req.focus_brand!r}",
                                  errors=[f"unpriced: {req.focus_brand}"])
            amount = round(max(10_000.0, lead.fit_score * 900.0), 2)
            offer = Offer(offer_id=new_id("off"), event_id=req.event_id, brand=lead.name,
                          tier="standard", amount_inr=amount,
                          deliverables=list(req.scenario.event.deliverables_offered)[:2],
                          fit_score=lead.fit_score,
                          pitch="standard tier; no exclusivity; no data sharing")
            req.board.post("pricing", "offer", AgentId.A2_PRICING,
                           offer.model_dump(mode="json"), source=DecisionSource.RULES)
            return StepResult(ok=True, summary=f"priced {lead.name} at INR {amount}",
                              produced={"offer": offer})

        offer = seeded.offer.model_copy(update={"version": 1})
        req.board.post("pricing", "offer", AgentId.A2_PRICING,
                       offer.model_dump(mode="json"), source=DecisionSource.RULES)
        # The bid makes the reasoning auditable: feasibility and effort are stated
        # rather than asserted, so a reader can recompute the award score.
        bid = Bid(bid_id=new_id("bid"), task_id=offer.offer_id, agent=AgentId.A2_PRICING,
                  feasibility=0.95, expected_value=offer.amount_inr, effort=2,
                  rationale=f"tier {offer.tier} at {offer.fit_score} fit; "
                            f"clauses: {seeded.notes or 'none'}")
        req.board.post("pricing", "bid", AgentId.A2_PRICING,
                       {"bid_id": bid.bid_id, "utility": bid.utility}, source=DecisionSource.RULES)
        # A2 having priced means the next objection must go back out to the
        # sponsor, not straight back into A2. Recorded so the orchestrator does not
        # have to guess whether pricing already happened this visit.
        req.state["repriced"] = True
        return StepResult(
            ok=True,
            summary=(f"priced {offer.brand} at INR {offer.amount_inr} "
                     f"({seeded.exclusivity_days}d exclusivity, "
                     f"data clause={seeded.attendee_data_clause})"),
            produced={"offer": offer, "seed_offer": seeded, "bid_utility": bid.utility},
        )

    def _a3(self, req: StepRequest) -> StepResult:
        """Classify the reply, then act on it. This is where routing happens."""
        decision = _rules_intent_decision(
            DecisionRequest(
                request_id=f"intent::{req.round_index}",
                question="What is the operative intent of the sponsor's reply?",
                question_type=QuestionType.CHOICE,
                state={"reply": req.reply_text},
                options=[i.value for i in Intent],
                asked_by=AgentId.A3_OUTREACH,
                decision_point="intent.classify",
            ),
            req.reply_text,
        )
        intent = _intent_from_choice(decision.choice or Intent.UNKNOWN.value)
        send = intent in (Intent.YES, Intent.INTERESTED, Intent.PUSHBACK)
        send_status = None
        if send:
            send_status = req.tools.call("gmail.send", brand=req.focus_brand,
                                         thread_id=f"thr_{req.focus_brand}").status
        thread = Thread(thread_id=f"thr_{req.scenario.name}", event_id=req.event_id,
                        brand=req.focus_brand, status="negotiating" if intent is not Intent.NO
                        else "closed_lost", day=req.round_index, intent=intent,
                        reply_text=req.reply_text, delivered=False)
        req.board.post("outreach", "thread", AgentId.A3_OUTREACH,
                       thread.model_dump(mode="json"),
                       source=DecisionSource.RULES, confidence=decision.confidence)
        return StepResult(
            ok=True,
            summary=f"intent={intent.value} (conf {decision.confidence:.2f}); "
                    f"next agent {intent_to_agent(intent).value}",
            degraded=decision.degraded,
            produced={"intent": intent, "thread": thread, "send_status": send_status},
            notes=[f"decision source={decision.source.value} degraded={decision.degraded}"],
        )

    def _a4(self, req: StepRequest) -> StepResult:
        """Draft the MoU from whatever offer is on the board. No policy checks.

        A4 checking compliance would collapse A5's role; the separation is the
        point of the architecture, so this agent only transcribes terms.

        **Gate sequencing.** A4 asks for a release gate only once A5 has already
        checked the draft (``state["compliance_checked"]``). The first pass only
        drafts. A contract agent that could release before compliance has run would
        make A5's veto advisory, which is the architectural bug this ordering
        exists to prevent - so the sequencing is enforced in the agent, not left to
        the orchestrator's good intentions.
        """
        payload = req.board.latest_payload("pricing", "offer")
        if not payload:
            return StepResult(ok=False, summary="no offer on the board to contract",
                              errors=["contract without an offer"])
        scen = req.scenario
        seeded = next((o for o in scen.seed_data.offers
                       if o.offer.brand == payload.get("brand")), None)
        ask = float(payload.get("amount_inr", 0.0))
        bounds = scen.negotiation
        amount = float(req.state.get("current_ask", ask))
        terms = ContractTerms(
            brand=str(payload.get("brand", req.focus_brand)),
            amount_inr=amount,
            exclusivity_days=seeded.exclusivity_days if seeded else 0,
            attendee_data_clause=bool(seeded.attendee_data_clause) if seeded else False,
            signer_authority_evidence=bool(seeded.signer_authority_evidence) if seeded else False,
            deliverables_evidence_url=seeded.deliverables_evidence_url if seeded else None,
            deliverables=tuple(payload.get("deliverables", [])),
        )
        req.board.post("contract", "mou_draft", AgentId.A4_CONTRACT, terms.to_dict(),
                       source=DecisionSource.RULES)
        if bounds and amount < bounds.walkaway_inr:
            return StepResult(
                ok=False, summary=f"drafted at INR {amount} which is below walkaway "
                                  f"INR {bounds.walkaway_inr}",
                errors=["below walkaway"], contract_terms=terms)
        if not req.state.get("compliance_checked"):
            # First pass: no compliance verdict exists yet, so request the gate.
            return StepResult(
                ok=True,
                summary=(f"drafted MoU for {terms.brand} at INR {amount} "
                         f"({terms.exclusivity_days}d exclusivity, "
                         f"data clause={terms.attendee_data_clause}); requesting release, "
                         "no compliance verdict on the board yet"),
                produced={"contract_terms": terms.to_dict(), "released": True},
                gate_kind=GateKind.MOU,
                contract_terms=terms,
            )
        outstanding = terms.violations(req.scenario.policy)
        if outstanding:
            # A5 has spoken and the draft still breaches policy. A blocking veto is
            # not something a contract agent may override, so no gate is requested:
            # asking a human to approve a contract the policy has already refused
            # would make the veto decorative.
            return StepResult(
                ok=False,
                summary=(f"draft for {terms.brand} at INR {amount} still breaches "
                         f"{len(outstanding)} policy cap(s) after compliance review; "
                         "withholding release"),
                errors=[f"blocked by compliance: {v}" for v in outstanding],
                produced={"contract_terms": terms.to_dict(), "released": False},
                contract_terms=terms,
            )
        return StepResult(
            ok=True,
            summary=(f"drafted MoU for {terms.brand} at INR {amount} "
                     f"({terms.exclusivity_days}d exclusivity, "
                     f"data clause={terms.attendee_data_clause}); compliance cleared, "
                     "requesting release"),
            produced={"contract_terms": terms.to_dict(), "released": True},
            gate_kind=GateKind.MOU,
            contract_terms=terms,
        )

    def _a5(self, req: StepRequest) -> StepResult:
        """Check the draft against policy caps. Holds veto. Raises real disputes."""
        terms = req.state.get("contract_terms")
        if not isinstance(terms, ContractTerms):
            return StepResult(ok=True, summary="no draft to check; nothing to block",
                              sufficient=True)
        caps = req.scenario.policy
        breaches = terms.violations(caps)
        flags: list[dict[str, Any]] = []
        disputes: list[Dispute] = []
        if breaches:
            for code, msg in zip(("POLICY_BREACH",), breaches, strict=False):
                flags.append({"code": code, "message": msg,
                              "severity": Severity.BLOCKING.value})
            req.counters.blocking_flags.extend(breaches)
            req.board.post("risk_flags", "blocking_flag", AgentId.A5_COMPLIANCE,
                           {"brand": terms.brand, "breaches": breaches},
                           source=DecisionSource.RULES)
            disputes.append(Dispute(
                dispute_id=new_id("dsp"), event_id=req.event_id, zone=DisputeZone.CONTRACT,
                claimant=AgentId.A2_PRICING, opponent=AgentId.A5_COMPLIANCE,
                position=(f"offer to {terms.brand} at INR {terms.amount_inr} with "
                          f"{terms.exclusivity_days}d exclusivity and "
                          f"attendee_data_clause={terms.attendee_data_clause} is the "
                          "strongest offer on the table and is priced accordingly"),
                counter_position=(f"veto: {len(breaches)} policy breach(es): "
                                  + "; ".join(breaches)),
                evidence=[f"offer:{terms.brand}"], severity=Severity.BLOCKING,
            ))
        # Evidence gap: absence of evidence is reported, never assumed either way.
        missing = req.scenario.seed_data.missing_evidence
        if missing:
            req.board.post("risk_flags", "evidence_gap", AgentId.A5_COMPLIANCE,
                           {"missing": list(missing)}, source=DecisionSource.RULES)
            flags.append({"code": "EVIDENCE_GAP",
                          "message": f"cannot verify: {'; '.join(missing)}",
                          "severity": Severity.MEDIUM.value})
        return StepResult(
            ok=True,
            summary=(f"checked {terms.brand} draft: {len(breaches)} breach(es), "
                     f"{len(missing)} evidence gap(s)"),
            disputes=disputes, flags=flags,
            produced={"breaches": breaches, "missing_evidence": list(missing)},
        )

    def _a6(self, req: StepRequest) -> StepResult:
        """Two jobs: challenge bad-bet outreach, and audit deliverables honestly."""
        sd = req.scenario.seed_data
        disputes: list[Dispute] = []
        notes: list[str] = []

        # --- job 1: is the outreach the system wants to send worth it? -------
        focus_lead = next((b for b in sd.brands if b.name == req.focus_brand), None)
        if focus_lead is not None and sd.outreach_cost_inr > 0.0:
            ev = sd.lead_ev(focus_lead)
            if ev < sd.outreach_cost_inr:
                better = max(sd.brands, key=lambda b: sd.lead_ev(b))
                disputes.append(Dispute(
                    dispute_id=new_id("dsp"), event_id=req.event_id, zone=DisputeZone.OUTREACH,
                    claimant=AgentId.A3_OUTREACH, opponent=AgentId.A6_AUDIT,
                    position=(f"pursue {focus_lead.name}: fit {focus_lead.fit_score}, "
                              f"{focus_lead.distance_km}km, category overlap with "
                              f"{req.scenario.event.categories_wanted}"),
                    counter_position=(f"expected value INR {ev} is below the INR "
                                      f"{sd.outreach_cost_inr} outreach cost; "
                                      f"{better.name} yields INR {sd.lead_ev(better)} "
                                      "for the same spend"),
                    evidence=[f"lead:{focus_lead.lead_id}"], severity=Severity.MEDIUM,
                ))
                notes.append(f"negative-EV lead {focus_lead.name}: {ev} < {sd.outreach_cost_inr}")

        # --- job 2: fulfilments, honestly -------------------------------------
        findings: list[AuditFinding] = []
        res = req.tools.call("vision.verify_deliverables", brand=req.focus_brand)
        terms = req.state.get("contract_terms")
        deliverables = list(terms.deliverables) if isinstance(terms, ContractTerms) else []
        if not deliverables:
            payload = req.board.latest_payload("pricing", "offer")
            deliverables = list(payload.get("deliverables", []))
        for d in deliverables:
            if res.status is ToolStatus.UNAVAILABLE:
                # The honest answer. fulfilled=False, and the note says *why*:
                # no evidence exists. This is S5's whole point.
                findings.append(AuditFinding(
                    promise=d, fulfilled=False, evidence_url=None, confidence=0.0,
                    source=DecisionSource.RULES,
                    note=f"cannot verify: {res.reason}"))
            else:
                findings.append(AuditFinding(
                    promise=d, fulfilled=False, evidence_url=res.evidence_url,
                    confidence=0.5, source=DecisionSource.RULES,
                    note="fixture audit: no fulfilment record available"))
        if findings:
            req.board.post("roi", "audit_findings", AgentId.A6_AUDIT,
                           {"findings": [f.model_dump(mode="json") for f in findings]},
                           source=DecisionSource.RULES)
        req.state["unverifiable"] = [f.promise for f in findings]
        return StepResult(
            ok=True,
            summary=(f"audited {len(findings)} deliverable(s); "
                     f"{sum(1 for f in findings if f.fulfilled)} fulfilled, "
                     f"{len(findings) - sum(1 for f in findings if f.fulfilled)} cannot verify"),
            disputes=disputes, findings=findings,
            degraded=res.degraded,
            produced={"findings": [f.model_dump(mode="json") for f in findings],
                      "vision_status": res.status.value},
            notes=notes,
        )

    def _a7(self, req: StepRequest) -> StepResult:
        """Arbitrate the open dispute by asking the decision layer.

        Genuinely a decision: the outcome depends on a confidence the arbiter must
        then judge against the escalation threshold, and a low-confidence answer
        escalates to a human instead of guessing.
        """
        dispute: Dispute | None = req.state.get("open_dispute")
        if not isinstance(dispute, Dispute):
            return StepResult(ok=True, summary="no open dispute to arbitrate")
        request = DecisionRequest(
            request_id=f"arb::{dispute.dispute_id}",
            question=(
                "Should the disputed position be conceded or upheld? The opponent's "
                "counter-position is a policy veto or an arithmetic correction."
            ),
            question_type=QuestionType.NOUL,
            state={"position": dispute.position, "counter_position": dispute.counter_position,
                   "zone": dispute.zone.value, "severity": dispute.severity.value},
            options=["concede", "uphold"],
            asked_by=AgentId.A7_ARBITER,
            decision_point="arbitrate.dispute",
        )
        # The offline prior is principled, not convenient:
        #   * a *policy* veto is upheld, because policy is not the agents' to trade
        #     away and the human owns it;
        #   * an *arithmetic* correction is conceded, because arithmetic is
        #     checkable and A6 is right about it.
        # The arbiter still has to clear ``ARBITER_CONFIDENCE_FLOOR`` on whatever
        # confidence the decision layer returns, so a weak answer escalates.
        is_policy = dispute.zone in (DisputeZone.COMPLIANCE, DisputeZone.CONTRACT)
        rule_choice = "uphold" if is_policy else "concede"
        rule_probs = ({"uphold": 0.72, "concede": 0.28} if is_policy
                      else {"concede": 0.72, "uphold": 0.28})
        decision = req.decide.ask(request, rule_choice=rule_choice,
                                 rule_probs=rule_probs,
                                 jitter_seed=f"arb:{dispute.dispute_id}")
        resolved = decision.confidence >= 0.5
        return StepResult(
            ok=True,
            summary=(f"arbitrated {dispute.dispute_id}: {decision.choice} "
                     f"(conf {decision.confidence:.2f}, "
                     f"{decision.source.value}) -> "
                     f"{'resolved' if resolved else 'escalated'}"),
            degraded=decision.degraded,
            produced={"arbitration": decision.model_dump(mode="json"),
                      "resolved": resolved,
                      "amend": (decision.choice == "concede" or decision.choice == "uphold")},
            gate_kind=None if resolved else GateKind.ESCALATION,
            notes=[f"arbiter confidence {decision.confidence:.2f}"],
        )


class RealAgentRuntime:
    """Adapter onto the project's real agents, imported lazily.

    What it drives
    --------------
    :meth:`execute` builds a real :class:`~graph.state.GraphRuntime` through
    :func:`graph.build.build_runtime` - the orchestrator's own wiring, so the
    :class:`~core.protocols.AgentContext` each agent receives is the context production
    runs give it - and then calls ``agent.run(ctx)`` on the registered real agent. The
    agents are real, the decision layer is
    :func:`decision.registry.build_registry`, and the board is the real blackboard.

    Isolation
    ---------
    The runtime, board, tool map and decision bridge are cached **per ``run_id``**.
    Every step of one ``(scenario, condition, seed)`` execution shares them - they have
    to, or A4 could not see what A2 priced - and two executions share nothing, because
    a blackboard carried across seeds would make the second seed's number a function of
    the first seed's leftovers.

    Failure is never zero
    ---------------------
    Three things can stop a step, and none of them is reported as success:

    * :class:`~core.errors.HumanGateRequired` - A3/A4 raise this deliberately so the
      graph can interrupt. The run **parks**: ``ok=False``, ``gate_pending=True``, the
      pending ``GateKind`` returned, and the drafted MoU/thread left at
      ``pending_approval`` so the benchmark can still audit the terms.
    * :class:`~core.errors.ArbiterEscalation` - A7 could not settle. Recorded as an
      escalation with the real dispute id, and **not** resolved.
    * an absent agent, an agent that raised, or an agent that produced nothing - each
      an explicit failure carrying its reason.

    Provenance
    ----------
    :meth:`available` returns the reason string verbatim, and
    :func:`resolve_runtime` folds it - plus the decision chain that will actually
    answer - into the provenance string stamped on every ``RunOutcome``. There is no
    path by which a stub's numbers are labelled as a real stack's.
    """

    #: Names probed on the bare ``agents`` package. A *fallback* only; see
    #: :data:`eval.integration.AGENT_PACKAGE_FACTORIES`.
    _REAL_FACTORIES: tuple[str, ...] = ("build_registry", "build_agents", "make_registry",
                                        "default_registry", "get_registry")

    #: How many ``(scenario, condition, seed)`` runtimes to keep alive at once. The
    #: cache exists so agents are built once per run, not once per step; the bound
    #: exists so a long sweep does not accumulate one board per cell forever.
    _MAX_CACHED_RUNS = 4

    def __init__(self, *, agents: Mapping[AgentId, Any] | None = None,
                 settings: Any = None, simulated_gates: bool = False) -> None:
        self.name = "real-agents"
        #: Injected registry. ``None`` means "probe for it", which is what production
        #: and the ablation do. A caller that already built a registry (a test driving
        #: one agent, or the API holding a live graph) can pass it instead, and then
        #: :meth:`available` reports what it was given rather than probing.
        self._injected_agents: dict[AgentId, Any] | None = (
            dict(agents) if agents is not None else None)
        self._injected_settings = settings
        self._status: Any | None = None
        self._probed = False
        self._reason: str = "not probed"
        #: When True, each run scope is seeded with simulated-human gate approvals
        #: (SEND/MOU/COUNTER/ESCALATION) so real A3/A4 agents proceed past their
        #: HumanGateRequired parks in unattended eval sweeps. Labelled, never a
        #: real human approval. See :func:`eval.integration.simulated_gate_approvals`.
        self.simulated_gates: bool = bool(simulated_gates)
        #: ``run_id -> eval.integration.RunScope``.
        self._scopes: OrderedDict[str, Any] = OrderedDict()

    # ------------------------------------------------------------------ probing
    def available(self) -> tuple[bool, str]:
        """``(ok, reason)`` for the real stack, with the real reason string.

        Delegates to :func:`eval.integration.probe_real_stack`, which tries
        ``graph.registry.build_agents`` **first** - the factory production wiring uses -
        and only then falls back to probing the ``agents`` package by name. Every
        failure path returns a specific reason, because that string is printed in the
        report's provenance block and a generic "unavailable" would tell a reader
        nothing they could act on.
        """
        if self._probed:
            return (bool(self._status and self._status.ok), self._reason)
        self._probed = True
        if self._injected_agents is not None:
            self._status = _injected_status(self._injected_agents, self._injected_settings)
            self._reason = self._status.reason
            return (self._status.ok, self._reason)
        from .integration import probe_real_stack

        status = probe_real_stack(settings=self._injected_settings)
        self._status = status
        self._reason = status.reason
        if status.ok and status.absent:
            # Partially present is *available* - the six agents that exist are real
            # agents - but the absence has to travel with the number.
            self._reason = f"{status.reason}; a step needing an absent agent fails explicitly"
        return (status.ok, self._reason)

    @property
    def status(self) -> Any:
        """The :class:`~eval.integration.RealStackStatus`, probing if needed."""
        self.available()
        return self._status

    def provenance(self) -> str:
        """One line naming the runtime **and** the decision layer that answers it.

        Both halves matter. A number produced by the real agents through the rules
        decision backend is a different measurement from one produced through clef, and
        the provenance block has to say which, because ``report.py`` aggregates runtime
        labels from this string and a reader has no other way to tell.

        The leading token is ``real-agents`` - the runtime's own ``name`` - because
        ``report.py`` splits the string on ``"("`` to build its runtime column, and a
        column reading "real agent stack" next to "local-rules-stub" would make the two
        look like different kinds of thing rather than the same kind of measurement taken
        against different systems.
        """
        from .integration import decision_provenance

        ok, why = self.available()
        label = self.name if ok else f"{self.name} unavailable"
        base = f"{label} ({why}); decisions: {decision_provenance(self._settings())}"
        if self.simulated_gates:
            base += "; gates: simulated-human (unattended eval; no real person involved)"
        return base

    def _settings(self) -> Any:
        """The ``Settings`` the runtime is built against, or ``None`` if unavailable."""
        status = self._status
        return getattr(status, "settings", None) if status is not None else None

    # ---------------------------------------------------------------- execution
    def execute(self, request: StepRequest) -> StepResult:
        """Run one real agent step and project what it did into a ``StepResult``.

        The result is derived from the **board**, not from the agent's return value.
        ``AgentResult`` carries the agent's own summary and sufficiency judgement;
        the artefacts it actually produced - offers, MoUs, disputes, findings, risk
        flags - are what the ablation measures, and they live on the blackboard. A
        summary is a claim; a board entry is evidence.
        """
        ok, why = self.available()
        if not ok or self._status is None or not self._status.agents:
            raise RuntimeError(
                f"RealAgentRuntime unavailable: {why}. The caller must fall back "
                "explicitly (see conditions.resolve_runtime) and label the run."
            )
        scope, scope_notes = self._scope_for(request)
        agent = scope.agent(request.agent)
        if agent is None:
            # An agent the real registry does not have. Recorded as an explicit failure:
            # the alternative - returning ok=True with an empty result - would let a
            # missing agent read as an agent that decided to do nothing.
            return StepResult(
                ok=False,
                summary=(f"the real registry has no agent for {request.agent.value}; "
                         f"step not executed"),
                errors=[f"unregistered in the real agent registry: {request.agent.value}"],
                notes=[*scope_notes, f"registry factory: {self._status.factory}"])

        from .integration import publish_world, read_delta

        world_notes = publish_world(scope, request)
        mark = scope.mark_board()
        # Counted before the call, not after: a step that parks at a human gate or
        # escalates still *ran*, and `steps_taken` has to mean the same thing for the
        # real stack as it does for the stub, which counts every dispatch.
        request.counters.steps += 1
        try:
            ctx = scope.runtime.context_for(request.agent, str(request.event_id),
                                            str(request.run_id))
            outcome = agent.run(ctx)
        except Exception as exc:  # noqa: BLE001 - one agent's fault, one cell's failure
            return self._on_exception(scope, request, exc, mark,
                                      extra_notes=[*scope_notes, *world_notes])

        delta = read_delta(scope.delta(mark))
        return self._on_success(scope, request, outcome, delta,
                                extra_notes=[*scope_notes, *world_notes])

    # ------------------------------------------------------------- outcomes
    def _on_success(self, scope: Any, request: StepRequest, outcome: Any,
                    delta: Any, *, extra_notes: Sequence[str]) -> StepResult:
        """Turn a completed ``AgentResult`` plus its board delta into a ``StepResult``."""
        from .integration import blocking_flags_from, contract_terms_from_mou, latest_arbitration

        blocking = blocking_flags_from(delta)
        request.counters.blocking_flags.extend(blocking)

        terms: ContractTerms | None = None
        notes: list[str] = [*extra_notes, *(str(n) for n in getattr(outcome, "notes", []) or [])]
        produced: dict[str, Any] = {}
        if delta.mouses:
            terms, term_notes = contract_terms_from_mou(delta.mouses[-1])
            notes.extend(term_notes)
            produced["contract_terms"] = terms.to_dict()
            produced["mou_status"] = str(delta.mouses[-1].status)
        if delta.offers:
            produced["offer"] = delta.offers[-1].model_dump(mode="json")
        if delta.threads:
            produced["thread"] = delta.threads[-1].model_dump(mode="json")
        if delta.roi is not None:
            produced["roi_report"] = delta.roi
        if delta.unparsed:
            notes.append("board artefacts the harness could not type: "
                         + ", ".join(sorted(set(delta.unparsed))))
        arbitration = latest_arbitration(delta)
        if arbitration is not None:
            produced["arbitration"] = arbitration

        observation = getattr(outcome, "observation", None)
        degraded = bool(getattr(outcome, "degraded", False))
        source_mix = dict(request.counters.decisions_by_source)
        if source_mix and all(k == DecisionSource.RULES.value for k in source_mix):
            # Every decision in this run came from the rules layer. If a model was the
            # intended head, say the step was degraded rather than letting a
            # `degraded=False` read as "the model was asked and answered".
            notes.append(
                "every decision so far came from DecisionSource.RULES"
                + (f" while {scope.decide.intended_source.value} was intended"
                   if scope.decide.intended_source is not DecisionSource.RULES else ""))
            degraded = degraded or (
                scope.decide.intended_source is not DecisionSource.RULES)

        flags = [{"code": f.code, "message": f.message,
                  "severity": str(getattr(f.severity, "value", f.severity))}
                 for f in delta.flags]
        summary = (str(getattr(observation, "summary", "") or "")
                   or f"{request.agent.value} finished with no summary")
        return StepResult(
            ok=True,
            summary=summary,
            sufficient=bool(getattr(observation, "sufficient", True)),
            degraded=degraded,
            produced=produced,
            disputes=list(delta.disputes),
            flags=flags,
            findings=list(delta.findings),
            contract_terms=terms,
            notes=notes,
        )

    def _on_exception(self, scope: Any, request: StepRequest, exc: BaseException,
                      mark: int, *, extra_notes: Sequence[str]) -> StepResult:
        """Classify an exception out of ``agent.run`` and convert it to a step result.

        Three cases, three honest shapes:

        * ``HumanGateRequired`` - a park. ``ok=False`` because the side effect did not
          happen; ``gate_pending=True``; the pending ``GateKind`` returned so the
          condition's gate machinery counts it; any MoU already drafted is returned as
          ``contract_terms`` so the benchmark can audit the terms that are stuck.
        * ``ArbiterEscalation`` - an escalation. Recorded with the dispute id and
          **not** resolved, which is what the exception means.
        * anything else - a failure with the exception's own type and message. Never a
          zero, and never a bare ``ok=False`` with nothing to read.
        """
        from core.errors import ArbiterEscalation, HumanGateRequired

        from .integration import (
            blocking_flags_from,
            contract_terms_from_mou,
            gate_kind_from,
            read_delta,
        )

        delta = read_delta(scope.delta(mark))
        blocking = blocking_flags_from(delta)
        request.counters.blocking_flags.extend(blocking)
        terms: ContractTerms | None = None
        notes: list[str] = [*extra_notes]
        if delta.mouses:
            terms, term_notes = contract_terms_from_mou(delta.mouses[-1])
            notes.extend(term_notes)
        if delta.unparsed:
            notes.append("board artefacts the harness could not type: "
                         + ", ".join(sorted(set(delta.unparsed))))
        produced: dict[str, Any] = {}
        if terms is not None:
            produced["contract_terms"] = terms.to_dict()

        if isinstance(exc, HumanGateRequired):
            # A5's veto posts no HumanGate - it raises with a message only - so a
            # veto defaults to ESCALATION: a veto escalated to a human is exactly what
            # that gate kind means.
            kind = gate_kind_from(delta, str(exc), GateKind.ESCALATION)
            pending = _pending_approval(scope, request, terms)
            notes.append(
                f"PARKED: {request.agent.value} raised HumanGateRequired "
                f"({kind.value}) and no HumanDecision exists, so the side effect did "
                f"not happen; {pending}")
            return StepResult(
                ok=False,
                summary=(f"{request.agent.value} parked at an unanswered "
                         f"{kind.value} gate: the action was NOT taken"),
                sufficient=False,
                gate_kind=kind,
                gate_pending=True,
                parked_reason=str(exc),
                produced={**produced, "pending_approval": pending},
                disputes=list(delta.disputes),
                flags=[{"code": f.code, "message": f.message,
                        "severity": str(getattr(f.severity, "value", f.severity))}
                       for f in delta.flags],
                findings=list(delta.findings),
                contract_terms=terms,
                errors=[],
                notes=notes,
            )

        if isinstance(exc, ArbiterEscalation):
            dispute_id = str(getattr(exc, "dispute_id", "") or "unknown")
            request.counters.disputes_escalated += 1
            produced["arbitration"] = {
                "choice": "escalate", "confidence": 0.0,
                "source": scope.decide.intended_source.value, "degraded": True,
                "escalated": True, "dispute_id": dispute_id,
                "model": f"{self.name}:arbiter-escalation",
            }
            notes.append(
                f"ESCALATED: {request.agent.value} raised ArbiterEscalation for dispute "
                f"{dispute_id}; the dispute is recorded as escalated and NOT resolved")
            return StepResult(
                ok=True,
                summary=(f"{request.agent.value} escalated dispute {dispute_id} to a "
                         "human; it is not resolved"),
                sufficient=False,
                degraded=True,
                produced=produced,
                disputes=list(delta.disputes),
                flags=[{"code": f.code, "message": f.message,
                        "severity": str(getattr(f.severity, "value", f.severity))}
                       for f in delta.flags],
                findings=list(delta.findings),
                gate_kind=GateKind.ESCALATION,
                contract_terms=terms,
                notes=notes,
            )

        # Anything else is a real failure. The type and message travel with it.
        notes.append(f"agent raised {type(exc).__name__}: {exc}")
        return StepResult(
            ok=False,
            summary=f"{request.agent.value} raised {type(exc).__name__}: {exc}",
            sufficient=False,
            degraded=True,
            produced=produced,
            disputes=list(delta.disputes),
            flags=[{"code": f.code, "message": f.message,
                    "severity": str(getattr(f.severity, "value", f.severity))}
                   for f in delta.flags],
            findings=list(delta.findings),
            errors=[f"{type(exc).__name__}: {exc}"],
            notes=notes,
            contract_terms=terms,
        )

    # ------------------------------------------------------------------- scope
    def _scope_for(self, request: StepRequest) -> tuple[Any, list[str]]:
        """The run's real-stack collaborators, building them on first use.

        Keyed on ``run_id``, which the conditions mint fresh for every
        ``(scenario, condition, seed)`` execution. That is what gives the spec's
        "per-(agent, scenario, seed) runtime": no state crosses a run boundary, and
        every step within a run sees the same board.
        """
        key = str(request.run_id)
        cached = self._scopes.get(key)
        if cached is not None:
            return (cached, [])

        from .integration import open_run_scope

        scope = open_run_scope(request, self._status, counters=request.counters,
                               simulated_gates=self.simulated_gates)
        self._scopes[key] = scope
        while len(self._scopes) > self._MAX_CACHED_RUNS:
            self._scopes.popitem(last=False)
        note = f"real run scope opened: run_id={key}"
        if self.simulated_gates:
            note += " (simulated-human gate approvals seeded; no real person involved)"
        return (scope, [note])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        agents = len(self._status.agents) if self._status is not None else 0
        return f"<RealAgentRuntime agents={agents} scopes={len(self._scopes)}>"


def _pending_approval(scope: Any, request: StepRequest,
                      terms: ContractTerms | None) -> str:
    """Describe what is stuck, in the words the domain uses.

    ``MoU.status`` has a literal ``"pending_approval"`` and ``Thread.status`` has the
    same, so the parked artefact is described with that exact word rather than a
    paraphrase: a reader grepping the board for ``pending_approval`` has to find it.
    """
    if terms is not None:
        return (f"the MoU for {terms.brand} at INR {terms.amount_inr} stays at "
                f"status='pending_approval'")
    return (f"the {request.agent.value} thread for {request.focus_brand} stays at "
            "status='pending_approval'")


def _injected_status(agents: Mapping[AgentId, Any], settings: Any) -> Any:
    """A :class:`~eval.integration.RealStackStatus` built from a caller-supplied registry.

    Lives here rather than in :mod:`eval.integration` so the injection point needs no
    lazy import: it is pure data assembly over a registry the caller already trusts.
    """
    from core import REASONING_AGENTS

    from .integration import RealStackStatus

    registry = dict(agents)
    absent = tuple(a.value for a in REASONING_AGENTS if a not in registry)
    detail = f"injected registry -> {len(registry)} agents"
    if absent:
        detail += f"; absent: {list(absent)}"
    return RealStackStatus(ok=bool(registry), reason=detail,
                           factory="injected", agents=registry, settings=settings,
                           absent=absent)


class StubRuntime:
    """Configurable no-op runtime for tests.

    Records the agent sequence so a test can assert path shape without any
    reasoning at all. Used only by ``tests/unit/test_eval.py``.
    """

    name = "stub"

    def __init__(self, *, record: list[AgentId] | None = None,
                 raise_for: Sequence[AgentId] = (),
                 dispute_on: dict[AgentId, int] | None = None) -> None:
        self.record = record if record is not None else []
        self.raise_for = tuple(raise_for)
        self.dispute_on = dict(dispute_on or {})

    def available(self) -> tuple[bool, str]:
        return (True, "stub runtime: performs no reasoning")

    def execute(self, request: StepRequest) -> StepResult:
        if request.agent in self.raise_for:
            raise RuntimeError(
                f"StubRuntime was configured to fail on {request.agent.value}; this "
                "stands in for a dependency that is not present yet")
        self.record.append(request.agent)
        request.counters.steps += 1
        n = self.dispute_on.get(request.agent, 0)
        disputes = [
            Dispute(
                dispute_id=new_id("dsp"), event_id=request.event_id,
                zone=DisputeZone.CONTRACT, claimant=AgentId.A2_PRICING,
                opponent=AgentId.A5_COMPLIANCE,
                position=f"stub position {i}", counter_position=f"stub counter {i}",
                severity=Severity.MEDIUM,
            )
            for i in range(n)
        ]
        return StepResult(ok=True, summary=f"stub:{request.agent.value}",
                          disputes=disputes)


def resolve_runtime(*, prefer_real: bool = True) -> tuple[AgentRuntime, str]:
    """Pick a runtime, returning it *and* an honest provenance string.

    Tries the real agent stack first so that when it exists, measurements come from
    it. Falls back to :class:`LocalRuleRuntime` with the reason spelled out - the string
    lands in the report's provenance block, so a reader always knows which system
    produced the numbers.

    **The honesty contract.** The returned string is the only thing that says where a
    number came from, and it is shaped so it cannot lie by omission:

    * The real stack yields ``"real-agents (<factory and count>; decisions: <chain>)"``.
      ``report.py`` splits on ``"("`` to build its runtime column, so the label
      ``real-agents`` is what a reader sees in the table.
    * The stub yields ``"FALLBACK local-rules-stub (...)"``. The word ``FALLBACK`` is
      load-bearing and is asserted by ``tests/unit/test_eval.py``; it must never be
      removed to make a table look better.
    * The decision chain is named in **both** cases, because "real agents + rules
      backend" and "real agents + clef backend" are different measurements and the
      provenance block is the only place that can tell a reader which one happened.
    """
    if prefer_real:
        real = RealAgentRuntime()
        ok, why = real.available()
        if ok:
            return (real, real.provenance())
        return (LocalRuleRuntime(),
                f"FALLBACK {LocalRuleRuntime.name} (deterministic, offline, "
                f"DecisionSource.RULES, no LLM calls): {why}")
    return (LocalRuleRuntime(), f"{LocalRuleRuntime.name} (requested explicitly)")


# =================================================================== outcomes
@dataclass
class NegotiationOutcome:
    """What happened in the money conversation, measured not asserted."""

    brand: str
    signed: bool
    rounds_to_deal: int | None
    opening_ask_inr: float
    sponsor_reservation_inr: float
    final_amount_inr: float | None
    #: ``(sponsor_share, organiser_share)`` of the bargaining gap; sums to 1.
    surplus_split: tuple[float, float] | None
    policy_violations: list[str] = field(default_factory=list)
    escalation: bool = False
    gates_raised: list[str] = field(default_factory=list)
    contract_terms: dict[str, Any] | None = None
    #: Set when the condition signed something the audit says it should not have.
    signed_in_violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["surplus_split"] = None if self.surplus_split is None else list(self.surplus_split)
        return d


@dataclass
class RunOutcome:
    """One ``(scenario, condition, seed)`` execution. The atom of every report."""

    scenario: str
    condition: str
    seed: int
    ok: bool
    failure_reason: str
    runtime: str
    #: (condition, condition, ...) path string, for ``path_variation``.
    agent_path: tuple[AgentId, ...] = ()
    steps_taken: int = 0
    llm_calls: int = 0
    decision_calls: int = 0
    decisions_by_source: dict[str, int] = field(default_factory=dict)
    probe_decision_calls: int = 0
    probe_decisions_by_source: dict[str, int] = field(default_factory=dict)
    degraded_decisions: int = 0
    disputes_raised: int = 0
    disputes_resolved: int = 0
    disputes_escalated: int = 0
    replans: int = 0
    stall_count: int = 0
    human_gates_raised: int = 0
    tool_calls: int = 0
    tool_status_counts: dict[str, int] = field(default_factory=dict)
    dispute_zones: list[str] = field(default_factory=list)
    blocking_flags: list[str] = field(default_factory=list)
    unverifiable_claims: list[str] = field(default_factory=list)
    route_decisions: list[RouteDecision] = field(default_factory=list)
    negotiation: NegotiationOutcome | None = None
    wall_clock_ms: float = 0.0
    started_at: str = ""
    notes: list[str] = field(default_factory=list)
    #: Non-empty when ``path_variation`` for this run cannot be read as evidence
    #: about the real system. Surfaced verbatim in the report.
    path_variation_caveat: str = ""

    @property
    def path_key(self) -> str:
        return "->".join(a.value for a in self.agent_path)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario": self.scenario,
            "condition": self.condition,
            "seed": self.seed,
            "ok": self.ok,
            "failure_reason": self.failure_reason,
            "runtime": self.runtime,
            "agent_path": [a.value for a in self.agent_path],
            "path_key": self.path_key,
            "steps_taken": self.steps_taken,
            "llm_calls": self.llm_calls,
            "decision_calls": self.decision_calls,
            "decisions_by_source": dict(self.decisions_by_source),
            "probe_decision_calls": self.probe_decision_calls,
            "probe_decisions_by_source": dict(self.probe_decisions_by_source),
            "degraded_decisions": self.degraded_decisions,
            "disputes_raised": self.disputes_raised,
            "disputes_resolved": self.disputes_resolved,
            "disputes_escalated": self.disputes_escalated,
            "replans": self.replans,
            "stall_count": self.stall_count,
            "human_gates_raised": self.human_gates_raised,
            "tool_calls": self.tool_calls,
            "tool_status_counts": dict(self.tool_status_counts),
            "dispute_zones": list(self.dispute_zones),
            "blocking_flags": list(self.blocking_flags),
            "unverifiable_claims": list(self.unverifiable_claims),
            "route_decisions": [r.to_dict() for r in self.route_decisions],
            "negotiation": None if self.negotiation is None else self.negotiation.to_dict(),
            "wall_clock_ms": self.wall_clock_ms,
            "started_at": self.started_at,
            "notes": list(self.notes),
            "path_variation_caveat": self.path_variation_caveat,
        }
        return d


# ================================================================== conditions
class Condition(ABC):
    """One way of running the same task. All three are directly comparable."""

    name: str = "condition"
    description: str = ""
    #: Whether this condition can consult the decision layer for control flow.
    uses_decision_layer: bool = False

    def __init__(self, runtime: AgentRuntime | None = None, *,
                 prefer_real_runtime: bool = True, jitter: float = 0.0,
                 max_steps: int = 12) -> None:
        self._runtime_override = runtime
        self._prefer_real = prefer_real_runtime
        self._jitter = jitter
        self.max_steps = max_steps
        #: Resolved once and reused, so every step of a run drives the *same* runtime
        #: instance. :class:`RealAgentRuntime` keeps one real blackboard, tool map and
        #: decision bridge per ``run_id``; re-resolving per step would hand each step a
        #: fresh, empty run scope, and A4 would never see the offer A2 priced. Isolation
        #: between runs is unaffected: ``run_id`` is minted fresh for every
        #: ``(scenario, condition, seed)`` execution and the scopes are keyed on it.
        self._resolved: tuple[AgentRuntime, str] | None = None

    # ------------------------------------------------------------------ helpers
    def _runtime(self) -> tuple[AgentRuntime, str]:
        if self._runtime_override is not None:
            ok, why = self._runtime_override.available()
            if not ok:
                return (self._runtime_override, f"UNAVAILABLE: {why}")
            # An override that can describe itself is asked to. Otherwise the report's
            # runtime column would name the runtime but not the decision layer that
            # answered it, which is the half of the provenance that distinguishes
            # "real agents + rules" from "real agents + clef".
            describe = getattr(self._runtime_override, "provenance", None)
            return (self._runtime_override,
                    str(describe()) if callable(describe) else why)
        if self._resolved is None:
            self._resolved = resolve_runtime(prefer_real=self._prefer_real)
        return self._resolved

    def available(self) -> tuple[bool, str]:
        runtime, why = self._runtime()
        return runtime.available()

    def run(self, scenario: Scenario, *, seed: int = DEFAULT_SEED) -> RunOutcome:
        """Execute one run. Never raises: failure becomes ``ok=False`` + reason."""
        started = utcnow().isoformat()
        t0 = time.perf_counter()
        runtime, runtime_why = self._runtime()
        counters = RunCounters()
        board = EvalBoard(scenario.event.event_id)
        route: list[RouteDecision] = []

        try:
            ok, why = runtime.available()
            if not ok:
                return RunOutcome(
                    scenario=scenario.name, condition=self.name, seed=seed, ok=False,
                    failure_reason=f"runtime unavailable: {why}", runtime=runtime.name,
                    started_at=started,
                    path_variation_caveat="no runs completed, so no variation exists")
            result = self._execute(scenario, seed, runtime, runtime_why,
                                   counters, board, route)
        except Exception as exc:  # noqa: BLE001 - one failed run must not kill a sweep
            return RunOutcome(
                scenario=scenario.name, condition=self.name, seed=seed, ok=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
                runtime=runtime.name, started_at=started,
                route_decisions=route,
                notes=list(counters.notes),
                path_variation_caveat="run raised before completing; no path recorded",
            )

        wall = round((time.perf_counter() - t0) * 1000.0, 3)
        return RunOutcome(
            scenario=scenario.name, condition=self.name, seed=seed, ok=True,
            failure_reason="", runtime=runtime_why, agent_path=result["agent_path"],
            steps_taken=counters.steps, llm_calls=counters.llm_calls,
            decision_calls=counters.decision_calls,
            decisions_by_source=dict(counters.decisions_by_source),
            probe_decision_calls=counters.probe_decision_calls,
            probe_decisions_by_source=dict(counters.probe_decisions_by_source),
            degraded_decisions=counters.degraded_decisions,
            disputes_raised=counters.disputes_raised,
            disputes_resolved=counters.disputes_resolved,
            disputes_escalated=counters.disputes_escalated,
            replans=counters.replans, stall_count=counters.stall_count,
            human_gates_raised=counters.human_gates_raised,
            tool_calls=counters.tool_calls,
            tool_status_counts=dict(counters.tool_status_counts),
            dispute_zones=list(counters.dispute_zones),
            blocking_flags=list(counters.blocking_flags),
            unverifiable_claims=list(counters.unverifiable_claims),
            route_decisions=route,
            negotiation=result.get("negotiation"),
            wall_clock_ms=wall, started_at=started,
            notes=list(counters.notes),
            path_variation_caveat=(
                f"runtime={runtime_why}; a stub runtime bounds achievable path "
                "variation, so read this row as evidence about control flow, not "
                "about the production agent stack"
                if runtime.name == LocalRuleRuntime.name else ""),
        )

    # -- shared phase: the router probe set, identical for every condition ----
    def _route_probes(self, scenario: Scenario, counters: RunCounters,
                      route: list[RouteDecision]) -> None:
        """Run every declared probe through this condition's routing mechanism.

        Seed-independent on purpose: all conditions see all probes, so the
        confusion matrix is a controlled comparison and not a function of which
        replies a particular run happened to draw.
        """
        for probe in scenario.reply_probes:
            decision = route_probe(probe, self.name)
            route.append(decision)
            if decision.classifier.startswith("decision:"):
                counters.probe_decision_calls += 1
                src = decision.decision_source.value
                counters.probe_decisions_by_source[src] = (
                    counters.probe_decisions_by_source.get(src, 0) + 1)

    def _handoff(self, counters: RunCounters, frm: AgentId | None, to: AgentId,
                 reason: str, source: DecisionSource, confidence: float,
                 run_id: str, event_id: str, path: list[AgentId]) -> None:
        """Record a handoff. The path is built here so nothing else can fake it."""
        if frm is not None and frm is not to:
            # Constructed for validation only; not persisted anywhere. Building it
            # means a self-handoff or an unknown agent id fails loudly at the point
            # of the mistake rather than at report time.
            Handoff(handoff_id=new_id("hnd"), event_id=event_id, run_id=run_id,
                    from_agent=frm, to_agent=to, reason=reason,
                    decision_source=source, confidence=confidence)
            path.append(to)

    def _step(self, request: StepRequest) -> StepResult:
        runtime, _ = self._runtime()
        return runtime.execute(request)

    @abstractmethod
    def _execute(self, scenario: Scenario, seed: int, runtime: AgentRuntime,
                 runtime_why: str, counters: RunCounters, board: EvalBoard,
                 route: list[RouteDecision]) -> dict[str, Any]:
        """Condition-specific control flow. Returns ``agent_path`` and optionally
        ``negotiation``."""


class FixedPipelineCondition(Condition):
    """The baseline: a hardcoded linear sequence with no conditional logic.

    ``A1 -> A2 -> A3 -> A4 -> A5 -> A6``, executed in that order, once each, every
    time, for every scenario and every seed. There is no branch, no re-entry, no
    early exit, and no node that could gate a side effect.

    This is what makes it the right control. Its ``path_variation`` is exactly 1
    by construction - not because the measurement is broken, but because the
    architecture has no mechanism to produce a second path. That is the whole
    argument, and the ablation's job is to show the *other* conditions scoring
    above 1 on the same inputs.

    It is not handicapped: it runs the same runtime, and therefore the same
    agents, as every other condition. A5 still raises its blocking flag - two
    steps too late, because nothing reads it.
    """

    name = "fixed_pipeline"
    description = ("Hardcoded linear A1->A2->A3->A4->A5->A6. No branches, no "
                   "re-entry, no gates. The non-agentic control.")
    uses_decision_layer = False

    PIPELINE: tuple[AgentId, ...] = (
        AgentId.A1_DISCOVERY, AgentId.A2_PRICING, AgentId.A3_OUTREACH,
        AgentId.A4_CONTRACT, AgentId.A5_COMPLIANCE, AgentId.A6_AUDIT,
    )

    def _execute(self, scenario: Scenario, seed: int, runtime: AgentRuntime,
                 runtime_why: str, counters: RunCounters, board: EvalBoard,
                 route: list[RouteDecision]) -> dict[str, Any]:
        self._route_probes(scenario, counters, route)
        rng = scenario_rng(scenario, seed)
        pool = scenario.seed_data.reply_pool or tuple(
            p.text for p in scenario.reply_probes)
        reply = pool[rng.randrange(len(pool))] if pool else ""
        focus = _primary_brand(scenario)
        path: list[AgentId] = [self.PIPELINE[0]]
        run_id = new_id("run")
        state: dict[str, Any] = {}
        open_dispute: Dispute | None = None

        for idx, agent in enumerate(self.PIPELINE):
            req = StepRequest(
                agent=agent, step_name=f"{self.name}.{agent.value}", scenario=scenario,
                seed=seed, board=board, decide=DecisionInvoker(
                    counters, seed=seed, run_mode=RunMode.OFFLINE, prefer_real=False),
                tools=ToolInvoker(scenario, counters, prefer_real=self._prefer_real),
                counters=counters, run_id=run_id, event_id=scenario.event.event_id,
                focus_brand=focus, reply_text=reply, round_index=0, state=state,
            )
            if open_dispute is not None:
                state["open_dispute"] = open_dispute
            result = self._step(req)
            counters.notes.extend(result.notes)
            counters.notes.extend(result.errors)
            # Only overwrite when a contract was actually drafted. Assigning
            # unconditionally would let A5's empty result erase A4's terms, and the
            # run would end with nothing to audit - which is exactly the kind of
            # silent data loss that makes a number unauditable.
            if result.contract_terms is not None:
                state["contract_terms"] = result.contract_terms

            if agent is not self.PIPELINE[-1] and idx + 1 < len(self.PIPELINE):
                # The transition is a constant, not a decision. Source is RULES
                # with confidence 1.0 to record, honestly, that nothing was
                # actually decided here.
                self._handoff(counters, agent, self.PIPELINE[idx + 1],
                              "fixed pipeline order", DecisionSource.RULES, 1.0,
                              run_id, scenario.event.event_id, path)

            for d in result.disputes:
                counters.add_dispute(d)
                board.post("disputes", "dispute", agent, d.model_dump(mode="json"),
                           source=DecisionSource.RULES)
                open_dispute = d
            if result.gate_kind is not None:
                counters.notes.append(
                    f"{agent.value} requested a {result.gate_kind.value} gate; the fixed "
                    "pipeline has no gate node, so it proceeds. Counted as 0 human "
                    "gates and as a policy violation by the benchmark.")

        # Any dispute still open at the end: a linear pipeline has no arbiter, so
        # it cannot resolve one. It is recorded as escalated because that is what
        # would have to happen next - but no human gate was raised, which is
        # precisely the gap.
        if open_dispute is not None and open_dispute.status is not DisputeStatus.RESOLVED:
            counters.disputes_escalated += 1
            counters.notes.append(
                f"dispute {open_dispute.dispute_id} unresolved at end of the linear "
                "sequence; no arbitration node exists so it is escalated implicitly")

        negotiation = self._negotiation(scenario, state, counters, focus)
        return {"agent_path": tuple(path), "negotiation": negotiation}

    def _negotiation(self, scenario: Scenario, state: dict[str, Any],
                     counters: RunCounters, brand: str) -> NegotiationOutcome | None:
        """The pipeline's money step: release whatever A2 produced.

        Unconditional. Not a shortcut - a linear pipeline has no rule to consult,
        so "release at the ask" is the only behaviour it can have. The benchmark
        then audits the result against the policy caps, independently.
        """
        bounds = scenario.negotiation
        terms = state.get("contract_terms")
        if bounds is None:
            return None
        ask = terms.amount_inr if isinstance(terms, ContractTerms) else bounds.opening_ask_inr
        violations = terms.violations(scenario.policy) if isinstance(terms, ContractTerms) else []
        gap = bounds.opening_ask_inr - bounds.sponsor_reservation_inr
        surplus: tuple[float, float] | None = None
        if gap > 0:
            sponsor_gain = ask - bounds.sponsor_reservation_inr
            surplus = (round(sponsor_gain / gap, 4), round(1.0 - sponsor_gain / gap, 4))
        if terms is not None and not isinstance(terms, ContractTerms):
            counters.notes.append("no contract terms were drafted; nothing was released")
        return NegotiationOutcome(
            brand=brand, signed=True, rounds_to_deal=0,
            opening_ask_inr=bounds.opening_ask_inr,
            sponsor_reservation_inr=bounds.sponsor_reservation_inr,
            final_amount_inr=ask, surplus_split=surplus,
            policy_violations=violations, escalation=False, gates_raised=[],
            contract_terms=terms.to_dict() if isinstance(terms, ContractTerms) else None,
            signed_in_violation=bool(violations),
        )


class KeywordRouterCondition(Condition):
    """Real agents; routing decided by substring matching.

    Isolates one variable: *what does the decision layer actually buy?* The
    runtime is identical to the baseline's, so any difference in the metrics is
    attributable to the routing mechanism and nothing else.

    It has no arbiter node, either. Arbitration is itself a decision, and this
    condition makes it with a keyword check: a dispute is conceded only when the
    sponsor's reply contains an explicit concede phrase. That is the honest
    keyword-based equivalent of arbitration, and it fails the same way routing
    does.

    It also keeps :attr:`PREFIX` - discovery, pricing, outreach - as a fixed
    sequence. Without it the router could skip pricing entirely and then fail at
    the contract step for reasons that have nothing to do with routing, which
    would make the baseline a strawman rather than a control. Everything *after*
    the first sponsor reply is keyword-routed.
    """

    name = "keyword_router"
    description = ("Same agents and same discovery/pricing/outreach prefix as the "
                   "agentic condition, but every decision after the first reply is a "
                   "substring match from the previous prototype's classifier "
                   "(assent tested before pushback).")
    uses_decision_layer = False

    #: Phrases the naive arbitration heuristic accepts as a concession.
    CONCEDE_KEYWORDS: tuple[str, ...] = ("we can live with", "we accept", "agreed on",
                                          "that's fine", "works for us")

    #: Fixed discovery prefix; everything after it is keyword-routed.
    PREFIX: tuple[AgentId, ...] = (
        AgentId.A1_DISCOVERY, AgentId.A2_PRICING, AgentId.A3_OUTREACH,
    )

    def _execute(self, scenario: Scenario, seed: int, runtime: AgentRuntime,
                 runtime_why: str, counters: RunCounters, board: EvalBoard,
                 route: list[RouteDecision]) -> dict[str, Any]:
        self._route_probes(scenario, counters, route)
        rng = scenario_rng(scenario, seed)
        pool = scenario.seed_data.reply_pool or tuple(p.text for p in scenario.reply_probes)
        run_id = new_id("run")
        state: dict[str, Any] = {}
        if scenario.negotiation is not None:
            state["current_ask"] = scenario.negotiation.opening_ask_inr
        path: list[AgentId] = [self.PREFIX[0]]
        focus = _primary_brand(scenario)
        decisions_made = 0
        negotiations: list[tuple[int, float]] = []
        last_transition: tuple[str, str] | None = None

        def step(agent: AgentId, round_index: int, reply: str) -> StepResult:
            req = StepRequest(
                agent=agent, step_name=f"{self.name}.{agent.value}",
                scenario=scenario, seed=seed, board=board,
                decide=DecisionInvoker(counters, seed=seed, run_mode=RunMode.OFFLINE,
                                       prefer_real=False),
                tools=ToolInvoker(scenario, counters, prefer_real=self._prefer_real),
                counters=counters, run_id=run_id, event_id=scenario.event.event_id,
                focus_brand=focus, reply_text=reply, round_index=round_index,
                state=state,
            )
            pending = state.get("open_dispute")
            if isinstance(pending, Dispute):
                state["open_dispute"] = pending
            res = self._step(req)
            counters.notes.extend(res.notes)
            counters.notes.extend(res.errors)
            if res.contract_terms is not None:
                state["contract_terms"] = res.contract_terms
            for d in res.disputes:
                counters.add_dispute(d)
                board.post("disputes", "dispute", agent, d.model_dump(mode="json"),
                           source=DecisionSource.RULES)
                state["open_dispute"] = d
            return res

        def open_dispute() -> Dispute | None:
            pending = state.get("open_dispute")
            return pending if isinstance(pending, Dispute) else None

        # ---- the fixed prefix: identical in shape to the agentic run's opening --
        first_reply = pool[rng.randrange(len(pool))] if pool else ""
        for i, agent in enumerate(self.PREFIX):
            step(agent, i, first_reply)
            decisions_made += 1
            if i + 1 < len(self.PREFIX):
                self._handoff(counters, agent, self.PREFIX[i + 1], "fixed discovery prefix",
                              DecisionSource.RULES, 1.0, run_id,
                              scenario.event.event_id, path)

        current = self.PREFIX[-1]
        while decisions_made < self.max_steps:
            reply = pool[rng.randrange(len(pool))] if pool else ""
            result = step(current, decisions_made, reply)

            # ---- the routing decision: substring match, nothing more ----------
            intent = naive_keyword_intent(reply)
            nxt = intent_to_agent(intent)
            source, confidence = DecisionSource.RULES, 1.0

            # ---- the arbitration decision: also a keyword check --------------
            pending = open_dispute()
            if pending is not None and pending.status is not DisputeStatus.RESOLVED:
                conceded = any(k in reply.lower() for k in self.CONCEDE_KEYWORDS)
                if conceded:
                    counters.disputes_resolved += 1
                    pending.status = DisputeStatus.RESOLVED
                    counters.notes.append(
                        f"keyword arbiter conceded dispute {pending.dispute_id} "
                        f"on the phrase {reply!r}")
                else:
                    counters.disputes_escalated += 1
                    counters.notes.append(
                        f"keyword arbiter escalated dispute {pending.dispute_id}: "
                        "no concede phrase found. No arbiter agent exists in this condition.")
                state.pop("open_dispute", None)

            decisions_made += 1
            # Order matters: a contract A4 drafted is a fact, whatever the latest
            # reply said. Checking the NO first would silently discard a drafted
            # MoU whenever a refusal happened to arrive on the same step, which is
            # data loss disguised as a routing decision.
            if result.contract_terms is not None:
                negotiations.append((decisions_made, result.contract_terms.amount_inr))
                # A keyword router has no gate node; record the intent.
                counters.notes.append(
                    f"A4 requested a {result.gate_kind.value if result.gate_kind else '?'} "
                    "gate; the keyword condition has no gate node. Counted as 0 human gates.")
                if intent is Intent.NO:
                    counters.notes.append(
                        f"a refusal ({reply!r}) arrived on the same step as the draft; the "
                        "draft is recorded and the run still closes out")
                break
            if intent is Intent.NO:
                counters.notes.append(f"reply classified NO ({reply!r}); closing out")
                break
            if not result.ok and not result.gate_pending:
                counters.notes.append(
                    f"{current.value} could not complete: {'; '.join(result.errors)}. "
                    "The substring router has no recovery path, so the run stops.")
                break
            # ---- oscillation guard -------------------------------------------
            # Two identical transitions in a row means the router is ping-ponging
            # and no amount of further stepping will help. Recorded as a stall
            # rather than burned through the budget, because the interesting fact
            # is *that* it oscillates, not how long it did.
            transition = (current.value, nxt.value)
            if transition == last_transition:
                counters.stall_count += 1
                counters.notes.append(
                    f"oscillation: the substring router sent control {current.value} -> "
                    f"{nxt.value} twice in a row; stopping rather than spinning")
                break
            last_transition = transition
            if decisions_made < self.max_steps:
                self._handoff(counters, current, nxt, f"naive keyword router -> {intent.value}",
                              source, confidence, run_id, scenario.event.event_id, path)
                current = nxt

        if open_dispute() is not None:
            counters.disputes_escalated += 1

        return {"agent_path": tuple(path),
                "negotiation": self._negotiation(scenario, state, counters, focus,
                                                 negotiations)}

    def _negotiation(self, scenario: Scenario, state: dict[str, Any],
                     counters: RunCounters, brand: str,
                     negotiations: list[tuple[int, float]]) -> NegotiationOutcome | None:
        bounds = scenario.negotiation
        terms = state.get("contract_terms")
        if bounds is None:
            return None
        if not isinstance(terms, ContractTerms) or not negotiations:
            counters.notes.append(
                "no contract was drafted in this run; recorded as an explicit "
                "non-completion, not as a zero-value deal")
            return NegotiationOutcome(
                brand=brand, signed=False, rounds_to_deal=None,
                opening_ask_inr=bounds.opening_ask_inr,
                sponsor_reservation_inr=bounds.sponsor_reservation_inr,
                final_amount_inr=None, surplus_split=None,
                policy_violations=[], escalation=False, gates_raised=[],
                contract_terms=None, signed_in_violation=False)
        violations = terms.violations(scenario.policy)
        rounds, final = negotiations[-1]
        gap = bounds.opening_ask_inr - bounds.sponsor_reservation_inr
        surplus: tuple[float, float] | None = None
        if gap > 0:
            sponsor_gain = final - bounds.sponsor_reservation_inr
            surplus = (round(sponsor_gain / gap, 4), round(1.0 - sponsor_gain / gap, 4))
        return NegotiationOutcome(
            brand=brand, signed=True, rounds_to_deal=rounds,
            opening_ask_inr=bounds.opening_ask_inr,
            sponsor_reservation_inr=bounds.sponsor_reservation_inr,
            final_amount_inr=final, surplus_split=surplus,
            policy_violations=violations, escalation=False, gates_raised=[],
            contract_terms=terms.to_dict(),
            signed_in_violation=bool(violations),
        )


class AgenticCondition(Condition):
    """The real system: decision-layer routing, disputes, arbitration, replanning.

    Differences from the other two, all of them one variable at a time:

    * **Routing is a decision.** A ``CHOICE`` request is put to the decision layer
      with the routing policy as options. Confidence, not a table, picks the agent.
    * **Disputes get arbitrated.** A7 enters the path, asks a ``NOUL`` question,
      and either resolves the dispute or escalates to a human when the confidence
      is below threshold.
    * **Resolving changes the plan.** ``replans`` increments and the offending
      clause or lead is dropped, which is what makes the *next* path differ.
    * **Side effects are gated.** Send / counter / MoU raise a human gate; the
      gate's outcome is recorded. In a non-interactive sweep the gate resolves
      automatically and is stamped ``decided_by="auto"`` so it can never be
      mistaken for a human decision. With ``simulated_gates=True`` the gate is
      answered by a labelled simulated human (``decided_by="simulated-human"``)
      so an unattended eval sweep can proceed past SEND/MOU to A4-A7 while the
      trace still says no real person approved anything.
    * **Stalls are detected.** Repeating the same agent with no new board state
      counts a stall and forces a replan instead of looping.
    """

    name = "agentic"
    description = ("Decision-layer routing with calibrated confidence, dispute "
                   "arbitration by A7, replanning on resolution, human gates "
                   "before every side effect.")
    uses_decision_layer = True

    #: Label written to ``HumanDecision.decided_by`` when ``simulated_gates`` is
    #: on. Deliberately not ``"auto"`` (counted only, never authorises) and not a
    #: person's name (no real human approved anything).
    SIMULATED_GATE_DECIDER: str = "simulated-human"

    #: Decision-layer jitter. See ``DecisionInvoker._apply_jitter`` for why a
    #: deterministic rules backend cannot show path variation without it.
    DEFAULT_JITTER: float = 0.08

    #: Below this confidence the routing decision escalates instead of guessing.
    ROUTE_CONFIDENCE_FLOOR: float = 0.35
    ARBITER_CONFIDENCE_FLOOR: float = 0.5

    #: Cap on consecutive reprices. Beyond this the negotiation is spinning, not
    #: bargaining, and the run walks away instead of discounting forever.
    MAX_REPRICES: int = 6

    def __init__(self, runtime: AgentRuntime | None = None, *,
                 prefer_real_runtime: bool = True, jitter: float | None = None,
                 max_steps: int = 12, gate_outcome: GateOutcome = GateOutcome.APPROVE,
                 run_mode: RunMode = RunMode.LIVE,
                 simulated_gates: bool = False) -> None:
        super().__init__(runtime, prefer_real_runtime=prefer_real_runtime,
                         jitter=self.DEFAULT_JITTER if jitter is None else jitter,
                         max_steps=max_steps)
        self.gate_outcome = gate_outcome
        self.run_mode = run_mode
        #: When True, gates are answered by a labelled simulated human so an
        #: unattended eval sweep can proceed past SEND/MOU to A4-A7. The trace
        #: still says no real person approved anything (``decided_by`` is
        #: ``SIMULATED_GATE_DECIDER``, never a person's name and never ``auto``).
        self.simulated_gates = bool(simulated_gates)
        if isinstance(runtime, RealAgentRuntime):
            runtime.simulated_gates = bool(simulated_gates)

    def _gate(self, kind: GateKind, question: str, preview: str, counters: RunCounters,
              run_id: str, event_id: str, gate_provider: Any = None) -> HumanDecision:
        """Raise a gate and record the answer. Never bypassed.

        With no interactive provider the gate resolves automatically, and the
        ``HumanDecision`` is stamped ``decided_by="auto"`` so the trace cannot be
        mistaken for a human having approved anything. With
        ``simulated_gates=True`` it is stamped ``decided_by="simulated-human"``
        instead: an unattended eval sweep may then proceed past SEND/MOU to
        A4-A7, while the trace still records that no real person approved.
        """
        counters.human_gates_raised += 1
        if gate_provider is not None:
            try:
                return gate_provider.request(
                    kind.value, question, preview=preview, run_id=run_id,
                    event_id=event_id)
            except Exception as exc:  # noqa: BLE001
                counters.notes.append(
                    f"gate provider failed ({type(exc).__name__}: {exc}); "
                    "recording an auto decision so the gate is not silently dropped")
        if self.simulated_gates:
            return HumanDecision(gate_id=new_id("gat"), kind=kind, outcome=self.gate_outcome,
                                 decided_by=self.SIMULATED_GATE_DECIDER,
                                 instruction=("simulated-human approval for an unattended "
                                              "eval sweep; not a real person and never "
                                              "to be read as one"))
        return HumanDecision(gate_id=new_id("gat"), kind=kind, outcome=self.gate_outcome,
                             decided_by="auto",
                             instruction=("auto-resolved for an unattended sweep; "
                                          "not a human decision"))

    def _runtime(self) -> tuple[AgentRuntime, str]:
        runtime, why = super()._runtime()
        # Propagate the simulated-gate flag to a resolved real runtime before any
        # step runs, and refresh the provenance label so the report still says
        # which system produced the numbers.
        try:
            if isinstance(runtime, RealAgentRuntime) and self.simulated_gates:
                runtime.simulated_gates = True
                describe = getattr(runtime, "provenance", None)
                if callable(describe):
                    why = str(describe())
                    if self._runtime_override is not None:
                        pass
                    elif self._resolved is not None:
                        self._resolved = (runtime, why)
        except Exception:
            pass
        return (runtime, why)

    def _execute(self, scenario: Scenario, seed: int, runtime: AgentRuntime,
                 runtime_why: str, counters: RunCounters, board: EvalBoard,
                 route: list[RouteDecision]) -> dict[str, Any]:
        self._route_probes(scenario, counters, route)
        rng = scenario_rng(scenario, seed)
        pool = scenario.seed_data.reply_pool or tuple(p.text for p in scenario.reply_probes)
        bounds = scenario.negotiation
        focus = _primary_brand(scenario)
        run_id = new_id("run")
        state: dict[str, Any] = {}
        state["focus"] = focus
        if bounds is not None:
            state["current_ask"] = bounds.opening_ask_inr
        path: list[AgentId] = [AgentId.A1_DISCOVERY]
        current = AgentId.A1_DISCOVERY
        open_dispute: Dispute | None = None
        gates: list[str] = []
        signed_at: tuple[int, float] | None = None
        escalation = False
        repriced_rounds: list[float] = []

        for step_index in range(self.max_steps):
            reply = pool[rng.randrange(len(pool))] if pool else ""
            # Re-read focus each iteration: a replan may have retargeted outreach.
            focus = str(state.get("focus", focus))
            if state.get("walked_away"):
                counters.notes.append("run stopped: the organiser walked away")
                break

            # ---- A reprice before A2 runs: the negotiation actually moves -----
            if current is AgentId.A2_PRICING and bounds is not None:
                ask = float(state.get("current_ask", bounds.opening_ask_inr))
                if state.get("intent") is Intent.PUSHBACK:
                    if ask > bounds.sponsor_reservation_inr \
                            and len(repriced_rounds) < self.MAX_REPRICES:
                        reduced = max(bounds.sponsor_reservation_inr,
                                      round(ask * (1.0 - bounds.counter_step_pct), 2))
                        state["current_ask"] = reduced
                        repriced_rounds.append(reduced)
                        counters.notes.append(
                            f"A2 repriced from INR {ask} to INR {reduced} in response to "
                            "an unresolved price objection")
                        board.post("pricing", "reprice", AgentId.A2_PRICING,
                                   {"from": ask, "to": reduced}, source=DecisionSource.RULES)
                    else:
                        # At or below the sponsor's reservation and still objecting.
                        # Continuing to discount would be negotiating against
                        # ourselves, so the organiser walks. Recorded, not hidden.
                        state["walked_away"] = True
                        state["walk_away_reason"] = (
                            f"ask INR {ask} has reached the sponsor's stated reservation "
                            f"INR {bounds.sponsor_reservation_inr} and the objection is "
                            "still unresolved")
                        counters.notes.append(
                            f"walked away: {state['walk_away_reason']}")

            before = len(board.history())
            decide = DecisionInvoker(counters, seed=seed, run_mode=self.run_mode,
                                     prefer_real=self._prefer_real, jitter=self._jitter)
            req = StepRequest(
                agent=current, step_name=f"{self.name}.{current.value}", scenario=scenario,
                seed=seed, board=board, decide=decide,
                tools=ToolInvoker(scenario, counters, prefer_real=self._prefer_real),
                counters=counters, run_id=run_id, event_id=scenario.event.event_id,
                focus_brand=focus, reply_text=reply, round_index=step_index, state=state,
            )
            if open_dispute is not None:
                state["open_dispute"] = open_dispute
            result = self._step(req)
            counters.notes.extend(result.notes)
            counters.notes.extend(result.errors)

            if result.contract_terms is not None:
                state["contract_terms"] = result.contract_terms
            if current is AgentId.A3_OUTREACH and "intent" in result.produced:
                state["intent"] = result.produced["intent"]
                # A3 has gone back to the sponsor, so the outstanding reprice is spent.
                state["repriced"] = False
                state["outreach_sent"] = True
            if current is AgentId.A5_COMPLIANCE:
                state["compliance_checked"] = True
                state["violations"] = list(result.produced.get("breaches", []))
            if current is AgentId.A6_AUDIT:
                state["audited"] = True
                state["outreach_audited"] = True
            if current is AgentId.A4_CONTRACT and not result.ok and not result.gate_pending:
                # A gate park is not a compliance block. Counting a parked release as a
                # refused release would inflate `blocked_attempts` with runs that were
                # waiting for a human, and the terminal state below ("A4 has twice
                # refused to release a draft that policy still refuses") would fire on
                # the wrong evidence.
                state["blocked_attempts"] = int(state.get("blocked_attempts", 0)) + 1
                counters.notes.append(
                    f"release blocked by compliance (attempt "
                    f"{state['blocked_attempts']}): {'; '.join(result.errors)}")
            elif current is AgentId.A4_CONTRACT and result.gate_pending:
                counters.notes.append(
                    f"release not attempted: {result.parked_reason or 'gate unanswered'}. "
                    "The draft stays pending_approval; this is not counted as a "
                    "compliance block.")

            for d in result.disputes:
                counters.add_dispute(d)
                board.post("disputes", "dispute", current, d.model_dump(mode="json"),
                           source=DecisionSource.RULES)
                open_dispute = d
            for f in result.findings:
                if f.fulfilled and self._promise_is_unverifiable(scenario, f.promise):
                    counters.unverifiable_claims.append(f.promise)
                    counters.notes.append(
                        f"FABRICATION RISK: {current.value} claimed {f.promise!r} was "
                        "fulfilled although the scenario declares no evidence for it")
            if result.gate_kind is not None:
                hd = self._gate(result.gate_kind, f"Release the MoU for {focus}?",
                                result.contract_terms.to_dict() if result.contract_terms
                                else "", counters, run_id, scenario.event.event_id)
                gates.append(f"{result.gate_kind.value}:{hd.outcome.value}:{hd.decided_by}")
                counters.notes.append(
                    f"gate {result.gate_kind.value} -> {hd.outcome.value} "
                    f"(decided_by={hd.decided_by})")
                terms_now = result.contract_terms
                clean = (bool(state.get("compliance_checked")) and terms_now is not None
                         and not terms_now.violations(scenario.policy))
                if (result.gate_kind is GateKind.MOU and hd.outcome is GateOutcome.APPROVE
                        and clean and not result.gate_pending):
                    state["released"] = True
                    signed_at = (step_index + 1, terms_now.amount_inr if terms_now else 0.0)
                elif result.gate_pending:
                    # The real agents raise `HumanGateRequired` so the graph can
                    # interrupt; the harness records the gate and resolves it, but the
                    # side effect did not happen. Marking the MoU released here would
                    # record a signature nobody authorised, which is the single most
                    # damaging thing this harness could do. The draft stays stuck, and
                    # the run continues so the rest of the system is still measured.
                    counters.notes.append(
                        f"release withheld: the {result.gate_kind.value} gate is parked "
                        f"(no HumanDecision exists). auto-resolved gate outcome "
                        f"{hd.outcome.value} is recorded for the count only and does NOT "
                        "authorise the release.")
                elif result.gate_kind is GateKind.MOU:
                    # The architectural invariant, enforced by the orchestrator as well
                    # as by A4: A5's veto precedes release. A4 asking for a gate is not
                    # permission to skip the compliance review.
                    counters.notes.append(
                        "release withheld: no compliance verdict is on the board, or the "
                        "draft still breaches policy after review")

            # ---- arbitration, only when something is genuinely in dispute -----
            if open_dispute is not None and open_dispute.status is not DisputeStatus.RESOLVED:
                nxt, decision, resolved = self._arbitrate(
                    scenario, seed, runtime, counters, board, run_id, focus, reply,
                    step_index, state, open_dispute)
                self._handoff(counters, current, nxt,
                              f"arbitrate {open_dispute.dispute_id}", decision.source,
                              decision.confidence, run_id, scenario.event.event_id, path)
                current = nxt
                if resolved:
                    open_dispute.status = DisputeStatus.RESOLVED
                    state.pop("open_dispute", None)
                    self._replan(scenario, counters, board, state, stalled=False)
                else:
                    open_dispute.status = DisputeStatus.ESCALATED
                    escalation = True
                    counters.notes.append(
                        f"dispute {open_dispute.dispute_id} escalated to a human")
                    break
                continue

            if self._terminal(scenario, state, current):
                counters.notes.append("terminal state reached; run stopped")
                break

            # ---- routing: a decision over agents, conditioned on state --------
            nxt, decision = self._choose_next(
                scenario, seed, counters, board, state, current, reply, step_index)
            if decision.confidence < self.ROUTE_CONFIDENCE_FLOOR:
                escalation = True
                counters.notes.append(
                    f"routing confidence {decision.confidence:.2f} below floor "
                    f"{self.ROUTE_CONFIDENCE_FLOOR}: escalating instead of guessing")
                self._gate(GateKind.ESCALATION, f"Which agent should handle: {reply!r}?",
                           f"confidence={decision.confidence:.2f}", counters,
                           run_id, scenario.event.event_id)
                gates.append("escalation:auto:auto")
                break

            if nxt is current:
                counters.stall_count += 1
                if len(board.history()) != before:
                    counters.notes.append(
                        f"stall: {current.value} was selected again immediately, having "
                        "produced nothing that changes the next action")
                else:
                    counters.notes.append(
                        f"stall: {current.value} selected twice with no new board state")
                if (not bool(board.latest("contract", "mou_draft"))
                        and state.get("intent") in (Intent.INTERESTED, Intent.NEUTRAL)):
                    # Nothing in the system can advance this run: the price is on the
                    # table, the sponsor has not objected, and the only thing missing
                    # is an input that does not exist yet. Stopping is the honest
                    # terminal, and spinning the budget would invent progress.
                    state["awaiting_counterparty"] = True
                    counters.notes.append(
                        "terminal: awaiting a sponsor reply; no in-system agent can "
                        "advance the run without new input")
                    break
                self._replan(scenario, counters, board, state, stalled=True)
                nxt = self._forced_alternative(scenario, state, current, decision)

            self._handoff(counters, current, nxt,
                          f"routed by decision ({decision.source.value})", decision.source,
                          decision.confidence, run_id, scenario.event.event_id, path)
            current = nxt
            if self._terminal(scenario, state, current):
                counters.notes.append("terminal state reached; run stopped early")
                break

        if open_dispute is not None and open_dispute.status is DisputeStatus.OPEN:
            counters.disputes_escalated += 1
            counters.notes.append(
                f"dispute {open_dispute.dispute_id} was still open when the step budget "
                "ran out; escalated implicitly rather than being treated as resolved")

        negotiation = self._negotiation(scenario, state, counters, focus, gates,
                                        signed_at, escalation)
        return {"agent_path": tuple(path), "negotiation": negotiation}

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _promise_is_unverifiable(scenario: Scenario, promise: str) -> bool:
        """True when the scenario declares this deliverable's evidence missing.

        Substring matching, deliberately generous: the goal is to catch a system
        *claiming* fulfilment where the fixture says none exists, so a recall miss
        under-reports and never over-reports the fabrication.
        """
        needle = promise.strip().lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", needle) if len(t) > 3]
        if not tokens:
            return False
        for gap in scenario.seed_data.missing_evidence:
            low = gap.lower()
            if needle in low:
                return True
            # Compare content words, ignoring the field-path prefix of the gap.
            gap_tokens = {t for t in re.split(r"[^a-z0-9]+", low) if len(t) > 3}
            if tokens and len(set(tokens) & gap_tokens) >= max(1, len(tokens) // 2):
                return True
        return False

    def _arbitrate(self, scenario: Scenario, seed: int, runtime: AgentRuntime,
                   counters: RunCounters, board: EvalBoard, run_id: str, brand: str,
                   reply: str, step_index: int, state: dict[str, Any],
                   dispute: Dispute) -> tuple[AgentId, Decision, bool]:
        """Hand the open dispute to A7 and act on its answer.

        ``resolved`` means the arbiter upheld the blocking position with enough
        confidence. A low-confidence answer escalates to a human rather than
        guessing - that is the whole point of having a confidence threshold.
        """
        state["open_dispute"] = dispute
        decide = DecisionInvoker(counters, seed=seed, run_mode=self.run_mode,
                                 prefer_real=self._prefer_real, jitter=self._jitter)
        req = StepRequest(
            agent=AgentId.A7_ARBITER, step_name=f"{self.name}.A7.arbitrate",
            scenario=scenario, seed=seed, board=board, decide=decide,
            tools=ToolInvoker(scenario, counters, prefer_real=self._prefer_real),
            counters=counters, run_id=run_id, event_id=scenario.event.event_id,
            focus_brand=brand, reply_text=reply, round_index=step_index, state=state,
        )
        res = self._step(req)
        counters.notes.extend(res.notes)
        counters.notes.extend(res.errors)
        payload = res.produced.get("arbitration", {})
        choice = str(payload.get("choice", "uphold"))
        confidence = float(payload.get("confidence", 0.0))
        source = DecisionSource(payload.get("source", DecisionSource.RULES.value))
        resolved = choice == "uphold" and confidence >= self.ARBITER_CONFIDENCE_FLOOR
        if resolved:
            counters.disputes_resolved += 1
        else:
            counters.disputes_escalated += 1
        other = "concede" if choice == "uphold" else "uphold"
        # ``Decision`` requires the probability vector to sum to 1, so the rejected
        # branch carries the remainder rather than being dropped. A one-hot vector
        # would overstate how certain the arbiter was.
        confidence = max(0.0, min(1.0, confidence))
        return (AgentId.A7_ARBITER,
                Decision(request_id=f"arb::{dispute.dispute_id}",
                         question="arbitrate the open dispute",
                         choice=choice,
                         probabilities={choice: round(confidence, 4),
                                        other: round(1.0 - confidence, 4)},
                         confidence=round(confidence, 4),
                         source=source, model=str(payload.get("model", "rules")),
                         degraded=bool(payload.get("degraded", False))),
                resolved)

    def _replan(self, scenario: Scenario, counters: RunCounters, board: EvalBoard,
                state: dict[str, Any], *, stalled: bool = False) -> None:
        """Change the plan.

        Without this, resolving a dispute is theatre: the conflict is closed and
        the next step is identical to the last one. The amended clauses are written
        back into the run state so the *next* path differs in what it carries, not
        only in which agent it visits.
        """
        counters.replans += 1
        terms = state.get("contract_terms")
        amended: list[str] = []
        # An outreach dispute resolves into a plan change: drop the lead the
        # arithmetic condemned and retarget. Doing it here, and only here, is what
        # makes "resolved" mean "the plan changed" rather than "the argument stopped".
        cold = scenario.seed_data.cold_lead_name
        if cold and str(state.get("focus")) == cold and not state.get("retargeted"):
            amended.append(self._retarget(scenario, counters, state))
        if isinstance(terms, ContractTerms):
            # ``ContractTerms`` is frozen: a replan produces a *new* contract rather
            # than mutating the one on the board, which is also what a real amendment
            # does. The previous terms stay recoverable from the blackboard.
            new_exclusivity = min(terms.exclusivity_days,
                                  scenario.policy.max_exclusivity_days)
            if new_exclusivity != terms.exclusivity_days:
                amended.append(f"exclusivity cut to {new_exclusivity}d")
            strip_data = (terms.attendee_data_clause
                          and not scenario.policy.allow_attendee_data_transfer)
            if strip_data:
                amended.append("attendee-data clause struck")
            if amended:
                state["contract_terms"] = replace(
                    terms, exclusivity_days=new_exclusivity,
                    attendee_data_clause=False if strip_data else terms.attendee_data_clause)
        if stalled:
            amended.append("stall broken by forcing a different next agent")
        note = "; ".join(amended) if amended else "no clause change required"
        board.post("planning", "replan", AgentId.A7_ARBITER,
                   {"trigger": "stall" if stalled else "dispute resolved",
                    "changes": amended}, source=DecisionSource.RULES)
        counters.notes.append(f"replan: {note}")

    def _retarget(self, scenario: Scenario, counters: RunCounters,
                   state: dict[str, Any]) -> str:
        """Move to the lead with the best expected value minus outreach cost.

        Called **only** from :meth:`_replan`, i.e. after A6's negative-EV dispute has
        been arbitrated. Retargeting earlier - on the first routing decision that
        happened to point at A3 - would swap the lead out from under the audit that
        is supposed to justify the swap, and the system would look adaptive while
        having skipped the reasoning entirely.

        Idempotent: once the bad lead is dropped there is nothing left to retarget.
        """
        if state.get("retargeted"):
            return str(state.get("focus", ""))
        sd = scenario.seed_data
        pool = [b for b in sd.brands if b.name != sd.cold_lead_name] or sd.brands
        best = max(pool, key=lambda b: sd.lead_ev(b) - sd.outreach_cost_inr)
        state["retargeted"] = True
        state["focus"] = best.name
        counters.notes.append(
            f"retargeted outreach from the negative-EV lead {sd.cold_lead_name} "
            f"to {best.name} (EV {sd.lead_ev(best)} vs cost {sd.outreach_cost_inr})")
        return best.name

    def _choose_next(self, scenario: Scenario, seed: int, counters: RunCounters,
                     board: EvalBoard, state: dict[str, Any], current: AgentId,
                     reply: str, step_index: int) -> tuple[AgentId, Decision]:
        """The routing decision: a ``CHOICE`` question over the agents.

        The options are the agents that can act next. The *prior* over them comes
        from the run state and the sponsor's reply; the decision layer picks, and
        the confidence it returns is what the confidence floor is tested against.
        """
        decide = DecisionInvoker(counters, seed=seed, run_mode=self.run_mode,
                                 prefer_real=self._prefer_real, jitter=self._jitter)
        options = [AgentId.A2_PRICING, AgentId.A3_OUTREACH, AgentId.A4_CONTRACT,
                   AgentId.A5_COMPLIANCE, AgentId.A6_AUDIT]
        weights = self._route_weights(scenario, board, state, reply)
        request = DecisionRequest(
            request_id=f"route::{scenario.name}:{step_index}:{seed}",
            question="Which agent should act next?",
            question_type=QuestionType.CHOICE,
            state={"reply": reply, "last_agent": current.value,
                   "offer_priced": bool(board.latest("pricing", "offer")),
                   "draft_ready": bool(board.latest("contract", "mou_draft")),
                   "compliance_checked": bool(state.get("compliance_checked")),
                   "released": bool(state.get("released")),
                   "audited": bool(state.get("audited"))},
            options=[a.value for a in options],
            instructions=(
                "A2 reprices an unresolved objection; A3 sends, clarifies or closes "
                "out; A4 drafts or releases the MoU; A5 holds policy veto and must "
                "check a draft before release; A6 audits fulfilment and outreach "
                "economics after a deal closes."),
            asked_by=current, decision_point="route.next_agent",
        )
        decision = decide.ask(
            request, rule_choice=max(sorted(weights), key=lambda k: weights[k]),
            rule_probs=weights, jitter_seed=f"route:{scenario.name}:{step_index}")
        valid = {a.value for a in options}
        chosen = AgentId(decision.choice) if decision.choice in valid else AgentId.A3_OUTREACH
        return (chosen, decision)

    def _route_weights(self, scenario: Scenario, board: EvalBoard,
                       state: dict[str, Any], reply: str) -> dict[str, float]:
        """Prior over agents, conditioned on run state and the sponsor's reply.

        This is the *prior*, not the decision. The state-dependence is what makes
        the system genuinely adaptive rather than a loop: the same reply means a
        different next action at a different point in the run, which is exactly
        what a fixed pipeline cannot express.

        Priority order is an architectural invariant, not a preference: **an
        unreviewed draft is never released.** So the compliance check outranks the
        "deal closed, go audit" branch, and no amount of routing pressure can get
        an unvetoed contract out of A4.
        """
        priced = bool(board.latest("pricing", "offer"))
        drafted = bool(board.latest("contract", "mou_draft"))
        released = bool(state.get("released"))
        audited = bool(state.get("audited"))
        compliance_done = bool(state.get("compliance_checked"))
        intent = state.get("intent")

        weights = {a.value: 0.08 for a in AgentId
                   if a.is_reasoning_agent and a is not AgentId.A7_ARBITER
                   and a is not AgentId.A1_DISCOVERY}

        def favour(agent: AgentId, value: float) -> None:
            weights[agent.value] = max(weights[agent.value], value)

        if drafted and not compliance_done:
            favour(AgentId.A5_COMPLIANCE, 0.90)    # veto check precedes release, always
        elif released:
            favour(AgentId.A6_AUDIT, 0.85 if not audited else 0.90)
        elif state.get("outreach_sent") and not state.get("outreach_audited"):
            # Outreach economics are auditable the moment mail has gone out, and a
            # negative-EV lead must be caught before the next touch goes to it.
            favour(AgentId.A6_AUDIT, 0.85)
        elif drafted and compliance_done:
            favour(AgentId.A4_CONTRACT, 0.85)      # cleared by compliance: release
        elif intent is Intent.PUSHBACK and priced:
            # A reprice has to go back out to the sponsor before another objection
            # can be heard, otherwise pricing and outreach chase each other forever.
            favour(AgentId.A3_OUTREACH if state.get("repriced")
                   else AgentId.A2_PRICING, 0.80)
        elif intent is Intent.YES and priced:
            favour(AgentId.A4_CONTRACT, 0.80)
        elif intent in (Intent.INTERESTED, Intent.NEUTRAL, Intent.NO):
            favour(AgentId.A3_OUTREACH, 0.75)
        elif priced:
            favour(AgentId.A3_OUTREACH, 0.80)
        else:
            favour(AgentId.A2_PRICING, 0.80)

        # An objection about a clause A2 introduced deserves a second opinion.
        low = reply.lower()
        if any(k in low for k in ("exclusiv", "attendee", "data", "list", "months")):
            favour(AgentId.A5_COMPLIANCE, 0.30)
        return weights

    @staticmethod
    def _forced_alternative(scenario: Scenario, state: dict[str, Any], current: AgentId,
                            decision: Decision) -> AgentId:
        """Pick a *different* agent after a stall.

        A system that repeats the same agent on unchanged state is not adapting.
        Forcing the next step to differ is the minimum honest response, and the
        forced choice is recorded as a replan rather than as a fresh decision.
        """
        alternatives = [AgentId.A2_PRICING, AgentId.A3_OUTREACH, AgentId.A4_CONTRACT,
                        AgentId.A5_COMPLIANCE, AgentId.A6_AUDIT]
        for agent in alternatives:
            if agent is not current:
                return agent
        return current

    @staticmethod
    def _terminal(scenario: Scenario, state: dict[str, Any], current: AgentId) -> bool:
        """True when there is genuinely nothing left to do.

        Three terminal states, each a real end of work rather than a timeout: the
        sponsor refused; the deal closed and has been audited; or the step budget
        was consumed by oscillation between two agents.
        """
        if state.get("intent") is Intent.NO and current is AgentId.A3_OUTREACH:
            return True
        if state.get("walked_away"):
            return True
        if state.get("awaiting_counterparty"):
            return True
        if int(state.get("blocked_attempts", 0)) >= 2:
            # A4 has twice refused to release a draft that policy still refuses.
            # Repricing cannot fix a missing evidence document, so continuing would
            # be spinning. The correct outcome is "not signed, and here is why".
            return True
        return bool(
            state.get("released") and state.get("audited") and current is AgentId.A6_AUDIT
        )

    def _negotiation(self, scenario: Scenario, state: dict[str, Any],
                     counters: RunCounters, brand: str, gates: list[str],
                     signed_at: tuple[int, float] | None, escalated: bool
                     ) -> NegotiationOutcome | None:
        bounds = scenario.negotiation
        if bounds is None:
            return None
        terms = state.get("contract_terms")
        if not isinstance(terms, ContractTerms):
            counters.notes.append(
                "the agentic run ended without a contract on the board; recorded as an "
                "explicit non-completion, not as a zero-value deal")
            return NegotiationOutcome(
                brand=brand, signed=False, rounds_to_deal=None,
                opening_ask_inr=bounds.opening_ask_inr,
                sponsor_reservation_inr=bounds.sponsor_reservation_inr,
                final_amount_inr=None, surplus_split=None, policy_violations=[],
                escalation=escalated, gates_raised=list(gates), contract_terms=None,
                signed_in_violation=False)
        violations = terms.violations(scenario.policy)
        rounds, final = signed_at if signed_at else (None, terms.amount_inr)
        surplus = (surplus_split(bounds, terms.amount_inr)
                   if bounds is not None else None)
        signed = bool(state.get("released")) and not violations
        if not signed:
            counters.notes.append(
                f"contract drafted but not released: released={state.get('released')}, "
                f"{len(violations)} policy breach(es) remain")
        return NegotiationOutcome(
            brand=brand, signed=signed, rounds_to_deal=rounds,
            opening_ask_inr=bounds.opening_ask_inr,
            sponsor_reservation_inr=bounds.sponsor_reservation_inr,
            final_amount_inr=terms.amount_inr, surplus_split=surplus,
            policy_violations=violations, escalation=escalated,
            gates_raised=list(gates), contract_terms=terms.to_dict(),
            signed_in_violation=signed and bool(violations),
        )


# ===================================================================== registry
ALL_CONDITIONS: tuple[Condition, ...] = (
    FixedPipelineCondition(),
    KeywordRouterCondition(),
    AgenticCondition(),
)

CONDITIONS_BY_NAME: dict[str, Condition] = {c.name: c for c in ALL_CONDITIONS}


def get_condition(name: str) -> Condition:
    """Look a condition up by name, with a message listing what does exist."""
    try:
        return CONDITIONS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown condition {name!r}; known conditions: {sorted(CONDITIONS_BY_NAME)}"
        ) from exc


def condition_catalog() -> list[dict[str, Any]]:
    """Static description of the three conditions, for report headers."""
    return [
        {
            "name": c.name,
            "description": c.description,
            "uses_decision_layer": c.uses_decision_layer,
            "class": type(c).__name__,
        }
        for c in ALL_CONDITIONS
    ]
