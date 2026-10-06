"""Unit tests for the ``graph`` orchestration layer.

Fully offline. No network, no model, no database file: every test supplies a stub
decision callable and either a fake agent or none at all.

What is being tested, and why each one matters more than it looks
-----------------------------------------------------------------
``test_reducers_merge_parallel_branches``
    The failure this prevents is silent. Two superstep branches both writing
    ``handoffs`` without a reducer means one branch's routing evidence vanishes
    and nothing errors.

``test_route_after_reply_*``
    The classification router is where a wrong answer costs a real sponsorship.
    Asserting *each* intent's destination pins the mapping that a reviewer will
    read off the diagram.

``test_stall_counter_fires_and_replans``
    A stall detector that cannot fire is a comment, not a control. This drives
    the counter past ``max_replans`` and asserts the run ends with a stated
    reason instead of looping.

``test_contract_net_awards_highest_utility``
    Proves the award follows ``Bid.utility`` and not agent ordering — i.e. the
    protocol is real.

``test_debate_produces_disputes_and_reaches_adjudication``
    Proves each side authors a distinct position and that the arbiter, not the
    last speaker, closes the dispute.

``test_gate_records_human_decision``
    Proves a HumanGate is raised, a HumanDecision is applied, and the run
    actually parks and resumes on the same thread id.

``test_graph_degrades_when_agent_missing``
    Proves the topology is stable when an agent is absent, which is the condition
    the whole package has to survive during concurrent development.
"""
from __future__ import annotations

from typing import Any

import pytest

from core import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    DisputeZone,
    EventProfile,
    GateKind,
    GateOutcome,
    Intent,
    RunMode,
    Settings,
    TraceKind,
)
from core.protocols import ActResult, Observation
from graph import (
    LABEL,
    NODE,
    PaytriqState,
    build_checkpointer,
    build_graph,
    contract_net,
    debate,
    default_config,
    edges,
    initial_state,
    interrupts,
    registry_status,
    replay,
)
from graph import state as gstate
from graph.checkpointer import CheckpointSetup


# ======================================================================= fixtures
def make_settings(**overrides: Any) -> Settings:
    """Offline settings: rules decision layer, non-interactive gates, tiny budgets."""
    base: dict[str, Any] = {
        "decision_backend": "rules",
        "run_mode": RunMode.OFFLINE,
        "human_gates_interactive": False,
        "auto_approve_outcome": "approve",
        "max_debate_rounds": 2,
        "max_replans": 2,
        "max_auction_rounds": 2,
        "agent_step_budget": 2,
        "agent_deadline_s": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------- stub decide
class StubDecide:
    """A scripted decision layer.

    ``answers`` maps a *substring of the decision_point* to either a choice string
    or a callable taking the :class:`DecisionRequest`. Anything unmatched gets
    ``default``. Every answer is recorded in :attr:`calls`, so a test can assert
    that a decision was actually asked for rather than inferred.
    """

    def __init__(self, answers: dict[str, Any] | None = None,
                 default: str = "proceed", source: DecisionSource = DecisionSource.RULES,
                 confidence: float = 0.8) -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.source = source
        self.confidence = confidence
        self.calls: list[DecisionRequest] = []

    def __call__(self, request: DecisionRequest) -> Decision:
        self.calls.append(request)
        point = request.decision_point or request.question
        choice: Any = self.default
        for needle, value in self.answers.items():
            if needle in point:
                choice = value(request) if callable(value) else value
                break
        if isinstance(choice, Decision):
            return choice
        choice = str(choice)
        options = [o for o in request.options if o] or [choice, "other"]
        if choice not in options:
            options = [*options, choice]
        # Distribute probabilities so the scripted choice holds at least
        # ``confidence``. ``Decision`` forbids a confidence above the chosen
        # option's probability, so a high scripted confidence must push
        # probability onto the choice — otherwise every multi-option question
        # would silently cap at 1/len(options) and confidence assertions would be
        # meaningless.
        others = [o for o in options if o != choice]
        target = min(max(self.confidence, 1.0 / len(options)), 1.0)
        rest = round((1.0 - target) / len(others), 4) if others else 0.0
        probabilities = {choice: round(target, 4)}
        for other in others:
            probabilities[other] = rest
        total = sum(probabilities.values()) or 1.0
        probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}
        confidence = min(probabilities[choice], self.confidence if self.confidence else 1.0)
        return Decision(
            request_id=request.request_id,
            question=request.question,
            choice=choice,
            probabilities=probabilities,
            confidence=confidence,
            source=self.source,
            model="tests.stub_decide",
            degraded=self.source is DecisionSource.RULES,
            raw={"stub": True, "decision_point": point},
        )

    def points(self) -> list[str]:
        return [c.decision_point for c in self.calls]


# ------------------------------------------------------------------- fake agents
class FakeAgent:
    """The smallest thing that satisfies the graph's needs of an agent.

    It is *not* a :class:`~agents.base.ReActAgent` subclass: these tests exercise
    the orchestrator, and the registry is tested separately against real
    subclasses. What matters here is ``id``, ``role`` and ``run(ctx)``.
    """

    def __init__(self, agent_id: AgentId, *, role: str = "fake",
                 posts: list[tuple[str, str, dict[str, Any]]] | None = None,
                 sufficient: bool = True, fail: bool = False) -> None:
        self.id = agent_id
        self.role = role
        self.step_budget = 2
        self.deadline_s = 5.0
        self.posts = posts or []
        self.sufficient = sufficient
        self.fail = fail
        self.calls = 0

    def run(self, ctx: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"fake {self.id.value} is configured to fail")
        for zone, kind, payload in self.posts:
            ctx.board.post(zone, kind, self.id, payload, source=DecisionSource.RULES)
        observation = Observation(summary=f"{self.id.value} fake step", sufficient=self.sufficient)
        from agents.base import AgentResult

        return AgentResult(self.id, observation, None, 1, 1.0, False, False, [])


def brand_lead(lead_id: str, name: str, fit: float = 70.0,
               email: str | None = "sponsor@example.invalid",
               author: str = AgentId.A1_DISCOVERY.value) -> dict[str, Any]:
    return {
        "lead_id": lead_id, "name": name, "category": "cafe", "distance_km": 1.2,
        "contact_email": email, "phone": None, "rating": 4.2, "fit_score": fit,
        "fit_breakdown": {}, "fit_rationale": "near the venue",
        "source": "rules", "author": author, "discovered_at": "2026-01-01T00:00:00Z",
    }


def offer(offer_id: str, brand: str, amount: float = 45000.0,
          version: int = 1) -> dict[str, Any]:
    return {
        "offer_id": offer_id, "event_id": "evt_test", "brand": brand, "tier": "silver",
        "amount_inr": amount, "deliverables": ["booth", "logo"], "pitch": "",
        "fit_score": 70.0, "version": version, "revised": False, "revision_note": "",
        "author": AgentId.A2_PRICING.value, "created_at": "2026-01-01T00:00:00Z",
    }


def thread(thread_id: str, brand: str, *, status: str = "sent",
           reply: str = "", intent: Intent | None = None) -> dict[str, Any]:
    return {
        "thread_id": thread_id, "event_id": "evt_test", "brand": brand,
        "email": "sponsor@example.invalid", "status": status, "day": 0,
        "intent": (intent.value if intent else Intent.UNKNOWN.value),
        "reply_text": reply, "offer_id": "off_test", "sent_at": None, "delivered": False,
        "author": AgentId.A3_OUTREACH.value,
    }


def risk_flag(flag_id: str, brand: str = "Cafe", *, severity: str = "medium",
              resolved: bool = False) -> dict[str, Any]:
    return {
        "flag_id": flag_id, "event_id": "evt_test", "brand": brand,
        "severity": severity, "code": "PROMISE_NOT_FULFILLED",
        "message": "no inspectable evidence for the booth deliverable",
        "evidence": ["be_1"], "raised_by": AgentId.A5_COMPLIANCE.value,
        "raised_at": "2026-01-01T00:00:00Z", "resolved": resolved, "resolution": "",
    }


def blocking_flag(flag_id: str, brand: str = "Cafe") -> dict[str, Any]:
    return risk_flag(flag_id, brand, severity="blocking")


def base_state(**overrides: Any) -> PaytriqState:
    """A state with enough world in it for the routers to have something to read."""
    profile = EventProfile(event_id="evt_test", name="TechFest", location="Pune",
                           footfall=5000, date="2026-11-15",
                           audience="college students")
    state = initial_state(event=profile, run_id="run_test")
    state.update({
        "event_id": "evt_test",
        "run_id": "run_test",
        "brands": [brand_lead("brd_1", "Cafe One"), brand_lead("brd_2", "Cafe Two"),
                   brand_lead("brd_3", "Cafe Three")],
        "offers": [offer("off_1", "Cafe One")],
        "threads": [thread("thr_1", "Cafe One")],
        "mous": [{"mou_id": "mou_1", "event_id": "evt_test", "brand": "Cafe One",
                  "amount_inr": 45000.0, "terms": "standard", "deliverables": ["booth"],
                  "status": "draft", "document_path": None, "version": 1,
                  "created_at": "2026-01-01T00:00:00Z"}],
        "agent_id": AgentId.A3_OUTREACH.value,
    })
    state.update(overrides)
    return state


def make_runtime(decide: Any | None = None, agents: dict | None = None,
                 *, settings: Settings | None = None,
                 interactive: bool | None = None,
                 thread_id: str = "t_test") -> gstate.GraphRuntime:
    cfg = settings or make_settings()
    board = gstate.InMemoryBoardLike()
    tracer = gstate.NullTracer()
    return gstate.GraphRuntime(
        settings=cfg,
        board=board,
        tracer=tracer,
        decide=decide or StubDecide(),
        agents=dict(agents or {}),
        mode=RunMode.OFFLINE,
        gate_policy=gstate.GatePolicy(
            interactive=bool(cfg.human_gates_interactive if interactive is None
                             else interactive),
            auto_outcome=cfg.auto_approve_outcome,
        ),
        thread_id=thread_id,
        run_id="run_test",
        event_id="evt_test",
    )


def memory_checkpointer() -> CheckpointSetup:
    from langgraph.checkpoint.memory import InMemorySaver

    return CheckpointSetup(saver=InMemorySaver(), kind="memory")


# ============================================================== 1. reducers / state
class TestReducers:
    def test_append_merges_two_parallel_branches(self) -> None:
        """The core LangGraph guarantee: concurrent writes concatenate."""
        left = [{"reason": "from branch g"}]
        right = [{"reason": "from branch h"}]
        assert gstate.append(left, right) == [{"reason": "from branch g"},
                                              {"reason": "from branch h"}]
        assert gstate.append(None, right) == right
        assert gstate.append(left, None) == left

    def test_keep_last_ignores_none(self) -> None:
        """A node with nothing to report must not erase a sibling's value."""
        assert gstate.keep_last(7, None) == 7
        assert gstate.keep_last(7, 9) == 9

    def test_merge_dicts_merges_ledgers(self) -> None:
        merged = gstate.merge_dicts({"revision": 1, "plan": ["a"]},
                                    {"revision": 2, "plan": ["b"]})
        assert merged == {"revision": 2, "plan": ["b"]}

    def test_upsert_by_converges_on_the_same_entity(self) -> None:
        """Two agents pricing one brand must produce one offer, not two."""
        reducer = gstate.upsert_by("offer_id")
        merged = reducer([offer("off_1", "Cafe", 45000.0, version=1)],
                         [offer("off_1", "Cafe", 40000.0, version=2)])
        assert len(merged) == 1
        assert merged[0]["version"] == 2

    def test_upsert_by_keeps_distinct_entities(self) -> None:
        reducer = gstate.upsert_by("offer_id")
        merged = reducer([offer("off_1", "Cafe")], [offer("off_2", "Tea")])
        assert [o["offer_id"] for o in merged] == ["off_1", "off_2"]

    def test_upsert_by_keeps_unidentifiable_records(self) -> None:
        """An artefact with no id is appended, never silently dropped."""
        reducer = gstate.upsert_by("offer_id")
        assert reducer([{"value": "x"}], [{"value": "y"}]) == [{"value": "x"},
                                                                {"value": "y"}]

    def test_state_channels_declare_the_expected_reducers(self) -> None:
        from typing import get_type_hints

        # ``from __future__ import annotations`` stringifies every annotation, so
        # the resolved hints (with extras) are what actually reaches LangGraph.
        hints = get_type_hints(gstate.PaytriqState, include_extras=True)
        assert hints["handoffs"].__metadata__[0] is gstate.append
        assert hints["disputes"].__metadata__[0] is gstate.append
        assert hints["lessons"].__metadata__[0] is gstate.append
        assert hints["approvals"].__metadata__[0] is gstate.append
        assert hints["trace_events"].__metadata__[0] is gstate.append
        assert hints["risk_flags"].__metadata__[0] is gstate.append
        assert hints["offers"].__metadata__[0].__name__ == "upsert_by_offer_id"
        assert hints["stall_count"].__metadata__[0] is gstate.keep_last

    def test_initial_state_seeds_every_channel(self) -> None:
        state = initial_state(event_id="evt_x")
        for channel in ("handoffs", "disputes", "brands", "stall_count",
                        "task_ledger", "progress_ledger"):
            assert channel in state, channel
        assert state["stall_count"] == 0
        assert state["task_ledger"]["revision"] == 0


class TestLedgers:
    def test_task_ledger_separates_facts_from_guesses(self) -> None:
        ledger = gstate.new_task_ledger(facts=["a venue exists"],
                                         guesses=["sponsors are nearby"],
                                         plan=["discover"])
        assert ledger["facts"] == ["a venue exists"]
        assert ledger["guesses"] == ["sponsors are nearby"]

    def test_progress_ledger_defaults_to_moving(self) -> None:
        ledger = gstate.new_progress_ledger()
        assert ledger["is_complete"] is False
        assert ledger["is_progress"] is True

    def test_blocking_flags_excludes_resolved_and_non_blocking(self) -> None:
        state = {"risk_flags": [risk_flag("rsk_1", severity="medium"),
                                blocking_flag("rsk_2"),
                                risk_flag("rsk_3", severity="blocking", resolved=True)]}
        assert [f["flag_id"] for f in gstate.blocking_flags(state)] == ["rsk_2"]

    def test_latest_offer_picks_the_highest_version(self) -> None:
        state = {"offers": [offer("off_1", "Cafe", 45000.0, 1),
                            offer("off_2", "Cafe", 38000.0, 2)]}
        assert gstate.latest_offer_for(state, "Cafe")["offer_id"] == "off_2"
        assert gstate.latest_offer_for(state, "Nope") is None


# =================================================================== 2. routing
class TestRouteAfterReply:
    @pytest.mark.parametrize("intent,expected_label,expected_target", [
        (Intent.YES.value, LABEL.CONTRACT, NODE.CONTRACT),
        (Intent.PUSHBACK.value, LABEL.REPRICE, NODE.REVISE),
        (Intent.INTERESTED.value, LABEL.FOLLOWUP, NODE.OUTREACH),
        (Intent.NO.value, LABEL.ARCHIVE, NODE.ARCHIVE),
        (Intent.NEUTRAL.value, LABEL.FOLLOWUP, NODE.OUTREACH),
    ])
    def test_every_intent_maps_to_the_right_node(self, intent: str,
                                                 expected_label: str,
                                                 expected_target: str) -> None:
        decide = StubDecide(answers={"route.after_reply": intent})
        rt = make_runtime(decide)
        state = base_state(threads=[thread("thr_1", "Cafe One", reply="some text")])

        label = edges.route_after_reply(state, rt)

        assert label == expected_label
        assert edges.PATH_MAPS[NODE.REPLY][label] == expected_target
        assert "route.after_reply" in " ".join(decide.points()), \
            "the router must ask the decision layer, not branch on a constant"

    def test_records_a_handoff_with_source_and_confidence(self) -> None:
        decide = StubDecide(answers={"route.after_reply": Intent.PUSHBACK.value},
                            source=DecisionSource.CLEF, confidence=0.91)
        rt = make_runtime(decide)
        edges.route_after_reply(
            base_state(threads=[thread("thr_1", "Cafe", reply="too expensive")]), rt)

        queued = rt.scratch.pending_handoffs
        assert len(queued) == 1
        handoff = queued[0]
        assert handoff.from_agent is AgentId.A3_OUTREACH
        assert handoff.to_agent is AgentId.A2_PRICING
        assert handoff.decision_source is DecisionSource.CLEF
        assert handoff.confidence == pytest.approx(0.91)
        assert handoff.reason

    def test_unclassifiable_reply_is_labelled_not_guessed(self) -> None:
        decide = StubDecide(answers={"route.after_reply": "wat"})
        rt = make_runtime(decide)
        label = edges.route_after_reply(
            base_state(threads=[thread("thr_1", "Cafe", reply="hmm")]), rt)
        handoff = rt.scratch.pending_handoffs[0]
        assert label == LABEL.ARCHIVE
        assert "unrecognised" in (handoff.summary or "")

    def test_absent_reply_routes_on_nothing_and_says_so(self) -> None:
        decide = StubDecide(default="no")
        rt = make_runtime(decide)
        edges.route_after_reply(base_state(threads=[thread("thr_1", "Cafe")]), rt)
        assert rt.scratch.pending_handoffs[0].decision_source is DecisionSource.RULES


class TestRouteAfterDiscovery:
    def test_proceeds_when_enough_leads_are_viable(self) -> None:
        decide = StubDecide(answers={"route.after_discovery": "proceed"})
        rt = make_runtime(decide)
        state = base_state(discovery_radius_km=10.0)
        assert edges.route_after_discovery(state, rt) == LABEL.PROCEED
        assert edges.PATH_MAPS[NODE.DISCOVER][LABEL.PROCEED] == NODE.PRICE

    def test_replans_when_the_pipeline_is_thin(self) -> None:
        decide = StubDecide(answers={"route.after_discovery": "replan"})
        rt = make_runtime(decide)
        state = base_state(brands=[brand_lead("brd_1", "Only One")])
        label = edges.route_after_discovery(state, rt)
        assert label == LABEL.REPLAN
        assert edges.PATH_MAPS[NODE.DISCOVER][label] == NODE.REPLAN

    def test_uncontactable_leads_do_not_count_as_viable(self) -> None:
        decide = StubDecide(answers={"route.after_discovery": "replan"})
        rt = make_runtime(decide)
        state = base_state(brands=[brand_lead("b1", f"Cafe {i}", email=None)
                                   for i in range(5)])
        assert edges.route_after_discovery(state, rt) == LABEL.REPLAN


class TestRouteAfterCompliance:
    def test_clear_goes_to_audit(self) -> None:
        decide = StubDecide(answers={"route.after_compliance": "audit"})
        rt = make_runtime(decide)
        state = base_state(compliance_score=100.0, risk_flags=[risk_flag("rsk_1")])
        label = edges.route_after_compliance(state, rt)
        assert label == LABEL.CLEAR
        assert edges.PATH_MAPS[NODE.COMPLIANCE][label] == NODE.AUDIT

    def test_blocking_flag_overrides_an_optimistic_decision(self) -> None:
        """A5's veto cannot be talked out of by a decision that says 'audit'."""
        decide = StubDecide(answers={"route.after_compliance": "audit"})
        rt = make_runtime(decide)
        state = base_state(compliance_score=95.0, risk_flags=[blocking_flag("rsk_2")])
        label = edges.route_after_compliance(state, rt)
        assert label == LABEL.BLOCKING
        assert edges.PATH_MAPS[NODE.COMPLIANCE][label] == NODE.ADJUDICATE


class TestRouteProgress:
    def _stalling_state(self, **overrides: Any) -> PaytriqState:
        state = base_state(**overrides)
        state["progress_ledger"] = gstate.new_progress_ledger(
            is_complete=False, is_progress=False,
            progress_reason="the last step produced no new evidence",
            changed="(nothing)")
        return state

    def test_progress_routes_to_finalize(self) -> None:
        decide = StubDecide(answers={"route.progress": "progress"})
        rt = make_runtime(decide)
        assert edges.route_progress(base_state(), rt) == LABEL.PROGRESS
        assert edges.PATH_MAPS[NODE.AUDIT][LABEL.PROGRESS] == NODE.FINALIZE

    def test_complete_routes_to_finalize(self) -> None:
        decide = StubDecide(answers={"route.progress": "complete"})
        rt = make_runtime(decide)
        assert edges.route_progress(base_state(), rt) == LABEL.COMPLETE
        assert edges.PATH_MAPS[NODE.AUDIT][LABEL.COMPLETE] == NODE.FINALIZE

    def test_stall_counter_reaches_the_router_and_fires(self) -> None:
        decide = StubDecide(answers={"route.progress": "stalled"})
        rt = make_runtime(decide, settings=make_settings(max_replans=2))

        assert edges.route_progress(self._stalling_state(stall_count=0), rt) \
            == LABEL.STALLED
        assert edges.route_progress(self._stalling_state(stall_count=1), rt) \
            == LABEL.STALLED
        # The third consecutive stall exceeds the budget of two replans.
        assert edges.route_progress(self._stalling_state(stall_count=2), rt) \
            == LABEL.GIVE_UP
        assert edges.PATH_MAPS[NODE.AUDIT][LABEL.GIVE_UP] == NODE.FINALIZE

    def test_stall_handoff_records_the_actual_reason(self) -> None:
        decide = StubDecide(answers={"route.progress": "stalled"})
        rt = make_runtime(decide)
        edges.route_progress(self._stalling_state(stall_count=0), rt)
        reason = rt.scratch.pending_handoffs[0].reason
        assert "no new evidence" in reason
        assert "1/2" in reason, "the handoff must name the replan budget position"

    def test_ledger_claiming_completion_wins_over_a_progress_answer(self) -> None:
        decide = StubDecide(answers={"route.progress": "progress"})
        rt = make_runtime(decide)
        state = base_state()
        state["progress_ledger"] = gstate.new_progress_ledger(
            is_complete=True, is_progress=True, is_complete_reason="MoU signed")
        assert edges.route_progress(state, rt) == LABEL.COMPLETE


class TestRouteGateOutcome:
    @pytest.mark.parametrize("outcome,expected", [
        ("approve", LABEL.APPROVED), ("revise", LABEL.REVISE), ("reject", LABEL.REJECTED),
    ])
    def test_each_outcome_has_a_destination(self, outcome: str, expected: str) -> None:
        rt = make_runtime()
        state = base_state(pending_gate={"kind": GateKind.SEND.value,
                                         "outcome": outcome, "decided_by": "human"})
        label = edges.route_gate_outcome(state, rt)
        assert label == expected
        assert label in edges.PATH_MAPS[NODE.ESCALATE]

    def test_revise_goes_back_to_replan_not_back_to_the_gate(self) -> None:
        rt = make_runtime()
        state = base_state(pending_gate={"kind": GateKind.SEND.value,
                                         "outcome": "revise", "decided_by": "human"})
        assert edges.PATH_MAPS[NODE.ESCALATE][edges.route_gate_outcome(state, rt)] \
            == NODE.REPLAN


def test_every_router_label_is_mapped_and_no_label_is_a_node_name() -> None:
    """Guards two silent wiring bugs.

    A label no node answers fails only at runtime, on a specific run. A label that
    *is* a node name renders in ``draw_mermaid`` as an unlabelled edge, hiding the
    fact that a router could have answered with a node name by accident.
    """
    node_names = {v for k, v in vars(NODE).items()
                  if not k.startswith("_") and isinstance(v, str)}
    declared_labels = {v for k, v in vars(LABEL).items()
                       if not k.startswith("_") and isinstance(v, str)}
    assert node_names, "NODE must declare its node names"
    assert declared_labels, "LABEL must declare its router labels"

    mapped: set[str] = set()
    for source, path_map in edges.PATH_MAPS.items():
        assert source in node_names, f"{source} is not a node"
        assert path_map, f"{source} has an empty path_map"
        for label, target in path_map.items():
            assert target in node_names, f"{source}: {label} -> {target} is not a node"
            assert label not in node_names, f"label {label!r} collides with a node name"
            mapped.add(label)

    unused = declared_labels - mapped
    assert not unused, f"these labels are declared but no router can return them: {unused}"


# ============================================================= 3. contract net
class TestContractNet:
    def test_awards_the_highest_utility_bid(self) -> None:
        """The award must follow ``Bid.utility``, not agent order."""
        rt = make_runtime(StubDecide())
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}

        def per_agent(request: DecisionRequest) -> str:
            asked_by = request.asked_by
            if request.decision_point.endswith("feasibility"):
                return "high" if asked_by is AgentId.A3_OUTREACH else "low"
            if request.decision_point.endswith("effort"):
                return "small" if asked_by is AgentId.A3_OUTREACH else "large"
            if request.decision_point.endswith("accept"):
                return "accept"
            return "100" if asked_by is AgentId.A3_OUTREACH else "0"

        decide = rt.decide
        decide.answers = {"contract_net.bid": per_agent,
                          "contract_net.expedite": "accept"}
        result = contract_net.run_auction(
            rt, base_state(), title="contact sponsors", description="day-0 wave",
            manager=AgentId.A2_PRICING)

        assert result.winner is AgentId.A3_OUTREACH
        assert result.contested, "differently framed bids should disagree"
        ranking = dict(result.ranking())
        assert ranking[AgentId.A3_OUTREACH] > ranking[AgentId.A2_PRICING]

    def test_each_agent_answers_a_differently_framed_question(self) -> None:
        decide = StubDecide(default="medium",
                            answers={"contract_net.expedite": "accept"})
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A6_AUDIT: FakeAgent(AgentId.A6_AUDIT)}
        contract_net.run_auction(rt, base_state(), title="t", description="d",
                                 manager=AgentId.A2_PRICING,
                                 bidders=[AgentId.A2_PRICING, AgentId.A6_AUDIT])
        questions = {c.decision_point: c.question for c in decide.calls}
        pricing_q = next(q for p, q in questions.items() if ".A2." in p)
        audit_q = next(q for p, q in questions.items() if ".A6." in p)
        assert pricing_q != audit_q
        assert "pricing" in pricing_q.lower()
        assert "evidence" in audit_q.lower()

    def test_bids_are_posted_to_the_board_with_rationale(self) -> None:
        decide = StubDecide(default="medium",
                            answers={"contract_net.expedite": "accept"})
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING)}
        result = contract_net.run_auction(rt, base_state(), title="t", description="d",
                                          manager=AgentId.A2_PRICING)
        bids = rt.board.read(contract_net.ZONE_BIDS, kind=contract_net.KIND_BID)
        assert len(bids) == 1
        assert bids[0].payload["bid_id"] == result.bids[0].bid_id
        assert "feasibility=" in result.bids[0].rationale
        assert "sources:" in result.bids[0].rationale.lower()

    def test_uncontracted_task_names_the_reason(self) -> None:
        decide = StubDecide(default="medium")
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING)}
        result = contract_net.run_auction(rt, {"event_id": "evt_test"}, title="t",
                                          description="d", manager=AgentId.A2_PRICING,
                                          bidders=[])
        assert result.winner is None
        assert any("no bids" in n or "self-execute" in n for n in result.notes)

    def test_a_contractor_that_declines_loses_the_award(self) -> None:
        """Smith's revocation path: a contractor may refuse the contract."""
        decide = StubDecide(default="medium",
                            answers={"contract_net.expedite": "decline"})
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING)}
        result = contract_net.run_auction(rt, base_state(), title="t", description="d",
                                          manager=AgentId.A2_PRICING)
        assert result.winner is None
        assert any("DECLINED" in n for n in result.notes)
        expedite_entries = rt.board.read(contract_net.ZONE_TASKS,
                                         kind=contract_net.KIND_EXPEDITE)
        assert expedite_entries[-1].payload["status"] == "declined"

    def test_decision_outage_produces_a_labeled_rules_bid(self) -> None:
        """A backend failure must degrade *visibly*, not produce a silent guess."""
        from core import DecisionUnavailable

        def boom(_: DecisionRequest) -> Decision:
            raise DecisionUnavailable("clef is down")

        rt = make_runtime(boom)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING)}
        result = contract_net.run_auction(rt, base_state(), title="t", description="d",
                                          manager=AgentId.A2_PRICING)
        assert result.degraded is True
        assert all("DEGRADED" in b.rationale for b in result.bids)

    def test_ties_break_deterministically_on_agent_id(self) -> None:
        decide = StubDecide(default="medium",
                            answers={"contract_net.expedite": "accept"})
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = contract_net.run_auction(rt, base_state(), title="t", description="d",
                                          manager=AgentId.A2_PRICING)
        assert result.bids[0].utility == result.bids[1].utility
        assert result.winner is AgentId.A2_PRICING, "lowest agent id wins a tie"
        assert any("tie on utility" in n for n in result.notes)


# ===================================================================== 4. debate
class TestDebate:
    def test_each_side_authors_a_position_and_evidence_is_recorded(self) -> None:
        decide = StubDecide(default="hold")
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        rt.board.post("offers", "offer", AgentId.A2_PRICING, offer("off_1", "Cafe"))
        rt.board.post("threads", "thread", AgentId.A3_OUTREACH,
                      thread("thr_1", "Cafe"))

        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH,
                                   topic="the price on the table",
                                   zone=DisputeZone.PRICING)

        assert result.dispute.position and result.dispute.counter_position
        assert result.dispute.position != result.dispute.counter_position
        assert result.dispute.evidence, "a dispute must cite board entry ids"
        kinds = {e.kind for e in rt.board.read(debate.ZONE_DEBATE)}
        assert debate.KIND_DISPUTE in kinds

    def test_sides_are_asked_differently_framed_questions(self) -> None:
        decide = StubDecide(default="hold")
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A5_COMPLIANCE: FakeAgent(AgentId.A5_COMPLIANCE)}
        debate.run_debate(rt, base_state(), claimant=AgentId.A2_PRICING,
                          opponent=AgentId.A5_COMPLIANCE, topic="margin")
        opens = {c.decision_point: c.question for c in decide.calls
                 if c.decision_point.startswith("debate.open.")}
        assert len({q for q in opens.values()}) == 2, "identical framing defeats the point"

    def test_concession_settles_the_dispute_without_arbitration(self) -> None:
        decide = StubDecide(answers={
            "debate.open.A3": "concede",
            "debate.round": "hold",
        }, default="hold")
        rt = make_runtime(decide)
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH, topic="price")
        assert result.settled
        assert result.conceded_by is AgentId.A3_OUTREACH
        assert "opening" in result.resolution
        assert "debate.adjudicate" not in " ".join(decide.points()), \
            "a settled debate must not spend an arbitration"

    def test_concession_mid_debate_stops_the_rounds_early(self) -> None:
        decide = StubDecide(answers={"debate.round2.A3": "concede"}, default="hold")
        rt = make_runtime(decide, settings=make_settings(max_debate_rounds=5))
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH, topic="price")
        assert result.settled
        assert result.rounds == 2, "rounds must stop on concession, not burn the budget"

    def test_exhausted_rounds_reach_the_arbiter(self) -> None:
        decide = StubDecide(default="hold")
        rt = make_runtime(decide, settings=make_settings(max_debate_rounds=3))
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH, topic="price")
        assert result.rounds == 3
        assert result.adjudicated_by is AgentId.A7_ARBITER
        assert "debate.adjudicate" in " ".join(decide.points())
        assert debate.KIND_ADJUDICATION in {
            e.kind for e in rt.board.read(debate.ZONE_DEBATE)}

    def test_low_confidence_arbiter_escalates_instead_of_guessing(self) -> None:
        decide = StubDecide(answers={"debate.adjudicate": "resolve"},
                            confidence=0.4, source=DecisionSource.CLEF)
        rt = make_runtime(decide, settings=make_settings(confidence_threshold=0.62))
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH, topic="price")
        assert result.status.value == "escalated"
        assert "escalated to a human gate" in result.resolution

    def test_degraded_arbiter_always_escalates(self) -> None:
        """Even a confident-looking answer escalates when the backend degraded."""
        decide = StubDecide(answers={"debate.adjudicate": "resolve"},
                            confidence=0.99, source=DecisionSource.RULES)
        rt = make_runtime(decide, settings=make_settings(confidence_threshold=0.62))
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH, topic="price")
        assert result.status.value == "escalated"
        assert "degraded" in result.resolution

    def test_confident_arbiter_resolves(self) -> None:
        decide = StubDecide(answers={"debate.adjudicate": "resolve"},
                            confidence=0.95, source=DecisionSource.CLEF)
        rt = make_runtime(decide, settings=make_settings(confidence_threshold=0.62))
        rt.agents = {AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
                     AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)}
        result = debate.run_debate(rt, base_state(),
                                   claimant=AgentId.A2_PRICING,
                                   opponent=AgentId.A3_OUTREACH, topic="price")
        assert result.status.value == "resolved"
        assert result.resolution.startswith("arbiter resolved")


# ================================================================== 5. interrupts
class TestInterrupts:
    def test_gate_records_a_human_decision_and_approval(self) -> None:
        decide = StubDecide()
        rt = make_runtime(decide, interactive=False)
        result = interrupts.gate_send(
            rt, base_state(), question="Send to 3 sponsors?",
            preview="Cafe One: sent; Cafe Two: sent")

        assert result.gate.kind is GateKind.SEND
        assert result.decision.gate_id == result.gate.gate_id
        assert result.approval.kind is GateKind.SEND
        assert result.approval.action_taken.startswith("send outreach mail")
        assert result.outcome is GateOutcome.APPROVE
        kinds = {e.kind for e in rt.board.read(interrupts.ZONE_GATES)}
        assert {interrupts.KIND_GATE, interrupts.KIND_DECISION,
                interrupts.KIND_APPROVAL} <= kinds

    def test_auto_resolution_is_labelled_auto_not_human(self) -> None:
        rt = make_runtime(StubDecide(), interactive=False)
        result = interrupts.gate_mou(rt, base_state(), question="Release the MoU?",
                                     preview="Cafe One: 45000 INR")
        assert result.human is False
        assert result.decision.decided_by == interrupts.AUTO_DECIDER
        assert interrupts.decided_by_is_human(result.decision) is False

    def test_human_resume_is_labelled_with_who_decided(self) -> None:
        rt = make_runtime(StubDecide(), interactive=True)
        result = interrupts.run_gate(
            rt, base_state(),
            interrupts.GateRequest(kind=GateKind.COUNTER, question="Reply?",
                                   payload_preview="Cafe: too expensive",
                                   action_taken="answer counter-offer"),
            interrupt_fn=lambda payload: {"outcome": "revise", "decided_by": "dr.iyer",
                                          "instruction": "hold at 40k"},
        )
        assert result.human is True
        assert result.decision.decided_by == "dr.iyer"
        assert result.outcome is GateOutcome.REVISE
        assert result.instruction == "hold at 40k"

    def test_auto_resume_value_is_still_reported_as_auto(self) -> None:
        """An ``auto`` marker arriving over the wire is still not a human."""
        rt = make_runtime(StubDecide(), interactive=True)
        result = interrupts.run_gate(
            rt, base_state(),
            interrupts.GateRequest(kind=GateKind.ESCALATION, question="Proceed?"),
            interrupt_fn=lambda payload: {"outcome": "approve", "decided_by": "auto"},
        )
        assert result.human is False
        assert result.decision.decided_by == interrupts.AUTO_DECIDER

    @pytest.mark.parametrize("raw,expected", [
        ("approve", GateOutcome.APPROVE),
        ("APPROVE", GateOutcome.APPROVE),
        ({"outcome": "reject"}, GateOutcome.REJECT),
        ({"action": "approve", "note": "fine"}, GateOutcome.APPROVE),
        ("please drop the discount to 40k", GateOutcome.REVISE),
        ("", GateOutcome.REJECT),
        ({"outcome": "banana"}, GateOutcome.REJECT),
    ])
    def test_resume_coercion_fails_closed(self, raw: Any, expected: GateOutcome) -> None:
        outcome, _instruction, _note = interrupts.coerce_resume(
            raw, kind=GateKind.SEND,
            options=[GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE])
        assert outcome is expected

    def test_every_gate_kind_is_reachable_through_a_shorthand(self) -> None:
        rt = make_runtime(StubDecide(), interactive=False)
        results = [
            interrupts.gate_send(rt, base_state(), question="q", preview="p"),
            interrupts.gate_counter(rt, base_state(), question="q", preview="p"),
            interrupts.gate_mou(rt, base_state(), question="q", preview="p"),
            interrupts.gate_escalation(rt, base_state(), question="q", preview="p"),
        ]
        assert [r.gate.kind for r in results] == [
            GateKind.SEND, GateKind.COUNTER, GateKind.MOU, GateKind.ESCALATION]

    def test_gate_summary_separates_human_from_auto(self) -> None:
        rt = make_runtime(StubDecide(), interactive=False)
        interrupts.gate_send(rt, base_state(), question="q1", preview="p1")
        interrupts.gate_mou(rt, base_state(), question="q2", preview="p2")
        approvals = rt.board.read(interrupts.ZONE_GATES, kind=interrupts.KIND_APPROVAL)
        summary = interrupts.gate_summary([e.payload for e in approvals])
        assert summary["total"] == 2
        assert summary["auto_count"] == 2
        assert summary["human_count"] == 0
        assert summary["all_human"] is False
        assert set(summary["by_kind"]) == {"send", "mou"}


# ================================================================ 6. graph build
class TestBuildGraph:
    def test_builds_with_no_agents_at_all(self) -> None:
        graph = build_graph(make_settings(), agents={},
                            checkpointer=memory_checkpointer())
        mermaid = graph.mermaid()
        for node in (NODE.DISCOVER, NODE.PRICE, NODE.OUTREACH, NODE.ADJUDICATE,
                     NODE.FINALIZE):
            assert node in mermaid
        assert len(graph.agents_absent) == 7
        assert graph.agents_present == ()

    def test_builds_with_a_subset_of_agents(self) -> None:
        graph = build_graph(make_settings(),
                            agents={AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH)},
                            checkpointer=memory_checkpointer())
        assert graph.agents_present == (AgentId.A3_OUTREACH.value,)
        assert AgentId.A2_PRICING.value in graph.agents_absent
        assert len(graph.mermaid()) > 0, "the diagram must still render"

    def test_mermaid_comes_from_the_compiled_graph(self) -> None:
        graph = build_graph(make_settings(), agents={},
                            checkpointer=memory_checkpointer())
        assert graph.mermaid() == graph.compiled.get_graph().draw_mermaid()
        assert "route_after" not in graph.mermaid()  # labels are edge annotations

    def test_mismatched_agent_class_cannot_break_the_build(self) -> None:
        """A registry whose classes cannot be constructed is reported, not fatal."""
        from core import ConfigError
        from graph.registry import AgentRecord

        class Unconstructible(FakeAgent):
            def __init__(self, *, required_kwarg: str) -> None:  # noqa: ARG002
                super().__init__(AgentId.A1_DISCOVERY)

        record = AgentRecord(agent_id=AgentId.A1_DISCOVERY, cls=Unconstructible,
                             module="tests.fake")
        with pytest.raises(ConfigError) as excinfo:
            record.instantiate(make_settings())
        assert "A1" in str(excinfo.value)
        assert "tried" in str(excinfo.value)


# ================================================================= 7. execution
class TestExecution:
    def test_end_to_end_run_passes_every_gate_as_auto(self) -> None:
        decide = StubDecide(answers={
            "route.after_discovery": "proceed",
            "route.after_proposal": "outreach",
            "route.after_auction": "execute",
            "route.after_reply": "yes",
            "route.after_compliance": "audit",
            "route.progress": "progress",
        }, default="proceed")
        settings = make_settings(max_replans=2)
        rt = make_runtime(decide, settings=settings)
        rt.agents = {
            AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, posts=[
                ("leads", "brand_lead", brand_lead("brd_9", "Cafe Nine"))]),
            AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING, posts=[
                ("offers", "offer", offer("off_9", "Cafe Nine"))]),
            AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH, posts=[
                ("threads", "thread", thread("thr_9", "Cafe Nine", reply="yes, let's proceed",
                                             intent=Intent.YES))]),
            AgentId.A4_CONTRACT: FakeAgent(AgentId.A4_CONTRACT, posts=[
                ("contracts", "mou", {"mou_id": "mou_9", "event_id": "evt_test",
                                      "brand": "Cafe Nine", "amount_inr": 45000.0,
                                      "terms": "standard", "deliverables": ["booth"],
                                      "status": "draft", "document_path": None,
                                      "version": 1,
                                      "created_at": "2026-01-01T00:00:00Z"})]),
        }
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())

        final = graph.run(event_id="evt_test", thread_id="t_e2e")

        assert final["status"] == "complete"
        assert final["phase"] == "done"
        assert len(final["handoffs"]) >= 4, "every routing decision is recorded"
        for handoff in final["handoffs"]:
            assert handoff["from_agent"] and handoff["to_agent"]
            assert handoff["from_agent"] != handoff["to_agent"]
            assert handoff["decision_source"]
            assert 0.0 <= handoff["confidence"] <= 1.0
            assert handoff["reason"]
        assert final["approvals"], "gates ran and recorded approvals"
        assert all(a["decided_by"] == "auto" for a in final["approvals"])

    def test_missing_agent_node_runs_as_a_logged_no_op(self) -> None:
        decide = StubDecide(answers={"route.progress": "complete"}, default="proceed")
        settings = make_settings()
        rt = make_runtime(decide, settings=settings)
        rt.agents = {AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY)}
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        final = graph.run(event_id="evt_test", thread_id="t_missing")
        assert final["status"] == "complete"
        joined = " ".join(final["notes"])
        assert "A2" in joined and "not registered" in joined

    def test_a_failing_agent_does_not_kill_the_run(self) -> None:
        decide = StubDecide(answers={"route.progress": "complete"}, default="proceed")
        settings = make_settings()
        rt = make_runtime(decide, settings=settings)
        rt.agents = {
            AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, fail=True),
            AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING),
        }
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        final = graph.run(event_id="evt_test", thread_id="t_failing")
        assert final["status"] == "complete"
        assert any("RuntimeError" in n for n in final["notes"])

    def test_replan_actually_loops_and_then_gives_up(self) -> None:
        """Drives the stall counter past ``max_replans`` through the real graph."""
        decide = StubDecide(answers={
            "route.after_discovery": "replan",
            "route.after_proposal": "outreach",
            "route.after_auction": "execute",
            "route.after_reply": "no",
            "route.after_compliance": "audit",
            "route.progress": "stalled",
        }, default="proceed")
        settings = make_settings(max_replans=1)
        rt = make_runtime(decide, settings=settings)
        rt.agents = {
            AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, posts=[
                ("leads", "brand_lead", brand_lead("brd_1", "Thin", fit=10.0))]),
        }
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        final = graph.run(event_id="evt_test", thread_id="t_replan")

        assert final["status"] == "complete"
        assert final["replan_count"] >= 1, "the replan node must actually have run"
        assert final["stall_count"] >= 1
        assert final["task_ledger"]["revision"] >= 1
        assert final["discovery_radius_km"] > 10.0, "a replan must widen the radius"
        assert any("giving up" in h["reason"] for h in final["handoffs"]) or \
            any("stalled" in h["reason"] for h in final["handoffs"])

    def test_updates_stream_yields_one_delta_per_node(self) -> None:
        decide = StubDecide(answers={"route.progress": "complete"}, default="proceed")
        settings = make_settings()
        rt = make_runtime(decide, settings=settings)
        rt.agents = {AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY)}
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        chunks = list(graph.stream(event_id="evt_test", thread_id="t_stream",
                                   stream_mode="updates"))
        nodes = [node for chunk in chunks for node in chunk]
        assert NODE.DISCOVER in nodes
        assert NODE.FINALIZE in nodes
        assert "__interrupt__" not in nodes, "non-interactive gates do not park"

    def test_interactive_gate_parks_and_resumes_on_the_same_thread(self) -> None:
        """The real ``interrupt``/``Command(resume=...)`` round trip."""
        decide = StubDecide(answers={
            "route.after_discovery": "proceed",
            "route.after_proposal": "outreach",
            "route.after_auction": "execute",
            "route.after_reply": "no",
            "route.progress": "complete",
        }, default="proceed")
        settings = make_settings(human_gates_interactive=True)
        rt = make_runtime(decide, settings=settings, interactive=True)
        rt.agents = {
            AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, posts=[
                ("leads", "brand_lead", brand_lead("brd_1", "Cafe One"))]),
            AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING, posts=[
                ("offers", "offer", offer("off_1", "Cafe One"))]),
            AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH, posts=[
                ("threads", "thread", thread("thr_1", "Cafe One", reply="not interested"))]),
        }
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        thread_id = "t_hitl"

        parked = graph.run(event_id="evt_test", thread_id=thread_id)
        assert parked["status"] != "complete"
        snapshot = graph.compiled.get_state(default_config(thread_id))
        assert snapshot.interrupts, "the run must be parked on a gate"
        payload = snapshot.interrupts[0].value
        assert payload["kind"] in {k.value for k in GateKind}
        assert payload["question"]

        final = graph.resume({"outcome": "approve", "decided_by": "demo.operator"},
                             thread_id=thread_id)

        assert final["status"] == "complete"
        decisions = final["human_decisions"]
        assert decisions
        assert decisions[-1]["decided_by"] == "demo.operator"
        assert decisions[-1]["outcome"] == "approve"
        assert rt.agents[AgentId.A3_OUTREACH].calls == 1, \
            "the agent above the gate must not run twice on resume"

    def test_board_projection_advances_instead_of_re_reading(self) -> None:
        """Re-projecting the whole board would duplicate every append-only record."""
        from graph.build import collect_board_delta

        rt = make_runtime(StubDecide())
        rt.board.post("leads", "brand_lead", AgentId.A1_DISCOVERY,
                      brand_lead("brd_p", "Cafe Projected"))
        first, seq = collect_board_delta(rt)
        assert len(first["brands"]) == 1
        second, seq2 = collect_board_delta(rt)
        assert second == {}, "the same entries must not be projected twice"
        assert seq2 == seq

    def test_unknown_board_zones_are_not_guessed_into_a_field(self) -> None:
        from graph.build import _classify_entry

        assert _classify_entry("risk_flags", "risk_flag") == "risk_flags"
        assert _classify_entry("some_new_zone", "some_new_kind") is None

    def test_orchestration_owned_zones_are_never_projected(self) -> None:
        """One record, one source of truth.

        The orchestration layer writes its own records (handoffs, gates,
        approvals, disputes, bids) straight into state. If the board projection
        also picked them up, every one would be written twice — silently, because
        the channels are append-only.
        """
        from graph.build import _CLASSIFY_RULES, ORCHESTRATION_OWNED_ZONES, _classify_entry

        owned_targets = {"handoffs", "human_decisions", "approvals", "disputes",
                         "bids", "run_events"}
        for zone, kind in (("handoffs", "handoff"), ("gates", "human_gate"),
                           ("gates", "human_decision"), ("gates", "approval"),
                           ("debate", "dispute"), ("debate", "rebuttal"),
                           ("debate", "concession"), ("tasks", "bid"),
                           ("tasks", "award"), ("tasks", "announcement"),
                           ("tasks", "expedite")):
            assert _classify_entry(zone, kind) not in owned_targets, (
                f"{zone}/{kind} must not be projected into state: the layer that "
                f"created it already writes it there")

        for zone in ORCHESTRATION_OWNED_ZONES:
            for needle, _target in _CLASSIFY_RULES:
                assert needle not in zone, f"{needle!r} would match owned zone {zone!r}"

    def test_gate_outcome_survives_into_the_router_that_reads_it(self) -> None:
        """``pending_gate`` must still hold the answer when the edge is evaluated."""
        decide = StubDecide(answers={"route.progress": "complete"},
                            default="proceed")
        rt = make_runtime(decide)
        graph = build_graph(make_settings(), agents={}, runtime=rt,
                            checkpointer=memory_checkpointer())
        final = graph.run(event_id="evt_test", thread_id="t_gate_pending")
        assert final["status"] == "complete"
        assert final["approvals"], "the gate did run"
        answered = final["pending_gate"]
        assert answered, "the answered gate must still be readable"
        assert answered["outcome"] == "approve"
        assert answered["decided_by"] == "auto"
        assert answered["human"] is False

    def test_escalation_gate_outcome_reaches_its_router(self) -> None:
        """A REJECT at the escalation gate must be attributed to its decider."""
        decide = StubDecide(answers={"route.after_reply": "yes",
                                     "route.after_compliance": "arbitrate",
                                     "route.after_adjudication": "escalate",
                                     "route.progress": "complete",
                                     "route.gate_outcome": "rejected"},
                            default="proceed")
        rt = make_runtime(decide)
        rt.agents = {
            AgentId.A5_COMPLIANCE: FakeAgent(AgentId.A5_COMPLIANCE, posts=[
                ("risk_flags", "risk_flag", blocking_flag("rsk_x"))]),
            AgentId.A7_ARBITER: FakeAgent(AgentId.A7_ARBITER),
        }
        graph = build_graph(make_settings(), agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        # The pipeline only reaches compliance after a sponsor accepts, so the
        # run is seeded with an acceptance rather than forcing the router.
        seeded = base_state(threads=[thread("thr_s", "Cafe One",
                                            reply="yes, approved, sign it",
                                            intent=Intent.YES)])
        final = graph.run(state=seeded, thread_id="t_escalate_reject")
        assert final["status"] == "complete"
        reasons = " ".join(h["reason"] for h in final["handoffs"])
        assert "escalation gate" in reasons.lower(), \
            f"the escalation gate must have been reached; handoffs were: {reasons}"
        assert "rejected by unknown" not in reasons, \
            "the gate router must know WHO rejected, not read 'unknown'"


# ==================================================================== 8. replay
class TestReplay:
    def _graph_with_history(self, thread_id: str = "t_replay") -> Any:
        decide = StubDecide(answers={
            "route.after_discovery": "proceed",
            "route.after_proposal": "outreach",
            "route.after_auction": "execute",
            "route.after_reply": "pushback",
            "route.progress": "progress",
        }, default="proceed")
        settings = make_settings()
        rt = make_runtime(decide, settings=settings, thread_id=thread_id)
        rt.agents = {
            AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, posts=[
                ("leads", "brand_lead", brand_lead("brd_1", "Cafe One"))]),
            AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING, posts=[
                ("offers", "offer", offer("off_1", "Cafe One"))]),
            AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH, posts=[
                ("threads", "thread", thread("thr_1", "Cafe One",
                                             reply="too expensive, reduce the price"))]),
        }
        graph = build_graph(settings, agents=rt.agents, runtime=rt,
                            checkpointer=memory_checkpointer())
        graph.run(event_id="evt_test", thread_id=thread_id)
        return graph

    def test_lists_checkpoints_with_the_pending_node(self) -> None:
        graph = self._graph_with_history()
        history = replay.list_checkpoints(graph, "t_replay")
        assert len(history) > 4
        assert history[0].next_nodes == (), "the newest checkpoint has run out"
        pendings = {c.next_nodes for c in history if c.next_nodes}
        assert (NODE.REPLY,) in pendings
        summaries = [c.summary() for c in history]
        assert all("checkpoint_id" in s for s in summaries)

    def test_find_checkpoint_names_the_available_ids_on_failure(self) -> None:
        graph = self._graph_with_history()
        with pytest.raises(Exception) as excinfo:
            replay.find_checkpoint(graph, "t_replay", "nope")
        assert "Available" in str(excinfo.value)

    def test_fork_leaves_the_original_thread_untouched(self) -> None:
        graph = self._graph_with_history()
        original_before = graph.compiled.get_state(
            default_config("t_replay")).values
        history = replay.list_checkpoints(graph, "t_replay")
        target = next(c for c in history if c.next_nodes == (NODE.REPLY,))

        new_thread, seeded = replay.fork_from_checkpoint(
            graph, "t_replay", target.checkpoint_id,
            overrides={"discovery_radius_km": 99.0})

        assert new_thread != "t_replay"
        assert seeded["discovery_radius_km"] == 99.0
        assert seeded["handoffs"] == [], "append-only channels must be reset in a fork"
        after = graph.compiled.get_state(default_config("t_replay")).values
        assert after == original_before, "the original run must not be mutated"

    def test_a_fork_takes_a_different_branch(self) -> None:
        """Rewind to the pushback, classify it differently, watch the path change."""
        graph = self._graph_with_history("t_fork")
        history = replay.list_checkpoints(graph, "t_fork")
        target = next(c for c in history if c.next_nodes == (NODE.REPLY,))
        assert target.next_nodes == (NODE.REPLY,)

        accepting = StubDecide(answers={
            "route.after_reply": "yes",
            "route.after_compliance": "audit",
            "route.progress": "complete",
        }, default="proceed")
        final = replay.fork_and_run(graph, "t_fork", target.checkpoint_id,
                                    target_thread="t_fork_yes", decide=accepting)
        assert final["status"] == "complete"
        reasons = " ".join(h["reason"] for h in final["handoffs"])
        assert "MoU" in reasons or "agreement" in reasons

        original = graph.compiled.get_state(default_config("t_fork")).values
        assert "too expensive" in " ".join(
            str(t.get("reply_text")) for t in original["threads"])

    def test_compare_branches_reports_the_differences(self) -> None:
        graph = self._graph_with_history("t_cmp")
        history = replay.list_checkpoints(graph, "t_cmp")
        target = next(c for c in history if c.next_nodes == (NODE.REPLY,))
        replay.fork_and_run(graph, "t_cmp", target.checkpoint_id,
                            target_thread="t_cmp_other",
                            decide=StubDecide(answers={"route.after_reply": "yes",
                                                       "route.progress": "complete"},
                                              default="proceed"))
        diff = replay.compare_branches(graph, "t_cmp", "t_cmp_other")
        assert diff["identical"] is False
        assert diff["differences"]

    def test_describe_run_summarises_gates_and_decisions(self) -> None:
        graph = self._graph_with_history("t_desc")
        described = replay.describe_run(graph, "t_desc")
        assert described["thread_id"] == "t_desc"
        assert described["checkpoint_count"] > 0
        assert described["gates"]["total"] > 0
        assert described["handoff_reasons"]

    def test_missing_checkpointer_raises_a_clear_error(self) -> None:
        graph = build_graph(make_settings(), agents={},
                            checkpointer=CheckpointSetup(saver=None, kind="none"))
        with pytest.raises(Exception) as excinfo:
            replay.list_checkpoints(graph, "t_none")
        assert "checkpointer" in str(excinfo.value).lower()


# ================================================================= 9. checkpointer
class TestCheckpointer:
    def test_default_config_carries_a_thread_id(self) -> None:
        cfg = default_config("abc")
        assert cfg["configurable"]["thread_id"] == "abc"
        assert default_config("abc", checkpoint_ns="ns")["configurable"]["checkpoint_ns"] \
            == "ns"

    def test_default_thread_id_is_deterministic_per_event(self) -> None:
        from graph.checkpointer import default_thread_id

        assert default_thread_id(event_id="evt_x") == "evt_evt_x"
        assert default_thread_id(event_id="evt_x") == default_thread_id(event_id="evt_x")

    def test_sqlite_checkpointer_lands_under_the_traces_dir(self, tmp_path: Any) -> None:
        setup = build_checkpointer(tmp_path / "cp.sqlite", settings=make_settings())
        try:
            assert setup.kind == "sqlite"
            assert setup.degraded is False
            assert (tmp_path / "cp.sqlite").exists()
        finally:
            setup.close()

    def test_falls_back_to_memory_with_a_recorded_reason(self, tmp_path: Any) -> None:
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("i am a file, not a directory")
        setup = build_checkpointer(blocker / "cp.sqlite", settings=make_settings())
        assert setup.kind == "memory"
        assert setup.degraded is True
        assert "sqlite unavailable" in setup.degraded_reason
        assert setup.status()["durable"] is False

    def test_refusing_the_fallback_is_a_config_error(self, tmp_path: Any) -> None:
        from core import ConfigError

        blocker = tmp_path / "file_not_dir"
        blocker.write_text("x")
        with pytest.raises(ConfigError):
            build_checkpointer(blocker / "cp.sqlite", settings=make_settings(),
                               allow_memory_fallback=False)


# =============================================================== 10. registry
class TestRegistry:
    def test_status_reports_what_was_found_without_raising(self) -> None:
        status = registry_status(refresh=True)
        assert status["expected"] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        assert isinstance(status["found"], dict)
        assert isinstance(status["complete"], bool)

    def test_discovery_is_keyed_on_the_declared_agent_id_not_the_class_name(self) -> None:
        """A class called anything at all still lands on its declared AgentId."""
        from agents.base import ReActAgent
        from core import Plan
        from graph.registry import AgentRecord, _declared_id

        class TotallyUnexpectedName(ReActAgent):
            id = AgentId.A4_CONTRACT
            role = "contract under a name nobody predicted"

            def _plan(self, ctx: Any, obs: Any) -> Plan:
                return Plan(goal="x")

            def _act(self, ctx: Any, plan: Plan) -> ActResult:
                return ActResult(ok=True)

            def _observe(self, ctx: Any, plan: Plan, result: ActResult) -> Observation:
                return Observation(summary="ok")

        record = AgentRecord(agent_id=_declared_id(TotallyUnexpectedName),
                             cls=TotallyUnexpectedName, module="tests.mystery")
        instance = record.instantiate(make_settings())
        assert instance.id is AgentId.A4_CONTRACT
        assert instance.step_budget == make_settings().agent_step_budget

        class SecondMystery(ReActAgent):
            id = "A6"  # declared as a bare string, not as the enum

            def _plan(self, ctx: Any, obs: Any) -> Plan:
                return Plan(goal="x")

            def _act(self, ctx: Any, plan: Plan) -> ActResult:
                return ActResult(ok=True)

            def _observe(self, ctx: Any, plan: Plan, result: ActResult) -> Observation:
                return Observation(summary="ok")

        assert _declared_id(SecondMystery) is AgentId.A6_AUDIT

    def test_declared_id_accepts_the_bare_string_form(self) -> None:
        from graph.registry import _declared_id

        class Bare:
            id = "A2"

        class Nonsense:
            id = "not-an-agent"

        assert _declared_id(Bare) is AgentId.A2_PRICING
        assert _declared_id(Nonsense) is None


# ================================================================ 11. fallback stubs
class TestFallbacks:
    def test_in_memory_board_is_append_only_with_monotonic_seq(self) -> None:
        board = gstate.InMemoryBoardLike()
        first = board.post("z", "k", AgentId.A1_DISCOVERY, {"a": 1})
        second = board.post("z", "k", AgentId.A2_PRICING, {"a": 2})
        assert second.seq == first.seq + 1
        assert [e.entry_id for e in board.history()] == [first.entry_id,
                                                         second.entry_id]
        assert board.read("z", kind="k") and board.zones() == ["z"]
        assert board.latest("z").entry_id == second.entry_id

    def test_null_tracer_marks_everything_degraded(self) -> None:
        tracer = gstate.NullTracer()
        tracer.configure("run_x")
        with tracer.agent(AgentId.A1_DISCOVERY, "x.plan"):
            pass
        tracer.event(TraceKind.HUMAN, "gate.send")
        assert tracer.finish()["degraded"] is True
        assert all(e.get("degraded") is True for e in tracer.events
                   if isinstance(e, dict) and "degraded" in e)

    def test_rules_decide_always_names_its_source(self) -> None:
        decide = gstate.rules_decide()
        decision = decide(DecisionRequest(request_id="dec_1", question="q",
                                          options=["yes", "no"]))
        assert decision.source is DecisionSource.RULES
        assert decision.degraded is True
        assert decision.choice in decision.probabilities
        assert abs(sum(decision.probabilities.values()) - 1.0) < 0.02

    def test_runtime_context_accepts_both_call_shapes(self) -> None:
        from core import ConfigError

        rt = make_runtime()
        assert gstate.runtime_context(rt) is rt

        class Wrapper:
            context = rt

        assert gstate.runtime_context(Wrapper()) is rt
        with pytest.raises(ConfigError):
            gstate.runtime_context(object())
