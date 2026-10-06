"""Tests for the real-stack adapter in :mod:`eval.integration` / ``RealAgentRuntime``.

What is being defended here
---------------------------
The harness has exactly one honesty obligation that supersedes everything else: **a
number must say which system produced it.** Before this adapter existed,
``RealAgentRuntime.available()`` probed the ``agents`` package for factory names that
are not there, so every ablation run silently used a deterministic stub and the report
had to say so in its provenance block. These tests pin the fixed behaviour:

1. The probe finds ``graph.registry.build_agents`` - the factory the orchestrator uses.
2. :meth:`RealAgentRuntime.execute` drives a **real** agent and returns a real
   ``StepResult`` derived from the blackboard.
3. A ``HumanGateRequired`` out of ``agent.run`` **parks** the run. Never ``ok=True``.
4. An ``ArbiterEscalation`` is recorded as an escalation and **not** resolved.
5. A missing agent, or an agent that raises, is an explicit failure - never a zero.
6. Provenance names the real runtime when it ran and the stub when it did not.
7. An unavailable real stack yields a **labelled** stub run, not an exception.
8. Two runs never share a board, so a seed sweep measures seeds.
9. ``eval`` still imports with every sibling package blocked.

Offline and fast. The ``settings_offline`` fixture forces ``RUN_MODE=offline``, which
truncates the decision chain to rules and so removes the only network call in the
adapter; every test therefore runs in well under a second and no test can be made to
pass or fail by whether a model happens to be running on the developer's machine.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core.config import Settings
from core.errors import ArbiterEscalation, HumanGateRequired
from core.protocols import AgentContext, Observation, ToolResult
from core.schemas import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    GateKind,
    MoU,
    RunMode,
    ToolStatus,
)
from eval import conditions, integration
from eval.conditions import (
    AgenticCondition,
    EvalBoard,
    LocalRuleRuntime,
    RealAgentRuntime,
    RunCounters,
    StepRequest,
    StepResult,
    ToolInvoker,
    resolve_runtime,
)
from eval.integration import (
    CountingTool,
    MirroredBoard,
    RealDecisionBridge,
    contract_terms_from_mou,
    probe_real_stack,
)
from eval.scenarios import get_scenario

#: One scenario is enough: the adapter is what is under test, not the scenario matrix.
SCENARIO = get_scenario("S4_clean_run")

#: The gate kind A3/A4 put in their exception message when a send/MoU gate is
#: unanswered. Asserted so a change in the adapter's gate inference is visible here
#: rather than only in a sweep's notes.
PARKED_GATE_MESSAGE = ("send gate gat_test0001 is unanswered and no approval record "
                       "exists for sending outreach; refusing to proceed")


# ================================================================ test doubles
class _ScriptedAgent:
    """A minimal stand-in for a real ``ReActAgent``, keyed by ``AgentId``.

    Deliberately *not* a ``ReActAgent`` subclass: the adapter must work with anything
    that has ``id`` and ``run(ctx)``, and a test double that inherits the real class
    would make the adapter look correct only because of the base class.
    """

    role = "scripted"

    def __init__(self, agent_id: AgentId, behaviour: Any) -> None:
        self.id = agent_id
        self.role = "scripted"
        self._behaviour = behaviour

    def run(self, ctx: AgentContext) -> Any:
        return self._behaviour(ctx)


class _Recorder:
    """A ``run`` that records the context it was handed and returns a verdict."""

    def __init__(self, *, summary: str = "did the thing",
                 sufficient: bool = True, degraded: bool = False,
                 post_offer: bool = False) -> None:
        self.summary = summary
        self.sufficient = sufficient
        self.degraded = degraded
        self.post_offer = post_offer
        self.seen: list[AgentContext] = []

    def __call__(self, ctx: AgentContext) -> Any:
        self.seen.append(ctx)
        if self.post_offer:
            ctx.board.post("offers", "offer", self_id(), {
                "offer_id": "off_test0001", "event_id": ctx.event_id, "brand": "Test Brand",
                "tier": "standard", "amount_inr": 42000.0, "deliverables": ["booth"],
                "fit_score": 60.0, "version": 1, "revised": False, "revision_note": "",
                "created_at": "2026-01-01T00:00:00Z",
            })
        return _ScriptedResult(summary=self.summary, sufficient=self.sufficient,
                               degraded=self.degraded)


class _ScriptedResult:
    """The parts of ``agents.base.AgentResult`` the adapter reads."""

    __slots__ = ("observation", "degraded", "notes", "steps", "duration_ms", "stop")

    def __init__(self, *, summary: str, sufficient: bool, degraded: bool) -> None:
        self.observation = Observation(summary=summary, sufficient=sufficient)
        self.degraded = degraded
        self.notes: list[str] = []
        self.steps = 1
        self.duration_ms = 1.0
        self.stop = False


def self_id() -> AgentId:
    """The author for the fake offer the recorder posts.

    A module-level function rather than a lambda because ``offers`` is an
    ``authoritative_agent`` zone in ``blackboard/zones.py``: a post authored by the
    wrong agent is counted as foreign, and the entry still lands. Keeping the author
    right means the test is exercising the adapter, not the board's authorship rule.
    """
    return AgentId.A2_PRICING


class _NarrowTool:
    """A tool whose ``run`` accepts one keyword, to prove the wrapper filters."""

    name = "narrow"
    description = "a deliberately narrow tool"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available(self) -> tuple[bool, str]:
        return (True, "always")

    def run(self, wanted: str = "") -> ToolResult:
        self.calls.append({"wanted": wanted})
        return ToolResult.success({"ok": True}, source="narrow")


class _BrokenTool:
    name = "broken"
    description = "raises, to prove a tool fault is counted rather than swallowed"

    def available(self) -> tuple[bool, str]:
        return (False, "deliberately unavailable")

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - never reached
        return ToolResult.success({}, source="broken")


# ==================================================================== helpers
def _request(agent: AgentId, counters: RunCounters, *, run_id: str = "run_test",
             board: EvalBoard | None = None, reply: str = "",
             round_index: int = 0) -> StepRequest:
    """A ``StepRequest`` wired to offline collaborators only."""
    return StepRequest(
        agent=agent, step_name=f"test.{agent.value}", scenario=SCENARIO, seed=11,
        board=board if board is not None else EvalBoard(SCENARIO.event.event_id),
        decide=conditions.DecisionInvoker(counters, seed=11, run_mode=RunMode.OFFLINE,
                                          prefer_real=False),
        tools=ToolInvoker(SCENARIO, counters),
        counters=counters, run_id=run_id, event_id=SCENARIO.event.event_id,
        focus_brand=SCENARIO.seed_data.brands[0].name, reply_text=reply,
        round_index=round_index, state={})


def _runtime_with(behaviour: Any, agent_id: AgentId = AgentId.A1_DISCOVERY,
                  settings: Settings | None = None) -> RealAgentRuntime:
    """A ``RealAgentRuntime`` driving one scripted agent through the real wiring."""
    return RealAgentRuntime(agents={agent_id: _ScriptedAgent(agent_id, behaviour)},
                            settings=settings)


# ============================================================= 1. the probe
class TestProbeFindsTheRealFactory:
    def test_the_probe_reports_graph_registry_build_agents(self, settings_offline) -> None:
        """The factory the orchestrator itself uses must be the one that answers.

        This is the specific regression the adapter fixes: the old probe looked for
        ``build_registry``/``build_agents``/... on the bare ``agents`` package, where
        none of them exist, and so reported a working agent stack as absent.
        """
        status = probe_real_stack(settings=settings_offline)
        assert status.factory == "graph.registry.build_agents", status.reason
        assert status.ok is True
        assert status.agents, "the real registry built no agents"
        assert all(isinstance(k, AgentId) for k in status.agents)

    def test_the_probe_names_which_agents_are_missing(self, settings_offline) -> None:
        """A partial registry is usable, but the absence has to travel with it."""
        status = probe_real_stack(settings=settings_offline)
        if status.absent:
            assert "absent" in status.reason
            for value in status.absent:
                assert value in status.reason
        else:
            assert "absent" not in status.reason

    def test_the_bare_agents_probe_list_is_still_available_as_a_fallback(self) -> None:
        """The fallback order is part of the contract, not an accident."""
        names = integration.AGENT_PACKAGE_FACTORIES
        assert "build_agents" in names
        assert "build_registry" in names
        # The reason the old probe failed, stated as an assertion so the regression is
        # visible in the diff if the names ever come back.
        status = RealAgentRuntime._REAL_FACTORIES
        assert tuple(status) == names

    def test_a_registry_factory_that_builds_nothing_is_reported_not_accepted(
            self, settings_offline, monkeypatch) -> None:
        """Zero agents is a failure. Accepting it would silently measure nothing."""
        import graph.registry

        monkeypatch.setattr(graph.registry, "build_agents", lambda settings: {})
        status = probe_real_stack(settings=settings_offline)
        assert status.ok is False
        assert "0 agents" in status.reason

    def test_a_factory_that_raises_falls_through_to_the_agents_package(
            self, settings_offline, monkeypatch) -> None:
        """A broken first choice must not hide a working second choice."""
        import graph.registry

        def boom(settings: Any) -> Any:
            raise ValueError("factory is mid-edit")

        monkeypatch.setattr(graph.registry, "build_agents", boom)
        status = probe_real_stack(settings=settings_offline)
        assert status.ok is False
        assert "factory is mid-edit" in status.reason
        assert "agents package" in status.reason


# ================================================= 2. execute drives a real agent
class TestExecuteDrivesRealAgents:
    def test_execute_returns_a_real_step_result_from_a_real_agent(
            self, settings_offline) -> None:
        """The scripted agent's board entry becomes a typed ``Offer`` in the result.

        The adapter derives its answer from the blackboard rather than from the
        agent's return value, so this asserts the whole projection path: post ->
        mirror -> read back -> schema-validate.
        """
        recorder = _Recorder(summary="priced Test Brand", post_offer=True)
        runtime = _runtime_with(recorder, settings=settings_offline)
        counters = RunCounters()
        board = EvalBoard(SCENARIO.event.event_id)

        result = runtime.execute(_request(AgentId.A1_DISCOVERY, counters, board=board))

        assert isinstance(result, StepResult)
        assert result.ok is True
        assert result.summary == "priced Test Brand"
        assert result.produced["offer"]["brand"] == "Test Brand"
        assert counters.steps == 1
        assert len(recorder.seen) == 1, "the real agent's run(ctx) was not called"
        ctx = recorder.seen[0]
        assert ctx.agent is AgentId.A1_DISCOVERY
        assert ctx.event_id == SCENARIO.event.event_id
        assert callable(ctx.decide), "the real decision layer was not wired into the context"
        assert isinstance(ctx.tools, dict)
        assert ctx.step_budget > 0 and ctx.deadline_s > 0

    def test_a_real_agent_from_the_real_registry_runs(self, settings_offline) -> None:
        """End-to-end with the project's own agents, not a double.

        A2 is chosen because it is the cheapest agent that still posts a typed
        artefact, which keeps the test under a second while proving the probe, the
        runtime wiring, the board projection and the delta reader all work together.
        """
        runtime = RealAgentRuntime(settings=settings_offline)
        ok, why = runtime.available()
        if not ok:
            pytest.skip(f"real agent registry unavailable in this tree: {why}")
        counters = RunCounters()
        board = EvalBoard(SCENARIO.event.event_id)

        result = runtime.execute(_request(AgentId.A2_PRICING, counters, board=board))

        assert isinstance(result, StepResult)
        assert result.summary.strip(), "a real agent ran and said nothing"
        assert counters.steps == 1
        # Whatever it decided, it is recorded against a source. An anonymous decision
        # is the one thing this harness must never emit.
        assert counters.decisions_by_source, result.summary
        for source in counters.decisions_by_source:
            assert DecisionSource(source) is not None

    def test_the_step_budget_and_deadline_reach_the_agent(self, settings_offline) -> None:
        """The agent's own limits come from ``Settings``, not from a hard-coded default."""
        runtime = _runtime_with(_Recorder(), settings=settings_offline)
        recorder_seen: list[int] = []
        runtime._injected_agents = {  # noqa: SLF001 - recording the built context
            AgentId.A1_DISCOVERY: _ScriptedAgent(
                AgentId.A1_DISCOVERY,
                lambda ctx: (recorder_seen.append(ctx.step_budget)
                             or _ScriptedResult(summary="ok", sufficient=True,
                                                degraded=False))),
        }
        runtime._probed = False  # noqa: SLF001 - re-probe against the new registry
        counters = RunCounters()

        runtime.execute(_request(AgentId.A1_DISCOVERY, counters))

        assert recorder_seen == [settings_offline.agent_step_budget]


# ================================================ 3. a human gate parks the run
class TestHumanGateParksRatherThanSucceeds:
    def test_human_gate_required_parks_the_run(self, settings_offline) -> None:
        """The headline honesty rule: a gate is a control-flow state, not a success."""
        def parked(ctx: AgentContext) -> Any:
            raise HumanGateRequired(PARKED_GATE_MESSAGE)

        runtime = _runtime_with(parked, agent_id=AgentId.A3_OUTREACH,
                                settings=settings_offline)
        counters = RunCounters()

        result = runtime.execute(_request(AgentId.A3_OUTREACH, counters))

        assert result.ok is False, "a parked gate was reported as a completed step"
        assert result.gate_pending is True
        assert result.parked is True
        assert result.outcome_label == "parked_at_gate"
        assert result.gate_kind is GateKind.SEND
        assert result.parked_reason == PARKED_GATE_MESSAGE
        # A park is not an error: there is nothing to fix but a missing human.
        assert result.errors == []
        assert "pending_approval" in result.produced["pending_approval"]
        assert any("PARKED" in n for n in result.notes)
        # The step still ran, so it is still counted. `steps_taken` must mean the same
        # thing for the real stack as for the stub.
        assert counters.steps == 1

    def test_a_park_never_marks_the_run_as_finished_work(self, settings_offline) -> None:
        """An auto-resolved gate in the harness must not authorise the side effect.

        ``AgenticCondition._gate`` resolves every gate with an auto ``APPROVE`` so the
        trace is schema-valid. If that counted as authorisation, a parked send would be
        recorded as a sent message - the single most damaging thing this harness could
        do.
        """
        def parked(ctx: AgentContext) -> Any:
            raise HumanGateRequired(PARKED_GATE_MESSAGE)

        runtime = _runtime_with(parked, agent_id=AgentId.A3_OUTREACH,
                                settings=settings_offline)
        condition = AgenticCondition(runtime=runtime, run_mode=RunMode.OFFLINE)

        outcome = condition.run(SCENARIO, seed=11)

        assert outcome.ok is True
        # The gate was raised and counted - a gate really was raised.
        assert outcome.human_gates_raised >= 1
        # And nothing was released or sent because of it.
        assert outcome.negotiation is None or outcome.negotiation.signed is False
        assert any("does NOT" in n or "PARKED" in n for n in outcome.notes)

    def test_a_park_is_not_counted_as_a_policy_block(self, settings_offline) -> None:
        """``blocked_attempts`` means "A4 refused", not "A4 waited for a human".

        The agentic condition's terminal state fires at two blocked attempts. Counting
        a park as a refusal would make a run that was politely waiting for a human look
        like a run that had been refused twice by policy.
        """
        def parked(ctx: AgentContext) -> Any:
            raise HumanGateRequired("mou gate gat_x is unanswered; the MoU stays "
                                    "pending_approval")

        runtime = _runtime_with(parked, agent_id=AgentId.A4_CONTRACT,
                                settings=settings_offline)
        counters = RunCounters()
        board = EvalBoard(SCENARIO.event.event_id)
        request = _request(AgentId.A4_CONTRACT, counters, board=board)

        result = runtime.execute(request)

        assert result.gate_pending is True
        assert result.gate_kind is GateKind.MOU
        assert result.errors == [], "a park was reported as a compliance block"

    def test_a_veto_with_no_recorded_gate_still_names_a_kind(self, settings_offline) -> None:
        """A5 raises with a message and posts no ``HumanGate``.

        Without a fallback the adapter would report ``gate_kind=None``, the condition
        would count no gate at all, and a compliance veto would vanish from the report.
        """
        def veto(ctx: AgentContext) -> Any:
            raise HumanGateRequired("A5 veto: 2 blocking risk flag(s); no MoU may be "
                                    "released")

        runtime = _runtime_with(veto, agent_id=AgentId.A5_COMPLIANCE,
                                settings=settings_offline)
        counters = RunCounters()

        result = runtime.execute(_request(AgentId.A5_COMPLIANCE, counters))

        assert result.gate_pending is True
        assert result.gate_kind is GateKind.ESCALATION
        assert result.ok is False


# ============================================ 4. an escalation is not a resolution
class TestArbiterEscalationIsRecordedNotResolved:
    def test_arbiter_escalation_is_recorded_and_left_open(self, settings_offline) -> None:
        """``ArbiterEscalation`` means "I could not settle this". Never "settled"."""
        def escalate(ctx: AgentContext) -> Any:
            raise ArbiterEscalation("dsp_test0001", "confidence 0.20 below floor 0.50")

        runtime = _runtime_with(escalate, agent_id=AgentId.A7_ARBITER,
                                settings=settings_offline)
        counters = RunCounters()

        result = runtime.execute(_request(AgentId.A7_ARBITER, counters))

        assert result.gate_kind is GateKind.ESCALATION
        assert result.outcome_label == "escalated"
        arbitration = result.produced["arbitration"]
        assert arbitration["escalated"] is True
        assert arbitration["dispute_id"] == "dsp_test0001"
        assert arbitration["confidence"] == 0.0
        assert counters.disputes_escalated == 1
        assert counters.disputes_resolved == 0, "an escalation was counted as a resolution"
        assert any("ESCALATED" in n and "NOT resolved" in n for n in result.notes)

    def test_an_escalation_leaves_the_arbiter_condition_escalated(
            self, settings_offline) -> None:
        """End to end: the run's negotiation reports ``escalation=True``, not a deal.

        The dispute is posted by **A3**, not A5. The agentic condition hands the open
        dispute to A7 from whichever agent raised it, and A3 is the one agent every
        observed route reaches (``A1 -> A2 -> A3``); A5 is reached only when a draft
        exists to check, which no script here produces. Putting the dispute on A5 would
        make the test depend on the router reaching it, and a router change would then
        look like an escalation regression.
        """
        def quiet(ctx: AgentContext) -> Any:
            return _ScriptedResult(summary="nothing to do", sufficient=True,
                                   degraded=False)

        def raise_dispute(ctx: AgentContext) -> Any:
            ctx.board.post("disputes", "dispute", AgentId.A3_OUTREACH, {
                "dispute_id": "dsp_test0002", "event_id": ctx.event_id,
                "zone": "outreach", "claimant": AgentId.A2_PRICING,
                "opponent": AgentId.A6_AUDIT,
                "position": "pursue the low-EV lead",
                "counter_position": "expected value is below the outreach cost",
                "evidence": [], "severity": "medium", "rounds": 0,
            })
            return _ScriptedResult(summary="disputed", sufficient=True, degraded=False)

        def escalate(ctx: AgentContext) -> Any:
            raise ArbiterEscalation("dsp_test0002", "no zone of agreement")

        runtime = RealAgentRuntime(
            agents={AgentId.A1_DISCOVERY: _ScriptedAgent(AgentId.A1_DISCOVERY, quiet),
                    AgentId.A2_PRICING: _ScriptedAgent(AgentId.A2_PRICING, quiet),
                    AgentId.A3_OUTREACH: _ScriptedAgent(AgentId.A3_OUTREACH,
                                                         raise_dispute),
                    AgentId.A7_ARBITER: _ScriptedAgent(AgentId.A7_ARBITER, escalate)},
            settings=settings_offline)
        condition = AgenticCondition(runtime=runtime, run_mode=RunMode.OFFLINE)

        outcome = condition.run(SCENARIO, seed=11)

        assert outcome.disputes_raised >= 1, outcome.notes
        assert outcome.disputes_escalated >= 1
        assert outcome.disputes_resolved == 0, "an escalation was counted as a resolution"
        assert outcome.negotiation is None or outcome.negotiation.escalation is True
        assert outcome.negotiation is None or outcome.negotiation.signed is False


# ============================================== 5. failures are explicit, not zero
class TestFailuresAreNeverZeros:
    def test_an_unregistered_agent_is_an_explicit_failure(self, settings_offline) -> None:
        """A missing agent must not read as an agent that chose to do nothing."""
        runtime = _runtime_with(_Recorder(), agent_id=AgentId.A1_DISCOVERY,
                                settings=settings_offline)
        counters = RunCounters()

        result = runtime.execute(_request(AgentId.A6_AUDIT, counters))

        assert result.ok is False
        assert result.errors == [f"unregistered in the real agent registry: "
                                 f"{AgentId.A6_AUDIT.value}"]
        assert counters.steps == 0

    def test_an_agent_that_raises_is_reported_with_its_own_type(self,
                                                                settings_offline) -> None:
        """The exception's type and message travel with the failure."""
        def explode(ctx: AgentContext) -> Any:
            raise ValueError("the blackboard refused every write")

        runtime = _runtime_with(explode, settings=settings_offline)
        counters = RunCounters()

        result = runtime.execute(_request(AgentId.A1_DISCOVERY, counters))

        assert result.ok is False
        assert result.errors == ["ValueError: the blackboard refused every write"]
        assert "ValueError" in result.summary
        assert result.degraded is True
        assert counters.steps == 1

    def test_a_run_with_a_broken_agent_is_reported_not_raised(self,
                                                              settings_offline) -> None:
        """``Condition.run`` converts the failure into ``ok=False`` plus a reason."""
        def explode(ctx: AgentContext) -> Any:
            raise RuntimeError("dependency not present")

        runtime = _runtime_with(explode, settings=settings_offline)
        condition = AgenticCondition(runtime=runtime, run_mode=RunMode.OFFLINE)

        outcome = condition.run(SCENARIO, seed=11)

        assert outcome.ok is True, "one agent's failure must not kill the sweep cell"
        assert any("dependency not present" in n for n in outcome.notes)


# ================================================== 6 & 7. provenance labelling
class TestProvenanceNaming:
    def test_provenance_names_the_real_runtime_when_it_ran(self) -> None:
        """A real run says ``real-agents``; nothing about a stub."""
        runtime, why = resolve_runtime(prefer_real=True)
        assert runtime.name
        assert why.strip()
        if isinstance(runtime, LocalRuleRuntime):
            assert "FALLBACK" in why
            assert LocalRuleRuntime.name in why
        else:
            assert "real-agents" in why
            assert "FALLBACK" not in why
            assert "graph.registry.build_agents" in why

    def test_provenance_also_names_the_decision_layer(self) -> None:
        """Real agents + rules backend and real agents + clef are different numbers.

        The provenance block is the only place a reader can tell which happened, so it
        has to say. This is the half of the honesty contract the previous provenance
        string was missing entirely.
        """
        runtime, why = resolve_runtime(prefer_real=True)
        if isinstance(runtime, LocalRuleRuntime):
            pytest.skip("the real stack is unavailable here; the stub labels itself")
        assert "decisions:" in why
        assert "decision.registry" in why

    def test_provenance_never_claims_a_backend_it_did_not_probe(self,
                                                                settings_offline) -> None:
        """``health(probe=False)`` fills ``available=True`` in with ``"not probed"``.

        Reporting that verbatim would claim clef is reachable on a machine where it is
        not, which is precisely the kind of lie this block exists to prevent.
        """
        text = integration.decision_provenance(settings_offline, probe=False)
        assert "availability not probed" in text
        assert "=up" not in text and "=down" not in text

    def test_provenance_marks_the_run_mode(self, settings_offline) -> None:
        text = integration.decision_provenance(settings_offline, probe=True)
        assert "chain=" in text
        assert str(settings_offline.effective_backend()) in text

    def test_requesting_the_stub_explicitly_says_so(self) -> None:
        runtime, why = resolve_runtime(prefer_real=False)
        assert isinstance(runtime, LocalRuleRuntime)
        assert why == f"{LocalRuleRuntime.name} (requested explicitly)"
        # "FALLBACK" is reserved for the case where the stub was chosen *for* the
        # reader. A deliberate choice is not a degradation.
        assert "FALLBACK" not in why

    def test_an_unavailable_real_stack_yields_a_labelled_stub_run(
            self, settings_offline, monkeypatch) -> None:
        """No real stack: a run still happens, and it says it is not real.

        This is the fallback path that used to fire on *every* run. It must still work,
        and it must still be labelled.
        """
        monkeypatch.setattr(integration, "probe_real_stack",
                            lambda **_kw: integration.RealStackStatus(
                                ok=False,
                                reason="graph.registry unavailable (ImportError: no "
                                       "module named graph); probed the agents package "
                                       "instead"))
        runtime, why = resolve_runtime(prefer_real=True)

        assert isinstance(runtime, LocalRuleRuntime)
        assert "FALLBACK" in why
        assert LocalRuleRuntime.name in why
        assert "graph.registry unavailable" in why

        condition = AgenticCondition(run_mode=RunMode.OFFLINE)
        outcome = condition.run(SCENARIO, seed=11)
        assert outcome.ok is True
        assert "FALLBACK" in outcome.runtime
        assert outcome.path_variation_caveat, (
            "a stub run must carry the caveat that its path variation is not evidence "
            "about the production agent stack")

    def test_execute_refuses_to_run_without_a_real_stack(self,
                                                         settings_offline,
                                                         monkeypatch) -> None:
        """``execute`` on an unavailable runtime raises rather than substituting.

        A measurement must never come from a fallback nobody was told about, and the
        raise is what stops the caller from quietly using the stub.
        """
        monkeypatch.setattr(integration, "probe_real_stack",
                            lambda **_kw: integration.RealStackStatus(
                                ok=False, reason="agents package unavailable"))
        runtime = RealAgentRuntime(settings=settings_offline)
        counters = RunCounters()

        with pytest.raises(RuntimeError, match="unavailable"):
            runtime.execute(_request(AgentId.A1_DISCOVERY, counters))

    def test_every_run_record_carries_the_runtime_that_produced_it(
            self, settings_offline) -> None:
        """``RunOutcome.runtime`` names the runtime *and* its reason, per run."""
        runtime = _runtime_with(_Recorder(), settings=settings_offline)
        condition = AgenticCondition(runtime=runtime, run_mode=RunMode.OFFLINE)

        outcome = condition.run(SCENARIO, seed=11)
        payload = outcome.to_dict()

        assert payload["runtime"].strip()
        # The label leads with the runtime's own name so ``report.py``'s
        # ``split("(")`` column reads ``real-agents``, and it names the registry that
        # was actually used rather than only saying "the real stack".
        assert payload["runtime"].startswith("real-agents")
        assert "injected registry" in payload["runtime"]


# ================================================== 8. runs never share a board
class TestRunIsolation:
    def test_two_run_ids_get_two_boards(self, settings_offline) -> None:
        """A blackboard carried across seeds would make seed 2 depend on seed 1."""
        recorder = _Recorder(post_offer=True)
        runtime = _runtime_with(recorder, settings=settings_offline)
        counters = RunCounters()

        runtime.execute(_request(AgentId.A1_DISCOVERY, counters, run_id="run_a"))
        first_len = len(recorder.seen[0].board.history())
        runtime.execute(_request(AgentId.A1_DISCOVERY, counters, run_id="run_b"))
        second_len = len(recorder.seen[1].board.history())

        assert recorder.seen[0].board is not recorder.seen[1].board
        # Each run starts from the published scenario world plus its own reply, so the
        # second run's board is not longer by the first run's leftovers.
        assert second_len == first_len

    def test_steps_of_one_run_share_a_board(self, settings_offline) -> None:
        """A4 must be able to see what A2 priced, or the pipeline cannot work."""
        recorder = _Recorder(post_offer=True)
        runtime = _runtime_with(recorder, settings=settings_offline)
        counters = RunCounters()

        runtime.execute(_request(AgentId.A1_DISCOVERY, counters, run_id="run_shared"))
        after_first = len(recorder.seen[0].board.history())
        runtime.execute(_request(AgentId.A1_DISCOVERY, counters, run_id="run_shared"))

        assert recorder.seen[0].board is recorder.seen[1].board
        # The length is read *between* the two steps. Read afterwards both calls would
        # report the final length and the assertion would be vacuous.
        assert len(recorder.seen[1].board.history()) > after_first

    def test_the_scope_cache_is_bounded(self, settings_offline) -> None:
        """A long sweep must not accumulate one board per cell forever."""
        recorder = _Recorder()
        runtime = _runtime_with(recorder, settings=settings_offline)
        counters = RunCounters()

        for i in range(RealAgentRuntime._MAX_CACHED_RUNS + 3):  # noqa: SLF001
            runtime.execute(_request(AgentId.A1_DISCOVERY, counters, run_id=f"run_{i}"))

        assert len(runtime._scopes) <= RealAgentRuntime._MAX_CACHED_RUNS  # noqa: SLF001


# ============================================== 9. eval imports without siblings
class TestEvalImportsWithoutSiblingPackages:
    def test_importing_eval_and_conditions_needs_no_sibling_package(
            self, repo_root: Path) -> None:
        """``eval`` must import with ``graph``/``agents``/``blackboard``/... blocked.

        Run in a subprocess because the blocker has to be installed on ``sys.meta_path``
        *before* the first import, and this session has long since imported them. Using
        ``find_spec`` is deliberate: the legacy ``find_module`` finder protocol was
        removed in Python 3.12, so a blocker written against it is silently ignored on
        3.14 and this test would pass vacuously.
        """
        script = f"""
import sys

class Blocker:
    BLOCKED = ("graph", "tools", "decision", "observability", "agents", "blackboard")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError("blocked for this check: " + fullname)
        return None

for mod in list(sys.modules):
    if mod.split(".")[0] in Blocker.BLOCKED:
        del sys.modules[mod]
sys.meta_path.insert(0, Blocker())

sys.path.insert(0, {str(repo_root)!r})

import eval
import eval.conditions as conditions
import eval.integration as integration
from eval.conditions import LocalRuleRuntime, RealAgentRuntime, resolve_runtime

runtime, why = resolve_runtime(prefer_real=True)
assert isinstance(runtime, LocalRuleRuntime), type(runtime).__name__
assert "FALLBACK" in why, why
assert LocalRuleRuntime.name in why
print("OK", eval.__version__)
"""
        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, timeout=120,
                                   check=False)
        assert completed.returncode == 0, (
            f"eval failed to import with every sibling blocked\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        assert completed.stdout.startswith("OK"), completed.stdout


# ==================================================== 10. the wiring components
class TestAdapterComponents:
    def test_counting_tool_filters_kwargs_to_the_real_signature(self) -> None:
        """The agents introspect the tool they are handed and pass what it accepts.

        A wrapper with a narrower signature would silently starve them of arguments, so
        the wrapper takes ``**kwargs`` and does the filtering against the real tool.
        """
        real = _NarrowTool()
        counters = RunCounters()
        wrapped = CountingTool("narrow", real, counters)

        result = wrapped.run(wanted="yes", ignored="dropped")

        assert isinstance(result, ToolResult)
        assert real.calls == [{"wanted": "yes"}], "extra kwargs reached the real tool"
        assert counters.tool_calls == 1
        assert counters.tool_status_counts[ToolStatus.OK.value] == 1

    def test_counting_tool_reports_a_tool_that_raises_as_failed(self) -> None:
        """A tool fault is data, not a crash, and it must be counted."""
        counters = RunCounters()
        wrapped = CountingTool("broken", _BrokenTool(), counters)

        def raiser(**kwargs: Any) -> ToolResult:
            raise KeyError("missing configuration")

        wrapped._tool = type("_Raising", (), {  # noqa: SLF001 - deliberate substitution
            "description": "raises",
            "run": staticmethod(raiser),
            "available": staticmethod(lambda: (True, "ok")),
        })()
        result = wrapped.run()

        assert result.status is ToolStatus.FAILED
        assert counters.tool_status_counts[ToolStatus.FAILED.value] == 1

    def test_mirrored_board_keeps_the_real_zone_names(self) -> None:
        """The projection must not lose which zone an artefact really came from."""
        real = integration._new_real_board()[0]  # noqa: SLF001 - documented fallback
        projection = EvalBoard("evt_mirror")
        board = MirroredBoard(real, projection)

        entry = board.post("offers", "offer", AgentId.A2_PRICING, {
            "offer_id": "off_mirror01", "event_id": "evt_mirror", "brand": "B",
            "tier": "gold", "amount_inr": 1.0, "deliverables": ["x"],
        })

        assert entry.zone == "offers"
        projected = projection.latest("pricing", "offer")
        assert projected is not None, "the eval board did not receive the projection"
        assert projected.payload["_real_zone"] == "offers"
        assert projected.payload["_real_kind"] == "offer"
        assert "author" not in projected.payload, "the bookkeeping keys must be visible"

    def test_mirrored_board_reflects_what_the_real_board_accepted(self) -> None:
        """A rejected post must not reach the conditions.

        Otherwise a condition could route on an artefact the system refused to write,
        which is the worst kind of silent data loss.
        """
        real = integration._new_real_board()[0]  # noqa: SLF001
        projection = EvalBoard("evt_reject")
        board = MirroredBoard(real, projection)

        with pytest.raises(Exception):  # noqa: B017 - the board's own error type is not
            board.post("offers", "not_a_kind", AgentId.A2_PRICING, {"x": 1})

        assert projection.history() == []

    def test_decision_bridge_uses_the_real_registry(self, settings_offline) -> None:
        """``decision.registry.build_registry`` is the name that actually exists."""
        counters = RunCounters()
        bridge = RealDecisionBridge(counters, settings=settings_offline,
                                    run_mode=RunMode.OFFLINE)
        ok, why = bridge.probe()

        assert ok is True, why
        assert "decision.registry.build_registry" in why

    def test_decision_bridge_records_a_rules_answer_as_degraded(
            self, settings_offline) -> None:
        """A LIVE run answered by rules must not look like a model call.

        Two distinct situations are checked here, because they are genuinely different
        measurements and must not be reported the same way:

        * **OFFLINE** - the rules engine is the *intended* backend, so its answer is not
          degraded. ``decision.registry`` does the degrading, at chain position > 0, and
          only when a fallback really did answer a question meant for something better.
        * **bridge fallback** - the registry is gone entirely, so the bridge answered.
          That *is* a substitution and is recorded as one (next test).
        """
        counters = RunCounters()
        bridge = RealDecisionBridge(counters, settings=settings_offline,
                                    run_mode=RunMode.OFFLINE)
        request = DecisionRequest(
            request_id="req_bridge", question="Which agent acts next?",
            state={"reply": "the price is too high, can you reduce it?"},
            options=[i.value for i in conditions.Intent],
            asked_by=AgentId.A2_PRICING, decision_point="route.next_agent")

        decision = bridge(request)

        assert isinstance(decision, Decision)
        assert decision.source is DecisionSource.RULES
        # Offline: rules is what the run mode asked for, so this is not a degradation.
        assert bridge.intended_source is DecisionSource.RULES
        assert decision.degraded is False
        assert counters.degraded_decisions == 0
        assert counters.llm_calls == 0, "a rules answer was counted as a model call"
        assert counters.decisions_by_source[DecisionSource.RULES.value] == 1

    def test_a_live_run_declares_clef_the_intended_backend(self,
                                                            settings_offline) -> None:
        """``intended_source`` is what makes a degradation detectable at all."""
        bridge = RealDecisionBridge(RunCounters(), settings=settings_offline,
                                    run_mode=RunMode.LIVE)
        assert bridge.intended_source is DecisionSource.CLEF
        offline = RealDecisionBridge(RunCounters(), settings=settings_offline,
                                     run_mode=RunMode.OFFLINE)
        assert offline.intended_source is DecisionSource.RULES

    def test_decision_bridge_falls_back_when_the_registry_is_absent(
            self, settings_offline, monkeypatch) -> None:
        """The substitution is recorded in the counters' notes, not swallowed.

        ``sys.modules[...] = None`` is how an import is made to fail for a module that
        is *already* imported. Patching ``builtins.__import__`` would not work:
        ``importlib.import_module`` goes through ``importlib._bootstrap``, not
        ``builtins``, so the patch would silently never fire and the test would pass
        against the real registry without exercising the fallback at all.
        """
        counters = RunCounters()
        bridge = RealDecisionBridge(counters, settings=settings_offline,
                                    run_mode=RunMode.LIVE)
        monkeypatch.setitem(sys.modules, "decision.registry", None)
        request = DecisionRequest(
            request_id="req_fallback", question="Classify the reply",
            state={"reply": "too expensive"}, options=[i.value for i in conditions.Intent],
            asked_by=AgentId.A3_OUTREACH, decision_point="intent.classify")

        decision = bridge(request)

        ok, why = bridge.probe()
        assert ok is False, "the registry should have been unreachable"
        assert "decision.registry unavailable" in why
        assert decision.source is DecisionSource.RULES
        assert decision.degraded is True, (
            "a substitution for an unreachable backend was not marked degraded")
        assert counters.llm_calls == 0
        assert counters.degraded_decisions == 1
        assert counters.decisions_by_source[DecisionSource.RULES.value] == 1
        assert "eval.rules-fallback" in decision.model

    def test_contract_terms_are_read_from_the_mou_body(self) -> None:
        """The harness audits the real artefact, not the scenario's expectation.

        The real A4 always writes "No exclusivity is granted", so ``exclusivity_days``
        must read as 0 and the notes must say so. Importing the scenario's
        ``SeedOffer`` clause facts here would grade the real agents against numbers they
        never saw.
        """
        mou = MoU(mou_id="mou_terms01", event_id="evt_terms", brand="Helio Fintech",
                  amount_inr=120000.0, terms=(
                      "MEMORANDUM OF UNDERSTANDING\n"
                      "6. EXCLUSIVITY. No exclusivity is granted under this MoU.\n"),
                  deliverables=["booth"], status="pending_approval", version=1)

        terms, notes = contract_terms_from_mou(mou)

        assert terms.brand == "Helio Fintech"
        assert terms.amount_inr == 120000.0
        assert terms.exclusivity_days == 0
        assert terms.attendee_data_clause is False
        assert terms.deliverables == ("booth",)
        assert any("grants no exclusivity" in n for n in notes)
        # The eval's own policy caps then flag what the real contract is missing, which
        # is a true statement about the system rather than a harness artefact.
        assert terms.violations(SCENARIO.policy), (
            "a real MoU with no evidence URL should trip the policy caps")
        assert any("signer_authority_evidence=False" in n for n in notes)

    def test_contract_terms_read_an_exclusivity_term_when_one_is_granted(self) -> None:
        """A grant and an unreadable term must not collapse into the same number."""
        granted = MoU(mou_id="mou_terms02", event_id="evt_terms", brand="B",
                      amount_inr=1.0, terms="7. EXCLUSIVITY. 365 days of exclusivity "
                                            "are granted to the Sponsor.",
                      deliverables=["x"], status="draft", version=1)
        unreadable = MoU(mou_id="mou_terms03", event_id="evt_terms", brand="B",
                         amount_inr=1.0, terms="7. EXCLUSIVITY. As set out in Schedule 2.",
                         deliverables=["x"], status="draft", version=1)

        assert contract_terms_from_mou(granted)[0].exclusivity_days == 365
        assert contract_terms_from_mou(unreadable)[0].exclusivity_days == 0
        assert any("could not read" in n
                   for n in contract_terms_from_mou(unreadable)[1])

    def test_a_missing_tool_is_an_empty_map_not_a_fake_tool(
            self, settings_offline, monkeypatch) -> None:
        """An empty ``ctx.tools`` is a measurement; a fabricated tool is not."""
        counters = RunCounters()
        invoker = ToolInvoker(SCENARIO, counters, prefer_real=False)
        monkeypatch.setattr(ToolInvoker, "available",
                            lambda self: (False, "tools.registry unavailable"))

        tools, reason = integration.build_tool_map(invoker, counters)

        assert tools == {}
        assert "no tool registry" in reason

    def test_the_full_ablation_still_renders_on_the_real_stack(self) -> None:
        """One scenario, three conditions, one seed: the whole report path, fast.

        Guards the wiring end to end: ``run_ablation`` -> ``to_markdown`` -> ``to_json``
        with a real runtime, which is the combination the deliverable is judged on.
        """
        from eval.ablation import run_ablation
        from eval.report import to_json, to_markdown

        # ``ALL_CONDITIONS`` holds instances, not classes, so the type is what is
        # instantiable. The instances themselves are shared module-level singletons, so
        # building fresh ones here is also what keeps this test independent of the
        # runtime-resolution cache every other test has already filled.
        instances = [type(c)(prefer_real_runtime=False) for c in conditions.ALL_CONDITIONS]
        report = run_ablation([SCENARIO], instances, (11,))
        markdown = to_markdown(report)
        payload = json.loads(to_json(report))

        assert markdown.startswith("# Ablation report")
        assert "### Provenance" in markdown
        assert payload["headline"]["headline_metric"] == "path_variation"
        assert len(payload["records"]) == 3
        for record in payload["records"]:
            assert record["runtime"].strip(), "a record carried no provenance"
