"""Unit tests for the ``eval`` package.

Fully offline and dependency-free. No network, no model, no tool registry, no
``agents``/``graph``/``decision`` package required. Where a test needs an agent
runtime it injects :class:`~eval.conditions.StubRuntime` or the offline
:class:`~eval.conditions.LocalRuleRuntime`, both of which live in ``eval`` itself,
so a change in another package cannot make these tests flake.

The five load-bearing assertions the project depends on:

1. The fixed pipeline's ``path_variation`` is exactly 1 on every scenario and every
   seed - the baseline is genuinely non-agentic.
2. ``AblationReport`` keeps its raw per-run records, so every aggregate is auditable.
3. ``to_markdown`` renders the provenance block.
4. A missing dependency produces an explicit ``unavailable`` / ``ok=False`` failure,
   never a fabricated number.
5. Every seeded scenario parses, validates, and says what it claims to say.
"""
from __future__ import annotations

import json
import math

import pytest

from core.protocols import classify_intent_fallback
from core.schemas import AgentId, DecisionSource, Intent, ToolStatus
from eval import ablation, benchmark, conditions, report, scenarios
from eval.ablation import (
    UNAVAILABLE,
    MetricStats,
    Provenance,
    RunRecord,
    run_ablation,
)
from eval.benchmark import run_benchmark, surplus_split
from eval.conditions import (
    AgenticCondition,
    Condition,
    ContractTerms,
    FixedPipelineCondition,
    KeywordRouterCondition,
    LocalRuleRuntime,
    RunCounters,
    StubRuntime,
    naive_keyword_intent,
    route_probe,
)
from eval.report import to_json, to_markdown
from eval.scenarios import (
    ALL_SCENARIOS,
    DEFAULT_SEED,
    FIXTURE_SOURCE,
    SIX_WEEKS_DAYS,
    NegotiationBounds,
    PolicyCaps,
    get_scenario,
    scenario_rng,
)

#: Small fixed seed set. Chosen once and frozen: every aggregate asserted below is
#: reproducible from this list, and the report prints it.
SEEDS = (11, 22, 33)

#: A three-scenario subset that keeps the suite fast while still covering the
#: policy conflict, the routing test, and the evidence gap.
FAST_SCENARIOS = (
    get_scenario("S1_exclusivity_conflict"),
    get_scenario("S3_ambiguous_reply"),
    get_scenario("S5_vision_gap"),
)


def _offline(*cond_classes, **kwargs):
    """Conditions pinned to the local runtime, with no probing of other packages."""
    return [cls(prefer_real_runtime=False, **kwargs) for cls in cond_classes]


# ============================================================ 5. fixtures parse
class TestScenariosParseAndValidate:
    def test_self_check_holds(self) -> None:
        """The provable claims about S1 and S2 are actually true of the fixtures."""
        scenarios._self_check()

    def test_all_five_scenarios_exist_with_the_spec_names(self) -> None:
        assert [s.name for s in ALL_SCENARIOS] == [
            "S1_exclusivity_conflict",
            "S2_low_ev_lead",
            "S3_ambiguous_reply",
            "S4_clean_run",
            "S5_vision_gap",
        ]

    def test_every_scenario_round_trips_through_to_dict(self) -> None:
        for s in ALL_SCENARIOS:
            payload = s.to_dict()
            # JSON round-trip proves nothing is a non-serialisable surprise.
            json.dumps(payload)
            assert payload["name"] == s.name
            assert payload["event"]["event_id"] == s.event.event_id

    def test_every_seeded_brand_declares_fixture_source(self) -> None:
        brands = [b for s in ALL_SCENARIOS for b in s.seed_data.brands]
        assert brands, "the fixtures must contain at least one brand"
        assert all(b.source == FIXTURE_SOURCE for b in brands)

    def test_every_seeded_offer_declares_fixture_source(self) -> None:
        offers = [o for s in ALL_SCENARIOS for o in s.seed_data.offers]
        assert offers, "the fixtures must contain at least one offer"
        assert all(o.source == FIXTURE_SOURCE for o in offers)

    def test_seed_offer_rejects_a_non_fixture_source(self) -> None:
        seeded = get_scenario("S1_exclusivity_conflict").offer_for("Helio Fintech")
        with pytest.raises(ValueError, match="fixture"):
            type(seeded)(offer=seeded.offer, source="google-maps")

    def test_seed_data_rejects_a_brand_without_fixture_source(self) -> None:
        s1 = get_scenario("S1_exclusivity_conflict")
        bad = s1.seed_data.brands[0].model_copy(update={"source": "live-api"})
        with pytest.raises(ValueError, match="fixture"):
            type(s1.seed_data)(brands=(bad,))

    def test_s1_exclusivity_really_breaches_the_cap(self) -> None:
        s1 = get_scenario("S1_exclusivity_conflict")
        offer = s1.offer_for("Helio Fintech")
        assert offer.exclusivity_days > s1.policy.max_exclusivity_days
        assert s1.policy.max_exclusivity_days == SIX_WEEKS_DAYS
        assert not s1.policy.allow_attendee_data_transfer
        breaches = offer.violations(s1.policy)
        assert any("exclusivity" in b for b in breaches)
        assert any("attendee-data" in b for b in breaches)

    def test_s4_offer_sits_inside_the_cap(self) -> None:
        s4 = get_scenario("S4_clean_run")
        offer = s4.offer_for("TerraWatt Grid")
        assert offer.exclusivity_days <= s4.policy.max_exclusivity_days
        assert offer.violations(s4.policy) == []

    def test_s2_lead_is_actually_value_destroying(self) -> None:
        s2 = get_scenario("S2_low_ev_lead")
        lead = s2.brand(s2.seed_data.cold_lead_name or "")
        ev = s2.seed_data.lead_ev(lead)
        assert ev < s2.seed_data.outreach_cost_inr
        assert s2.seed_data.is_bad_bet(lead)

    def test_s2_declares_the_expected_a3_vs_a6_dispute(self) -> None:
        s2 = get_scenario("S2_low_ev_lead")
        assert any(c.parties == (AgentId.A3_OUTREACH, AgentId.A6_AUDIT)
                   for c in s2.expected_conflicts)

    def test_s1_declares_the_expected_a2_vs_a5_dispute(self) -> None:
        s1 = get_scenario("S1_exclusivity_conflict")
        assert any(c.parties == (AgentId.A2_PRICING, AgentId.A5_COMPLIANCE)
                   for c in s1.expected_conflicts)

    def test_s5_declares_no_fulfilment_evidence(self) -> None:
        s5 = get_scenario("S5_vision_gap")
        assert s5.seed_data.missing_evidence
        assert all(o.deliverables_evidence_url is None for o in s5.seed_data.offers)

    def test_every_reply_pool_entry_is_a_labelled_reply(self) -> None:
        """A run may only send a reply the scenario has declared ground truth for."""
        for s in ALL_SCENARIOS:
            labelled = {p.text for p in s.reply_probes}
            assert s.seed_data.reply_pool, f"{s.name} has an empty reply pool"
            for text in s.seed_data.reply_pool:
                assert text in labelled, f"{s.name} reply pool entry has no probe label"

    def test_probe_labels_are_unique_within_a_scenario(self) -> None:
        for s in ALL_SCENARIOS:
            labels = [p.label for p in s.reply_probes]
            assert len(labels) == len(set(labels)), s.name

    def test_probe_labels_agree_with_the_frozen_fallback_except_where_documented(self) -> None:
        """Ground truth is stated independently of the routing policy.

        The one place the harness asserts a *better* label than the frozen fallback
        is the NAUTRAL-vs-INTERESTED distinction, which is a statement about what a
        competent negotiator concludes, not about what a substring list contains.
        Intent agreement with the frozen fallback is checked here so that a future
        edit to a probe text cannot quietly invalidate the comparison.
        """
        for s in ALL_SCENARIOS:
            for p in s.reply_probes:
                fallback = classify_intent_fallback(p.text)
                assert isinstance(p.expected_intent, Intent)
                assert isinstance(p.expected_agent, AgentId)
                assert p.rationale.strip(), f"{p.label} has no stated rationale"
                # A PUSHBACK or NO label must never contradict the frozen ordering.
                if p.expected_intent in (Intent.PUSHBACK, Intent.NO):
                    assert fallback in (p.expected_intent, Intent.NEUTRAL), (
                        f"{p.label}: expected {p.expected_intent.value} but the frozen "
                        f"fallback returns {fallback.value}; the harness must not claim "
                        "a label the frozen contract contradicts")

    def test_every_probe_states_whether_it_is_adversarial(self) -> None:
        assert any(p.adversarial for s in ALL_SCENARIOS for p in s.reply_probes)
        assert any(not p.adversarial for s in ALL_SCENARIOS for p in s.reply_probes)

    def test_negotiation_bounds_validate(self) -> None:
        with pytest.raises(ValueError, match="opening_ask"):
            NegotiationBounds(opening_ask_inr=10_000.0, sponsor_reservation_inr=20_000.0,
                              walkaway_inr=5_000.0).validate()
        with pytest.raises(ValueError, match="walkaway"):
            NegotiationBounds(opening_ask_inr=100_000.0, sponsor_reservation_inr=50_000.0,
                              walkaway_inr=60_000.0).validate()

    def test_scenario_rng_is_deterministic_and_keyed_by_scenario(self) -> None:
        a = get_scenario("S1_exclusivity_conflict")
        b = get_scenario("S3_ambiguous_reply")
        assert [scenario_rng(a, 5).random() for _ in range(4)] == \
               [scenario_rng(a, 5).random() for _ in range(4)]
        # Different scenarios at the same seed must not share a draw sequence,
        # otherwise adding a scenario would shift an existing one's numbers.
        assert scenario_rng(a, 5).random() != scenario_rng(b, 5).random()

    def test_default_seed_is_documented_and_fixed(self) -> None:
        assert DEFAULT_SEED == 20260210
        assert all(s.seed == DEFAULT_SEED for s in ALL_SCENARIOS)

    def test_unknown_scenario_name_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="S1_exclusivity_conflict"):
            get_scenario("S9_does_not_exist")


# ==================================================== 1. fixed pipeline pv == 1
class TestFixedPipelineIsNonAgentic:
    @pytest.mark.slow
    @pytest.mark.parametrize("scenario", FAST_SCENARIOS, ids=lambda s: s.name)
    def test_path_variation_is_exactly_one(self, scenario) -> None:
        """The load-bearing claim: the baseline cannot produce a second path."""
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        report_ = run_ablation([scenario], [cond], SEEDS)
        agg = report_.condition(scenario.name, cond.name)
        assert agg.failed_runs == []
        assert agg.path_variation == 1
        assert len(agg.distinct_paths) == 1
        assert agg.distinct_paths[0] == "A1->A2->A3->A4->A5->A6"

    @pytest.mark.parametrize("scenario", FAST_SCENARIOS, ids=lambda s: s.name)
    def test_it_makes_no_decision_calls_at_all(self, scenario) -> None:
        """No conditional logic means no decision layer. Zero, not one."""
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        report_ = run_ablation([scenario], [cond], SEEDS)
        agg = report_.condition(scenario.name, cond.name)
        assert agg.stats["decision_calls"].total == 0
        assert agg.decision_source_mix == {}
        assert agg.stats["human_gates_raised"].total == 0
        assert agg.stats["replans"].total == 0

    def test_its_path_is_independent_of_the_seed(self) -> None:
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        s = get_scenario("S1_exclusivity_conflict")
        paths = {cond.run(s, seed=seed).path_key for seed in SEEDS}
        assert paths == {"A1->A2->A3->A4->A5->A6"}

    def test_it_runs_every_agent_in_the_declared_order(self) -> None:
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        s = get_scenario("S1_exclusivity_conflict")
        outcome = cond.run(s, seed=1)
        assert outcome.agent_path[0] == AgentId.A1_DISCOVERY
        assert outcome.agent_path[-1] == AgentId.A6_AUDIT
        assert outcome.steps_taken == len(FixedPipelineCondition.PIPELINE)


# ============================================== 2. report keeps raw records
@pytest.fixture(scope="module")
def report_():
    """One sweep shared by every assertion in this class.

    Module-scoped because the sweep is pure and deterministic; re-running it per
    test would add seconds without adding coverage.
    """
    return run_ablation(FAST_SCENARIOS,
                        _offline(FixedPipelineCondition, KeywordRouterCondition),
                        SEEDS)


class TestAblationReportKeepsRawRecords:

    def test_every_cell_produced_exactly_one_record(self, report_) -> None:
        expected = len(FAST_SCENARIOS) * 2 * len(SEEDS)
        assert len(report_.records) == expected
        seen = {(r.scenario, r.condition, r.seed) for r in report_.records}
        assert len(seen) == expected

    def test_records_are_run_records_holding_full_outcomes(self, report_) -> None:
        assert all(isinstance(r, RunRecord) for r in report_.records)
        first = report_.records[0]
        assert first.batch_id == report_.batch_id
        assert first.outcome.path_key.startswith("A1")
        assert first.outcome.started_at
        assert first.outcome.wall_clock_ms >= 0.0

    def test_raw_records_are_serialisable_and_complete(self, report_) -> None:
        payload = json.loads(to_json(report_))
        assert len(payload["records"]) == len(report_.records)
        row = payload["records"][0]
        for key in ("scenario", "condition", "seed", "ok", "agent_path", "steps_taken",
                    "decision_calls", "decisions_by_source", "tool_status_counts",
                    "route_decisions", "runtime", "path_variation_caveat"):
            assert key in row, f"raw record is missing {key!r}"

    def test_aggregates_are_recomputable_from_the_raw_records(self, report_) -> None:
        """Audit test: the headline number must be derivable from the raw rows."""
        for sname in report_.scenario_names:
            for cname in report_.condition_names:
                agg = report_.condition(sname, cname)
                paths = {r.path_key for r in report_.records
                         if r.scenario == sname and r.condition == cname and r.ok}
                assert agg.path_variation == len(paths)
                assert sorted(agg.distinct_paths) == sorted(paths)
                total = sum(r.outcome.steps_taken for r in report_.records
                            if r.scenario == sname and r.condition == cname and r.ok)
                assert agg.stats["steps_taken"].total == total

    def test_raw_records_carry_every_decision_source(self, report_) -> None:
        for rec in report_.records:
            for src in rec.outcome.decisions_by_source:
                assert src in {s.value for s in DecisionSource}

    def test_router_probes_are_recorded_per_run(self, report_) -> None:
        """Each run routes every probe, so the comparison is seed-independent."""
        for rec in report_.records:
            if not rec.outcome.ok:
                continue
            scenario = get_scenario(rec.scenario)
            assert len(rec.outcome.route_decisions) == len(scenario.reply_probes)

    def test_router_probes_are_counted_once_per_pair_not_once_per_seed(self, report_) -> None:
        for sname in report_.scenario_names:
            for cname in report_.condition_names:
                agg = report_.condition(sname, cname)
                assert agg.router.probes == len(get_scenario(sname).reply_probes)

    def test_headline_reports_paths_and_health(self, report_) -> None:
        head = report_.headline
        assert head["headline_metric"] == "path_variation"
        assert "necessary but NOT sufficient" in head["headline_caveat"]
        pv = head["path_variation_by_condition"]["fixed_pipeline"]
        assert pv["mean_path_variation"] == 1.0
        assert pv["scenarios_measured"] == len(FAST_SCENARIOS)
        for cond, health in head["run_health_by_condition"].items():
            assert health["completion_rate"] == 1.0, cond
            assert health["stall_rate"] is not None, cond


# =============================================== 3. markdown renders provenance
@pytest.fixture(scope="module")
def markdown():
    rep = run_ablation(FAST_SCENARIOS,
                       _offline(FixedPipelineCondition, KeywordRouterCondition),
                       SEEDS, batch_id="test_batch")
    return to_markdown(rep)


class TestMarkdownRendersProvenance:

    @pytest.mark.parametrize("needle", [
        "### Provenance",
        "**Git commit**:",
        "**Code hash (sha256)**:",
        "**Hashed packages**:",
        "**Run mode**:",
        "**Intended decision backend**:",
        "**Decision backends probed**:",
        "**Runtimes used**:",
        "**Seeds**:",
        "Every figure in this report was produced by executing the Paytriq",
        "synthetic fixtures",
    ])
    def test_provenance_block_contains(self, markdown, needle) -> None:
        assert needle in markdown

    def test_provenance_appears_before_the_results(self, markdown) -> None:
        assert markdown.index("### Provenance") < markdown.index("## Per-scenario results")

    def test_path_variation_is_a_prominent_column(self, markdown) -> None:
        header = [ln for ln in markdown.splitlines() if ln.startswith("| Scenario | Condition |")]
        assert header, "the per-scenario table header is missing"
        assert "`path_variation`" in header[0]
        # Second column of the body, immediately after the scenario name.
        body = markdown.split("## Per-scenario results", 1)[1]
        first_row = next(ln for ln in body.splitlines()
                         if ln.startswith("| `S1"))
        # | scenario | condition | path_variation | ...
        assert first_row.split("|")[3].strip() == "**1**"

    def test_it_reports_the_router_misroutes(self, markdown) -> None:
        assert "## Routing comparison" in markdown
        assert "Cross-condition disagreement" in markdown
        assert "MISS" in markdown

    def test_it_renders_the_raw_record_index(self, markdown) -> None:
        assert "## Raw per-run records" in markdown
        assert "test_batch" in markdown

    def test_json_and_markdown_render_both_report_types(self) -> None:
        rep = run_ablation([get_scenario("S4_clean_run")],
                           _offline(FixedPipelineCondition), SEEDS)
        bench = benchmark.benchmark_from_records(
            rep.records, [get_scenario("S4_clean_run")], ["fixed_pipeline"],
            rep.provenance)
        for doc in (to_markdown(rep), to_markdown(bench)):
            assert "### Provenance" in doc
            assert "Every figure in this report was produced by executing" in doc
        for blob in (to_json(rep), to_json(bench)):
            assert json.loads(blob)["provenance"]["statement"].startswith(
                "Every figure in this report")

    def test_a_wrong_type_is_rejected_rather_than_guessed(self) -> None:
        with pytest.raises(TypeError, match="AblationReport or BenchmarkReport"):
            to_markdown({"not": "a report"})
        with pytest.raises(TypeError, match="AblationReport or BenchmarkReport"):
            to_json({"not": "a report"})

    def test_large_amounts_are_not_mangled_by_trailing_zero_stripping(self) -> None:
        """Regression: ``f"{120000.0:.0f}"`` is "120000"; rstrip would give "12"."""
        assert report.format_value(120_000.0, digits=0) == "120000"
        assert report.format_value(11_160.0, digits=0) == "11160"
        assert report.format_value(85_000.0, digits=0) == "85000"
        assert report.format_value(0.5, digits=3) == "0.5"
        assert report.format_value(1.0, digits=3) == "1"

    def test_repeated_violations_are_grouped_without_being_dropped(self) -> None:
        items = [f"seed={s}: exclusivity 365d exceeds cap 42d" for s in SEEDS]
        items += [f"seed={s}: no evidence of signer authority" for s in SEEDS]
        grouped = report.group_violations(items)
        assert "exclusivity 365d exceeds cap 42d (x3)" in grouped
        assert "no evidence of signer authority (x3)" in grouped
        assert report.group_violations([]) == "none"
        # Overflow is announced, not silently truncated.
        many = [f"seed={s}: violation number {s}" for s in range(20)]
        assert "more" in report.group_violations(many, limit=3)

    def test_a_measured_zero_decision_mix_is_not_rendered_as_unavailable(self) -> None:
        """The fixed pipeline makes no decisions; that is a result, not a gap."""
        rep = run_ablation([get_scenario("S4_clean_run")],
                           _offline(FixedPipelineCondition), (1,))
        doc = to_markdown(rep)
        assert "none - measured 0 decision calls" in doc
        assert "There is no conditional logic" not in doc

    def test_missing_provenance_renders_as_unavailable(self) -> None:
        doc = report.provenance_markdown(None)
        assert UNAVAILABLE in doc
        assert "**Git commit**: `unavailable`" in doc


# ============================= 4. missing dependency -> explicit unavailable
class TestMissingDependencyIsReportedNotFaked:
    def test_a_runtime_that_cannot_run_produces_ok_false_with_a_reason(self) -> None:
        """The headline honesty test: no run, no numbers, and the reason recorded."""
        cond = FixedPipelineCondition(
            runtime=StubRuntime(raise_for=(AgentId.A1_DISCOVERY,)))
        outcome = cond.run(get_scenario("S4_clean_run"), seed=1)
        assert outcome.ok is False
        assert "StubRuntime was configured to fail" in outcome.failure_reason
        # Nothing was measured, so nothing is reported. Not zeros.
        assert outcome.steps_taken == 0
        assert outcome.agent_path == ()
        assert outcome.negotiation is None
        assert "no path recorded" in outcome.path_variation_caveat

    def test_an_unavailable_cell_renders_path_variation_as_unavailable(self) -> None:
        """Not 0. Zero would claim the system took exactly one path."""
        cond = FixedPipelineCondition(runtime=StubRuntime(raise_for=(AgentId.A2_PRICING,)))
        rep = run_ablation([get_scenario("S4_clean_run")], [cond], SEEDS)
        agg = rep.condition("S4_clean_run", cond.name)
        assert agg.ok_runs == 0
        assert agg.path_variation is None
        assert agg.stats["steps_taken"].mean is None
        assert agg.stats["steps_taken"].stdev is None
        assert agg.failed_runs
        assert rep.failures
        assert "unavailable" in to_markdown(rep)

    def test_failures_are_listed_with_their_reason(self) -> None:
        cond = KeywordRouterCondition(runtime=StubRuntime(raise_for=(AgentId.A3_OUTREACH,)))
        rep = run_ablation([get_scenario("S3_ambiguous_reply")], [cond], (1, 2))
        assert len(rep.failures) == 2
        assert all("StubRuntime was configured to fail" in f for f in rep.failures)
        assert "## Run failures" in to_markdown(rep)

    def test_a_metric_with_one_sample_has_no_stdev(self) -> None:
        """A stdev from one observation is noise presented as precision."""
        stats = MetricStats(name="steps_taken", values=[7.0])
        assert stats.mean == 7.0
        assert stats.stdev is None
        assert "unavailable sd" in report.format_stat(stats)

    def test_an_empty_metric_renders_as_unavailable(self) -> None:
        assert report.format_stat(MetricStats(name="x")) == UNAVAILABLE
        assert report.format_value(None) == UNAVAILABLE
        assert report.format_value(float("nan")) == UNAVAILABLE

    def test_benchmark_records_a_failure_rather_than_a_zero_deal_rate(self) -> None:
        cond = FixedPipelineCondition(runtime=StubRuntime(raise_for=(AgentId.A1_DISCOVERY,)))
        rep = run_benchmark([get_scenario("S4_clean_run")], [cond], SEEDS)
        cb = rep.condition(cond.name)
        assert cb.signed_runs == 0
        assert cb.completed_runs == 0
        assert cb.deal_rate == 0.0        # 0/3 attempted: honest, and explained below
        assert cb.explicit_failures
        assert "completed" in cb.explicit_failures[0]
        # Metrics with nothing behind them are None, not 0.
        assert cb.mean_rounds_to_deal is None
        assert cb.surplus_split_mean is None
        assert cb.escalation_rate is None

    def test_benchmark_never_invents_a_surplus_share_outside_the_range(self) -> None:
        bounds = NegotiationBounds(opening_ask_inr=100.0, sponsor_reservation_inr=80.0,
                                   walkaway_inr=70.0)
        assert surplus_split(bounds, 100.0) == (1.0, 0.0)
        assert surplus_split(bounds, 80.0) == (0.0, 1.0)
        # Below the reservation is not "extra surplus captured", it is a data error.
        assert surplus_split(bounds, 10.0) is None
        assert surplus_split(bounds, 500.0) is None

    def test_real_agent_runtime_reports_why_it_is_unavailable(self) -> None:
        """``agents`` may be mid-author by another package; say so, do not fake it."""
        runtime = conditions.RealAgentRuntime()
        ok, why = runtime.available()
        assert isinstance(ok, bool)
        assert why.strip()
        if not ok:
            with pytest.raises(RuntimeError, match="available"):
                runtime.execute(conditions.StepRequest(
                    agent=AgentId.A1_DISCOVERY, step_name="probe",
                    scenario=get_scenario("S4_clean_run"), seed=1,
                    board=conditions.EvalBoard("evt"), decide=None, tools=None,
                    counters=RunCounters(), run_id="run_test",
                    event_id="evt", focus_brand="x", reply_text="", round_index=0,
                    state={}))

    def test_resolve_runtime_labels_its_fallback(self) -> None:
        runtime, why = conditions.resolve_runtime(prefer_real=True)
        assert callable(runtime.available) and callable(runtime.execute)
        assert runtime.name
        assert why.strip()
        if runtime.name == LocalRuleRuntime.name:
            assert "FALLBACK" in why
            assert LocalRuleRuntime.name in why

    def test_provenance_reports_an_unavailable_git_commit_rather_than_a_fake_one(self) -> None:
        commit = ablation.git_commit()
        assert commit
        assert commit == ablation.UNAVAILABLE or len(commit) >= 7

    def test_code_hash_covers_the_eval_package_itself(self) -> None:
        digest, included = ablation.code_hash()
        assert digest != UNAVAILABLE
        assert "eval" in included
        assert "core" in included
        # Content-addressed: identical tree, identical digest.
        assert ablation.code_hash()[0] == digest


# =============================================== routing: the measured bug
class TestRouterComparison:
    def test_the_naive_router_reproduces_the_documented_bug(self) -> None:
        """The exact sentence named in ``core.schemas.Intent``.

        ``"yes"`` is a substring of ``"ye-s-terday"``, so the previous prototype's
        assent-first router classified a price objection as acceptance and sent it to
        contract signing. This is asserted against the frozen docstring's claim, not
        against our own fixture.
        """
        text = "yesterday we thought the price was too high, but let's proceed"
        assert naive_keyword_intent(text) is Intent.YES
        assert classify_intent_fallback(text) is Intent.PUSHBACK

    def test_the_naive_router_tests_assent_before_pushback(self) -> None:
        assert naive_keyword_intent("we agree the price is too high, can you reduce it?") \
            is Intent.YES
        assert classify_intent_fallback(
            "we agree the price is too high, can you reduce it?") is Intent.PUSHBACK

    def test_the_naive_router_defaults_any_unrecognised_message_to_interest(self) -> None:
        """The second documented defect: silence becomes a warm lead."""
        assert naive_keyword_intent("the committee reconvenes after the recess") \
            is Intent.INTERESTED
        assert classify_intent_fallback("the committee reconvenes after the recess") \
            is Intent.NEUTRAL

    def test_the_naive_router_knows_vocabulary_the_frozen_fallback_does_not(self) -> None:
        """An honest counterexample: the naive list is not simply worse at everything."""
        text = "Approved on our side, go ahead."
        assert naive_keyword_intent(text) is Intent.INTERESTED   # 'approved' missing
        assert classify_intent_fallback(text) is Intent.YES

    def test_the_keyword_condition_misroutes_every_adversarial_s3_probe(self) -> None:
        s3 = get_scenario("S3_ambiguous_reply")
        adversarial = [p for p in s3.reply_probes if p.adversarial]
        assert adversarial, "S3 must contain adversarial probes"
        for p in adversarial:
            d = route_probe(p, KeywordRouterCondition.name)
            assert d.routed_agent is not p.expected_agent
            assert d.classifier == "keyword"
            assert d.decision_source is DecisionSource.RULES

    def test_the_decision_routing_gets_the_adversarial_probes_right(self) -> None:
        s3 = get_scenario("S3_ambiguous_reply")
        for p in [x for x in s3.reply_probes if x.adversarial]:
            d = route_probe(p, AgenticCondition.name)
            assert d.routed_agent is p.expected_agent, p.label
            assert d.routed_intent is p.expected_intent, p.label
            assert d.classifier.startswith("decision:")
            assert d.decision_source is DecisionSource.RULES

    def test_routing_is_a_pure_function_of_the_probe_text(self) -> None:
        for s in ALL_SCENARIOS:
            for p in s.reply_probes:
                a = route_probe(p, KeywordRouterCondition.name)
                b = route_probe(p, KeywordRouterCondition.name)
                assert a.to_dict() == b.to_dict()

    def test_router_confusion_counts_are_internally_consistent(self) -> None:
        rep = run_ablation([get_scenario("S3_ambiguous_reply")],
                           _offline(FixedPipelineCondition, KeywordRouterCondition,
                                    AgenticCondition),
                           (1,))
        agg = rep.condition("S3_ambiguous_reply", KeywordRouterCondition.name)
        r = agg.router
        assert r.probes == 5
        assert r.agent_agreements == 2
        assert r.adversarial_probes == 3
        assert r.adversarial_agent_misroutes == 3
        assert len(r.per_probe) == 5
        assert r.to_dict()["probes"] == 5

    def test_cross_condition_disagreement_is_reported_per_probe(self) -> None:
        rep = run_ablation([get_scenario("S3_ambiguous_reply")],
                           _offline(FixedPipelineCondition, KeywordRouterCondition,
                                    AgenticCondition),
                           (1,))
        sa = rep.per_scenario["S3_ambiguous_reply"]
        assert sa.router_split
        misrouted = [label for label, agreed in sa.router_unanimous.items() if not agreed]
        assert misrouted, "the conditions must disagree on at least one probe"
        for label in misrouted:
            split = sa.router_split[label]
            assert len(set(split.values())) > 1
            assert len(split) == 3

    def test_every_probe_the_naive_router_misses_is_marked_adversarial(self) -> None:
        """Cross-check: we did not quietly label failures as non-adversarial."""
        for s in ALL_SCENARIOS:
            for p in s.reply_probes:
                d = route_probe(p, KeywordRouterCondition.name)
                if not d.agent_agreed:
                    assert p.adversarial, (
                        f"{s.name}/{p.label} is missed by the naive router but is not "
                        "marked adversarial; the fixture would be rigged")


# =============================================== every decision has a source
class TestNoAnonymousDecisions:
    def test_the_agentic_condition_labels_every_decision(self) -> None:
        cond = AgenticCondition(prefer_real_runtime=False)
        outcome = cond.run(get_scenario("S1_exclusivity_conflict"), seed=7)
        assert outcome.ok
        assert outcome.decision_calls > 0
        assert sum(outcome.decisions_by_source.values()) == outcome.decision_calls
        for src in outcome.decisions_by_source:
            assert src in {s.value for s in DecisionSource}
        # Offline: every decision is rules, and the report must say so rather than
        # implying a model was consulted.
        assert set(outcome.decisions_by_source) == {DecisionSource.RULES.value}
        assert outcome.llm_calls == 0

    def test_a_rules_answer_in_a_live_run_is_marked_degraded(self) -> None:
        """Degradation is labelled, never disguised as a model call."""
        cond = AgenticCondition(prefer_real_runtime=False, run_mode=conditions.RunMode.LIVE)
        outcome = cond.run(get_scenario("S4_clean_run"), seed=3)
        assert outcome.degraded_decisions > 0
        assert outcome.degraded_decisions == outcome.decision_calls

    def test_an_offline_run_is_not_marked_degraded(self) -> None:
        cond = AgenticCondition(prefer_real_runtime=False, run_mode=conditions.RunMode.OFFLINE)
        outcome = cond.run(get_scenario("S4_clean_run"), seed=3)
        assert outcome.degraded_decisions == 0

    def test_fixture_backed_tool_calls_are_visible_as_cached(self) -> None:
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        outcome = cond.run(get_scenario("S1_exclusivity_conflict"), seed=1)
        assert outcome.tool_calls > 0
        assert outcome.tool_status_counts
        assert ToolStatus.CACHED.value in outcome.tool_status_counts
        assert sum(outcome.tool_status_counts.values()) == outcome.tool_calls

    def test_a_missing_vision_evidence_tool_reports_unavailable_not_success(self) -> None:
        """S5: absence of evidence must surface as UNAVAILABLE, not as a pass."""
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        outcome = cond.run(get_scenario("S5_vision_gap"), seed=1)
        assert outcome.tool_status_counts.get(ToolStatus.UNAVAILABLE.value, 0) > 0
        assert ToolStatus.OK.value not in outcome.tool_status_counts

    def test_no_condition_claims_fulfilment_it_cannot_evidence(self) -> None:
        """The S5 honesty check, across every condition and seed."""
        for cond in _offline(FixedPipelineCondition, KeywordRouterCondition,
                             AgenticCondition):
            for seed in SEEDS:
                outcome = cond.run(get_scenario("S5_vision_gap"), seed=seed)
                assert outcome.unverifiable_claims == [], (
                    f"{cond.name} claimed fulfilment with no evidence: "
                    f"{outcome.unverifiable_claims}")


# =============================================== determinism and auditability
class TestDeterminism:
    def test_a_seed_reproduces_a_run_exactly(self) -> None:
        for cond in _offline(FixedPipelineCondition, KeywordRouterCondition,
                             AgenticCondition):
            a = cond.run(get_scenario("S1_exclusivity_conflict"), seed=99)
            b = cond.run(get_scenario("S1_exclusivity_conflict"), seed=99)
            assert a.path_key == b.path_key
            assert a.steps_taken == b.steps_taken
            assert a.decision_calls == b.decision_calls
            assert a.disputes_raised == b.disputes_raised
            assert a.replans == b.replans
            assert a.human_gates_raised == b.human_gates_raised
            assert a.tool_status_counts == b.tool_status_counts

    def test_two_sweeps_with_the_same_seeds_agree(self) -> None:
        """Determinism, with one honest exception.

        ``wall_clock_ms`` is a measurement of the machine, not of the decision, so it
        is the one field allowed to differ between two sweeps. Everything that
        describes *what the system did* must be identical.
        """
        args = ([get_scenario("S3_ambiguous_reply")],
                _offline(FixedPipelineCondition, AgenticCondition), SEEDS)
        first = run_ablation(*args)
        second = run_ablation(*args)
        for s in first.scenario_names:
            for c in first.condition_names:
                a, b = first.condition(s, c).to_dict(), second.condition(s, c).to_dict()
                assert a["path_variation"] == b["path_variation"]
                assert a["distinct_paths"] == b["distinct_paths"]
                assert a["decision_source_mix"] == b["decision_source_mix"]
                assert a["tool_status_mix"] == b["tool_status_mix"]
                assert a["router"] == b["router"]
                for name, stats in a["stats"].items():
                    if name == "wall_clock_ms":
                        continue
                    assert stats["values"] == b["stats"][name]["values"], name

    def test_different_seeds_produce_different_inputs(self) -> None:
        """If every seed produced the same reply, path variation would be vacuous."""
        s = get_scenario("S3_ambiguous_reply")
        replies = {scenario_rng(s, seed).choice(s.seed_data.reply_pool) for seed in SEEDS}
        assert len(replies) > 1


# =============================================== the benchmark, measured
@pytest.fixture(scope="module")
def bench():
    return run_benchmark(FAST_SCENARIOS,
                         _offline(FixedPipelineCondition, KeywordRouterCondition,
                                  AgenticCondition),
                         SEEDS)


class TestBenchmark:

    def test_it_covers_every_scenario_and_condition(self, bench) -> None:
        assert set(bench.conditions) == {"fixed_pipeline", "keyword_router", "agentic"}
        for cb in bench.conditions.values():
            assert set(cb.cells) == {s.name for s in FAST_SCENARIOS}
            assert cb.seeds_run == len(FAST_SCENARIOS) * len(SEEDS)

    def test_the_fixed_pipeline_signs_a_policy_breaching_contract(self) -> None:
        """The finding the baseline exists to produce, asserted as a measurement."""
        rep = run_benchmark([get_scenario("S1_exclusivity_conflict")],
                            _offline(FixedPipelineCondition), SEEDS)
        cb = rep.condition("fixed_pipeline")
        assert cb.signed_runs == len(SEEDS)
        assert cb.policy_violations_total > 0
        assert cb.signed_in_violation_total > 0
        # ...and it never escalated, which is why the breach reached the signature.
        assert cb.escalation_rate == 0.0

    def test_surplus_shares_sum_to_one(self, bench) -> None:
        for cb in bench.conditions.values():
            split = cb.surplus_split_mean
            if split is None:
                continue
            assert math.isclose(split[0] + split[1], 1.0, abs_tol=0.001)

    def test_a_condition_that_signs_at_the_ask_captures_all_the_surplus(self, bench) -> None:
        """No negotiation means the sponsor wins the whole gap - by construction."""
        cb = bench.condition("fixed_pipeline")
        cell = cb.cells["S1_exclusivity_conflict"]
        assert cell.splits, "the baseline signs at the opening ask, so a split exists"
        for sponsor_share, organiser_share in cell.splits:
            assert sponsor_share > organiser_share

    def test_every_cell_records_its_own_outcome(self, bench) -> None:
        for cb in bench.conditions.values():
            for cell in cb.cells.values():
                assert cell.seeds_run == len(SEEDS)
                assert cell.signed_runs <= cell.completed_runs <= cell.seeds_run
                assert 0.0 <= (cell.deal_rate or 0.0) <= 1.0

    def test_no_fabrication_warnings_anywhere(self, bench) -> None:
        for cb in bench.conditions.values():
            assert cb.fabrication_warnings_total == 0

    def test_signing_below_the_walkaway_is_counted_not_hidden(self) -> None:
        """A low deal rate must not conceal deals taken below the walk-away.

        The fixed pipeline signs S2 at INR 11,160 against a stated walk-away of
        INR 55,000. That is a failure, and it has to be visible next to a 100%
        deal rate rather than showing up only as a missing surplus figure.
        """
        s2 = get_scenario("S2_low_ev_lead")
        rep = run_benchmark([s2], _offline(FixedPipelineCondition, AgenticCondition),
                            SEEDS)
        cb = rep.condition("fixed_pipeline")
        assert cb.deal_rate == 1.0
        assert cb.below_walkaway_total > 0
        cell = cb.cells["S2_low_ev_lead"]
        assert cell.below_walkaway
        assert "walk-away" in cell.below_walkaway[0]
        # Signing below the walk-away means the surplus share is undefined, so it is
        # reported as unavailable rather than as a computed number.
        assert cell.splits == []
        assert cb.surplus_split_mean is None
        # A condition that respects its walk-away never trips this.
        assert rep.condition("agentic").below_walkaway_total == 0

    def test_json_is_full_fidelity(self, bench) -> None:
        payload = json.loads(to_json(bench))
        assert payload["provenance"]["git_commit"]
        assert payload["conditions"]["fixed_pipeline"]["cells"]
        assert "definitions" in payload["headline"]


# =============================================== contracts and invariants
class TestInvariants:
    def test_the_three_conditions_are_distinct_and_registered(self) -> None:
        names = {c.name for c in conditions.ALL_CONDITIONS}
        assert names == {"fixed_pipeline", "keyword_router", "agentic"}
        assert conditions.get_condition("agentic").name == "agentic"
        with pytest.raises(KeyError, match="fixed_pipeline"):
            conditions.get_condition("nope")

    def test_only_the_agentic_condition_uses_the_decision_layer(self) -> None:
        assert conditions.AgenticCondition.uses_decision_layer is True
        assert conditions.FixedPipelineCondition.uses_decision_layer is False
        assert conditions.KeywordRouterCondition.uses_decision_layer is False

    def test_every_condition_is_a_condition(self) -> None:
        for c in conditions.ALL_CONDITIONS:
            assert isinstance(c, Condition)

    def test_a_self_handoff_is_never_recorded(self) -> None:
        """The path builder refuses to log a handoff to oneself.

        ``core.schemas.Handoff`` raises on ``from_agent == to_agent``; the builder
        simply never produces one, so the invariant holds without an exception being
        raised in the middle of a run.
        """
        cond = FixedPipelineCondition(prefer_real_runtime=False)
        counters = RunCounters()
        path: list[AgentId] = []
        cond._handoff(counters, AgentId.A2_PRICING, AgentId.A2_PRICING, "self",
                      DecisionSource.RULES, 1.0, "run", "evt", path)
        assert path == []
        cond._handoff(counters, AgentId.A2_PRICING, AgentId.A4_CONTRACT, "real",
                      DecisionSource.RULES, 1.0, "run", "evt", path)
        assert path == [AgentId.A4_CONTRACT]

    def test_the_frozen_contract_rejects_a_self_handoff_directly(self) -> None:
        from core.schemas import Handoff

        with pytest.raises(Exception, match="handoff must change"):
            Handoff(handoff_id="hnd", event_id="evt", run_id="run",
                    from_agent=AgentId.A2_PRICING, to_agent=AgentId.A2_PRICING,
                    reason="x", decision_source=DecisionSource.RULES, confidence=1.0)

    def test_the_board_is_append_only_and_ordered(self) -> None:
        board = conditions.EvalBoard("evt")
        a = board.post("z", "k", AgentId.A1_DISCOVERY, {"i": 0})
        b = board.post("z", "k", AgentId.A2_PRICING, {"i": 1})
        history = board.history()
        assert [h.entry_id for h in history] == [a.entry_id, b.entry_id]
        assert board.zones() == ["z"]
        assert board.latest("z").entry_id == b.entry_id
        assert board.read("z", limit=1)[0].entry_id == b.entry_id

    def test_counters_sum_to_their_breakdowns(self) -> None:
        counters = RunCounters()
        assert sum(counters.decisions_by_source.values()) == 0
        counters.add_dispute(_dispute())
        assert counters.disputes_raised == 1
        assert counters.dispute_zones == ["contract"]
        counters.add_tool(ToolStatus.CACHED)
        counters.add_tool(ToolStatus.UNAVAILABLE)
        assert counters.tool_calls == 2
        assert sum(counters.tool_status_counts.values()) == 2

    def test_contract_terms_audit_against_the_caps(self) -> None:
        caps = PolicyCaps()
        clean = ContractTerms(brand="b", amount_inr=1.0, exclusivity_days=10,
                              attendee_data_clause=False, signer_authority_evidence=True,
                              deliverables_evidence_url="fixture://x")
        assert clean.violations(caps) == []
        dirty = ContractTerms(brand="b", amount_inr=1.0, exclusivity_days=400,
                              attendee_data_clause=True, signer_authority_evidence=False,
                              deliverables_evidence_url=None)
        breaches = dirty.violations(caps)
        assert len(breaches) == 4

    def test_run_ablation_rejects_an_empty_sweep(self) -> None:
        with pytest.raises(ValueError, match="scenario"):
            run_ablation([], [FixedPipelineCondition(prefer_real_runtime=False)], (1,))
        with pytest.raises(ValueError, match="condition"):
            run_ablation(ALL_SCENARIOS, [], (1,))
        with pytest.raises(ValueError, match="seed"):
            run_ablation(ALL_SCENARIOS, [FixedPipelineCondition()], ())

    def test_the_condition_lookup_error_names_the_scenario_and_condition(self) -> None:
        rep = run_ablation([get_scenario("S4_clean_run")],
                           _offline(FixedPipelineCondition), (1,))
        with pytest.raises(KeyError, match="fixed_pipeline"):
            rep.condition("S4_clean_run", "not_a_condition")
        with pytest.raises(KeyError, match="not_a_scenario"):
            rep.condition("not_a_scenario", "fixed_pipeline")

    def test_scenario_provenance_note_is_repeated_in_every_report(self) -> None:
        rep = run_ablation([get_scenario("S4_clean_run")],
                           _offline(FixedPipelineCondition), (1,))
        assert scenarios.SYNDETIC_BRAND_NOTE in rep.provenance.data_note
        assert scenarios.SYNDETIC_BRAND_NOTE in to_markdown(rep)
        assert scenarios.FIXTURE_SOURCE in rep.provenance.fixture_source

    def test_the_provenance_statement_is_never_edited_to_suit_a_result(self) -> None:
        assert "No trace, count, or outcome here was hand-written" in Provenance().statement
        assert "never replaced with a plausible substitute" in Provenance().statement

    def test_lazy_package_import_exposes_the_public_surface(self) -> None:
        import eval as ev

        assert ev.DEFAULT_SEED == DEFAULT_SEED
        assert ev.UNAVAILABLE == UNAVAILABLE
        assert callable(ev.run_ablation)
        assert callable(ev.run_benchmark)
        assert callable(ev.to_markdown)
        with pytest.raises(AttributeError):
            _ = ev.not_a_real_export


def _dispute():
    """A minimal valid ``Dispute``, built through the frozen contract."""
    from core.schemas import Dispute, DisputeZone, Severity

    return Dispute(
        dispute_id="dsp_test", event_id="evt_test", zone=DisputeZone.CONTRACT,
        claimant=AgentId.A2_PRICING, opponent=AgentId.A5_COMPLIANCE,
        position="priced 365-day exclusivity", counter_position="policy caps it at 42 days",
        severity=Severity.BLOCKING,
    )
