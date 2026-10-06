"""P2 #9-#11: simulated gates, canonical trace, golden provenance.

Fast and offline. The canonical trace itself is produced by
``scripts/capture_canonical_trace.py`` (real graph, simulated CLEF backend);
these tests only assert the committed artefacts it left behind load with the
repo's own tooling and carry honest provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.schemas import GateKind, RunMode
from eval.conditions import AgenticCondition, RealAgentRuntime, RunCounters
from eval.integration import SIMULATED_GATE_DECIDER, simulated_gate_approvals

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_RUN = REPO_ROOT / "docs" / "samples" / "golden_run.jsonl"
GOLDEN_SUMMARY = REPO_ROOT / "docs" / "samples" / "golden_summary.json"


class TestSimulatedGateEvalMode:
    def test_auto_is_the_default_and_simulated_is_opt_in(self) -> None:
        auto = AgenticCondition(prefer_real_runtime=False, run_mode=RunMode.OFFLINE)
        assert auto.simulated_gates is False
        decided = auto._gate(GateKind.SEND, "q?", "preview", RunCounters(),
                             "run_x", "evt_x")
        assert decided.decided_by == "auto"

        sim = AgenticCondition(prefer_real_runtime=False, run_mode=RunMode.OFFLINE,
                               simulated_gates=True)
        assert sim.simulated_gates is True
        got = sim._gate(GateKind.SEND, "q?", "preview", RunCounters(),
                        "run_x", "evt_x")
        assert got.decided_by == SIMULATED_GATE_DECIDER == "simulated-human"
        assert got.decided_by != "auto"

    def test_simulated_approvals_cover_every_gate_kind(self) -> None:
        approvals = simulated_gate_approvals()
        kinds = {a["kind"] for a in approvals}
        assert {"send", "mou", "counter", "escalation"} <= kinds
        assert all(a["outcome"] == "approve" for a in approvals)
        assert all(a["decided_by"] == SIMULATED_GATE_DECIDER for a in approvals)
        assert all("not a real person" in a["instruction"] for a in approvals)

    def test_real_runtime_labels_simulated_gates_in_provenance(self,
                                                               settings_offline) -> None:
        rt = RealAgentRuntime(simulated_gates=True, settings=settings_offline)
        assert rt.simulated_gates is True
        assert "simulated-human" in rt.provenance()

    def test_simulated_gates_reach_a4_to_a7_on_the_stub(self) -> None:
        from eval.scenarios import get_scenario

        cond = AgenticCondition(prefer_real_runtime=False, run_mode=RunMode.OFFLINE,
                               simulated_gates=True)
        outcome = cond.run(get_scenario("S4_clean_run"), seed=11)
        assert outcome.ok
        assert "A4" in outcome.path_key and "A6" in outcome.path_key
        assert any("simulated-human" in n for n in outcome.notes)


class TestCanonicalTraceArtefact:
    def test_golden_files_exist(self) -> None:
        assert GOLDEN_RUN.exists(), "committed canonical trace is missing"
        assert GOLDEN_SUMMARY.exists(), "regenerated golden_summary.json is missing"

    def test_trace_loads_with_the_repo_tooling_and_covers_all_agents(self) -> None:
        from observability.lint import assert_clean
        from observability.summary import compute_summary

        summary = compute_summary(GOLDEN_RUN, "golden_check")
        assert summary.handoff_count >= 5
        assert summary.message_count >= 1
        assert summary.tool_call_count >= 1
        assert summary.llm_call_count >= 1
        assert summary.human_gate_count >= 2
        assert summary.distinct_gap_values > 1

        agents: set[str] = set()
        handoffs = 0
        for line in GOLDEN_RUN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ag = rec.get("agent")
            if isinstance(ag, str) and ag.startswith("A"):
                agents.add(ag)
            if rec.get("kind") == "handoff" and "seq" in rec:
                handoffs += 1
                assert rec.get("attributes", {}).get("from_agent")
                assert rec.get("attributes", {}).get("to_agent")
        assert agents == {"A1", "A2", "A3", "A4", "A5", "A6", "A7"}
        assert handoffs == summary.handoff_count
        assert_clean(GOLDEN_RUN)

    def test_golden_summary_is_honest_about_a_zero_commit_repo(self) -> None:
        payload = json.loads(GOLDEN_SUMMARY.read_text(encoding="utf-8"))
        assert payload["sha"] == "uncommitted-working-tree"
        assert payload["git_commit"] == "uncommitted-working-tree"
        # trace_file names the source canonical file (bare filename, no path),
        # while the committed copy lives as golden_run.jsonl.
        assert "/" not in payload["trace_file"] and "\\" not in payload["trace_file"]
        assert payload["trace_file"].endswith(".jsonl")
        assert payload["run_id"] in payload["trace_file"]
        assert payload["run_id"]
        # Sanitized, not hand-written: host_id redacted, payloads intact.
        first = json.loads(GOLDEN_RUN.read_text(encoding="utf-8").splitlines()[0])
        assert first["attributes"]["host_id"] == "redacted"

    def test_build_wiring_injects_the_real_sha(self) -> None:
        docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "ARG GIT_COMMIT" in docker and "ARG CODE_SHA256" in docker
        assert "GIT_COMMIT=${GIT_COMMIT}" in docker or "GIT_COMMIT" in docker
        ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "github.sha" in ci and "GIT_COMMIT" in ci
        render = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")
        assert "GIT_COMMIT" in render and "CODE_SHA256" in render
