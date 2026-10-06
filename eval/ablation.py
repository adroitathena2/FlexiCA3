"""The ablation: does the architecture adapt, and can we prove it?

Why this is the most important file in the project
--------------------------------------------------
A project can *claim* it is agentic. Evidence requires a measurement that could
have come out the other way. So this module runs the same task under three
control-flow regimes and reports what actually happened, including the numbers
that are unflattering.

The headline metric is :attr:`ConditionAggregate.path_variation` - the number of
*distinct* agent paths observed across seeds on identical inputs.

* A fixed pipeline scores exactly 1, and not by accident: it has no mechanism to
  produce a second path.
* A genuinely adaptive system scores above 1, because different inputs reach
  different agents.

Everything else here exists to stop the reader from having to take that on trust:

* Raw per-run records are retained (:attr:`RunRecord`), so every aggregate can be
  recomputed from them.
* Every decision records its ``DecisionSource``; there is no anonymous number.
* Tool calls are broken down by ``ToolStatus`` so fixture-backed calls are visible
  as such.
* Missing data renders as ``"unavailable"``, never as a plausible substitute.
* :attr:`Provenance` states which runtime and which decision backends produced the
  numbers, so a reader can judge what the measurement is worth.

Two caveats this module surfaces rather than hides
--------------------------------------------------
1. If the runs came from :class:`~eval.conditions.LocalRuleRuntime`, path variation
   measures *control flow under a deterministic stub*, not the production agent
   stack. :attr:`RunRecord.path_variation_caveat` says so on every record and
   ``report.py`` prints it.
2. ``stdev`` over a single seed is ``None`` and renders as ``unavailable``. A
   mean and a standard deviation from one observation is a lie, not a shorthand.
"""
from __future__ import annotations

import hashlib
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.ids import utcnow

from .conditions import (
    ALL_CONDITIONS,
    Condition,
    RouteDecision,
    RunOutcome,
)
from .scenarios import (
    ALL_SCENARIOS,
    DEFAULT_SEED,
    FIXTURE_SOURCE,
    SYNDETIC_BRAND_NOTE,
    Scenario,
)

__all__ = [
    "UNAVAILABLE",
    "Provenance",
    "RunRecord",
    "MetricStats",
    "RouterProbeResult",
    "RouterConfusion",
    "ConditionAggregate",
    "ScenarioAggregate",
    "AblationReport",
    "run_ablation",
    "collect_provenance",
    "git_commit",
    "code_hash",
]

#: The one string this project uses for "we did not measure this".
UNAVAILABLE: str = "unavailable"

#: Numeric metrics aggregated across seeds. Order fixes the report's row order.
METRIC_FIELDS: tuple[str, ...] = (
    "steps_taken",
    "llm_calls",
    "decision_calls",
    "probe_decision_calls",
    "degraded_decisions",
    "disputes_raised",
    "disputes_resolved",
    "disputes_escalated",
    "replans",
    "stall_count",
    "human_gates_raised",
    "tool_calls",
    "wall_clock_ms",
)


# ================================================================= provenance
def git_commit(repo_root: Path | None = None) -> str:
    """Current git commit, or ``"unavailable"``.

    Deliberately not a cached value from settings: the provenance block must
    describe the tree as it is now. An empty result string from git means "no
    commits yet", which is a real state and is reported as ``unavailable`` rather
    than papered over with a hash of nothing.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{UNAVAILABLE} (git invocation failed: {type(exc).__name__})"
    commit = out.stdout.strip()
    stderr = out.stderr.strip()
    if out.returncode != 0 or commit == "HEAD":
        # ``git rev-parse HEAD`` prints its *usage* text, not a diagnostic, when the
        # repository has no commits yet - and echoes "HEAD" on stdout. Reporting
        # either verbatim would put a fake hash or a paragraph of git help into a
        # provenance block. Detect the case and say what is actually true.
        if "unknown revision" in stderr or "does not have any commits" in stderr \
                or commit == "HEAD":
            return f"{UNAVAILABLE} (repository has no commits yet)"
        detail = stderr.splitlines()
        why = detail[-1] if detail else f"git exited {out.returncode}"
        return f"{UNAVAILABLE} ({why})"
    return commit


def code_hash(repo_root: Path | None = None,
              packages: Sequence[str] = ("core", "agents", "decision", "blackboard",
                                         "tools", "observability", "graph", "api", "eval")
              ) -> tuple[str, list[str]]:
    """``(sha256, included_paths)`` over the source of the packages that exist.

    Hashes file *contents*, not mtimes, so the value is stable across checkouts
    and changes whenever any contributing source changes. Packages absent from the
    tree are omitted and listed as such by the caller - a hash over a subset would
    otherwise look like a hash over the whole system.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    included: list[str] = []
    hasher = hashlib.sha256()
    for pkg in packages:
        pkg_dir = root / pkg
        if not pkg_dir.is_dir():
            continue
        included.append(pkg)
        for path in sorted(pkg_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
    if not included:
        return (UNAVAILABLE, [])
    return (hasher.hexdigest(), included)


def _probe_decision_backends(prefer_real: bool) -> dict[str, str]:
    """Ask the decision layer what it can actually reach, per backend.

    Probed through the same module the runs use, so this cannot disagree with what
    happened during the sweep.
    """
    if not prefer_real:
        return {"policy": "not probed (prefer_real=False; rules is the intended backend)"}
    out: dict[str, str] = {}
    try:
        import importlib

        mod = importlib.import_module("decision.registry")
        # ``build_registry`` is what decision.registry actually exposes. Probing
        # only names that do not exist would print a technically-true but
        # misleading "exposes none of [...]" for a module that is present and
        # working, which is the kind of false negative a reader cannot check.
        for name in ("build_registry", "build_backend", "get_backend", "resolve_backend"):
            factory = getattr(mod, name, None)
            if callable(factory):
                try:
                    backend = factory()
                    ok, why = backend.available()
                    out[name] = f"{backend.name}/{backend.model}: " + (
                        "available" if ok else f"unavailable ({why})")
                    # Name the chain that will actually answer, so a report reader
                    # can see *which* backend is live rather than only that one is.
                    chain = getattr(backend, "describe", lambda: {})()
                    order = chain.get("order") if isinstance(chain, dict) else None
                    first = chain.get("first_available") if isinstance(chain, dict) else None
                    if order:
                        out[name + ":chain"] = (
                            f"order={list(order)} first_available={first}")
                except Exception as exc:  # noqa: BLE001
                    out[name] = f"raised {type(exc).__name__}: {exc}"
                break
        else:
            out["decision.registry"] = (
                "present but exposes none of "
                "['build_registry', 'build_backend', 'get_backend', 'resolve_backend']")
    except Exception as exc:  # noqa: BLE001
        out["decision.registry"] = f"import failed: {type(exc).__name__}: {exc}"
    return out


@dataclass
class Provenance:
    """Everything needed to judge whether a number in a report is trustworthy."""

    git_commit: str = UNAVAILABLE
    code_sha256: str = UNAVAILABLE
    code_hash_inputs: list[str] = field(default_factory=list)
    packages_missing: list[str] = field(default_factory=list)
    run_mode: str = UNAVAILABLE
    intended_decision_backend: str = UNAVAILABLE
    decision_backends: dict[str, str] = field(default_factory=dict)
    runtimes: dict[str, str] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    default_seed: int = DEFAULT_SEED
    fixture_source: str = FIXTURE_SOURCE
    generated_at: str = ""
    #: Fixed text. Never edited to suit a result.
    statement: str = (
        "Every figure in this report was produced by executing the Paytriq "
        "system. No trace, count, or outcome here was hand-written. Any figure "
        "that could not be measured is rendered as 'unavailable' and is never "
        "replaced with a plausible substitute."
    )
    data_note: str = SYNDETIC_BRAND_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "code_sha256": self.code_sha256,
            "code_hash_inputs": list(self.code_hash_inputs),
            "packages_missing": list(self.packages_missing),
            "run_mode": self.run_mode,
            "intended_decision_backend": self.intended_decision_backend,
            "decision_backends": dict(self.decision_backends),
            "runtimes": dict(self.runtimes),
            "seeds": list(self.seeds),
            "default_seed": self.default_seed,
            "fixture_source": self.fixture_source,
            "generated_at": self.generated_at or utcnow().isoformat(),
            "statement": self.statement,
            "data_note": self.data_note,
        }


def collect_provenance(runtimes: Mapping[str, str], seeds: Sequence[int], *,
                       repo_root: Path | None = None,
                       prefer_real: bool = True) -> Provenance:
    """Assemble the provenance block for a sweep that already ran."""
    digest, included = code_hash(repo_root)
    all_pkgs = ("core", "agents", "decision", "blackboard", "tools",
                "observability", "graph", "api", "eval")
    root = repo_root or Path(__file__).resolve().parent.parent
    missing = [p for p in all_pkgs if not (root / p).is_dir()]
    try:
        from core.config import get_settings

        settings = get_settings()
        run_mode = settings.run_mode.value
        intended = settings.effective_backend()
    except Exception as exc:  # noqa: BLE001
        run_mode = f"{UNAVAILABLE} (settings import failed: {type(exc).__name__})"
        intended = UNAVAILABLE
    return Provenance(
        git_commit=git_commit(repo_root),
        code_sha256=digest,
        code_hash_inputs=included,
        packages_missing=missing,
        run_mode=run_mode,
        intended_decision_backend=intended,
        decision_backends=_probe_decision_backends(prefer_real),
        runtimes=dict(runtimes),
        seeds=list(seeds),
        generated_at=utcnow().isoformat(),
    )


# ==================================================================== records
@dataclass
class RunRecord:
    """One ``(scenario, condition, seed)`` execution plus the batch's context.

    Raw. The aggregates in :class:`AblationReport` are all recomputable from a list
    of these, and the report keeps them so a reader does not have to trust the
    aggregation.
    """

    batch_id: str
    scenario: str
    condition: str
    seed: int
    outcome: RunOutcome

    @property
    def ok(self) -> bool:
        return self.outcome.ok

    @property
    def path_key(self) -> str:
        return self.outcome.path_key

    def metric(self, name: str) -> float | None:
        """Numeric value of a metric, or ``None`` if this record has no value.

        Returns ``None`` rather than 0 for a field the run never produced, so an
        aggregate can tell "zero" from "not measured".
        """
        if not self.ok:
            return None
        value = getattr(self.outcome, name, None)
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        return {"batch_id": self.batch_id, **self.outcome.to_dict()}


@dataclass
class MetricStats:
    """Mean and sample standard deviation of one metric across seeds."""

    name: str
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float | None:
        if not self.values:
            return None
        return round(statistics.fmean(self.values), 4)

    @property
    def stdev(self) -> float | None:
        """Sample stdev. ``None`` for n < 2 - a stdev from one sample is noise
        presented as precision."""
        if len(self.values) < 2:
            return None
        return round(statistics.stdev(self.values), 4)

    @property
    def total(self) -> float | None:
        return round(sum(self.values), 4) if self.values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "mean": self.mean,
            "stdev": self.stdev,
            "total": self.total,
            "values": list(self.values),
        }


@dataclass
class RouterProbeResult:
    """How one condition routed one probe."""

    probe_label: str
    reply_text: str
    expected_agent: str
    expected_intent: str
    adversarial: bool
    routed_agent: str
    routed_intent: str
    agent_agreed: bool
    intent_agreed: bool
    decision_source: str
    classifier: str

    @classmethod
    def from_decision(cls, d: RouteDecision) -> RouterProbeResult:
        return cls(
            probe_label=d.probe_label, reply_text=d.reply_text,
            expected_agent=d.expected_agent.value, expected_intent=d.expected_intent.value,
            adversarial=d.adversarial, routed_agent=d.routed_agent.value,
            routed_intent=d.routed_intent.value, agent_agreed=d.agent_agreed,
            intent_agreed=d.intent_agreed, decision_source=d.decision_source.value,
            classifier=d.classifier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_label": self.probe_label,
            "reply_text": self.reply_text,
            "expected_agent": self.expected_agent,
            "expected_intent": self.expected_intent,
            "adversarial": self.adversarial,
            "routed_agent": self.routed_agent,
            "routed_intent": self.routed_intent,
            "agent_agreed": self.agent_agreed,
            "intent_agreed": self.intent_agreed,
            "decision_source": self.decision_source,
            "classifier": self.classifier,
        }


@dataclass
class RouterConfusion:
    """Routing comparison for one ``(scenario, condition)`` pair.

    Counts, not percentages. The spec is explicit about this, and it is also the
    honest choice: with 3-5 probes per scenario a percentage implies a sample size
    it does not have.
    """

    probes: int = 0
    agent_agreements: int = 0
    intent_agreements: int = 0
    adversarial_probes: int = 0
    adversarial_agent_misroutes: int = 0
    per_probe: list[RouterProbeResult] = field(default_factory=list)

    def add(self, r: RouterProbeResult) -> None:
        self.probes += 1
        self.agent_agreements += int(r.agent_agreed)
        self.intent_agreements += int(r.intent_agreed)
        self.adversarial_probes += int(r.adversarial)
        self.adversarial_agent_misroutes += int(r.adversarial and not r.agent_agreed)
        self.per_probe.append(r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probes": self.probes,
            "agent_agreements": self.agent_agreements,
            "intent_agreements": self.intent_agreements,
            "adversarial_probes": self.adversarial_probes,
            "adversarial_agent_misroutes": self.adversarial_agent_misroutes,
            "per_probe": [p.to_dict() for p in self.per_probe],
        }


@dataclass
class ConditionAggregate:
    """Every measurement for one ``(scenario, condition)`` pair."""

    scenario: str
    condition: str
    seeds_run: int = 0
    ok_runs: int = 0
    failed_runs: list[str] = field(default_factory=list)
    stats: dict[str, MetricStats] = field(default_factory=dict)
    #: THE headline number: distinct agent paths across seeds on identical input.
    path_variation: int | None = None
    distinct_paths: list[str] = field(default_factory=list)
    decision_source_mix: dict[str, int] = field(default_factory=dict)
    tool_status_mix: dict[str, int] = field(default_factory=dict)
    dispute_zones: list[str] = field(default_factory=list)
    blocking_flags: list[str] = field(default_factory=list)
    unverifiable_claims: list[str] = field(default_factory=list)
    router: RouterConfusion = field(default_factory=RouterConfusion)
    runtimes: dict[str, int] = field(default_factory=dict)
    path_variation_caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "condition": self.condition,
            "seeds_run": self.seeds_run,
            "ok_runs": self.ok_runs,
            "failed_runs": list(self.failed_runs),
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "path_variation": self.path_variation,
            "distinct_paths": list(self.distinct_paths),
            "decision_source_mix": dict(self.decision_source_mix),
            "tool_status_mix": dict(self.tool_status_mix),
            "dispute_zones": list(self.dispute_zones),
            "blocking_flags": list(self.blocking_flags),
            "unverifiable_claims": list(self.unverifiable_claims),
            "router": self.router.to_dict(),
            "runtimes": dict(self.runtimes),
            "path_variation_caveats": list(self.path_variation_caveats),
        }


@dataclass
class ScenarioAggregate:
    """All conditions on one scenario, plus the cross-condition router split."""

    scenario: str
    conditions: dict[str, ConditionAggregate] = field(default_factory=dict)
    #: ``probe_label -> {condition: routed_agent}``. Cross-condition disagreement
    #: is the direct answer to "did each condition route it to the same agent?".
    router_split: dict[str, dict[str, str]] = field(default_factory=dict)
    router_unanimous: dict[str, bool] = field(default_factory=dict)

    def conditions_in_order(self) -> list[str]:
        return list(self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
            "router_split": {k: dict(v) for k, v in self.router_split.items()},
            "router_unanimous": dict(self.router_unanimous),
        }


@dataclass
class AblationReport:
    """The complete result: raw records, per-pair aggregates, and a headline."""

    batch_id: str
    generated_at: str
    provenance: Provenance
    seeds: list[int]
    scenario_names: list[str]
    condition_names: list[str]
    #: Raw records. Present so every aggregate below is auditable.
    records: list[RunRecord] = field(default_factory=list)
    per_scenario: dict[str, ScenarioAggregate] = field(default_factory=dict)
    headline: dict[str, Any] = field(default_factory=dict)
    #: Set when any run failed, with the reasons. Never silently dropped.
    failures: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    def ok_records(self) -> list[RunRecord]:
        return [r for r in self.records if r.ok]

    def condition(self, scenario: str, condition: str) -> ConditionAggregate:
        try:
            return self.per_scenario[scenario].conditions[condition]
        except KeyError as exc:
            raise KeyError(
                f"no aggregate for scenario={scenario!r} condition={condition!r}; "
                f"have scenarios {self.scenario_names} x conditions {self.condition_names}"
            ) from exc

    def paths(self) -> list[str]:
        """Every distinct agent path observed, in order of first appearance."""
        seen: dict[str, None] = {}
        for r in self.ok_records():
            seen.setdefault(r.path_key, None)
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at,
            "seeds": list(self.seeds),
            "scenario_names": list(self.scenario_names),
            "condition_names": list(self.condition_names),
            "provenance": self.provenance.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "per_scenario": {k: v.to_dict() for k, v in self.per_scenario.items()},
            "headline": dict(self.headline),
            "failures": list(self.failures),
        }


# ================================================================= aggregation
def _aggregate_pair(scenario: Scenario, condition: Condition,
                    records: list[RunRecord]) -> ConditionAggregate:
    agg = ConditionAggregate(scenario=scenario.name, condition=condition.name)
    agg.seeds_run = len(records)
    ok = [r for r in records if r.ok]
    agg.ok_runs = len(ok)
    for r in records:
        if not r.ok:
            agg.failed_runs.append(f"seed={r.seed}: {r.outcome.failure_reason}")

    for metric in METRIC_FIELDS:
        stats = MetricStats(name=metric)
        stats.values = [v for v in (r.metric(metric) for r in ok) if v is not None]
        agg.stats[metric] = stats

    if ok:
        seen: dict[str, None] = {}
        for r in ok:
            seen.setdefault(r.path_key, None)
        agg.distinct_paths = list(seen)
        agg.path_variation = len(seen)
    else:
        # No completed run means no variation to report. 0 would claim the system
        # took exactly one path; None says we do not know.
        agg.path_variation = None

    for r in ok:
        for src, n in r.outcome.decisions_by_source.items():
            agg.decision_source_mix[src] = agg.decision_source_mix.get(src, 0) + n
        for st, n in r.outcome.tool_status_counts.items():
            agg.tool_status_mix[st] = agg.tool_status_mix.get(st, 0) + n
        agg.dispute_zones.extend(r.outcome.dispute_zones)
        agg.blocking_flags.extend(r.outcome.blocking_flags)
        agg.unverifiable_claims.extend(r.outcome.unverifiable_claims)
        if r.outcome.path_variation_caveat:
            agg.path_variation_caveats.append(r.outcome.path_variation_caveat)
        agg.runtimes[r.outcome.runtime.split("(")[0].strip()] = \
            agg.runtimes.get(r.outcome.runtime.split("(")[0].strip(), 0) + 1
        # Router probes are seed-independent by construction, so they are counted
        # once per (scenario, condition). Recording them once per seed would make
        # the probe count a function of how many seeds happened to be run, which is
        # a reporting artefact, not a finding.
        if not agg.router.per_probe:
            for d in r.outcome.route_decisions:
                agg.router.add(RouterProbeResult.from_decision(d))
    agg.dispute_zones = sorted(set(agg.dispute_zones))
    agg.blocking_flags = sorted(set(agg.blocking_flags))
    agg.unverifiable_claims = sorted(set(agg.unverifiable_claims))
    agg.path_variation_caveats = sorted(set(agg.path_variation_caveats))
    return agg


def _headline(report_records: list[RunRecord], condition_names: list[str],
              scenario_names: list[str]) -> dict[str, Any]:
    """The claim-shaped summary, with the shape of the claim stated explicitly.

    ``path_variation_by_condition`` is mean distinct paths per scenario. A fixed
    pipeline is 1.0 everywhere by construction; anything higher is evidence that
    the architecture produced more than one trajectory on identical input.

    **Path variation alone is not evidence of adaptivity.** A router that ping-pongs
    between two agents also varies, and varies more. So the headline block pairs
    every variation number with the three measurements that distinguish productive
    variation from oscillation:

    * ``stall_rate`` - runs that repeated an agent with no new state
    * ``dispute_rate`` - runs in which a disagreement was actually surfaced
    * ``completion_rate`` - runs that finished rather than crashing

    A condition only has a case if it varies *and* resolves: variation with a high
    stall rate is oscillation.
    """
    by_condition: dict[str, list[int | None]] = {c: [] for c in condition_names}
    for scenario in scenario_names:
        for cond in condition_names:
            paths: set[str] = set()
            complete = False
            for r in report_records:
                if r.scenario != scenario or r.condition != cond:
                    continue
                if r.ok:
                    paths.add(r.path_key)
                    complete = True
            by_condition[cond].append(len(paths) if complete else None)

    means: dict[str, Any] = {}
    for cond, vals in by_condition.items():
        measured = [v for v in vals if v is not None]
        means[cond] = {
            "per_scenario_path_variation": vals,
            "mean_path_variation": (round(statistics.fmean(measured), 4)
                                    if measured else None),
            "scenarios_measured": len(measured),
            "scenarios_unmeasured": len(vals) - len(measured),
        }

    health: dict[str, dict[str, Any]] = {}
    for cond in condition_names:
        cells = [r for r in report_records if r.condition == cond]
        ok = [r for r in cells if r.ok]
        stalls = [r for r in ok if r.outcome.stall_count > 0]
        disputed = [r for r in ok if r.outcome.disputes_raised > 0]
        resolved = [r for r in ok if r.outcome.disputes_resolved > 0]
        escalated = [r for r in ok if r.outcome.disputes_escalated > 0]
        replans = [r for r in ok if r.outcome.replans > 0]
        gated = [r for r in ok if r.outcome.human_gates_raised > 0]
        signed = [r for r in ok
                  if r.outcome.negotiation is not None and r.outcome.negotiation.signed]
        health[cond] = {
            "runs": len(cells),
            "ok_runs": len(ok),
            "completion_rate": round(len(ok) / len(cells), 4) if cells else None,
            "stall_rate": round(len(stalls) / len(ok), 4) if ok else None,
            "dispute_rate": round(len(disputed) / len(ok), 4) if ok else None,
            "dispute_resolution_rate": round(len(resolved) / len(disputed), 4)
            if disputed else None,
            "dispute_escalation_rate": round(len(escalated) / len(disputed), 4)
            if disputed else None,
            "replan_rate": round(len(replans) / len(ok), 4) if ok else None,
            "gate_rate": round(len(gated) / len(ok), 4) if ok else None,
            "signed_rate": round(len(signed) / len(ok), 4) if ok else None,
        }

    router_misroutes: dict[str, int] = {}
    router_adversarial: dict[str, int] = {}
    for r in report_records:
        if not r.ok:
            continue
        for d in r.outcome.route_decisions:
            if d.adversarial:
                router_adversarial[r.condition] = router_adversarial.get(r.condition, 0) + 1
            if not d.agent_agreed:
                router_misroutes[r.condition] = router_misroutes.get(r.condition, 0) + 1

    return {
        "headline_metric": "path_variation",
        "headline_definition": (
            "number of DISTINCT agent paths observed across seeds on byte-identical "
            "scenario inputs. A fixed pipeline has exactly 1 by construction; an "
            "adaptive system can exceed 1."
        ),
        "headline_caveat": (
            "path_variation is necessary but NOT sufficient evidence of adaptivity: a "
            "router that ping-pongs between two agents also varies. Read it "
            "together with stall_rate (oscillation), dispute_resolution_rate "
            "(does variation lead anywhere?) and completion_rate."
        ),
        "path_variation_by_condition": means,
        "run_health_by_condition": health,
        "router_agent_misroutes": router_misroutes,
        "router_adversarial_probes_evaluated": router_adversarial,
        "totals": {
            "runs": len(report_records),
            "ok_runs": sum(1 for r in report_records if r.ok),
            "failed_runs": sum(1 for r in report_records if not r.ok),
            "decision_calls": sum(r.outcome.decision_calls for r in report_records),
            "llm_calls": sum(r.outcome.llm_calls for r in report_records),
            "degraded_decisions": sum(r.outcome.degraded_decisions
                                      for r in report_records),
            "tool_calls": sum(r.outcome.tool_calls for r in report_records),
            "disputes_raised": sum(r.outcome.disputes_raised for r in report_records),
            "human_gates_raised": sum(r.outcome.human_gates_raised
                                      for r in report_records),
        },
    }


# ====================================================================== runner
def run_ablation(
    scenarios: Sequence[Scenario] = ALL_SCENARIOS,
    conditions: Sequence[Condition] = ALL_CONDITIONS,
    seeds: Sequence[int] = (DEFAULT_SEED,),
    *,
    batch_id: str | None = None,
    repo_root: Path | None = None,
) -> AblationReport:
    """Run every ``(scenario, condition, seed)`` cell and aggregate.

    Parameters
    ----------
    scenarios:
        Cases to run. Defaults to all five seeded scenarios.
    conditions:
        Control-flow regimes. Defaults to the three under comparison.
    seeds:
        Repeats per cell. One seed gives no spread, so the aggregate reports
        ``stdev = unavailable`` rather than 0.0.

    Returns
    -------
    AblationReport
        Raw records plus per-pair aggregates, cross-condition router splits, a
        headline block, and an explicit failure list.

    Never raises for a failed cell: a condition that cannot run contributes a
    record with ``ok=False`` and a reason. That is the difference between a
    measurement and a guess.
    """
    scen_list = list(scenarios)
    cond_list = list(conditions)
    seed_list = [int(s) for s in seeds]
    if not scen_list:
        raise ValueError("run_ablation needs at least one scenario")
    if not cond_list:
        raise ValueError("run_ablation needs at least one condition")
    if not seed_list:
        raise ValueError("run_ablation needs at least one seed")

    from core.ids import new_id

    bid = batch_id or new_id("abl")
    records: list[RunRecord] = []
    runtime_labels: dict[str, str] = {}

    for scenario in scen_list:
        for condition in cond_list:
            for seed in seed_list:
                outcome = condition.run(scenario, seed=seed)
                runtime_labels.setdefault(condition.name, outcome.runtime)
                records.append(RunRecord(batch_id=bid, scenario=scenario.name,
                                         condition=condition.name, seed=seed,
                                         outcome=outcome))

    per_scenario: dict[str, ScenarioAggregate] = {}
    for scenario in scen_list:
        sa = ScenarioAggregate(scenario=scenario.name)
        for condition in cond_list:
            cell = [r for r in records
                    if r.scenario == scenario.name and r.condition == condition.name]
            sa.conditions[condition.name] = _aggregate_pair(scenario, condition, cell)
        # Cross-condition router split for this scenario.
        for probe in scenario.reply_probes:
            split: dict[str, str] = {}
            for condition in cond_list:
                for r in records:
                    if r.scenario != scenario.name or r.condition != condition.name or not r.ok:
                        continue
                    for d in r.outcome.route_decisions:
                        if d.probe_label == probe.label:
                            split[condition.name] = d.routed_agent.value
                            break
            if split:
                sa.router_split[probe.label] = split
                sa.router_unanimous[probe.label] = len(set(split.values())) == 1
        per_scenario[scenario.name] = sa

    report = AblationReport(
        batch_id=bid,
        generated_at=utcnow().isoformat(),
        provenance=collect_provenance(runtime_labels, seed_list, repo_root=repo_root),
        seeds=seed_list,
        scenario_names=[s.name for s in scen_list],
        condition_names=[c.name for c in cond_list],
        records=records,
        per_scenario=per_scenario,
        failures=[f"{r.scenario}/{r.condition}/seed={r.seed}: {r.outcome.failure_reason}"
                  for r in records if not r.ok],
    )
    report.headline = _headline(records, report.condition_names, report.scenario_names)
    return report
