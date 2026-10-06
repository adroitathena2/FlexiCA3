"""The negotiation benchmark: can each condition actually close a deal correctly?

Different question from the ablation. The ablation asks whether the architecture
adapts; this asks what the adaptation is *worth* in money, compliance, and human
attention.

Five measures, all computed from executed runs:

``deal_rate``
    Signed runs / runs attempted. Denominator includes failures, because a run
    that crashed did not produce a signed contract. The explicit failure list is
    reported beside it so a low deal rate cannot be mistaken for a refusal to
    deal when it is really a system that could not finish.
``mean_rounds_to_deal``
    Mean rounds to signature over signed runs only, with ``n`` printed. Unavailable
    when nothing was signed - not 0.0, which would read as "signed immediately".
``surplus_split``
    The bargaining gap ``opening_ask - sponsor_reservation`` divided between the
    two sides. The sponsor captures ``final - reservation``; the organiser
    captures ``ask - final``; the two sum to the gap, so the split is zero-sum and
    the shares sum to 1. A condition that signs at the opening ask has not
    negotiated; the number shows that.
``policy_violations``
    Clause breaches found by an auditor that does **not** trust the condition:
    :meth:`~eval.conditions.ContractTerms.violations` is evaluated against the
    scenario's caps after the run. Signed contracts with breaches are additionally
    flagged ``signed_in_violation``, which is the finding that matters.
``escalation_rate``
    Runs in which a human gate was actually required, over completed runs. Gated
    runs are also listed individually, and a gate resolved automatically in an
    unattended sweep is stamped ``decided_by="auto"`` so it is never counted as a
    human decision.

Honesty rules specific to this module
-------------------------------------
* A run that raises is a ``failure`` string. It is never folded into a zero.
* ``fabrication_warnings`` counts deliverables a system claimed were fulfilled
  while the scenario declared no evidence for them. On S5 this should be zero for
  every condition; a non-zero value is a fabricated result, not a metric.
* Surplus is only computed over runs where it is well-defined; the count ``n`` is
  always printed beside it.
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.ids import new_id, utcnow

from .ablation import UNAVAILABLE, Provenance, RunRecord
from .conditions import (
    ALL_CONDITIONS,
    Condition,
    RunOutcome,
    surplus_split,
)
from .scenarios import ALL_SCENARIOS, DEFAULT_SEED, Scenario

__all__ = [
    "ScenarioDeals",
    "ConditionBenchmark",
    "BenchmarkReport",
    "run_benchmark",
    "benchmark_from_records",
    "surplus_split",
]


@dataclass
class ScenarioDeals:
    """Benchmark results for one ``(scenario, condition)`` pair."""

    scenario: str
    condition: str
    seeds_run: int = 0
    completed_runs: int = 0
    signed_runs: int = 0
    rounds: list[int] = field(default_factory=list)
    #: ``(sponsor_share, organiser_share)`` per signed run, in run order.
    splits: list[tuple[float, float]] = field(default_factory=list)
    final_amounts: list[float] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    signed_in_violation: list[str] = field(default_factory=list)
    #: Runs that signed *below* the organiser's own walk-away. Not a policy
    #: breach, but strictly worse: the system took less than it had already decided
    #: it could not afford to take. Counted separately so it cannot hide inside a
    #: healthy-looking deal rate.
    below_walkaway: list[str] = field(default_factory=list)
    gated_runs: list[str] = field(default_factory=list)
    escalation_flags: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    fabrication_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def deal_rate(self) -> float | None:
        if self.seeds_run == 0:
            return None
        return round(self.signed_runs / self.seeds_run, 4)

    @property
    def escalation_rate(self) -> float | None:
        if self.completed_runs == 0:
            return None
        return round(len(self.gated_runs) / self.completed_runs, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "condition": self.condition,
            "seeds_run": self.seeds_run,
            "completed_runs": self.completed_runs,
            "signed_runs": self.signed_runs,
            "deal_rate": self.deal_rate,
            "escalation_rate": self.escalation_rate,
            "rounds": list(self.rounds),
            "mean_rounds_to_deal": (round(statistics.fmean(self.rounds), 4)
                                    if self.rounds else None),
            "n_rounds_measured": len(self.rounds),
            "splits": [list(s) for s in self.splits],
            "n_splits_measured": len(self.splits),
            "final_amounts": list(self.final_amounts),
            "policy_violations": list(self.policy_violations),
            "signed_in_violation": list(self.signed_in_violation),
            "below_walkaway": list(self.below_walkaway),
            "gated_runs": list(self.gated_runs),
            "escalation_flags": list(self.escalation_flags),
            "failures": list(self.failures),
            "fabrication_warnings": list(self.fabrication_warnings),
            "notes": list(self.notes),
        }


@dataclass
class ConditionBenchmark:
    """Aggregate benchmark results for one condition, across all scenarios."""

    condition: str
    cells: dict[str, ScenarioDeals] = field(default_factory=dict)
    #: Failures that prevented a scenario from being completed at all. Promoted
    #: out of the cells so they cannot be lost in an aggregate.
    explicit_failures: list[str] = field(default_factory=list)

    @property
    def seeds_run(self) -> int:
        return sum(c.seeds_run for c in self.cells.values())

    @property
    def completed_runs(self) -> int:
        return sum(c.completed_runs for c in self.cells.values())

    @property
    def signed_runs(self) -> int:
        return sum(c.signed_runs for c in self.cells.values())

    @property
    def deal_rate(self) -> float | None:
        return round(self.signed_runs / self.seeds_run, 4) if self.seeds_run else None

    @property
    def escalation_rate(self) -> float | None:
        """Runs requiring at least one human gate, over completed runs."""
        gated = sum(len(c.gated_runs) for c in self.cells.values())
        return round(gated / self.completed_runs, 4) if self.completed_runs else None

    @property
    def mean_rounds_to_deal(self) -> float | None:
        rounds = [r for c in self.cells.values() for r in c.rounds]
        return round(statistics.fmean(rounds), 4) if rounds else None

    @property
    def n_rounds_measured(self) -> int:
        return sum(len(c.rounds) for c in self.cells.values())

    @property
    def surplus_split_mean(self) -> tuple[float, float] | None:
        splits = [s for c in self.cells.values() for s in c.splits]
        if not splits:
            return None
        return (round(statistics.fmean(s[0] for s in splits), 4),
                round(statistics.fmean(s[1] for s in splits), 4))

    @property
    def n_splits_measured(self) -> int:
        return sum(len(c.splits) for c in self.cells.values())

    @property
    def policy_violations_total(self) -> int:
        return sum(len(c.policy_violations) for c in self.cells.values())

    @property
    def signed_in_violation_total(self) -> int:
        return sum(len(c.signed_in_violation) for c in self.cells.values())

    @property
    def below_walkaway_total(self) -> int:
        return sum(len(c.below_walkaway) for c in self.cells.values())

    @property
    def fabrication_warnings_total(self) -> int:
        return sum(len(c.fabrication_warnings) for c in self.cells.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "seeds_run": self.seeds_run,
            "completed_runs": self.completed_runs,
            "signed_runs": self.signed_runs,
            "deal_rate": self.deal_rate,
            "escalation_rate": self.escalation_rate,
            "mean_rounds_to_deal": self.mean_rounds_to_deal,
            "n_rounds_measured": self.n_rounds_measured,
            "surplus_split_mean": (None if self.surplus_split_mean is None
                                   else list(self.surplus_split_mean)),
            "n_splits_measured": self.n_splits_measured,
            "policy_violations_total": self.policy_violations_total,
            "signed_in_violation_total": self.signed_in_violation_total,
            "below_walkaway_total": self.below_walkaway_total,
            "fabrication_warnings_total": self.fabrication_warnings_total,
            "explicit_failures": list(self.explicit_failures),
            "cells": {k: v.to_dict() for k, v in self.cells.items()},
        }


@dataclass
class BenchmarkReport:
    """Full benchmark: per-condition aggregates, per-cell detail, provenance."""

    batch_id: str
    generated_at: str
    provenance: Provenance
    seeds: list[int]
    scenario_names: list[str]
    conditions: dict[str, ConditionBenchmark] = field(default_factory=dict)
    headline: dict[str, Any] = field(default_factory=dict)

    def condition(self, name: str) -> ConditionBenchmark:
        try:
            return self.conditions[name]
        except KeyError as exc:
            raise KeyError(
                f"no benchmark for condition {name!r}; have {sorted(self.conditions)}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at,
            "seeds": list(self.seeds),
            "scenario_names": list(self.scenario_names),
            "provenance": self.provenance.to_dict(),
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
            "headline": dict(self.headline),
        }


# ======================================================================= core
def _cell_for(scenario: Scenario, records: list[RunRecord]) -> ScenarioDeals:
    cell = ScenarioDeals(scenario=scenario.name, condition=records[0].condition
                         if records else UNAVAILABLE)
    cell.seeds_run = len(records)
    bounds = scenario.negotiation
    for rec in records:
        label = f"seed={rec.seed}"
        out: RunOutcome = rec.outcome
        if not out.ok:
            cell.failures.append(f"{label}: {out.failure_reason}")
            continue
        cell.completed_runs += 1
        if out.unverifiable_claims:
            cell.fabrication_warnings.extend(
                f"{label}: claimed fulfilled with no evidence: {p}"
                for p in out.unverifiable_claims)
        neg = out.negotiation
        if neg is None:
            if bounds is None:
                cell.notes.append(
                    f"{label}: scenario defines no negotiation bounds; nothing to sign")
            else:
                cell.failures.append(
                    f"{label}: run completed but produced no negotiation outcome")
            continue
        if neg.policy_violations:
            cell.policy_violations.extend(
                f"{label}: {v}" for v in neg.policy_violations)
        if neg.gates_raised:
            cell.gated_runs.extend(f"{label}: {g}" for g in neg.gates_raised)
        if neg.escalation:
            cell.escalation_flags.append(label)
        if not neg.signed:
            cell.notes.append(
                f"{label}: not signed "
                f"(violations={len(neg.policy_violations)}, "
                f"gates={len(neg.gates_raised)}, "
                f"amount={neg.final_amount_inr})")
            continue
        cell.signed_runs += 1
        if bounds is not None and neg.final_amount_inr is not None \
                and neg.final_amount_inr < bounds.walkaway_inr:
            cell.below_walkaway.append(
                f"{label}: signed at INR {neg.final_amount_inr}, below the organiser's "
                f"own walk-away of INR {bounds.walkaway_inr}")
        if neg.rounds_to_deal is not None:
            cell.rounds.append(neg.rounds_to_deal)
        if neg.final_amount_inr is not None:
            cell.final_amounts.append(neg.final_amount_inr)
        if neg.signed_in_violation:
            cell.signed_in_violation.append(
                f"{label}: signed at INR {neg.final_amount_inr} with "
                f"{len(neg.policy_violations)} policy breach(es)")
        if bounds is not None and neg.final_amount_inr is not None:
            split = surplus_split(bounds, neg.final_amount_inr)
            if split is not None:
                cell.splits.append(split)
            else:
                cell.notes.append(
                    f"{label}: final INR {neg.final_amount_inr} is outside "
                    f"[{bounds.sponsor_reservation_inr}, {bounds.opening_ask_inr}]; "
                    "surplus share is undefined and is not reported for this run")
    return cell


def benchmark_from_records(records: list[RunRecord], scenarios: Sequence[Scenario],
                           condition_names: Sequence[str],
                           provenance: Provenance) -> BenchmarkReport:
    """Aggregate an *existing* set of runs into a benchmark.

    Takes records rather than running the system itself so the benchmark and the
    ablation of the same batch cannot disagree. ``run_benchmark`` is the
    convenience wrapper that executes and then calls this.
    """
    bench = BenchmarkReport(
        batch_id=records[0].batch_id if records else new_id("bnc"),
        generated_at=utcnow().isoformat(),
        provenance=provenance,
        seeds=sorted({r.seed for r in records}),
        scenario_names=[s.name for s in scenarios],
    )
    for cond in condition_names:
        cb = ConditionBenchmark(condition=cond)
        for scenario in scenarios:
            cell_records = [r for r in records
                            if r.scenario == scenario.name and r.condition == cond]
            if not cell_records:
                cb.explicit_failures.append(
                    f"{scenario.name}: no runs were executed for condition {cond!r}")
                cb.cells[scenario.name] = ScenarioDeals(
                    scenario=scenario.name, condition=cond, seeds_run=0)
                continue
            cell = _cell_for(scenario, cell_records)
            if cell.seeds_run and cell.completed_runs == 0:
                cb.explicit_failures.append(
                    f"{scenario.name}/{cond}: {cell.seeds_run} run(s) attempted, "
                    f"{cell.completed_runs} completed; failures: "
                    + "; ".join(cell.failures))
            cb.cells[scenario.name] = cell
        bench.conditions[cond] = cb

    bench.headline = {
        "deal_rate": {c: bench.conditions[c].deal_rate for c in bench.conditions},
        "escalation_rate": {c: bench.conditions[c].escalation_rate
                            for c in bench.conditions},
        "mean_rounds_to_deal": {c: bench.conditions[c].mean_rounds_to_deal
                                for c in bench.conditions},
        "policy_violations_total": {c: bench.conditions[c].policy_violations_total
                                    for c in bench.conditions},
        "signed_in_violation_total": {c: bench.conditions[c].signed_in_violation_total
                                      for c in bench.conditions},
        "below_walkaway_total": {c: bench.conditions[c].below_walkaway_total
                                 for c in bench.conditions},
        "fabrication_warnings_total": {c: bench.conditions[c].fabrication_warnings_total
                                       for c in bench.conditions},
        "explicit_failures": {c: bench.conditions[c].explicit_failures
                              for c in bench.conditions},
        "definitions": {
            "deal_rate": "signed runs / runs attempted (failures included in the denominator)",
            "escalation_rate": "runs with >=1 human gate / completed runs",
            "surplus_split": "(sponsor_share, organiser_share) of "
                             "(opening_ask - sponsor_reservation)",
            "below_walkaway": "signed runs whose final amount is below the "
                              "organiser's own stated walk-away",
            "unavailable": UNAVAILABLE,
        },
    }
    return bench


def run_benchmark(
    scenarios: Sequence[Scenario] = ALL_SCENARIOS,
    conditions: Sequence[Condition] = ALL_CONDITIONS,
    seeds: Sequence[int] = (DEFAULT_SEED,),
    *,
    batch_id: str | None = None,
    repo_root: Path | None = None,
) -> BenchmarkReport:
    """Execute the sweep and benchmark it.

    Runs the system once and aggregates, so the benchmark costs the same wall-clock
    as an ablation of the same cells and the two reports can be reconciled.
    """
    from .ablation import run_ablation

    report = run_ablation(scenarios, conditions, seeds, batch_id=batch_id,
                          repo_root=repo_root)
    return benchmark_from_records(report.records, list(scenarios),
                                  [c.name for c in conditions], report.provenance)
