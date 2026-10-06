"""Rendering: markdown for reading, JSON for auditing.

Both renderers share one rule that overrides tidiness: **a figure that was not
measured is printed as ``unavailable``.** Not ``0``, not ``N/A``, not omitted,
and never a plausible-looking substitute. The string is defined once, in
``eval.ablation.UNAVAILABLE``, so it cannot drift between the two formats.

Every report opens with a provenance block: git commit, code hash, run mode, the
decision backends that were actually reachable, and the runtime that produced the
runs. Without it a number in a report is anonymous, and an anonymous number in a
project that once shipped a hand-written trace is exactly the thing a reader
cannot check.

``to_markdown`` puts ``path_variation`` in the second column of the main table,
because it is the number the whole module exists to produce.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .ablation import (
    UNAVAILABLE,
    AblationReport,
    ConditionAggregate,
    MetricStats,
    Provenance,
    ScenarioAggregate,
)
from .benchmark import BenchmarkReport

__all__ = ["to_markdown", "to_json", "provenance_markdown", "provenance_dict",
           "format_value", "format_stat", "group_violations"]


# ==================================================================== helpers
def format_value(value: float | int | str | None, *, digits: int = 3) -> str:
    """Render a measured value, or ``unavailable`` when it was not measured."""
    if value is None or value == "":
        return UNAVAILABLE
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN: a computation failed, not a measurement
            return UNAVAILABLE
        if digits <= 0:
            # No stripping here: ``f"{120000.0:.0f}"`` is "120000", and rstrip-ing the
            # zeros would leave "12" - a plausible-looking but wrong rupee figure.
            return f"{value:.0f}"
        text = f"{value:.{digits}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def group_violations(items: Sequence[str], *, limit: int = 6) -> str:
    """Collapse ``"seed=11: X"`` entries into ``"X (x5)"`` for the markdown table.

    Groups by message and counts, because the per-seed prefix adds width without
    adding information once the count is shown. Nothing is dropped: if there are
    more than ``limit`` distinct messages, the overflow is summarised as a count
    rather than silently omitted, and the JSON export always carries the full list.
    """
    counts: dict[str, int] = {}
    for raw in items:
        message = raw.split(": ", 1)[1] if ": " in raw else raw
        counts[message] = counts.get(message, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"{msg} (x{n})" if n > 1 else msg for msg, n in ordered[:limit]]
    if len(ordered) > limit:
        parts.append(f"+{len(ordered) - limit} more (full list in the JSON export)")
    return "; ".join(parts) if parts else "none"


def format_stat(stats: MetricStats | None, *, digits: int = 3) -> str:
    """``mean ± stdev``, or just the mean, or ``unavailable``."""
    if stats is None or stats.n == 0:
        return UNAVAILABLE
    if stats.mean is None:
        return UNAVAILABLE
    mean = format_value(stats.mean, digits=digits)
    if stats.stdev is None:
        return f"{mean} ({UNAVAILABLE} sd)"
    return f"{mean} ± {format_value(stats.stdev, digits=digits)}"


def _escape(text: str) -> str:
    """Make a value safe inside a markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _mix_counts(mix: Mapping[str, int]) -> str:
    if not mix:
        return UNAVAILABLE
    return ", ".join(f"{k}={v}" for k, v in sorted(mix.items()))


def _pct(value: float | None) -> str:
    return UNAVAILABLE if value is None else f"{value * 100:.1f}%"


# ================================================================== provenance
def provenance_dict(prov: Provenance | None) -> dict[str, Any]:
    """Provenance as a dict, with every unknown rendered explicitly.

    Used by both renderers so the two can never disagree about what was measured.
    """
    if prov is None:
        return {
            "git_commit": UNAVAILABLE,
            "code_sha256": UNAVAILABLE,
            "code_hash_inputs": UNAVAILABLE,
            "run_mode": UNAVAILABLE,
            "intended_decision_backend": UNAVAILABLE,
            "decision_backends": {"status": UNAVAILABLE},
            "runtimes": {"status": UNAVAILABLE},
            "seeds": UNAVAILABLE,
            "statement": UNAVAILABLE,
            "data_note": UNAVAILABLE,
        }
    return prov.to_dict()


def provenance_markdown(prov: Provenance | None, *, title: str = "Provenance") -> str:
    """The provenance block. Every report starts with this."""
    p = provenance_dict(prov)
    backends = p.get("decision_backends") or {}
    backend_lines = "\n".join(f"  - `{k}`: {_escape(v)}" for k, v in sorted(backends.items())) \
        or f"  - `{UNAVAILABLE}`"
    runtimes = p.get("runtimes") or {}
    runtime_lines = "\n".join(f"  - `{k}`: {_escape(v)}" for k, v in sorted(runtimes.items())) \
        or f"  - `{UNAVAILABLE}`"
    inputs = p.get("code_hash_inputs")
    inputs_line = ", ".join(inputs) if isinstance(inputs, list) else str(inputs)
    missing = p.get("packages_missing")
    missing_line = ", ".join(missing) if isinstance(missing, list) else str(missing)
    seeds = p.get("seeds")
    seeds_line = ", ".join(str(s) for s in seeds) if isinstance(seeds, list) else str(seeds)

    return "\n".join([
        f"### {title}",
        "",
        f"- **Git commit**: `{_escape(p['git_commit'])}`",
        f"- **Code hash (sha256)**: `{_escape(p['code_sha256'])}`",
        f"- **Hashed packages**: {_escape(inputs_line)}",
        f"- **Packages absent from the tree**: {_escape(missing_line)}",
        f"- **Run mode**: `{_escape(p['run_mode'])}`",
        f"- **Intended decision backend**: `{_escape(p['intended_decision_backend'])}`",
        "- **Decision backends probed**:",
        backend_lines,
        "- **Runtimes used**:",
        runtime_lines,
        f"- **Seeds**: `{_escape(seeds_line)}`",
        f"- **Fixture source marker**: `{_escape(p.get('fixture_source', UNAVAILABLE))}`",
        f"- **Generated at**: `{_escape(p.get('generated_at', UNAVAILABLE))}`",
        "",
        f"**{_escape(p['statement'])}**",
        "",
        f"> {_escape(p['data_note'])}",
        "",
    ])


# ==================================================================== markdown
def _ablation_markdown(report: AblationReport) -> str:
    L: list[str] = []
    add = L.append

    add("# Ablation report: is the control flow adaptive?")
    add("")
    add(f"- **Batch**: `{report.batch_id}`")
    add(f"- **Generated**: `{report.generated_at}`")
    add(f"- **Seeds**: `{', '.join(str(s) for s in report.seeds)}`")
    add(f"- **Scenarios**: {', '.join(f'`{s}`' for s in report.scenario_names)}")
    add(f"- **Conditions**: {', '.join(f'`{c}`' for c in report.condition_names)}")
    add("")
    add(provenance_markdown(report.provenance))

    # ---- headline -------------------------------------------------------
    head = report.headline or {}
    add("## Headline: `path_variation`")
    add("")
    add("**Definition.** "
        f"{_escape(str(head.get('headline_definition', UNAVAILABLE)))}")
    add("")
    if head.get("headline_caveat"):
        add(f"**Read with.** {_escape(str(head['headline_caveat']))}")
        add("")
    add("A fixed pipeline scores exactly 1 *by construction* - it has no mechanism to "
        "produce a second path. Anything above 1 is evidence that the architecture "
        "produced more than one trajectory on byte-identical input.")
    add("")
    pv = head.get("path_variation_by_condition") or {}
    if pv:
        add("| Condition | Mean `path_variation` | Per-scenario | Scenarios measured |")
        add("| --- | --- | --- | --- |")
        for cond, data in pv.items():
            per = data.get("per_scenario_path_variation") or []
            per_txt = ", ".join(f"`{s.split('_')[0]}`={format_value(v)}" for s, v in
                                zip(report.scenario_names, per, strict=False))
            add(f"| `{cond}` | **{format_value(data.get('mean_path_variation'))}** "
                f"| {per_txt} | {data.get('scenarios_measured')} "
                f"({data.get('scenarios_unmeasured')} {UNAVAILABLE}) |")
        add("")
    else:
        add(f"_{UNAVAILABLE}: the sweep produced no path-variation data._")
        add("")

    health = head.get("run_health_by_condition") or {}
    if health:
        add("### Run health: what the variation actually consists of")
        add("")
        add("Variation on its own is weak evidence. A router that ping-pongs between two "
            "agents also varies. These three columns separate adaptation from oscillation: "
            "a system only has a case if it varies *and* resolves.")
        add("")
        add("| Condition | Runs | Completion | Stall rate (oscillation) | Dispute rate "
            "| Dispute resolution rate | Replan rate | Gate rate | Signed rate |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for cond, h in health.items():
            add(f"| `{cond}` | {h.get('runs', UNAVAILABLE)} "
                f"| {_pct(h.get('completion_rate'))} | {_pct(h.get('stall_rate'))} "
                f"| {_pct(h.get('dispute_rate'))} "
                f"| {_pct(h.get('dispute_resolution_rate'))} "
                f"| {_pct(h.get('replan_rate'))} | {_pct(h.get('gate_rate'))} "
                f"| {_pct(h.get('signed_rate'))} |")
        add("")
        add("A `dispute_resolution_rate` of `unavailable` means no run raised a dispute, "
            "not that resolution failed.")
        add("")

    totals = head.get("totals") or {}
    if totals:
        add("### Totals across all runs")
        add("")
        add("| Runs | Completed | Failed | Decision calls | LLM calls | Degraded decisions "
            "| Tool calls | Disputes | Human gates |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        add(f"| {totals.get('runs', UNAVAILABLE)} | {totals.get('ok_runs', UNAVAILABLE)} "
            f"| {totals.get('failed_runs', UNAVAILABLE)} "
            f"| {totals.get('decision_calls', UNAVAILABLE)} "
            f"| {totals.get('llm_calls', UNAVAILABLE)} "
            f"| {totals.get('degraded_decisions', UNAVAILABLE)} "
            f"| {totals.get('tool_calls', UNAVAILABLE)} "
            f"| {totals.get('disputes_raised', UNAVAILABLE)} "
            f"| {totals.get('human_gates_raised', UNAVAILABLE)} |")
        add("")

    # ---- per scenario ---------------------------------------------------
    add("## Per-scenario results")
    add("")
    add("`path_variation` is the second column deliberately.")
    add("")
    add("| Scenario | Condition | `path_variation` | Distinct paths | Steps | Decision calls "
        "| Disputes (r/x/e) | Replans | Stalls | Gates | Wall clock ms |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for sname in report.scenario_names:
        sa: ScenarioAggregate = report.per_scenario[sname]
        for cname in report.condition_names:
            agg: ConditionAggregate | None = sa.conditions.get(cname)
            if agg is None:
                add(f"| `{sname}` | `{cname}` | {UNAVAILABLE} | {UNAVAILABLE} | "
                    f"{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} | "
                    f"{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} |")
                continue
            paths = "<br>".join(f"`{_escape(p)}`" for p in agg.distinct_paths) or UNAVAILABLE
            if agg.failed_runs:
                add(f"| `{sname}` | `{cname}` | **{format_value(agg.path_variation)}** "
                    f"| {paths} | {format_stat(agg.stats.get('steps_taken'))} "
                    f"| {format_stat(agg.stats.get('decision_calls'))} "
                    f"| {format_stat(agg.stats.get('disputes_raised'))}/"
                    f"{format_stat(agg.stats.get('disputes_resolved'))}/"
                    f"{format_stat(agg.stats.get('disputes_escalated'))} "
                    f"| {format_stat(agg.stats.get('replans'))} "
                    f"| {format_stat(agg.stats.get('stall_count'))} "
                    f"| {format_stat(agg.stats.get('human_gates_raised'))} "
                    f"| {format_stat(agg.stats.get('wall_clock_ms'))} |")
            else:
                add(f"| `{sname}` | `{cname}` | **{format_value(agg.path_variation)}** "
                    f"| {paths} | {format_stat(agg.stats.get('steps_taken'))} "
                    f"| {format_stat(agg.stats.get('decision_calls'))} "
                    f"| {format_stat(agg.stats.get('disputes_raised'))}/"
                    f"{format_stat(agg.stats.get('disputes_resolved'))}/"
                    f"{format_stat(agg.stats.get('disputes_escalated'))} "
                    f"| {format_stat(agg.stats.get('replans'))} "
                    f"| {format_stat(agg.stats.get('stall_count'))} "
                    f"| {format_stat(agg.stats.get('human_gates_raised'))} "
                    f"| {format_stat(agg.stats.get('wall_clock_ms'))} |")
    add("")

    # ---- decision source / tool status ----------------------------------
    add("## Decision provenance and tool status")
    add("")
    add("Every decision names its source; there is no anonymous decision. Tool calls are "
        "broken down by status so fixture-backed calls are visible as such.")
    add("")
    add("| Scenario | Condition | Decision sources | Tool statuses | Runtimes |")
    add("| --- | --- | --- | --- | --- |")
    for sname in report.scenario_names:
        sa = report.per_scenario[sname]
        for cname in report.condition_names:
            agg = sa.conditions.get(cname)
            if agg is None:
                add(f"| `{sname}` | `{cname}` | {UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} |")
                continue
            # An empty mix with a *measured* zero is a finding, not a gap: the fixed
            # pipeline makes no decisions because it has no conditional logic. Only
            # an unmeasured cell earns 'unavailable', so the two are never confused.
            calls = agg.stats.get("decision_calls")
            if agg.decision_source_mix:
                src = _escape(_mix_counts(agg.decision_source_mix))
            elif calls is not None and calls.total == 0:
                src = "none - measured 0 decision calls"
            else:
                src = UNAVAILABLE
            add(f"| `{sname}` | `{cname}` | {src} "
                f"| {_escape(_mix_counts(agg.tool_status_mix))} "
                f"| {_escape(_mix_counts(agg.runtimes))} |")
    add("")
    add("`none - measured 0 decision calls` is a result, not a gap: with no "
        "conditional logic there is nothing to decide, and the count was measured.")
    add("")

    # ---- router comparison ------------------------------------------------
    add("## Routing comparison (the keyword-router failure, measured)")
    add("")
    add("Each scenario's probes were routed by every condition. Counts, not percentages: "
        "with a handful of probes per scenario a percentage would imply a sample size "
        "that does not exist.")
    add("")
    add("| Scenario | Condition | Probes | Agent agreed | Intent agreed | Adversarial probes "
        "| Adversarial misroutes |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for sname in report.scenario_names:
        sa = report.per_scenario[sname]
        for cname in report.condition_names:
            agg = sa.conditions.get(cname)
            if agg is None:
                add(f"| `{sname}` | `{cname}` | {UNAVAILABLE} | {UNAVAILABLE} | "
                    f"{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} |")
                continue
            r = agg.router
            add(f"| `{sname}` | `{cname}` | {r.probes} | {r.agent_agreements}/{r.probes} "
                f"| {r.intent_agreements}/{r.probes} | {r.adversarial_probes} "
                f"| **{r.adversarial_agent_misroutes}** |")
    add("")

    add("### Cross-condition disagreement, probe by probe")
    add("")
    add("`|>` means the conditions did not agree on which agent handles that reply.")
    add("")
    conds = report.condition_names
    add("| Scenario | Probe | " + " | ".join(f"`{c}`" for c in conds) + " | Unanimous |")
    add("| --- | --- | " + " | ".join("---" for _ in conds) + " | --- |")
    for sname in report.scenario_names:
        sa = report.per_scenario[sname]
        for probe, split in sa.router_split.items():
            cells = [f"`{split.get(c, UNAVAILABLE)}`" for c in conds]
            unan = "yes" if sa.router_unanimous.get(probe) else "**no**"
            add(f"| `{sname}` | `{probe}` | " + " | ".join(cells) + f" | {unan} |")
    add("")

    # ---- per-condition detail -------------------------------------------
    add("### Every probe, every condition, verbatim")
    add("")
    add("| Condition | Probe | Reply | Expected | Routed | Agent OK | Intent OK | Source |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    seen_probes: set[tuple[str, str]] = set()
    for sname in report.scenario_names:
        sa = report.per_scenario[sname]
        for cname in report.condition_names:
            agg = sa.conditions.get(cname)
            if agg is None:
                continue
            for p in agg.router.per_probe:
                key = (cname, f"{sname}:{p.probe_label}")
                if key in seen_probes:
                    continue
                seen_probes.add(key)
                add(f"| `{cname}` | `{p.probe_label}` | {_escape(p.reply_text[:70])} "
                    f"| `{p.expected_agent}` | `{p.routed_agent}` "
                    f"| {'OK' if p.agent_agreed else '**MISS**'} "
                    f"| {'OK' if p.intent_agreed else 'miss'} | `{p.decision_source}` |")
    add("")

    # ---- conflicts noticed ----------------------------------------------
    add("## Conflicts raised, per condition")
    add("")
    add("| Scenario | Condition | Dispute zones | Blocking flags | Unverifiable fulfilment claims |")
    add("| --- | --- | --- | --- | --- |")
    for sname in report.scenario_names:
        sa = report.per_scenario[sname]
        for cname in report.condition_names:
            agg = sa.conditions.get(cname)
            if agg is None:
                add(f"| `{sname}` | `{cname}` | {UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} |")
                continue
            zones = ", ".join(f"`{z}`" for z in agg.dispute_zones) or "none"
            flags = "<br>".join(f"`{_escape(f)}`" for f in agg.blocking_flags) or "none"
            claims = ", ".join(agg.unverifiable_claims) or "none"
            add(f"| `{sname}` | `{cname}` | {zones} | {flags} | {claims or 'none'} |")
    add("")

    # ---- caveats ---------------------------------------------------------
    caveats = sorted({c for sname in report.scenario_names
                      for cname in report.condition_names
                      for c in (report.per_scenario[sname].conditions.get(cname)
                                .path_variation_caveats
                                if report.per_scenario[sname].conditions.get(cname) else [])})
    add("## Interpretation limits")
    add("")
    if caveats:
        for c in caveats:
            add(f"- {_escape(c)}")
    else:
        add(f"- {_UNAVAILABLE_NOTE}")
    add("")

    # ---- failures --------------------------------------------------------
    add("## Run failures")
    add("")
    if report.failures:
        add(f"{len(report.failures)} run(s) did not complete. They are reported, not "
            "imputed:")
        add("")
        for f in report.failures:
            add(f"- {_escape(f)}")
    else:
        add("None. Every cell completed.")
    add("")

    # ---- raw records -----------------------------------------------------
    add("## Raw per-run records")
    add("")
    add("Retained in full so every aggregate above can be recomputed. The JSON export "
        "carries the complete records; the table below is the audit index.")
    add("")
    add("| # | Scenario | Condition | Seed | OK | Path | Steps | Dec | Disputes | Gates | "
        "Wall ms | Runtime |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, rec in enumerate(report.records):
        o = rec.outcome
        add(f"| {i} | `{rec.scenario}` | `{rec.condition}` | {rec.seed} "
            f"| {'yes' if o.ok else '**NO**'} | `{_escape(o.path_key) or 'n/a'}` "
            f"| {o.steps_taken if o.ok else UNAVAILABLE} "
            f"| {o.decision_calls if o.ok else UNAVAILABLE} "
            f"| {o.disputes_raised if o.ok else UNAVAILABLE} "
            f"| {o.human_gates_raised if o.ok else UNAVAILABLE} "
            f"| {format_value(o.wall_clock_ms) if o.ok else UNAVAILABLE} "
            f"| {_escape(o.runtime)[:80]} |")
    add("")
    return "\n".join(L)


_UNAVAILABLE_NOTE = (
    f"_{UNAVAILABLE}: no caveat text was recorded, which means the runs used a "
    "runtime whose results can be read directly as evidence about the system._"
)


def _benchmark_markdown(report: BenchmarkReport) -> str:
    L: list[str] = []
    add = L.append

    add("# Negotiation benchmark")
    add("")
    add(f"- **Batch**: `{report.batch_id}`")
    add(f"- **Generated**: `{report.generated_at}`")
    add(f"- **Seeds**: `{', '.join(str(s) for s in report.seeds)}`")
    add("")
    add(provenance_markdown(report.provenance))

    add("## Per-condition aggregate")
    add("")
    add("| Condition | Deal rate | Signed / attempted | Mean rounds to deal (n) "
        "| Surplus split (sponsor / organiser, n) | Policy violations | Signed in violation "
        "| Signed below walk-away | Escalation rate (gated / completed) "
        "| Fabrication warnings |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, cb in report.conditions.items():
        split = cb.surplus_split_mean
        split_txt = (f"{format_value(split[0])} / {format_value(split[1])} "
                     f"({cb.n_splits_measured})" if split else f"{UNAVAILABLE} (0)")
        add(f"| `{name}` | **{_pct(cb.deal_rate)}** | {cb.signed_runs}/{cb.seeds_run} "
            f"| {format_value(cb.mean_rounds_to_deal)} ({cb.n_rounds_measured}) "
            f"| {split_txt} | {cb.policy_violations_total} "
            f"| **{cb.signed_in_violation_total}** | **{cb.below_walkaway_total}** "
            f"| {_pct(cb.escalation_rate)} | {cb.fabrication_warnings_total} |")
    add("")

    add("### Definitions")
    add("")
    head = report.headline or {}
    defs = head.get("definitions") or {}
    for k in ("deal_rate", "escalation_rate", "surplus_split"):
        add(f"- **{k}**: {_escape(str(defs.get(k, UNAVAILABLE)))}")
    add(f"- **{UNAVAILABLE}**: a figure that could not be measured. It is never replaced "
        "with a substitute.")
    add("")

    add("## Per-scenario detail")
    add("")
    add("| Scenario | Condition | Runs | Completed | Signed | Deal rate | Rounds "
        "| Final amounts (INR) | Violations | Below walk-away | Gated | Failures |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for sname in report.scenario_names:
        for cname, cb in report.conditions.items():
            cell = cb.cells.get(sname)
            if cell is None:
                add(f"| `{sname}` | `{cname}` | {UNAVAILABLE} | {UNAVAILABLE} | "
                    f"{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} | "
                    f"{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE} |")
                continue
            amounts = ", ".join(format_value(a, digits=0) for a in cell.final_amounts) or "-"
            rounds = ", ".join(str(r) for r in cell.rounds) or "-"
            viol = _escape(group_violations(cell.policy_violations))
            # 'none' only when a final amount actually existed to compare; otherwise
            # the comparison was not made and 'unavailable' is the honest cell.
            walk = (str(len(cell.below_walkaway)) if cell.below_walkaway
                    else "none" if cell.final_amounts else UNAVAILABLE)
            gated = len(cell.gated_runs)
            fails = "; ".join(_escape(f) for f in cell.failures) or "none"
            add(f"| `{sname}` | `{cname}` | {cell.seeds_run} | {cell.completed_runs} "
                f"| {cell.signed_runs} | {_pct(cell.deal_rate)} | {rounds} | {amounts} "
                f"| {viol} | {walk} | {gated} | {fails} |")
    add("")

    add("## Explicit failures")
    add("")
    any_fail = False
    for cname, cb in report.conditions.items():
        if cb.explicit_failures:
            any_fail = True
            add(f"**`{cname}`**")
            add("")
            for f in cb.explicit_failures:
                add(f"- {_escape(f)}")
            add("")
    if not any_fail:
        add("None: every condition completed every cell.")
        add("")

    add("## Fabrication warnings")
    add("")
    found = False
    for cname, cb in report.conditions.items():
        for sname, cell in cb.cells.items():
            if cell.fabrication_warnings:
                found = True
                add(f"- `{cname}` / `{sname}`: " + "; ".join(_escape(w) for w in
                                                            cell.fabrication_warnings))
    if not found:
        add("None: no condition claimed a deliverable was fulfilled in a scenario that "
            "declares no evidence for it.")
    add("")
    return "\n".join(L)


# ======================================================================= entry
def to_markdown(report: AblationReport | BenchmarkReport) -> str:
    """Render either report type as markdown, provenance block first.

    Accepts an :class:`~eval.ablation.AblationReport` or a
    :class:`~eval.benchmark.BenchmarkReport`. Anything else raises ``TypeError``
    rather than guessing, because a renderer that silently accepts the wrong type
    will produce a confidently wrong document.
    """
    if isinstance(report, AblationReport):
        return _ablation_markdown(report)
    if isinstance(report, BenchmarkReport):
        return _benchmark_markdown(report)
    raise TypeError(
        f"to_markdown expects an AblationReport or BenchmarkReport, got "
        f"{type(report).__name__}")


def to_json(report: AblationReport | BenchmarkReport) -> str:
    """Full-fidelity JSON: raw records, every aggregate, full provenance."""
    if isinstance(report, (AblationReport, BenchmarkReport)):
        payload = report.to_dict()
    else:
        raise TypeError(
            f"to_json expects an AblationReport or BenchmarkReport, got "
            f"{type(report).__name__}")
    return json.dumps(payload, indent=2, sort_keys=False, default=str)
