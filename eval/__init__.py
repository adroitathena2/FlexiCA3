"""Paytriq evaluation harness.

This package exists to produce the project's primary piece of evidence: a
measurement, not a claim, that the system is genuinely agentic rather than a
fixed pipeline with extra steps.

Layout
------
``scenarios``   Deterministic seeded fixtures. Every sponsored artefact carries
                ``source="fixture"`` and every brand is synthetic.
``conditions``  Three implementations of the same task that differ only in how
                control flow is chosen: a hardcoded pipeline, a naive keyword
                router, and the real decision-layer system.
``ablation``    Runs every ``(scenario, condition, seed)`` cell, aggregates
                across seeds, and reports ``path_variation`` - the headline
                number.
``benchmark``   Negotiation quality: deal rate, surplus split, policy violations,
                escalation rate.
``report``      Markdown and JSON rendering, both opening with a provenance block.

Integrity contract
------------------
Every figure in a report was produced by executing the system. Nothing is
hand-written. A figure that could not be measured renders as
:data:`~eval.ablation.UNAVAILABLE` (``"unavailable"``) and is never replaced with a
plausible substitute. A condition that cannot run is reported as a failure with
its reason, not as a row of zeros.

Importing this package is cheap and side-effect free: submodules are not imported
until you ask for them, so ``import eval`` never drags in the agent stack.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_SEED",
    "UNAVAILABLE",
    "Scenario",
    "Condition",
    "RunOutcome",
    "AblationReport",
    "BenchmarkReport",
    "ALL_SCENARIOS",
    "ALL_CONDITIONS",
    "run_ablation",
    "run_benchmark",
    "to_markdown",
    "to_json",
    "provenance_markdown",
    "__getattr__",
]


def __getattr__(name: str):
    """Lazy re-export.

    Submodules import each other and probe for optional packages; importing them
    eagerly at package import time would make ``import eval`` fail whenever any
    transitive optional dependency is mid-edit by another author. PEP 562 keeps
    the convenient ``eval.run_ablation`` surface without that coupling.
    """
    if name in ("DEFAULT_SEED", "Scenario", "ALL_SCENARIOS", "get_scenario"):
        from . import scenarios as _scenarios

        return getattr(_scenarios, name)
    if name in ("Condition", "RunOutcome", "ALL_CONDITIONS", "get_condition"):
        from . import conditions as _conditions

        return getattr(_conditions, name)
    if name in ("UNAVAILABLE", "AblationReport", "run_ablation", "Provenance"):
        from . import ablation as _ablation

        return getattr(_ablation, name)
    if name in ("BenchmarkReport", "run_benchmark"):
        from . import benchmark as _benchmark

        return getattr(_benchmark, name)
    if name in ("to_markdown", "to_json", "provenance_markdown"):
        from . import report as _report

        return getattr(_report, name)
    raise AttributeError(f"module 'eval' has no attribute {name!r}")
