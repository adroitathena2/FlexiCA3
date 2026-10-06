"""Typed exception hierarchy.

Design rule: **fail loud in the domain, degrade gracefully at the edges.**

Agent decision logic raises these. The decision layer converts them into a
`DecisionSource.RULES` fallback *and records that it did so*, so a degraded run
is visible in the trace rather than silently indistinguishable from a real one.

This is the whole point. A system that quietly substitutes a fake for a model
call cannot be audited; one that substitutes *and labels the substitution* can.
"""
from __future__ import annotations

__all__ = [
    "PaytriqError",
    "ConfigError",
    "SchemaError",
    "DecisionUnavailable",
    "DecisionFailed",
    "LowConfidence",
    "ToolUnavailable",
    "ToolFailed",
    "ToolTimeout",
    "AgentError",
    "NoValidPlan",
    "BudgetExhausted",
    "BlackboardError",
    "ZoneNotFound",
    "DisputeUnresolved",
    "ArbiterEscalation",
    "HumanGateRequired",
    "ReplayIntegrityError",
    "BudgetExceeded",
]


class PaytriqError(Exception):
    """Base for everything this package raises deliberately."""


# --------------------------------------------------------------------------- config
class ConfigError(PaytriqError):
    """Invalid or contradictory configuration; refuse to start."""


class SchemaError(PaytriqError):
    """An artefact failed validation."""


# ----------------------------------------------------------------------- decision
class DecisionUnavailable(PaytriqError):
    """A decision backend cannot be reached (no key, no server, network down).

    Callers should fall back and set ``Decision.source = RULES``.
    """


class DecisionFailed(PaytriqError):
    """A decision backend was reachable but returned an unusable result."""


class LowConfidence(PaytriqError):
    """Confidence fell below the configured gate; escalation is required."""

    def __init__(self, confidence: float, threshold: float, question: str = "") -> None:
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"confidence {confidence:.3f} below threshold {threshold:.3f}"
            + (f" for question {question!r}" if question else "")
        )


# ---------------------------------------------------------------------------- tool
class ToolUnavailable(PaytriqError):
    """A tool is not installed or not configured for this run."""


class ToolFailed(PaytriqError):
    """A tool ran and failed."""


class ToolTimeout(ToolFailed):
    """A tool exceeded its deadline."""


# --------------------------------------------------------------------------- agent
class AgentError(PaytriqError):
    """An agent failed in a way its supervisor cannot route around."""


class NoValidPlan(AgentError):
    """The planner could not produce a schema-valid plan."""


class BudgetExhausted(AgentError):
    """The agent hit its step or token budget before reaching a verdict."""


# ---------------------------------------------------------------------- blackboard
class BlackboardError(PaytriqError):
    """Base for blackboard faults."""


class ZoneNotFound(BlackboardError):
    """Referenced zone does not exist."""


# ------------------------------------------------------------------- coordination
class DisputeUnresolved(PaytriqError):
    """A dispute survived the maximum number of debate rounds."""


class ArbiterEscalation(PaytriqError):
    """The arbiter could not find a zone of agreement and escalated to a human."""

    def __init__(self, dispute_id: str, reason: str) -> None:
        self.dispute_id = dispute_id
        super().__init__(f"dispute {dispute_id} escalated: {reason}")


class HumanGateRequired(PaytriqError):
    """A side-effecting action was attempted without human approval.

    This is a *control-flow* signal, not a bug: the graph catches it, calls
    ``interrupt()``, and resumes once a human decides.
    """


# ------------------------------------------------------------------- replay/budget
class ReplayIntegrityError(PaytriqError):
    """A recorded run failed its integrity check and must not be presented as real."""


class BudgetExceeded(PaytriqError):
    """Run exceeded a configured wall-clock, token, or cost ceiling."""
