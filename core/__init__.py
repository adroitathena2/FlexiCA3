"""Paytriq core: typed contracts shared by every module.

Import rule enforced across the codebase: **modules import from ``core``, never
from each other.** ``core`` imports nothing internal and performs no I/O.
"""
from __future__ import annotations

from .config import REPO_ROOT, Settings, get_settings, reset_settings_cache
from .errors import (
    AgentError,
    ArbiterEscalation,
    BlackboardError,
    BudgetExceeded,
    BudgetExhausted,
    ConfigError,
    DecisionFailed,
    DecisionUnavailable,
    DisputeUnresolved,
    HumanGateRequired,
    LowConfidence,
    NoValidPlan,
    PaytriqError,
    ReplayIntegrityError,
    SchemaError,
    ToolFailed,
    ToolTimeout,
    ToolUnavailable,
    ZoneNotFound,
)
from .ids import new_id, run_id, span_id, trace_id, utcnow
from .protocols import (
    REASONING_AGENT_IDS,
    TOOL_REGISTRY,
    ActResult,
    Agent,
    AgentContext,
    Blackboard,
    BoardEntry,
    DecisionBackend,
    HumanGateProvider,
    Observation,
    Plan,
    Reflection,
    ReplayStore,
    Tool,
    ToolResult,
    Tracer,
    classify_intent_fallback,
)
from .schemas import (
    REASONING_AGENTS,
    AgentId,
    Approval,
    AuditFinding,
    Bid,
    BrandLead,
    Decision,
    DecisionRequest,
    DecisionSource,
    Deliverable,
    Dispute,
    DisputeStatus,
    DisputeZone,
    EventProfile,
    GateKind,
    GateOutcome,
    Handoff,
    HumanDecision,
    HumanGate,
    Intent,
    Lesson,
    MoU,
    Offer,
    QuestionType,
    RiskFlag,
    ROIReport,
    RunMode,
    Severity,
    SpanRecord,
    SpanStatus,
    Thread,
    ToolStatus,
    TraceEvent,
    TraceKind,
    TraceSummary,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "Settings", "get_settings", "reset_settings_cache", "REPO_ROOT",
    # ids
    "new_id", "utcnow", "run_id", "span_id", "trace_id",
    # errors
    "PaytriqError", "ConfigError", "SchemaError", "DecisionUnavailable",
    "DecisionFailed", "LowConfidence", "ToolUnavailable", "ToolFailed",
    "ToolTimeout", "AgentError", "NoValidPlan", "BudgetExhausted",
    "BlackboardError", "ZoneNotFound", "DisputeUnresolved", "ArbiterEscalation",
    "HumanGateRequired", "ReplayIntegrityError", "BudgetExceeded",
    # protocols
    "Tool", "ToolResult", "TOOL_REGISTRY", "DecisionBackend", "Blackboard",
    "BoardEntry", "Tracer", "Agent", "AgentContext", "Plan", "ActResult",
    "Observation", "Reflection", "ReplayStore", "HumanGateProvider",
    "classify_intent_fallback", "REASONING_AGENT_IDS",
    # schemas
    "AgentId", "REASONING_AGENTS", "Intent", "DisputeZone", "Severity",
    "DisputeStatus", "DecisionSource", "RunMode", "QuestionType", "TraceKind",
    "SpanStatus", "GateKind", "GateOutcome", "ToolStatus",
    "EventProfile", "BrandLead", "Offer", "Thread", "MoU", "Deliverable",
    "ROIReport", "AuditFinding", "RiskFlag", "Dispute", "Bid", "Lesson",
    "Handoff", "DecisionRequest", "Decision", "TraceEvent", "SpanRecord",
    "TraceSummary", "HumanGate", "HumanDecision", "Approval",
]
