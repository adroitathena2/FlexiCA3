"""``graph`` — the LangGraph orchestration layer for Paytriq.

What lives here
---------------
The seven agents (``agents/``) know *how* to price, contact, contract, verify and
audit. This package knows *when*: what runs next, what that decision was, and
what must be proven before a side effect is allowed. It owns four things no
individual agent can own.

**Coordination that is recorded, not asserted.** Every routing decision produces a
:class:`~core.schemas.Handoff` naming its ``decision_source`` and ``confidence`
(:mod:`graph.edges`). That single field is the difference between "the agent
decided" as a claim and as evidence.

**Contested allocation.** :mod:`graph.contract_net` runs Smith's Contract Net
Protocol — announcement, bidding, awarding, expediting — so who executes a task is
an *award against competing, differently-framed bids* rather than a routing table
nobody can question.

**Contested disagreement.** :mod:`graph.debate` runs Du et al.'s multi-agent
debate with evidence that references blackboard entry ids, and escalates to a
human when the arbiter cannot find a zone of agreement. Included for
auditability; see that module's docstring for the honest caveat about accuracy.

**Gates before every side effect.** :mod:`graph.interrupts` uses LangGraph v1's
dynamic ``interrupt`` for SEND, COUNTER, MOU and ESCALATION. Auto-resolutions are
recorded with ``decided_by="auto"`` so an unattended run is never mistaken for an
attended one.

Also here: :mod:`graph.state` (typed state, reducers, Magentic-One ledgers,
runtime), :mod:`graph.registry` (defensive agent discovery), :mod:`graph.checkpointer`
(SQLite/in-memory), :mod:`graph.build` (wiring, compilation, ``mermaid()``), and
:mod:`graph.replay` (time travel, forking a run to take a different branch).

Dependency direction
--------------------
``core -> agents -> graph -> api``, enforced by ``tests/unit/test_contracts.py``.
This package imports from ``core`` and, for the base class only, from
``agents.base``. It never imports a named agent module — :mod:`graph.registry`
scans the package and reads each agent's declared ``id`` — so the orchestrator does
not break when an agent module is renamed, moved, or has not been written yet.
"""
from __future__ import annotations

from .build import PaytriqGraph, build_graph, build_runtime, initial_state
from .checkpointer import (
    CheckpointSetup,
    build_checkpointer,
    checkpoint_status,
    default_config,
    default_thread_id,
)
from .contract_net import AuctionResult, TaskAnnouncement, run_auction
from .debate import DebateResult, adjudicate, run_debate
from .edges import (
    LABEL,
    NODE,
    PATH_MAPS,
    route_after_compliance,
    route_after_discovery,
    route_after_proposal,
    route_after_reply,
    route_progress,
)
from .interrupts import (
    GateRequest,
    GateResult,
    gate_summary,
    resume_command,
    run_gate,
)
from .registry import (
    available_agents,
    build_agents,
    discover_agent_classes,
    discover_agents,
    missing_agents,
    registry_status,
)
from .replay import (
    CheckpointInfo,
    compare_branches,
    describe_run,
    find_checkpoint,
    fork_and_run,
    fork_from_checkpoint,
    list_checkpoints,
    load_checkpoint,
)
from .state import (
    GatePolicy,
    GraphRuntime,
    PaytriqState,
    ProgressLedger,
    RuntimeScratch,
    StepMemo,
    TaskLedger,
    blocking_flags,
    new_progress_ledger,
    new_task_ledger,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # build / run
    "PaytriqGraph", "build_graph", "build_runtime", "initial_state",
    # state
    "PaytriqState", "GraphRuntime", "GatePolicy", "TaskLedger", "ProgressLedger",
    "RuntimeScratch", "StepMemo", "new_task_ledger", "new_progress_ledger",
    "blocking_flags",
    # registry
    "discover_agents", "discover_agent_classes", "build_agents",
    "available_agents", "missing_agents", "registry_status",
    # routing
    "NODE", "LABEL", "PATH_MAPS",
    "route_after_discovery", "route_after_proposal", "route_after_reply",
    "route_after_compliance", "route_progress",
    # contract net
    "run_auction", "AuctionResult", "TaskAnnouncement",
    # debate
    "run_debate", "adjudicate", "DebateResult",
    # interrupts
    "run_gate", "GateRequest", "GateResult", "gate_summary", "resume_command",
    # checkpointer
    "build_checkpointer", "CheckpointSetup", "default_config", "default_thread_id",
    "checkpoint_status",
    # replay
    "list_checkpoints", "load_checkpoint", "find_checkpoint", "fork_from_checkpoint",
    "fork_and_run", "compare_branches", "describe_run", "CheckpointInfo",
]
