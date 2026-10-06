"""Time travel: rewind to a decision, then take the branch you did not take.

What this module buys the demo
------------------------------
The strongest claim a multi-agent system can make is not "it decided well" but
"you can check it decided for the stated reason". This module makes that checkable
interactively: pick the checkpoint where A3 classified a sponsor's pushback, fork
the run from that exact state with a *different* decision available, and watch
the graph go somewhere else while the original run stays exactly as it was.

How it works, precisely
-----------------------
LangGraph checkpoints every superstep. :meth:`CompiledStateGraph.get_state_history`
returns them newest-first, each with a ``checkpoint_id`` and a ``next`` tuple
naming the node that was about to run. Two operations are possible:

* **Replay in place** — ``invoke(None, {..., "checkpoint_id": cid})`` re-executes
  from that checkpoint on the *same* thread, appending new checkpoints. The
  previous branch's history is preserved in the checkpoint list, but the thread's
  head moves. Useful for "show me that step again", destructive for "compare".
* **Fork** — copy the checkpoint's values into a *new* thread with
  ``update_state(..., as_node=<the node that was pending>)``, then run. The
  original thread is untouched and both branches can be read side by side.

:func:`fork_from_checkpoint` implements the fork, because "rewind to the pushback
and take a different branch" means *keep* the branch you already have. It also
accepts a replacement :class:`~core.schemas.DecisionRequest` answer list, so the
fork differs by **decision**, not just by field value — which is the honest kind
of counterfactual. Overriding ``offers[0].amount_inr`` is a what-if; swapping in a
decision layer that classifies the same reply as ``pushback`` is a what-if about
the reasoning, and only the second one demonstrates anything.

Both operations require a checkpointer. Without one, ``get_state_history`` is
empty and every function here raises :class:`~core.errors.ReplayIntegrityError`
with the reason, rather than silently returning an empty list that a caller might
mistake for "there were no decisions".
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from core import Decision, DecisionRequest, ReplayIntegrityError

from .checkpointer import default_config
from .edges import CONDITIONAL_SOURCES, static_predecessor

__all__ = [
    "CheckpointInfo", "list_checkpoints", "load_checkpoint", "find_checkpoint",
    "fork_from_checkpoint", "fork_and_run", "compare_branches",
    "describe_run", "ReplayError",
]

log = logging.getLogger("paytriq.graph.replay")

#: Alias so callers can catch either the Paytriq-native error or this module's.
ReplayError = ReplayIntegrityError


@dataclass(slots=True)
class CheckpointInfo:
    """One point a run can be resumed from, described for a human.

    ``next_nodes`` is the important field: it names the step that was *pending*
    when the checkpoint was written, which is what "rewind to the pushback" means
    in practice.
    """

    checkpoint_id: str
    thread_id: str
    step: int
    source: str
    next_nodes: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    created_at: str = ""
    values: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """A compact, JSON-safe description for the demo UI."""
        progress = self.values.get("progress_ledger") or {}
        task = self.values.get("task_ledger") or {}
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "step": self.step,
            "source": self.source,
            "next_nodes": list(self.next_nodes),
            "created_at": self.created_at,
            "phase": self.values.get("phase"),
            "status": self.values.get("status"),
            "agent_id": self.values.get("agent_id"),
            "stall_count": self.values.get("stall_count"),
            "replan_count": self.values.get("replan_count"),
            "handoffs": len(self.values.get("handoffs") or []),
            "approvals": len(self.values.get("approvals") or []),
            "disputes": len(self.values.get("disputes") or []),
            "pending_gate": (self.values.get("pending_gate") or {}).get("kind"),
            "task_ledger_revision": task.get("revision"),
            "progress_reason": progress.get("progress_reason"),
        }


def _compiled(graph: Any) -> Any:
    """Accept a :class:`~graph.build.PaytriqGraph` or a raw compiled graph."""
    return getattr(graph, "compiled", graph)


def _runtime_of(graph: Any) -> Any | None:
    return getattr(graph, "runtime", None)


def list_checkpoints(graph: Any, thread_id: str, *, limit: int | None = None) -> list[CheckpointInfo]:
    """Every checkpoint for ``thread_id``, newest first.

    Raises :class:`ReplayIntegrityError` when the graph has no checkpointer,
    because an empty history and a missing checkpointer look identical otherwise
    and only one of them means "nothing happened".
    """
    compiled = _compiled(graph)
    if getattr(compiled, "checkpointer", None) is None:
        raise ReplayIntegrityError(
            "this graph was compiled without a checkpointer, so there is no "
            "state history to rewind. Build it with graph.build.build_graph(...) "
            "or pass checkpointer= to compile."
        )
    config = default_config(thread_id)
    try:
        snapshots = list(compiled.get_state_history(config))
    except Exception as exc:  # noqa: BLE001 - surfaced with its cause
        raise ReplayIntegrityError(
            f"could not read checkpoint history for thread {thread_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    out: list[CheckpointInfo] = []
    for snapshot in snapshots:
        configurable = (getattr(snapshot, "config", {}) or {}).get("configurable", {})
        checkpoint_id = configurable.get("checkpoint_id")
        if not checkpoint_id:
            continue
        metadata = getattr(snapshot, "metadata", {}) or {}
        out.append(CheckpointInfo(
            checkpoint_id=str(checkpoint_id),
            thread_id=str(configurable.get("thread_id") or thread_id),
            step=int(metadata.get("step") or 0),
            source=str(metadata.get("source") or "unknown"),
            next_nodes=tuple(str(n) for n in (getattr(snapshot, "next", ()) or ())),
            parents=tuple(str(p) for p in (metadata.get("parents") or {})),
            created_at=str(getattr(snapshot, "created_at", "") or ""),
            values=dict(getattr(snapshot, "values", {}) or {}),
        ))
    return out[:limit] if limit else out


def find_checkpoint(graph: Any, thread_id: str, checkpoint_id: str) -> CheckpointInfo:
    """One checkpoint by id, with a clear error naming the ids that do exist."""
    history = list_checkpoints(graph, thread_id)
    for info in history:
        if info.checkpoint_id == checkpoint_id:
            return info
    available = ", ".join(i.checkpoint_id for i in history[:8]) or "(none)"
    raise ReplayIntegrityError(
        f"no checkpoint {checkpoint_id!r} on thread {thread_id!r}. "
        f"Available (newest first): {available}"
    )


def load_checkpoint(graph: Any, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
    """The full state values at a checkpoint, as a plain dict."""
    return dict(find_checkpoint(graph, thread_id, checkpoint_id).values)


def fork_from_checkpoint(graph: Any, source_thread: str, checkpoint_id: str, *,
                         target_thread: str | None = None,
                         overrides: dict[str, Any] | None = None,
                         as_node: str | None = None,
                         clear_logs: bool = True) -> tuple[str, dict[str, Any]]:
    """Copy a checkpoint's state into a new thread, ready to run differently.

    ``as_node`` names the node the forked state *looks like it just produced*, so
    LangGraph knows what to run next.

    When it is omitted the seed point is derived, and the derivation has a real
    constraint behind it. ``update_state`` **cannot evaluate a conditional edge**:
    it has no runtime context to hand the router, so seeding at a router's own
    node fails with ``Missing required config key`` (verified on langgraph 1.2.12).
    The fork is therefore seeded at :func:`graph.edges.static_predecessor` — the
    last node before the pending one that leaves by a *fixed* edge — so LangGraph
    re-enters the pending node and the router runs later, during ``invoke``, with
    the real context.

    ``clear_logs`` resets the append-only channels. Without it the fork inherits
    the original run's ``handoffs``, ``disputes``, ``bids`` and ``approvals``, and
    because those channels use :func:`graph.state.append` they would appear to
    have happened twice in one run — which reads as a fabricated duplicate. The
    original thread keeps its own copies untouched.

    Returns ``(new_thread_id, forked_state)``.
    """
    compiled = _compiled(graph)
    if getattr(compiled, "checkpointer", None) is None:
        raise ReplayIntegrityError(
            "forking requires a checkpointer; this graph was compiled without one"
        )
    source = find_checkpoint(graph, source_thread, checkpoint_id)
    target = target_thread or f"{source_thread}_fork_{checkpoint_id[:8]}"
    pending = source.next_nodes[0] if source.next_nodes else None
    if not pending and not as_node:
        raise ReplayIntegrityError(
            f"checkpoint {checkpoint_id!r} has no pending node (next={source.next_nodes}); "
            "it is the end of the run, so there is nothing to fork from. Pass "
            "as_node= explicitly to force a fork from the final state."
        )

    node = as_node
    if node is None:
        node = pending if pending not in CONDITIONAL_SOURCES else static_predecessor(pending)
    if not node:
        raise ReplayIntegrityError(
            f"cannot seed a fork before {pending!r}: it is the entry node of the "
            "graph and LangGraph's update_state cannot evaluate a conditional edge "
            "without a runtime context (langgraph 1.2.12). Fork from a later "
            "checkpoint, or pass as_node= explicitly."
        )

    forked = dict(source.values)
    forked.pop("_progress", None)
    if clear_logs:
        for channel in ("handoffs", "disputes", "lessons", "approvals",
                        "human_decisions", "trace_events", "notes", "bids",
                        "risk_flags", "run_events", "steps_done"):
            forked[channel] = []
    forked["status"] = "forked"
    forked["stall_count"] = 0
    forked["replan_count"] = 0
    forked["negotiation_rounds"] = 0
    if overrides:
        forked.update(overrides)

    config = default_config(target)
    try:
        compiled.update_state(config, forked, as_node=node)
    except Exception as exc:  # noqa: BLE001 - surfaced with its cause
        raise ReplayIntegrityError(
            f"could not seed thread {target!r} from checkpoint {checkpoint_id!r} "
            f"as_node={node!r}: {type(exc).__name__}: {exc}"
        ) from exc

    log.info("forked checkpoint %s (step %s, pending %s) into thread %s as_node=%s",
             checkpoint_id, source.step, pending, target, node)
    return target, forked


def fork_and_run(graph: Any, source_thread: str, checkpoint_id: str, *,
                 target_thread: str | None = None,
                 overrides: dict[str, Any] | None = None,
                 decide: Callable[[DecisionRequest], Decision] | None = None,
                 resume: Any | None = None,
                 recursion_limit: int = 60,
                 as_node: str | None = None) -> dict[str, Any]:
    """Fork, then run the fork to completion with a possibly different ``decide``.

    Passing ``decide`` is the interesting parameter. A fork with only
    ``overrides`` answers "what if the state were different"; a fork with a new
    ``decide`` answers "what if the *decision layer* had answered differently",
    which is the counterfactual the whole architecture is built to support. The
    replacement callable is installed on the runtime's ``extras`` for the run and
    restored afterwards, so the original graph keeps its original decision layer.
    """
    runtime = _runtime_of(graph)
    target, _seeded = fork_from_checkpoint(
        graph, source_thread, checkpoint_id, target_thread=target_thread,
        overrides=overrides, as_node=as_node)

    original_decide = runtime.decide if runtime is not None else None
    if decide is not None and runtime is not None:
        runtime.decide = decide
    try:
        config = default_config(target)
        compiled = _compiled(graph)
        stream = compiled.stream(
            Command(resume=resume) if resume is not None else None,
            config=config, stream_mode="updates", recursion_limit=recursion_limit,
            **({"context": runtime} if runtime is not None else {}),
        )
        for chunk in stream:
            if "__interrupt__" in chunk:
                log.info("forked run parked at a gate on thread %s", target)
                break
        # Read the persisted state rather than trusting the last streamed delta:
        # a fork that parks at a human gate emits no final state, and the
        # interrupted state is exactly what the caller needs to render the gate.
        return dict(compiled.get_state(config).values)
    finally:
        if decide is not None and runtime is not None and original_decide is not None:
            runtime.decide = original_decide


def compare_branches(graph: Any, left_thread: str, right_thread: str) -> dict[str, Any]:
    """Diff two threads' final states. Makes the fork legible in the demo.

    Compares the fields a reviewer actually cares about — routing outcome,
    approvals, money, and whether the run completed — rather than dumping two
    state dicts side by side and asking the reader to find the difference.
    """
    left = _final_state(graph, left_thread)
    right = _final_state(graph, right_thread)

    def dig(state: dict[str, Any], *path: str) -> Any:
        cursor: Any = state
        for key in path:
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(key)
        return cursor

    differences: list[dict[str, Any]] = []
    probes = (
        ("status", ("status",)),
        ("agent_id", ("agent_id",)),
        ("compliance_score", ("compliance_score",)),
        ("handoff_count", ("handoffs",)),
        ("approval_count", ("approvals",)),
        ("dispute_count", ("disputes",)),
        ("bid_count", ("bids",)),
        ("signed_mous", ("mous",)),
        ("threads", ("threads",)),
        ("offers", ("offers",)),
        ("risk_flags", ("risk_flags",)),
        ("roi_multiple", ("roi_report", "roi_multiple")),
        ("stall_count", ("stall_count",)),
        ("replan_count", ("replan_count",)),
    )
    for name, path in probes:
        lval, rval = dig(left, *path), dig(right, *path)
        if isinstance(lval, list):
            lval = len(lval)
        if isinstance(rval, list):
            rval = len(rval)
        if lval != rval:
            differences.append({"field": name, "path": list(path),
                                "left": _jsonable(lval), "right": _jsonable(rval),
                                "same": False})
    return {
        "left_thread": left_thread,
        "right_thread": right_thread,
        "identical": not differences,
        "differences": differences,
        "left_handoffs": [h.get("reason") for h in (left.get("handoffs") or [])[-6:]],
        "right_handoffs": [h.get("reason") for h in (right.get("handoffs") or [])[-6:]],
    }


def _jsonable(value: Any) -> Any:
    """Coerce a probe result into something a JSON response can carry.

    ``roi_report`` and friends can hold lists of dicts; a diff endpoint must not
    500 on a nested structure, and must not silently drop one either.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _final_state(graph: Any, thread_id: str) -> dict[str, Any]:
    compiled = _compiled(graph)
    if getattr(compiled, "checkpointer", None) is None:
        raise ReplayIntegrityError("cannot read state without a checkpointer")
    snapshot = compiled.get_state(default_config(thread_id))
    return dict(getattr(snapshot, "values", {}) or {})


def describe_run(graph: Any, thread_id: str) -> dict[str, Any]:
    """A demo-ready narrative of one thread: its checkpoints, decisions, gates."""
    history = list_checkpoints(graph, thread_id)
    state = _final_state(graph, thread_id)
    from .interrupts import gate_summary

    pending = state.get("pending_gate") or {}
    return {
        "thread_id": thread_id,
        "checkpoint_count": len(history),
        "checkpoints": [c.summary() for c in history],
        "final_status": state.get("status"),
        "phase": state.get("phase"),
        "pending_gate": pending.get("kind"),
        "gates": gate_summary(state.get("approvals") or []),
        "handoff_reasons": [h.get("reason") for h in (state.get("handoffs") or [])],
        "dispute_statuses": [d.get("status") for d in (state.get("disputes") or [])],
        "stall_count": state.get("stall_count"),
        "replan_count": state.get("replan_count"),
    }
