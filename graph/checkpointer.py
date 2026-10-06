"""Checkpointer setup: SQLite by default, in-memory as a logged fallback.

Why a checkpointer is not optional here
--------------------------------------
``langgraph.types.interrupt`` parks a run *in the checkpointer*. Without one
there is nowhere to park it, so every human gate would raise and the run would
die at the first approval. ``update_state`` and ``get_state_history`` — which
:mod:`graph.replay` uses to fork a run from a past decision — are also
checkpointer operations. The checkpointer is therefore part of the orchestration
layer's contract, not an optional extra.

Why SQLite, and where the file goes
----------------------------------
``langgraph-checkpoint-sqlite`` is chosen over the Postgres saver because this
is a single-process student system with one demo operator: durability across a
crash is worth having, and a database server is not. The file lives under
``settings.traces_dir`` (which defaults to ``<repo_root>/traces``) so that
checkpoints sit next to the traces they produced — one artefact directory to
delete, one to archive, one to attach to a bug report.

Fallback policy
---------------
:func:`build_checkpointer` never raises for an environmental problem. If SQLite
cannot be opened — read-only volume, missing driver, locked file — it returns
:class:`~langgraph.checkpoint.memory.InMemorySaver` and records **why**, both
in the log and on the returned object's ``degraded_reason``. Silent substitution
would be the exact failure mode the rest of this codebase is built to avoid: a
run that quietly lost its history would still produce output that looks fine.
"""
from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import ConfigError, Settings

__all__ = [
    "CheckpointSetup", "build_checkpointer", "default_config", "default_thread_id",
    "checkpoint_path", "close_checkpointer", "checkpoint_status", "STATE_KEY",
]

log = logging.getLogger("paytriq.graph.checkpointer")

#: The configurable key LangGraph requires on every invoke for interrupts and
#: time travel to work. Named here so the API layer cannot invent a second one.
STATE_KEY = "thread_id"

#: Filename under ``settings.traces_dir``. One database per repository, not per
#: run: threads are the unit of resumption, and a per-run file would make a
#: resumed run start a different conversation.
DB_FILENAME = "paytriq_checkpoints.sqlite"


def checkpoint_path(settings: Settings | None = None,
                    filename: str | None = None) -> Path:
    """Resolve the checkpoint file path, creating the directory if needed."""
    cfg = settings or Settings()
    base = Path(cfg.traces_dir)
    target = base / (filename or DB_FILENAME)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("could not create %s (%s); SQLite checkpointing will be "
                    "attempted anyway and may fail", target.parent, exc)
    return target


def default_thread_id(*, event_id: str = "", run_id: str = "") -> str:
    """A deterministic thread id.

    Deterministic because a thread id is the conversation key: deriving it from
    the event means "resume this event" needs no stored pointer, and deriving it
    from a UUID would mean every page reload starts a fresh run.
    """
    if event_id:
        return f"evt_{event_id}"
    if run_id:
        return f"run_{run_id}"
    from core import run_id as new_run_id

    return f"run_{new_run_id()}"


def default_config(thread_id: str | None = None, **extra: Any) -> dict[str, Any]:
    """The ``config`` dict every ``invoke``/``stream``/``update_state`` needs.

    ``configurable.thread_id`` is the only required entry for the features this
    package uses. Extra ``configurable`` keys are merged, and top-level keys are
    accepted, because LangGraph's own helpers pass both shapes and callers should
    not have to know which.
    """
    configurable: dict[str, Any] = {STATE_KEY: thread_id or default_thread_id()}
    for key, value in extra.items():
        configurable[key] = value
    return {"configurable": configurable}


@dataclass
class CheckpointSetup:
    """The saver plus everything a caller needs to know about how it was made.

    ``degraded`` and ``degraded_reason`` exist so the API's status panel can say
    "history is in memory, this process will forget it" instead of implying
    durability it does not have.
    """

    saver: Any
    kind: str
    path: Path | None = None
    degraded: bool = False
    degraded_reason: str = ""
    _closer: AbstractContextManager | None = field(default=None, repr=False)

    def close(self) -> None:
        """Release the SQLite connection. Safe to call more than once."""
        if self._closer is not None:
            try:
                self._closer.__exit__(None, None, None)
            finally:
                self._closer = None

    def status(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path) if self.path else None,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "durable": not self.degraded,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CheckpointSetup(kind={self.kind!r}, path={self.path}, "
                f"degraded={self.degraded})")


def build_checkpointer(path: os.PathLike[str] | str | None = None, *,
                       settings: Settings | None = None,
                       allow_memory_fallback: bool = True) -> CheckpointSetup:
    """Open a :class:`SqliteSaver`, or an :class:`InMemorySaver` if that fails.

    ``allow_memory_fallback=False`` is for tests that need to *prove* the real
    saver was selected; production uses the default ``True``.
    """
    cfg = settings or Settings()
    target = Path(path) if path is not None else checkpoint_path(cfg)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        return _memory_fallback(
            f"langgraph-checkpoint-sqlite is not installed ({exc}); "
            f"install langgraph-checkpoint-sqlite for durable checkpoints",
            target=target, settings=cfg, allow_memory_fallback=allow_memory_fallback)

    closer: AbstractContextManager | None = None
    try:
        # ``from_conn_string`` is a context manager that must stay open for the
        # life of the saver: it owns the sqlite3 connection.
        closer = SqliteSaver.from_conn_string(str(target))
        saver = closer.__enter__()
    except Exception as exc:  # noqa: BLE001 - any sqlite/driver fault degrades
        if closer is not None:
            try:
                closer.__exit__(None, None, None)
            except Exception as cleanup_exc:  # noqa: BLE001
                log.warning("sqlite closer failed during cleanup: %s", cleanup_exc)
        log.warning("could not open SQLite checkpoint at %s (%s: %s)",
                    target, type(exc).__name__, exc)
        return _memory_fallback(
            f"sqlite unavailable at {target}: {type(exc).__name__}: {exc}",
            target=target, settings=cfg, allow_memory_fallback=allow_memory_fallback)

    log.info("checkpointing to SQLite at %s", target)
    return CheckpointSetup(saver=saver, kind="sqlite", path=target, _closer=closer)


def _memory_fallback(reason: str, *, target: Path, settings: Settings,
                     allow_memory_fallback: bool) -> CheckpointSetup:
    if not allow_memory_fallback:
        raise ConfigError(
            f"could not build a durable checkpointer: {reason}. Set "
            f"allow_memory_fallback=True (the default) to continue in memory."
        )
    from langgraph.checkpoint.memory import InMemorySaver

    log.warning("FALLING BACK to InMemorySaver: %s. History will not survive this "
                "process.", reason)
    return CheckpointSetup(
        saver=InMemorySaver(),
        kind="memory",
        path=target,
        degraded=True,
        degraded_reason=reason,
    )


def close_checkpointer(setup: CheckpointSetup) -> None:
    """Idempotent close, for ``finally`` blocks."""
    setup.close()


def checkpoint_status(setup: CheckpointSetup) -> dict[str, Any]:
    """Alias for :meth:`CheckpointSetup.status`, for symmetry with the registry."""
    return setup.status()
