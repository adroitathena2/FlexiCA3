"""Trace, summary, blackboard and coordination-artefact inspection.

These are the read endpoints that make a run auditable after the fact, and they
exist because a system whose only evidence is its own console output cannot be
checked by anyone else.

The one that matters most: ``distinct_gap_values``
---------------------------------------------------
``GET /api/runs/{run_id}/summary`` surfaces
:attr:`core.schemas.TraceSummary.distinct_gap_values` at the top level, next to
the raw gaps it was computed from, because it is arithmetic that anyone can
reproduce:

    a genuine capture   ->  many distinct gap values (typically 20-50+)
    a hand-written one  ->  exactly 1

A hand-written JSON trace advances ``ts`` by a round number each line, because a
constant is the easiest thing to type. Real work does not: a tool call waits on a
socket, a model retries, a debate round finishes early. The gap distribution is
therefore a fabrication tripwire that needs no trust in the project's own
claims -- see ``observability/summary.py`` for the derivation and its two
caveats (a trace of fewer than three operations cannot support the inference, and
this detects synthetic *cadence*, not a sophisticated fabricator who measured
real timings).

Every other endpoint here is an ordinary read, with one rule: **an absent
artefact is reported as absent.** ``GET /api/runs/{run_id}/disputes`` on a run
with no disputes returns ``{"count": 0, "disputes": []}``, not a fabricated
sample dispute, and a run that never existed is a 404 rather than an empty 200.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from core.errors import BlackboardError
from core.schemas import SpanRecord, TraceEvent, TraceSummary

from .deps import RunContext, unavailable_payload
from .schemas import ResumeRequest

__all__ = ["router"]

log = logging.getLogger("paytriq.api.trace")

router = APIRouter(prefix="/api/runs", tags=["trace"])


def _run_or_404(run_id: str) -> RunContext:
    from .deps import get_run

    context = get_run(run_id)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"unknown run_id {run_id!r}",
                    "hint": "POST /api/events returns a run_id; the process must "
                            "still be running for in-memory artefacts to exist"},
        )
    return context


def _board_or_503(context: RunContext, run_id: str) -> Any:
    if context.board is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=unavailable_payload(
                context.board_error or "this run has no blackboard", where=run_id),
        )
    return context.board


def _trace_file(context: RunContext) -> Path | None:
    path = context.trace_path()
    return Path(path) if path else None


def _events_from_disk(path: Path) -> list[TraceEvent]:
    """Reconstruct ``TraceEvent``s from a trace file.

    Used when the process that produced the run is gone (or a different one). The
    file is the durable record, so reading it back is more honest than reporting
    "no trace" -- and if the file is damaged, the linter endpoint reports that
    rather than this silently returning fewer events.
    """
    from observability.exporter import classify_record, iter_trace_records

    events: list[TraceEvent] = []
    for record in iter_trace_records(path):
        if not record.ok or classify_record(record.data or {}) != "event":
            continue
        try:
            events.append(TraceEvent(**(record.data or {})))
        except (ValidationError, TypeError) as exc:
            log.warning("trace %s line %s is not a TraceEvent: %s",
                        path.name, record.line, exc)
    events.sort(key=lambda e: e.seq)
    return events


def _spans_from_disk(path: Path) -> list[SpanRecord]:
    from observability.exporter import classify_record, iter_trace_records

    spans: list[SpanRecord] = []
    for record in iter_trace_records(path):
        if not record.ok or classify_record(record.data or {}) != "span":
            continue
        try:
            spans.append(SpanRecord(**(record.data or {})))
        except (ValidationError, TypeError) as exc:
            log.warning("trace %s line %s is not a SpanRecord: %s",
                        path.name, record.line, exc)
    return spans


@router.get("/{run_id}/trace", summary="Every TraceEvent in a run, in order")
def get_trace(run_id: str, request: Request,
              kind: str | None = None,
              limit: int | None = None) -> dict:
    """The flat event log.

    Prefers the tracer's in-memory list (exact, and already what the SSE feed is
    tailing) and falls back to parsing the trace file, so a run survives the
    process that produced it. ``kind`` filters on ``TraceKind``; ``limit`` keeps
    the **most recent** N.
    """
    context = _run_or_404(run_id)
    in_memory = context.events()
    source = "tracer.events"
    events = list(in_memory)
    if not events:
        path = _trace_file(context)
        if path is not None and path.exists():
            events = _events_from_disk(path)
            source = "trace file"
        else:
            source = "unavailable"

    if kind:
        wanted = kind.strip().lower()
        events = [e for e in events if e.kind.value == wanted]
    if limit is not None:
        events = events[-limit:] if limit else []

    return {
        "run_id": run_id,
        "event_id": context.event_id,
        "source": source,
        "count": len(events),
        "trace_file": context.trace_path(),
        "events": [e.model_dump(mode="json") for e in events],
        "note": ("events are read from the tracer's in-memory log; the same "
                 "records are on disk in trace_file"
                 if source == "tracer.events" else
                 "no in-memory events: this run was read back from its trace file"
                 if source == "trace file" else
                 "no trace exists for this run; nothing was recorded, which is a "
                 "fact about the run rather than an empty success"),
    }


@router.get("/{run_id}/summary", summary="TraceSummary, including distinct_gap_values")
def get_summary(run_id: str, request: Request) -> dict:
    """Derived statistics, with the anti-fabrication metric at the top level.

    ``distinct_gap_values`` is returned twice on purpose: inside ``summary``
    (where ``core.schemas.TraceSummary`` puts it, for a consumer that reads the
    whole object) and as a top-level integer, because it is the number a sceptical
    reader should look at first and it should not be buried one level down in a
    field called ``summary``.

    ``anti_fabrication`` restates the check in words and includes the raw gaps, so
    the claim can be audited rather than believed.
    """
    context = _run_or_404(run_id)
    path = _trace_file(context)

    summary: TraceSummary | None = None
    error = ""
    if path is not None and path.exists():
        try:
            from observability.summary import compute_summary

            summary = compute_summary(path, run_id)
        except Exception as exc:  # noqa: BLE001 - a damaged trace must not 500
            error = f"{type(exc).__name__}: {exc}"
            log.warning("could not summarise %s: %s", path, error)
    elif path is None:
        error = context.tracer_error or "this run has no tracer"
    else:
        error = f"no trace file at {path}; the tracer writes it on finish()"

    if summary is None:
        # Honest, and still useful: report what *is* known rather than inventing
        # zeros that read like "a real run with no activity".
        return {
            "run_id": run_id,
            "available": False,
            "unavailable": True,
            "error": "no trace summary is available for this run",
            "reason": error,
            "trace_file": str(path) if path else None,
            "in_memory_event_count": len(context.events()),
            "finished": context.finished,
            "hint": "traces are written when tracer.finish() runs, on stage "
                    "completion or at application shutdown",
        }

    gaps = list(summary.consecutive_start_gaps_ms)
    payload = summary.model_dump(mode="json")
    payload.update({
        "available": True,
        "unavailable": False,
        "trace_file": str(path),
        "summary": payload.copy(),
        # Surfaced prominently on purpose -- see the module docstring.
        "distinct_gap_values": summary.distinct_gap_values,
        "anti_fabrication": {
            "distinct_gap_values": summary.distinct_gap_values,
            "consecutive_start_gaps_ms": gaps,
            "operation_count": summary.span_count or summary.event_count,
            "verdict": _gap_verdict(summary.distinct_gap_values,
                                    summary.span_count or summary.event_count),
            "how_to_check": (
                "compute len(set(gaps)) from consecutive_start_gaps_ms yourself; "
                "a genuine run has many distinct values, a hand-written trace with "
                "a constant ts cadence has exactly 1"
            ),
            "caveats": [
                "fewer than three operations cannot support the inference",
                "this detects synthetic cadence, not a fabricator who measured "
                "real timings",
            ],
        },
    })
    return payload


def _gap_verdict(distinct: int, operations: int) -> str:
    """Plain-language reading of the gap metric, with the caveat attached."""
    if operations < 3:
        return ("inconclusive: too few operations for the gap distribution to "
                "mean anything")
    if distinct <= 1:
        return ("SUSPECT: every inter-arrival gap is identical, which is the "
                "signature of a typed constant rather than a measured run")
    if distinct < 5:
        return ("weak: some timing variation, but fewer distinct gaps than a "
                "multi-step run normally produces")
    return ("consistent with a genuine capture: irregular timing across "
            f"{operations} operations")


@router.get("/{run_id}/board", summary="Blackboard snapshot for one run")
def get_board(run_id: str, request: Request) -> dict:
    """Full snapshot via ``blackboard.serialize.snapshot``.

    A snapshot, not a live read, so the response is self-contained and can be
    diffed against another run. ``stats`` is included from the board's own
    accounting, which reports per-zone counts *including empty zones* -- a panel
    that hides the zones nobody wrote to is the panel that misses the run that
    only reached discovery.
    """
    context = _run_or_404(run_id)
    board = _board_or_503(context, run_id)
    try:
        from blackboard.serialize import snapshot

        data = snapshot(board)
    except BlackboardError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "could not snapshot the blackboard", "reason": str(exc)},
        ) from exc
    data.update({"run_id": run_id, "event_id": context.event_id,
                 "available": True, "unavailable": False})
    return data


@router.get("/{run_id}/board/tree", summary="Derivation view: the citation graph")
def get_board_tree(run_id: str, request: Request) -> dict:
    """``blackboard.serialize.render_tree``: every entry, indented by derivation depth.

    Purely derived from the ``refs`` edges each author recorded -- no model, no
    inference. Walking up the page from a ruling walks back through the exact
    chain of reasoning that produced it, which is the answer to "why did A7 rule
    that way?" that a pile of records cannot give.
    """
    context = _run_or_404(run_id)
    board = _board_or_503(context, run_id)
    try:
        from blackboard.serialize import render_tree

        text = render_tree(board)
    except BlackboardError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "could not render the derivation view",
                    "reason": str(exc)},
        ) from exc
    return {
        "run_id": run_id,
        "event_id": context.event_id,
        "tree": text,
        "line_count": text.count("\n"),
        "entry_count": context.board_high_water(),
        "available": True,
        "unavailable": False,
    }


def _zone_rows(context: RunContext, run_id: str, zone: str,
               kinds: tuple[str, ...]) -> dict:
    board = _board_or_503(context, run_id)
    rows: list[dict[str, Any]] = []
    for entry in board.history():  # type: ignore[attr-defined]
        if entry.zone != zone or entry.kind not in kinds:
            continue
        rows.append({
            "entry_id": entry.entry_id,
            "seq": entry.seq,
            "kind": entry.kind,
            "author": entry.author.value,
            "confidence": entry.confidence,
            "source": entry.source.value,
            "refs": list(entry.refs),
            "at": entry.at,
            "artifact": dict(entry.payload),
        })
    return {
        "run_id": run_id,
        "event_id": context.event_id,
        "zone": zone,
        "count": len(rows),
        "available": True,
        "unavailable": False,
        "note": (f"no {zone} entries exist for this run; that is a fact about the "
                 f"run, not a missing response"),
    } | {zone: rows}


@router.get("/{run_id}/disputes", summary="Disputes raised between agents")
def get_disputes(run_id: str, request: Request) -> dict:
    """Schema-validated disagreements between two reasoning agents."""
    context = _run_or_404(run_id)
    return _zone_rows(context, run_id, "disputes", ("dispute",))


@router.get("/{run_id}/lessons", summary="Reflexion lessons written after a dispute")
def get_lessons(run_id: str, request: Request) -> dict:
    """Reflexion-style verbal lessons (Shinn et al. 2023). No weights change; the
    lesson is retrieved as context on the next cycle, so behaviour changes
    observably between runs on identical input."""
    context = _run_or_404(run_id)
    return _zone_rows(context, run_id, "lessons", ("lesson",))


@router.get("/{run_id}/handoffs", summary="Every agent-to-agent handoff")
def get_handoffs(run_id: str, request: Request) -> dict:
    """The routing record: who handed to whom, why, decided by what, and how sure.

    ``decision_source`` and ``confidence`` are mandatory on ``Handoff`` in
    ``core.schemas``, so every row here names what made the routing decision. That
    is what turns "the agent decided" from an assertion into evidence.
    """
    context = _run_or_404(run_id)
    return _zone_rows(context, run_id, "handoffs", ("handoff",))


@router.get("/{run_id}/bids", summary="Contract Net bids and the award")
def get_bids(run_id: str, request: Request) -> dict:
    """Bids against the announced outreach task, with feasibility-weighted utility."""
    context = _run_or_404(run_id)
    return _zone_rows(context, run_id, "bids", ("bid", "award", "announcement"))


@router.post("/{run_id}/lint", summary="Run the trace linter and report problems")
def lint_run(run_id: str, request: Request) -> dict:
    """``observability.lint.lint_trace``: internal consistency of the trace file.

    Checks valid lines, contiguous ``seq``, resolvable parents, an acyclic tree,
    and a named source on every decision. These are properties of the *evidence*,
    so finding a violation means the run's own account of itself is already
    compromised -- which is exactly why it is worth running and returning.
    """
    context = _run_or_404(run_id)
    path = _trace_file(context)
    if path is None or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "there is no trace file to lint yet",
                "reason": (context.tracer_error or
                           f"no trace at {path}" if path else
                           "this run has no tracer"),
                "trace_file": str(path) if path else None,
                "hint": "the trace file is written when tracer.finish() runs -- "
                        "at application shutdown, or when a stage completes",
                "in_memory_event_count": len(context.events()),
            },
        )
    try:
        from observability.lint import lint_trace

        problems = lint_trace(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail={"error": str(exc)}) from exc
    except Exception as exc:  # noqa: BLE001 - a linter fault is a finding, not a 500
        log.error("lint_trace failed on %s: %s", path, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "the trace linter could not run",
                    "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc

    rows = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in problems]
    errors = [r for r in rows if str(r.get("severity")) == "error"]
    return {
        "run_id": run_id,
        "trace_file": str(path),
        "problem_count": len(rows),
        "error_count": len(errors),
        "clean": not errors,
        "problems": rows,
        "note": ("no integrity problems were found; the trace is internally "
                 "consistent" if not rows else
                 f"{len(errors)} error(s) and {len(rows) - len(errors)} warning(s)"),
    }


@router.post("/{run_id}/resume", summary="Resume a parked run on the same thread")
def resume_run(run_id: str, payload: ResumeRequest, request: Request) -> dict:
    """Resume a run parked on a LangGraph human gate.

    Wired to ``PaytriqGraph.resume()`` with ``Command(resume=...)`` on the
    run's stable thread, so a gate retry continues the parked run instead of
    re-running the pipeline. When ``gate_id`` names an outstanding API-ledger
    gate, the same human answer is recorded there too, bridging the two
    ledgers with one decision. A run that is still parked after the resume
    reports ``parked: true`` with its pending gates; nothing is ever reported
    as sent while a gate is pending.
    """
    from core.errors import PaytriqError

    from . import deps as _deps

    context = _run_or_404(run_id)
    tid = (payload.thread_id or "").strip() or context.thread_id.strip() or run_id
    context.thread_id = tid
    since_seq = context.board_high_water()
    with context.lock:
        context.finished = False

    try:
        runner = _deps.get_graph_runner(context)
    except _deps.SubsystemUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_deps.unavailable_payload(exc, where=f"POST /api/runs/{run_id}/resume"),
        ) from exc

    try:
        pending = list(runner.pending_interrupts(thread_id=tid))
    except Exception:
        pending = []
    if not pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": f"run {run_id} has no pending gate on thread {tid}",
                    "hint": "the run is not parked; run a stage instead",
                    "thread_id": tid},
        )

    # Bridge the ledgers: the same human answer satisfies the API gate (if
    # named) and the graph interrupt.
    bridged_gate_id: str | None = None
    if payload.gate_id:
        gate = context.gates.gate(payload.gate_id)
        if gate is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"unknown gate_id {payload.gate_id!r}",
                        "hint": "GET /api/gates/{run_id} lists this run's gates"},
            )
        if context.gates.decision(payload.gate_id) is None:
            try:
                from core.ids import new_id as _new_id
                from core.schemas import Approval as _Approval
                from core.schemas import GateOutcome as _Outcome
                from core.schemas import HumanDecision as _Decision
                from core.schemas import utcnow as _utcnow

                from .routes_gates import record_decision
                outcome = _Outcome(payload.outcome)
                at = _utcnow()
                decision = _Decision(gate_id=gate.gate_id, kind=gate.kind,
                                     outcome=outcome, decided_by=payload.decided_by,
                                     instruction=payload.instruction or "", at=at)
                approval = _Approval(approval_id=_new_id("apr"), gate_id=gate.gate_id,
                                     kind=gate.kind, outcome=outcome,
                                     decided_by=payload.decided_by,
                                     action_taken=(f"{gate.kind.value} gate answered "
                                                   f"{outcome.value} by {payload.decided_by}"),
                                     at=at)
                record_decision(context, gate, decision, approval)
                bridged_gate_id = gate.gate_id
            except HTTPException:
                raise
            except (PaytriqError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": "could not bridge the API ledger",
                            "reason": f"{type(exc).__name__}: {exc}"},
                ) from exc

    resume_value = {"outcome": payload.outcome, "decided_by": payload.decided_by,
                    "instruction": payload.instruction or ""}
    if payload.gate_id:
        resume_value["gate_id"] = payload.gate_id
    state: dict[str, Any] = {}
    try:
        state = runner.resume(resume_value, thread_id=tid) or {}
    except PaytriqError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "resume failed", "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - one resume must not 500 opaquely
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "resume failed", "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc

    with context.lock:
        context.state = dict(state or {})
    context.status = "resumed_complete" if state.get("status") == "complete" else "resumed"
    try:
        still_pending = list(runner.pending_interrupts(thread_id=tid))
    except Exception:
        still_pending = []
    pending_rows: list[dict[str, Any]] = []
    for intr in still_pending:
        value = getattr(intr, "value", intr)
        if isinstance(value, dict):
            pending_rows.append({k: value.get(k) for k in
                                 ("gate_id", "kind", "question", "payload_preview")
                                 if value.get(k)})
        else:
            pending_rows.append({"interrupt": str(value)[:300]})
    parked = bool(still_pending)
    if not parked and (state.get("status") == "complete" or state.get("phase") == "done"):
        with context.lock:
            context.finished = True

    from .routes_pipeline import _decision_provenance, board_delta
    delta = board_delta(context.board, since_seq)
    return {
        "run_id": run_id,
        "event_id": context.event_id,
        "thread_id": tid,
        "resumed_with": resume_value,
        "bridged_gate_id": bridged_gate_id,
        "parked": parked,
        "pending_gates": pending_rows,
        "status": state.get("status", ""),
        "phase": state.get("phase", ""),
        "finished": context.finished,
        "state": state,
        "board_delta": delta,
        "board_delta_count": len(delta),
        "decision": _decision_provenance(state or {}, context),
        "notes": [("the run is still parked on a human gate and nothing was "
                   "sent; answer the pending gate to continue")
                  if parked else "resume completed the run"],
    }


@router.get("/{run_id}", summary="Everything known about one run")
def describe_run(run_id: str, request: Request) -> dict:
    """One screen's worth of context: status, board size, trace, gates."""
    context = _run_or_404(run_id)
    path = _trace_file(context)
    spans: list[SpanRecord] = []
    if path is not None and path.exists():
        spans = _spans_from_disk(path)
    data = context.describe()
    data.update({
        "available": True,
        "unavailable": False,
        "trace_file_exists": bool(path and path.exists()),
        "span_count_on_disk": len(spans),
        "approvals": [a.model_dump(mode="json")
                      for a in context.gates.approvals(run_id=run_id)],
        "outstanding_gates": [g.gate_id for g in context.gates.outstanding(run_id=run_id)],
    })
    return data
