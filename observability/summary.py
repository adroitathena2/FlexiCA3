"""Derived statistics over a trace file: the evidence that it is a real capture.

The headline number
-------------------
``TraceSummary.distinct_gap_values`` is the count of *distinct* gaps between the
sorted start times of consecutive operations in the trace.

Why this is the project's anti-fabrication check
------------------------------------------------
A trace is easy to fake and hard to fake *well*. Hand-written JSON always looks
plausible: the keys are right, the ids have the right prefixes, the schema
validates. What it cannot easily reproduce is **timing**.

Real work has irregular timing. An agent span wraps a tool call that hits a
network timeout, which wraps a model call that has to retry, which is followed by
a debate round that finished early because the arbiter found agreement. The
inter-arrival times of those operations are all different, to sub-millisecond
resolution, every single run.

A hand-written trace does not have that. It has whatever cadence the author
typed — typically ``ts`` advancing by a round ``250`` ms or ``1000`` ms each
line, because a constant is the easiest thing to type. So:

    a genuine capture  ->  many distinct gap values (typically 20-50+)
    a hand-written one ->  exactly 1

That difference is arithmetic, not judgement. Anyone who can read the file can
compute it and check it themselves; nobody has to take the project's word for it.
Publishing the number *and* the raw gaps it was derived from
(``consecutive_start_gaps_ms``) means the claim can be audited, not just believed.

Two honest caveats, stated here rather than buried:

1. A very short trace cannot support the inference. With fewer than three
   operations there are fewer than three gaps, so ``distinct_gap_values`` is
   small for reasons that have nothing to do with honesty. Read it alongside
   ``span_count``.
2. This detects *synthetic cadence*. It does not prove a run was genuine — a
   sophisticated fabricator who measured real timings would pass. It is a
   tripwire against the common and cheap failure mode, not a proof. It is the
   job of :mod:`observability.lint` to say whether that is even in scope.

Where the numbers come from
---------------------------
A trace file holds two views of the same run: one ``TraceEvent`` per logical
occurrence, and one ``SpanRecord`` per span. Counting both would double-count
every handoff and every decision. So the rule is:

* **semantic counts** (handoffs, decisions, LLM calls, tool calls, tool status,
  human gates) come from **events** when the file has any, because an event is
  one-per-occurrence by construction. If a file has no events at all, they fall
  back to spans so a span-only file still summarises.
* **structural counts** (``span_count``, ``root_span_count``) come from spans,
  because only spans carry parent links.
* **timing** comes from span ``start_unix_nano`` when available — nanosecond
  resolution beats millisecond datetimes, and a span's start is the true
  inter-arrival time. With no spans, event start is reconstructed as
  ``ts - duration_ms``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.schemas import SpanRecord, TraceEvent, TraceKind, TraceSummary

from .exporter import (
    ATTR_CODE_SHA256,
    ATTR_GIT_COMMIT,
    HOST_ID_SALT,
    TraceRecord,
    classify_record,
    iter_trace_records,
)

__all__ = ["compute_summary", "gap_statistics", "GAP_DECIMALS"]

logger = logging.getLogger("paytriq.observability")

#: Decimal places gaps are rounded to before de-duplication. Three decimals is
#: one microsecond: far coarser than the noise floor of a real run (so real
#: jitter is never accidentally collapsed) and far finer than any interval a
#: human would type (so a fabricated constant still de-duplicates to one value).
GAP_DECIMALS = 3

_HEX_TRACE = re.compile(r"^[0-9a-f]{32}$")
_HEX_SPAN = re.compile(r"^[0-9a-f]{16}$")


# ===================================================================== helpers
def _load(path: str | Path) -> tuple[list[TraceEvent], list[SpanRecord],
                                          list[TraceRecord]]:
    """Parse a trace file into validated events and spans.

    Unparseable lines are counted and logged rather than raised on: a summary of
    a slightly-damaged trace is still worth having, and :func:`lint_trace`
    exists to report the damage in detail.
    """
    events: list[TraceEvent] = []
    spans: list[SpanRecord] = []
    raw: list[TraceRecord] = []
    bad = 0
    for record in iter_trace_records(path):
        raw.append(record)
        if not record.ok:
            bad += 1
            continue
        kind = classify_record(record.data or {})
        try:
            if kind == "event":
                events.append(TraceEvent(**(record.data or {})))
            elif kind == "span":
                spans.append(SpanRecord(**(record.data or {})))
            else:
                bad += 1
        except (ValidationError, TypeError) as exc:
            bad += 1
            logger.warning("line %d is not a valid Paytriq record: %s",
                           record.line, exc)
    if bad:
        logger.warning("%d line(s) of %s could not be summarised", bad, path)
    return events, spans, raw


def _attribute(record: Any, *names: str) -> Any:
    """First present attribute among ``names``.

    Two naming conventions coexist by design and this is the one place that
    reconciles them: ``TraceEvent.attributes`` uses schema-shaped flat keys
    (``source``, ``confidence``) because events are read next to the domain
    objects, while span attributes use namespaced semantic-convention keys
    (``decision.source``) because spans are read in an OTel viewer.
    """
    attributes = getattr(record, "attributes", None) or {}
    for name in names:
        if name in attributes:
            return attributes[name]
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def gap_statistics(starts_ns: list[int]) -> tuple[list[float], int]:
    """Return ``(consecutive gaps in ms, number of distinct gap values)``.

    ``starts_ns`` need not be sorted or deduplicated — sorting is done here, so
    callers cannot accidentally compute gaps in file order and get an artefact
    of the exporter's completion-order writes.

    With fewer than two operations there are no gaps, so both outputs are empty
    and ``distinct_gap_values`` is 0. That is arithmetically honest and means
    nothing about authenticity; see the module docstring.
    """
    ordered = sorted(starts_ns)
    gaps: list[float] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gaps.append(round((current - previous) / 1e6, GAP_DECIMALS))
    return gaps, len({gap for gap in gaps})


def _start_times(events: list[TraceEvent], spans: list[SpanRecord]) -> list[int]:
    """Nanosecond start instants for every operation in the run.

    Prefers spans: they carry the true start recorded by the OTel clock at
    nanosecond resolution, and a span's start is exactly the inter-arrival time
    we want to measure. Falls back to ``ts - duration_ms`` on events, which is
    the same quantity at coarser resolution.
    """
    if spans:
        return [span.start_unix_nano for span in spans]
    instants: list[int] = []
    for event in events:
        start = event.ts - timedelta(milliseconds=event.duration_ms or 0.0)
        instants.append(int(start.timestamp() * 1e9))
    return instants


def _wall_clock_ms(events: list[TraceEvent], spans: list[SpanRecord]) -> float:
    """First start to last end, in milliseconds.

    Uses the span window when spans exist (spans record both ends) and otherwise
    the event window. Returns 0.0 for an empty trace rather than pretending.
    """
    if spans:
        first = min(span.start_unix_nano for span in spans)
        last = max(span.end_unix_nano for span in spans)
        return round((last - first) / 1e6, 3)
    if events:
        first = min(events, key=lambda e: e.ts - timedelta(
            milliseconds=e.duration_ms or 0.0)).ts
        last = max(events, key=lambda e: e.ts).ts
        return round((last - first).total_seconds() * 1000, 3)
    return 0.0


def _provenance(events: list[TraceEvent], spans: list[SpanRecord]) -> tuple[str, str]:
    """Recover the commit and code digest the run was produced from.

    The OTel ``Resource`` is not carried on each span record — it would repeat
    the same six strings on every line of the file — so the tracer emits them once
    as a ``run.configure`` event. Both namespaced and flat keys are accepted so
    the lookup works whichever spelling a given trace used. When the trace
    carries nothing (``unknown``), the current environment's ``GIT_COMMIT`` /
    ``CODE_SHA256`` are used when set (falling back to the platform names
    ``RENDER_GIT_COMMIT`` / ``GITHUB_SHA`` that CI/Render inject), otherwise
    ``unknown`` is kept. Never fabricated: an unset value stays ``unknown``.
    A zero-commit checkout records ``"uncommitted-working-tree"`` explicitly at
    capture time (see scripts/capture_canonical_trace.py) rather than inventing
    a hash.
    """
    commit = "unknown"
    digest = "unknown"
    for record in list(events) + list(spans):
        found_commit = _text(_attribute(record, ATTR_GIT_COMMIT, "git_commit"))
        if found_commit and commit == "unknown":
            commit = found_commit
        found_digest = _text(_attribute(record, ATTR_CODE_SHA256, "code_sha256"))
        if found_digest and digest == "unknown":
            digest = found_digest
        if commit != "unknown" and digest != "unknown":
            break
    if commit == "unknown":
        for _key in ("GIT_COMMIT", "RENDER_GIT_COMMIT", "GITHUB_SHA"):
            env_commit = _text(os.getenv(_key, ""))
            if env_commit:
                commit = env_commit
                break
    if digest == "unknown":
        for _key in ("CODE_SHA256", "RENDER_GIT_COMMIT", "GITHUB_SHA"):
            env_digest = _text(os.getenv(_key, ""))
            if env_digest:
                digest = env_digest
                break
    return commit, digest


def host_id(node: str | None = None) -> str:
    """Salted SHA-256 of the hostname.

    ``host.id`` exists so two traces can be compared — "same machine, two runs"
    versus "one run, two machines" — without publishing a hostname, which on a
    personal machine is very often the owner's own name. A salt keeps the digest
    from being a trivial rainbow-table lookup of a known machine name.
    """
    raw = f"{HOST_ID_SALT}|{node if node is not None else platform.node()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ==================================================================== the summary
def compute_summary(path: str | Path, run_id: str) -> TraceSummary:
    """Summarise one trace file.

    Every field of ``TraceSummary`` is populated; nothing is left at a default
    for lack of effort. Fields that genuinely do not apply (a run with no tools
    has no tool status counts) stay empty rather than being invented — an empty
    dict is a fact, a fabricated one is not.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"no trace at {file_path}; a summary of nothing would be a fabrication"
        )

    events, spans, raw = _load(file_path)
    primary: list[Any] = events if events else spans

    root_span_count = sum(1 for span in spans if span.parent_span_id is None)
    if not spans:
        root_span_count = sum(1 for event in events if event.parent_span_id is None)

    handoff_count = sum(1 for record in primary if record.kind is TraceKind.HANDOFF)
    message_count = sum(1 for record in primary if record.kind is TraceKind.MESSAGE)
    llm_call_count = sum(1 for record in primary if record.kind is TraceKind.LLM)
    tool_call_count = sum(1 for record in primary if record.kind is TraceKind.TOOL)
    human_gate_count = sum(1 for record in primary if record.kind is TraceKind.HUMAN)

    decision_sources: Counter[str] = Counter()
    degraded = 0
    tool_status: Counter[str] = Counter()
    for record in primary:
        if record.kind is TraceKind.DECISION:
            source = _text(_attribute(record, "source", "decision.source"))
            decision_sources[source or "unnamed"] += 1
            if _truthy(_attribute(record, "degraded", "decision.degraded")):
                degraded += 1
        if record.kind is TraceKind.TOOL:
            status = _text(_attribute(record, "tool_status", "tool.status", "status"))
            tool_status[status or "unstated"] += 1

    # Legacy fallback for traces that recorded routing only as handoffs (no
    # DECISION spans): when no decision was counted but handoffs carry a
    # decision_source, count those sources so a run with clef handoffs does not
    # report decision_counts={}. Only a fallback — when decisions exist they
    # are the count, never double-counted with handoffs.
    if not decision_sources and handoff_count:
        for record in primary:
            if record.kind is TraceKind.HANDOFF:
                source = _text(_attribute(
                    record, "decision_source", "handoff.decision_source",
                    "source", "decision.source"))
                if source:
                    decision_sources[source] += 1
    # Same legacy fallback for model calls: a trace with clef/gemini handoffs
    # but no LLM spans still made model calls to route. Count model-sourced
    # handoffs only when no LLM span exists, so explicit LLM spans stay the
    # count whenever they are present.
    if llm_call_count == 0 and handoff_count:
        model_handoffs = 0
        for record in primary:
            if record.kind is TraceKind.HANDOFF:
                source = _text(_attribute(
                    record, "decision_source", "handoff.decision_source",
                    "source", "decision.source")).lower()
                if source in ("clef", "gemini"):
                    model_handoffs += 1
        if model_handoffs:
            llm_call_count = model_handoffs

    gaps, distinct = gap_statistics(_start_times(events, spans))
    commit, digest = _provenance(events, spans)

    summary = TraceSummary(
        run_id=run_id,
        trace_file=str(file_path),
        git_commit=commit,
        code_sha256=digest,
        event_count=len(events),
        span_count=len(spans),
        root_span_count=root_span_count,
        handoff_count=handoff_count,
        message_count=message_count,
        # Sorted so two runs of the same shape produce comparable summaries.
        decision_counts=dict(sorted(decision_sources.items())),
        degraded_decision_count=degraded,
        wall_clock_ms=_wall_clock_ms(events, spans),
        consecutive_start_gaps_ms=gaps,
        distinct_gap_values=distinct,
        llm_call_count=llm_call_count,
        tool_call_count=tool_call_count,
        tool_status_counts=dict(sorted(tool_status.items())),
        human_gate_count=human_gate_count,
    )
    if raw and summary.event_count + summary.span_count < len(raw):
        logger.warning("%s: %d line(s) present but not summarised",
                       file_path.name, len(raw))
    logger.info("summarised %s: %d span(s), %d event(s), %d distinct gap value(s)",
                file_path.name, summary.span_count, summary.event_count,
                summary.distinct_gap_values)
    return summary


def assert_ids_are_hex(traces: list[SpanRecord]) -> None:  # pragma: no cover
    """Optional integrity helper: assert every id is a correctly sized hex string.

    Kept here next to the loader because it is a property of the *file format*,
    not of any one operation. ``tests/unit/test_observability.py`` uses it.
    """
    for span in traces:
        if not _HEX_TRACE.match(span.trace_id):
            raise ValueError(f"trace_id is not 32 hex chars: {span.trace_id!r}")
        if not _HEX_SPAN.match(span.span_id):
            raise ValueError(f"span_id is not 16 hex chars: {span.span_id!r}")
        if span.parent_span_id is not None and not _HEX_SPAN.match(span.parent_span_id):
            raise ValueError(f"parent_span_id is not 16 hex chars: "
                             f"{span.parent_span_id!r}")


def environment_snapshot() -> dict[str, Any]:
    """Host facts recorded alongside a trace.

    Exposed for the API's status panel. Kept here so the tracer does not import
    :mod:`platform` directly and so the shape is documented in one place.
    """
    return {
        "host_arch": platform.machine(),
        "host_id": host_id(),
        "process_runtime_version": platform.python_version(),
        "process_pid": os.getpid(),
        "platform": platform.platform(terse=True),
    }
