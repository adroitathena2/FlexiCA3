"""Trace integrity checking.

What this is
------------
A **trace integrity checker**. It answers one question: *is this file internally
consistent with itself?* Is every line valid JSON that validates against the
frozen schema? Is ``seq`` contiguous? Does every ``parent_span_id`` point at a
record that is actually in the file? Is the tree acyclic? Did every decision name
a source?

What this is **not**
-------------------
An anti-fraud device, and the distinction is not pedantry — it decides what the
tool is allowed to be trusted for. A checker that could prove a run "was real"
would be a claim about the world; a checker that proves a file "is self-consistent"
is a claim about arithmetic, and arithmetic is all it does here. A file can pass
every check below and still be fiction: fabrication that happens to be internally
tidy is fabrication.

So the cadence check (:data:`PROBLEM_CONSTANT_CADENCE`) is a **warning**, not an
error, and :func:`assert_clean` only fails on errors unless a caller asks for
``strict=True``. A warning is a nudge to look, not a verdict. The one number that
*is* meant to be published is ``TraceSummary.distinct_gap_values`` — not because
it proves anything, but because it is published so a reader can form their own
judgement from raw data.

Problem codes
-------------
``invalid_json``, ``not_an_object``, ``unknown_record``, ``schema_invalid``,
``invalid_agent``, ``environment_as_agent``, ``seq_not_zero_based``,
``seq_not_monotonic``, ``seq_gap``, ``timestamp_regression``, ``end_before_start``,
``orphan_parent``, ``cycle``, ``decision_missing_source``,
``decision_confidence_range``, ``llm_missing_model``, ``constant_cadence``.

A note on the monotonicity check
--------------------------------
Span *start* times are **not** monotonic in file order and are not expected to
be: ``SimpleSpanProcessor`` writes a span when it ends, so a child always
precedes its parent. Checking start monotonicity would report every real trace as
broken. What must hold, and what is checked, is that span *end* times are
non-decreasing in file order — that is the order the file was written in.
``TraceEvent.ts`` is likewise non-decreasing, because an event is timestamped when
it is created and events are appended as they are created.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.errors import PaytriqError
from core.schemas import AgentId, SpanRecord, TraceEvent, TraceKind, TraceSummary

from .exporter import classify_record, iter_trace_records
from .summary import compute_summary

__all__ = ["Problem", "TraceLintError", "lint_trace", "assert_clean",
           "format_problems", "PROBLEM_CODES"]

logger = logging.getLogger("paytriq.observability")

PROBLEM_INVALID_JSON = "invalid_json"
PROBLEM_NOT_AN_OBJECT = "not_an_object"
PROBLEM_UNKNOWN_RECORD = "unknown_record"
PROBLEM_SCHEMA_INVALID = "schema_invalid"
PROBLEM_INVALID_AGENT = "invalid_agent"
PROBLEM_ENVIRONMENT_AS_AGENT = "environment_as_agent"
PROBLEM_SEQ_NOT_ZERO = "seq_not_zero_based"
PROBLEM_SEQ_NOT_MONOTONIC = "seq_not_monotonic"
PROBLEM_SEQ_GAP = "seq_gap"
PROBLEM_TIMESTAMP_REGRESSION = "timestamp_regression"
PROBLEM_END_BEFORE_START = "end_before_start"
PROBLEM_ORPHAN_PARENT = "orphan_parent"
PROBLEM_CYCLE = "cycle"
PROBLEM_DECISION_MISSING_SOURCE = "decision_missing_source"
PROBLEM_DECISION_CONFIDENCE = "decision_confidence_range"
PROBLEM_LLM_MISSING_MODEL = "llm_missing_model"
PROBLEM_CONSTANT_CADENCE = "constant_cadence"

#: Every code this module can emit. Exposed so a test can assert the union.
PROBLEM_CODES: frozenset[str] = frozenset({
    PROBLEM_INVALID_JSON, PROBLEM_NOT_AN_OBJECT, PROBLEM_UNKNOWN_RECORD,
    PROBLEM_SCHEMA_INVALID, PROBLEM_INVALID_AGENT, PROBLEM_ENVIRONMENT_AS_AGENT,
    PROBLEM_SEQ_NOT_ZERO, PROBLEM_SEQ_NOT_MONOTONIC, PROBLEM_SEQ_GAP,
    PROBLEM_TIMESTAMP_REGRESSION, PROBLEM_END_BEFORE_START,
    PROBLEM_ORPHAN_PARENT, PROBLEM_CYCLE, PROBLEM_DECISION_MISSING_SOURCE,
    PROBLEM_DECISION_CONFIDENCE, PROBLEM_LLM_MISSING_MODEL,
    PROBLEM_CONSTANT_CADENCE,
})

#: Constant cadence is only *interesting* with enough gaps to be a pattern. Two
#: spans have one gap, and one gap is trivially a single distinct value; calling
#: that suspicious would make the warning meaningless on every short trace.
MIN_GAPS_FOR_CADENCE_CHECK = 3


class TraceLintError(PaytriqError):
    """A trace failed its integrity check and must not be presented as clean.

    Subclasses :class:`core.errors.PaytriqError` so a caller can catch the whole
    project's deliberate failures with one ``except``.
    """


@dataclass(frozen=True, slots=True)
class Problem:
    """One integrity finding. Structured so it can be filtered, not just printed."""

    code: str
    message: str
    line: int | None = None
    severity: str = "error"
    record: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
            "record": self.record,
        }

    def __str__(self) -> str:
        where = f" line {self.line}" if self.line is not None else ""
        which = f" [{self.record}]" if self.record else ""
        return f"{self.severity.upper()} {self.code}{where}{which}: {self.message}"


def format_problems(problems: Iterable[Problem], *, limit: int = 20) -> str:
    """Render problems for an exception message or a log line."""
    listed = list(problems)
    if not listed:
        return "no problems"
    head = [str(problem) for problem in listed[:limit]]
    if len(listed) > limit:
        head.append(f"... and {len(listed) - limit} more")
    return "\n".join(head)


# =================================================================== predicates
def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _attribute(record: Any, *names: str) -> Any:
    """First present attribute among ``names``.

    Accepts both naming conventions in use — flat schema keys on events
    (``source``) and namespaced semantic-convention keys on spans
    (``decision.source``) — so one check serves both views of the run.
    """
    attributes = getattr(record, "attributes", None)
    if not isinstance(attributes, Mapping):
        return None
    for name in names:
        if name in attributes:
            return attributes[name]
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


# ======================================================================= checks
def _check_agent(data: Mapping[str, Any], line: int,
                 problems: list[Problem]) -> None:
    """Validate the raw ``agent`` field before schema validation sees it.

    Done separately so an unknown agent is reported as *that*, rather than as a
    generic schema failure that says only "input should be a valid AgentId".
    ``ENVIRONMENT`` gets its own code: it is the simulated sponsor, and a trace
    that presents it as a reasoning agent is misrepresenting who decided what.
    """
    raw = data.get("agent")
    if raw is None:
        return
    try:
        agent = AgentId(str(raw))
    except ValueError:
        problems.append(Problem(
            PROBLEM_INVALID_AGENT,
            f"agent={raw!r} is not one of {sorted(a.value for a in AgentId)}",
            line=line, record=str(raw),
        ))
        return
    if not agent.is_reasoning_agent:
        problems.append(Problem(
            PROBLEM_ENVIRONMENT_AS_AGENT,
            f"agent={agent.value} is the simulated sponsor and must not appear "
            f"as a reasoning agent; use attributes to note sponsor interaction",
            line=line, record=agent.value,
        ))


def _check_events(events: list[tuple[int, TraceEvent]],
                  problems: list[Problem]) -> None:
    """``seq`` is 0-based, contiguous and increasing; ``ts`` never goes backwards."""
    expected = 0
    previous_ts = None
    for line, event in events:
        if event.seq != expected:
            if expected == 0:
                problems.append(Problem(
                    PROBLEM_SEQ_NOT_ZERO,
                    f"first event has seq={event.seq}; an append-only log starts "
                    f"at 0", line=line, record=f"seq={event.seq}",
                ))
            elif event.seq < expected:
                problems.append(Problem(
                    PROBLEM_SEQ_NOT_MONOTONIC,
                    f"seq went backwards: expected {expected}, got {event.seq}",
                    line=line, record=f"seq={event.seq}",
                ))
            else:
                problems.append(Problem(
                    PROBLEM_SEQ_GAP,
                    f"seq jumped from {expected - 1} to {event.seq}; "
                    f"{event.seq - expected} event(s) are missing",
                    line=line, record=f"seq={event.seq}",
                ))
        expected = max(expected, event.seq) + 1

        if previous_ts is not None and event.ts < previous_ts:
            problems.append(Problem(
                PROBLEM_TIMESTAMP_REGRESSION,
                f"ts moved backwards: {previous_ts.isoformat()} -> "
                f"{event.ts.isoformat()}",
                line=line, record=f"seq={event.seq}",
            ))
        previous_ts = event.ts


def _check_spans(spans: list[tuple[int, SpanRecord]],
                 problems: list[Problem]) -> None:
    """``end >= start``, and end times non-decreasing in file order."""
    previous_end: int | None = None
    for line, span in spans:
        if span.start_unix_nano < 0:
            problems.append(Problem(
                PROBLEM_SCHEMA_INVALID,
                f"start_unix_nano is negative ({span.start_unix_nano})",
                line=line, record=span.span_id,
            ))
        if span.end_unix_nano < span.start_unix_nano:
            problems.append(Problem(
                PROBLEM_END_BEFORE_START,
                f"span {span.name!r} ends {span.start_unix_nano - span.end_unix_nano} "
                f"ns before it starts", line=line, record=span.span_id,
            ))
        if previous_end is not None and span.end_unix_nano < previous_end:
            # See the module docstring: start order is *not* expected to be
            # monotonic; end order is, because that is the write order.
            problems.append(Problem(
                PROBLEM_TIMESTAMP_REGRESSION,
                f"span {span.name!r} ends before an earlier-written span ended; "
                f"the file is not in completion order",
                line=line, record=span.span_id,
            ))
        previous_end = span.end_unix_nano


def _check_parents(records: Iterable[tuple[int, Any]], known: set[str],
                   problems: list[Problem]) -> None:
    """Every non-None ``parent_span_id`` must name a record present in the file.

    Run *after* the whole file is read, which matters: a parent is written after
    its children, so a single-pass check would flag every valid parent as
    dangling.
    """
    for line, record in records:
        parent = record.parent_span_id
        if parent is None or parent in known:
            continue
        problems.append(Problem(
            PROBLEM_ORPHAN_PARENT,
            f"parent_span_id={parent} is not present in this file; the span tree "
            f"is not self-contained", line=line, record=record.span_id,
        ))


def _check_cycles(spans: list[SpanRecord], problems: list[Problem]) -> None:
    """Detect a cycle in the span tree.

    A real trace is acyclic by construction — OTel spans nest by stack depth. A
    cycle can therefore only come from a hand-edited or concatenated file, which
    is exactly why it is worth checking. Iterative with an explicit stack so a
    deep tree cannot blow the interpreter's recursion limit.
    """
    parents = {span.span_id: span.parent_span_id for span in spans}
    for start_id in parents:
        if start_id in parents and parents[start_id] == start_id:
            problems.append(Problem(
                PROBLEM_CYCLE, f"span {start_id} is its own parent",
                record=start_id,
            ))
            continue
        seen = {start_id}
        cursor = parents.get(start_id)
        while cursor is not None and cursor in parents:
            if cursor in seen:
                problems.append(Problem(
                    PROBLEM_CYCLE,
                    f"span {start_id} takes part in a parent cycle through {cursor}",
                    record=start_id,
                ))
                break
            seen.add(cursor)
            cursor = parents.get(cursor)


def _check_semantics(records: list[tuple[int, Any]],
                     problems: list[Problem]) -> None:
    """Decision provenance and model naming.

    Both checks are about *auditability*, not correctness of the run's output. A
    decision with no ``source`` cannot be attributed to a subsystem, so no
    ablation can be run over the trace; a model call with no model name cannot be
    compared between configurations. Both would pass schema validation, which is
    the point — the schema is permissive here on purpose, so this layer can be
    the one that says no.
    """
    for line, record in records:
        if record.kind is TraceKind.DECISION:
            source = _text(_attribute(record, "source", "decision.source"))
            if not source:
                problems.append(Problem(
                    PROBLEM_DECISION_MISSING_SOURCE,
                    "decision record has no non-empty source; every decision must "
                    "name the subsystem that produced it",
                    line=line, record=record.span_id,
                ))
            confidence = _as_float(_attribute(record, "confidence",
                                              "decision.confidence"))
            if confidence is None:
                problems.append(Problem(
                    PROBLEM_DECISION_CONFIDENCE,
                    f"decision confidence is "
                    f"{_attribute(record, 'confidence', 'decision.confidence')!r}, "
                    f"which is not a number",
                    line=line, record=record.span_id,
                ))
            elif not 0.0 <= confidence <= 1.0:
                problems.append(Problem(
                    PROBLEM_DECISION_CONFIDENCE,
                    f"decision confidence {confidence} is outside [0, 1]",
                    line=line, record=record.span_id,
                ))
        if record.kind is TraceKind.LLM:
            model = _text(_attribute(record, "model", "gen_ai.request.model"))
            if not model:
                problems.append(Problem(
                    PROBLEM_LLM_MISSING_MODEL,
                    "llm record names no model; a model call without a model id "
                    "cannot be compared across configurations",
                    line=line, record=record.span_id,
                ))


def _check_cadence(summary: TraceSummary, problems: list[Problem]) -> None:
    """Flag a suspiciously constant start cadence. **Warning, never an error.**

    See the module docstring: this is a nudge, not a verdict. Genuinely constant
    inter-arrival timing is a strong hint that timings were typed rather than
    measured — but the check cannot tell a fabricator from a fast, perfectly
    deterministic program, so it must not be able to fail a trace on its own.
    """
    gaps = summary.consecutive_start_gaps_ms
    if len(gaps) < MIN_GAPS_FOR_CADENCE_CHECK:
        return
    if summary.distinct_gap_values <= 1:
        interval = gaps[0]
        problems.append(Problem(
            PROBLEM_CONSTANT_CADENCE,
            f"every consecutive start gap is {interval} ms across {len(gaps)} "
            f"gap(s); real execution has jitter, so this looks synthesised. "
            f"Reported as a warning because it is not proof either way.",
            severity="warning", record=f"{len(gaps)} gaps",
        ))


# ==================================================================== entry point
def lint_trace(path: str | Path) -> list[Problem]:
    """Check one trace file and return every problem found.

    Never raises for a bad trace: reporting the problems *is* the contract, so a
    malformed line is a return value. Only an unreadable path or a directory
    raises, and those are programming errors.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"no trace at {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"{file_path} is a directory, not a trace file")

    problems: list[Problem] = []
    events: list[tuple[int, TraceEvent]] = []
    spans: list[tuple[int, SpanRecord]] = []

    for record in iter_trace_records(file_path):
        if not record.ok:
            problems.append(Problem(
                PROBLEM_INVALID_JSON, record.error or "unparseable line",
                line=record.line,
            ))
            continue
        data = record.data or {}
        kind = classify_record(data)
        if kind is None:
            problems.append(Problem(
                PROBLEM_UNKNOWN_RECORD,
                f"line is neither a TraceEvent (has seq) nor a SpanRecord "
                f"(has start_unix_nano); keys were {sorted(data)[:8]}",
                line=record.line,
            ))
            continue
        _check_agent(data, record.line, problems)
        try:
            if kind == "event":
                events.append((record.line, TraceEvent(**data)))
            else:
                spans.append((record.line, SpanRecord(**data)))
        except ValidationError as exc:
            problems.append(Problem(
                PROBLEM_SCHEMA_INVALID, _summarise_validation_error(exc),
                line=record.line,
            ))

    _check_events(events, problems)
    _check_spans(spans, problems)

    validated_spans = [span for _, span in spans]
    known_ids = {span.span_id for span in validated_spans}
    _check_parents(
        [(line, span) for line, span in spans], known_ids, problems)
    # An event's parent is the enclosing span, so it must be a span id.
    _check_parents(
        [(line, event) for line, event in events], known_ids, problems)
    _check_cycles(validated_spans, problems)
    _check_semantics(
        [(line, event) for line, event in events]
        + [(line, span) for line, span in spans],
        problems,
    )

    summary = compute_summary(file_path, "")
    _check_cadence(summary, problems)

    errors = sum(1 for problem in problems if problem.is_error)
    logger.info("lint %s: %d problem(s) (%d error(s))",
                file_path.name, len(problems), errors)
    return problems


def _summarise_validation_error(exc: ValidationError) -> str:
    """One-line pydantic error summary.

    ``ValidationError.errors()`` carries absolute ``loc`` paths and a 30-sentence
    ``ctx`` dump; for a trace reader the message alone is what matters.
    """
    parts: list[str] = []
    for error in exc.errors()[:4]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    if len(exc.errors()) > 4:
        parts.append(f"... and {len(exc.errors()) - 4} more")
    return "; ".join(parts)


def assert_clean(path: str | Path, *, strict: bool = False) -> TraceSummary:
    """Return the trace's summary, or raise :class:`TraceLintError`.

    Args:
        path: the trace file.
        strict: also fail on warnings. Off by default, because the cadence
            warning is a heuristic — see the module docstring. A caller that
            genuinely wants the strictest reading (a submission check) can ask
            for it; a CI gate on every trace should not.

    Raises:
        TraceLintError: with every problem listed, not just the first. A linter
            that stops at the first problem makes a fix loop ten times longer.
    """
    problems = lint_trace(path)
    blocking = [problem for problem in problems
                if problem.is_error or (strict and not problem.is_error)]
    if blocking:
        raise TraceLintError(
            f"{len(blocking)} blocking problem(s) in trace {Path(path)}:\n"
            f"{format_problems(blocking)}"
        )
    return compute_summary(path, "")


def lint_to_json(path: str | Path) -> str:  # pragma: no cover - CLI helper
    """Lint a trace and print the problems as one JSON array."""
    problems = [problem.to_dict() for problem in lint_trace(path)]
    return json.dumps(problems, indent=2, sort_keys=True)
