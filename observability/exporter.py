"""On-disk format for the Paytriq execution trace.

One JSON object per line, append-only, no wrapper. That format is not a
stylistic choice: a trace is the primary evidence an assessor reads, and JSONL
is the only format that can be ``grep``-ed, ``wc -l``-ed, and diffed without
loading it into a program. It is also trivially streamable, which is what lets
the API replay a run over SSE line by line while it is still executing.

Two record types share the file
-------------------------------
* ``TraceEvent``  — one per logical occurrence, carrying ``seq`` and ``ts``.
* ``SpanRecord``  — one per OTel span, carrying ``start_unix_nano``/``end_unix_nano``.

They are told apart structurally (a span record has ``start_unix_nano``; an
event has ``seq``) rather than by a discriminator field, because
``core.schemas`` sets ``extra="forbid"`` — inventing a new key would mean
mutating a frozen contract. :func:`classify_record` is the single place that
decision is made, so the reader, the summariser and the linter cannot disagree.

Why this module also owns the *event* sink
------------------------------------------
``JsonlSpanExporter`` is the sole writer of a trace file. Having the tracer keep
a second, independently-buffered handle open on the same path would be two
writers and one lock-free file: lines from a span export and lines from an event
could interleave mid-line, and ``wc -l`` would then disagree with the number of
parseable records. Rather than open that race, the exporter exposes
:meth:`JsonlSpanExporter.append_event` and the tracer calls it. One lock, one
handle, one ordering.

Ordering caveat that callers must know about
--------------------------------------------
``SimpleSpanProcessor`` exports on span **end**, so the file is written in
completion order: a child span's line always precedes its parent's.
:meth:`JsonlSpanExporter.export` therefore sorts each batch by
``start_unix_nano`` before writing, which restores start order *within* a batch
but cannot restore it across batches. Downstream code must sort before drawing
conclusions — :func:`observability.summary.compute_summary` does exactly that,
and :mod:`observability.lint` checks the invariant that actually holds (span
*end* times are monotonic, span start times are not).
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime
from datetime import time as datetime_time
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, StatusCode

from core.schemas import AgentId, SpanRecord, SpanStatus, TraceEvent, TraceKind

__all__ = [
    "JsonlSpanExporter",
    "record_from_span",
    "iter_trace_records",
    "classify_record",
    "json_safe",
    "attr_key",
    "write_trace_record",
    "TraceRecord",
    # reserved attribute names, shared by tracer.py and summary.py
    "ATTR_TRACE_KIND",
    "ATTR_AGENT",
    "ATTR_STATUS",
    "ATTR_RUN_ID",
    "ATTR_EVENT_ID",
    "ATTR_GIT_COMMIT",
    "ATTR_CODE_SHA256",
    "ATTR_SERVICE",
    "ATTR_ENVIRONMENT",
    "HOST_ID_SALT",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_AGENT_NAME",
    "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_TOOL_NAME",
    "ATTRIBUTE_PREFIX",
]

logger = logging.getLogger("paytriq.observability")

# --------------------------------------------------------------------- naming
#: Namespace for Paytriq's own attributes. Anything under this prefix is
#: Paytriq-specific and is *not* part of any OpenTelemetry semantic convention.
ATTRIBUTE_PREFIX = "paytriq."

#: Which ``TraceKind`` a span belongs to. ``SpanKind`` alone is too coarse:
#: INTERNAL covers agents, tools, decisions and handoffs alike.
ATTR_TRACE_KIND = f"{ATTRIBUTE_PREFIX}trace_kind"
#: The ``AgentId`` value, kept separately from ``gen_ai.agent.name`` (which is a
#: free-form label) so the summary can parse it back into the enum.
ATTR_AGENT = f"{ATTRIBUTE_PREFIX}agent"
#: ``SpanStatus`` as a string. Needed because OTel's StatusCode has no notion of
#: "a real failure was caught and a fallback answered instead" — which is the
#: single most important thing to know about a degraded run.
ATTR_STATUS = f"{ATTRIBUTE_PREFIX}status"
ATTR_RUN_ID = f"{ATTRIBUTE_PREFIX}run_id"
ATTR_EVENT_ID = f"{ATTRIBUTE_PREFIX}event_id"
ATTR_GIT_COMMIT = f"{ATTRIBUTE_PREFIX}git_commit"
ATTR_CODE_SHA256 = f"{ATTRIBUTE_PREFIX}code_sha256"
ATTR_SERVICE = f"{ATTRIBUTE_PREFIX}service"
ATTR_ENVIRONMENT = f"{ATTRIBUTE_PREFIX}environment"

#: OTel GenAI semantic-convention attribute names used by this project.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

#: Public salt for :func:`host_id`. Hard-coded and versioned on purpose: the
#: purpose of ``host.id`` in a trace is to let you tell "two runs, one machine"
#: apart from "one run, two machines" without publishing the hostname, which on
#: a laptop is often the owner's name. A per-run random salt would be strictly
#: better for privacy and strictly worse for that correlation, so the salt is
#: fixed and the digest is stable across runs of the same checkout.
HOST_ID_SALT = "paytriq.observability.host.v1"

_BYTES_PREFIX = "base64:"
_MAX_DEPTH = 12
_ATTR_KEY_INVALID = re.compile(r"[^A-Za-z0-9_.\-/]")

#: OTel span kind -> our trace kind, used only when a span carries no explicit
#: ``paytriq.trace_kind``. Coarse on purpose: every non-client, non-producer,
#: non-consumer span is orchestration, which is ``AGENT`` in our vocabulary.
_TRACE_KIND_BY_SPAN_KIND: dict[SpanKind, TraceKind] = {
    SpanKind.CLIENT: TraceKind.LLM,
    SpanKind.SERVER: TraceKind.AGENT,
    SpanKind.PRODUCER: TraceKind.AGENT,
    SpanKind.CONSUMER: TraceKind.AGENT,
    SpanKind.INTERNAL: TraceKind.AGENT,
}

_STATUS_BY_CODE: dict[StatusCode, SpanStatus] = {
    StatusCode.UNSET: SpanStatus.OK,
    StatusCode.OK: SpanStatus.OK,
    StatusCode.ERROR: SpanStatus.ERROR,
}

#: Span-name prefixes, used as a last-resort fallback when a span was created
#: outside :class:`~observability.tracer.OtelTracer` (e.g. by third-party
#: instrumentation) and therefore carries no Paytriq attributes.
_NAME_PREFIX_KINDS: tuple[tuple[str, TraceKind], ...] = (
    ("execute_tool ", TraceKind.TOOL),
    ("chat ", TraceKind.LLM),
    ("record_decision ", TraceKind.DECISION),
    ("handoff ", TraceKind.HANDOFF),
    ("message ", TraceKind.MESSAGE),
    ("invoke_agent ", TraceKind.AGENT),
)


# ==================================================================== JSON safety
def _truncate_repr(value: Any) -> str:
    """Last-resort stringification, bounded so one odd object cannot bloat a trace."""
    try:
        text = repr(value)
    except (TypeError, ValueError):  # a broken __repr__ must not kill the export
        text = f"<unrepresentable {type(value).__name__}>"
    return text if len(text) <= 200 else text[:197] + "..."


def json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Recursively coerce ``value`` into something ``json.dumps`` accepts losslessly.

    Callers pass free-form attribute dicts straight from agent code, so this
    function must be *total*: it may not raise on a hostile or exotic object, or
    a single ``Decimal`` in a probability map would lose the whole run's trace.
    Three cases need real handling rather than ``str()``:

    * ``NaN``/``Infinity`` — Python's ``json`` emits the bare tokens ``NaN`` and
      ``Infinity``, which are **not** valid JSON. Any strict parser downstream
      (including ``jq`` and most browsers) rejects the entire file. They become
      the strings ``"NaN"``/``"Infinity"``.
    * ``bytes`` — OTel attributes permit them. Base64 with a prefix, so a reader
      can tell an encoded blob from a string that happens to look like one.
    * ``datetime``/``Enum``/``Path`` — serialised structurally, not stringified,
      so a reader can parse them back.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if _depth >= _MAX_DEPTH:
        return _truncate_repr(value)
    if isinstance(value, Enum):
        return json_safe(value.value, _depth=_depth + 1)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _BYTES_PREFIX + base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Mapping):
        return {str(k): json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        # Sorted for determinism: two runs with the same inputs must produce
        # byte-identical lines or the trace stops being a reproducible record.
        return [json_safe(v, _depth=_depth + 1) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return _truncate_repr(value)


def attr_key(key: str) -> str:
    """Coerce an attribute name into one OTel will actually keep.

    OTel silently drops any key outside ``[A-Za-z0-9_.-/]`` with a log warning.
    "Silently" is the operative word: a lost attribute is a gap in the evidence,
    so keys are repaired here instead, and the repair is visible in the value.
    """
    cleaned = _ATTR_KEY_INVALID.sub("_", str(key)).strip()
    if not cleaned:
        return "_"
    if len(cleaned) > 256:
        logger.warning("attribute key %r truncated to 256 chars", str(key)[:64])
        cleaned = cleaned[:256]
    return cleaned


def safe_attributes(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise a caller-supplied attribute dict into OTel-safe, JSON-safe pairs."""
    if not attrs:
        return {}
    return {attr_key(k): json_safe(v) for k, v in attrs.items()}


# ================================================================== span mapping
def _trace_kind_for(span: ReadableSpan, attributes: Mapping[str, Any]) -> TraceKind:
    raw = attributes.get(ATTR_TRACE_KIND)
    if isinstance(raw, str):
        try:
            return TraceKind(raw)
        except ValueError:
            logger.warning("unknown %s=%r on span %r; falling back to span kind",
                           ATTR_TRACE_KIND, raw, span.name)
    mapped = _TRACE_KIND_BY_SPAN_KIND.get(span.kind, TraceKind.AGENT)
    if mapped is TraceKind.AGENT:
        for prefix, kind in _NAME_PREFIX_KINDS:
            if span.name.startswith(prefix):
                return kind
    return mapped


def _agent_for(attributes: Mapping[str, Any]) -> AgentId | None:
    raw = attributes.get(ATTR_AGENT, attributes.get(GEN_AI_AGENT_NAME))
    if raw is None:
        return None
    try:
        return AgentId(str(raw))
    except ValueError:
        # A non-AgentId label is not fatal: the span still exports, minus the
        # structured agent field. Losing the trace entirely would be worse.
        logger.warning("span attribute %s=%r is not an AgentId", ATTR_AGENT, raw)
        return None


def span_status_for(span: ReadableSpan, attributes: Mapping[str, Any]) -> SpanStatus:
    """Resolve a span's ``SpanStatus``.

    The explicit ``paytriq.status`` attribute wins over OTel's ``StatusCode``
    because ``SpanStatus.FALLBACK`` — "this failed and something else answered"
    — has no ``StatusCode`` equivalent, and conflating it with ``OK`` would hide
    exactly the degradation the project exists to make visible.
    """
    raw = attributes.get(ATTR_STATUS)
    if isinstance(raw, str):
        try:
            return SpanStatus(raw)
        except ValueError:
            logger.warning("unknown %s=%r on span %r; deriving from StatusCode",
                           ATTR_STATUS, raw, span.name)
    return _STATUS_BY_CODE.get(span.status.status_code, SpanStatus.OK)


def record_from_span(span: ReadableSpan) -> SpanRecord:
    """Project one ended OTel span onto the frozen :class:`SpanRecord` schema.

    ``trace_id``/``span_id`` are hex-encoded to the widths OTel specifies (32/16)
    so they can be pasted straight into a Jaeger/OTLP trace viewer even though
    this project never exports over the network.
    """
    attributes = json_safe(dict(span.attributes or {}))
    context = span.get_span_context()
    parent = span.parent
    trace_hex = f"{context.trace_id:032x}" if context is not None else ""
    parent_hex: str | None = None
    if parent is not None:
        parent_hex = f"{parent.span_id:016x}"
        if parent.trace_id != context.trace_id:
            # Cross-trace parenting (a span created under a provider that never
            # exported here) is not a parent *in this file*; recording it as one
            # would make the linter report a dangling reference that is real but
            # meaningless. Documented as: one tracer per run.
            parent_hex = None
    start = int(span.start_time)
    end = int(span.end_time) if span.end_time is not None else start
    return SpanRecord(
        trace_id=trace_hex,
        span_id=f"{context.span_id:016x}" if context is not None else "",
        parent_span_id=parent_hex,
        name=span.name,
        kind=_trace_kind_for(span, dict(span.attributes or {})),
        agent=_agent_for(dict(span.attributes or {})),
        start_unix_nano=start,
        end_unix_nano=max(start, end),
        status=span_status_for(span, dict(span.attributes or {})),
        attributes=attributes,
    )


# ==================================================================== trace file
class TraceRecord(NamedTuple):
    """One physical line of a trace file, parsed or not."""

    line: int                                   # 1-based, as a human would count
    data: dict[str, Any] | None              # None when the line is not JSON
    error: str | None                        # why parsing failed

    @property
    def ok(self) -> bool:
        return self.data is not None


def classify_record(data: Mapping[str, Any]) -> str | None:
    """Return ``"event"``, ``"span"``, or ``None`` if neither schema fits.

    Structural, not declarative — see the module docstring for why the schema
    cannot carry a discriminator field.
    """
    if "start_unix_nano" in data:
        return "span"
    if "seq" in data:
        return "event"
    return None


def iter_trace_records(path: str | Path) -> Iterator[TraceRecord]:
    """Stream a trace file line by line.

    Deliberately *not* raising on a malformed line: the linter's job is to
    report bad lines as problems, so a parse failure has to be a value it can
    iterate over, not an exception that aborts the scan at line 3 of 5000.
    Blank lines are skipped (trailing newlines, hand-edited files).
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        for lineno, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError) as exc:
                yield TraceRecord(lineno, None, f"invalid JSON: {exc}")
                continue
            if not isinstance(parsed, dict):
                yield TraceRecord(
                    lineno, None,
                    f"line is {type(parsed).__name__}, expected a JSON object",
                )
                continue
            yield TraceRecord(lineno, parsed, None)


def write_trace_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one already-serialised record. Used by tests and tooling only."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


class JsonlSpanExporter(SpanExporter):
    """Append-only JSONL span exporter, and sole owner of the trace file.

    Design commitments, in order of importance:

    * **Never truncate.** The file is opened in append mode for the lifetime of
      the exporter and reopened the same way on construction. Overwriting a
      previous attempt at the same ``run_id`` would destroy evidence; keeping
      both attempts in one file is auditable, and ``summary`` will report both.
    * **Flush on every export.** ``SimpleSpanProcessor`` hands us one span at a
      time, and the process may be killed at any point (a demo should be able to
      ``Ctrl-C`` and still have a trace). Buffering would defeat that.
    * **Fail the export, never the run.** If a span cannot be projected onto
      ``SpanRecord`` we log and return ``SpanExportResult.FAILURE``. Losing one
      span is bad; taking down the agent loop because of it is worse.
    """

    def __init__(self, path: str | Path, *, encoding: str = "utf-8") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._encoding = encoding
        # newline="\n" pins the line terminator: without it Windows would write
        # CRLF and a byte-diff of two traces would differ for no substantive
        # reason.
        self._handle = self._path.open("a", encoding=encoding, newline="\n")
        self._lock = threading.Lock()
        self._closed = False
        self._span_count = 0
        self._event_count = 0
        self._bytes_written = 0

    # ------------------------------------------------------------- properties
    @property
    def path(self) -> Path:
        return self._path

    @property
    def span_count(self) -> int:
        return self._span_count

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------- OTel
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Write one JSON object per span, sorted by start time.

        ``SimpleSpanProcessor`` exports at span *end*, so the incoming sequence is
        in completion order. Sorting by ``start_unix_nano`` before writing makes
        each batch read in the order the work actually happened, which is what a
        reader assumes when they open the file. Ties break on ``span_id`` purely
        to keep the output byte-stable across runs.
        """
        if self._closed:
            logger.error("export after shutdown; %d span(s) dropped", len(spans))
            return SpanExportResult.FAILURE
        try:
            records = [record_from_span(span) for span in spans]
        except (ValueError, TypeError, AttributeError) as exc:
            # ValueError covers pydantic's ValidationError, which is what an
            # out-of-range or missing SpanRecord field raises.
            logger.error("cannot project batch of %d span(s): %s", len(spans), exc)
            return SpanExportResult.FAILURE
        records.sort(key=lambda record: (record.start_unix_nano, record.span_id))
        with self._lock:
            if self._closed:
                return SpanExportResult.FAILURE
            for record in records:
                if self._write_locked(record.model_dump(mode="json")):
                    self._span_count += 1
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Flush and close. Idempotent, because OTel may call it during teardown."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._handle.flush()
                self._handle.close()
            except OSError as exc:
                logger.error("failed to close trace file %s: %s", self._path, exc)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush: writes are already synchronous. Returns False if closed."""
        if self._closed:
            return False
        with self._lock:
            if self._closed:
                return False
            try:
                self._handle.flush()
            except OSError as exc:
                logger.error("flush of %s failed: %s", self._path, exc)
                return False
        return True

    # ------------------------------------------------------------------ writes
    def append_event(self, event: TraceEvent) -> None:
        """Append one ``TraceEvent`` line.

        Called by :class:`~observability.tracer.OtelTracer` so that events reach
        disk through this single lock rather than a second file handle.
        """
        self.append_mapping(event.model_dump(mode="json"))

    def append_record(self, record: SpanRecord) -> None:
        """Append one ``SpanRecord`` line directly (bypasses the span pipeline)."""
        self.append_mapping(record.model_dump(mode="json"))

    def append_mapping(self, record: Mapping[str, Any]) -> None:
        """Append an already-serialised mapping. Serialisation errors are logged."""
        with self._lock:
            if self._closed:
                logger.error("append after shutdown dropped: %s", sorted(record)[:6])
                return
            if self._write_locked(record):
                self._event_count += 1

    def _write_locked(self, record: Mapping[str, Any]) -> bool:
        """Serialise and write one line. Caller holds ``self._lock``.

        Returns True on success. Every failure mode is logged rather than raised:
        a trace write must never be the thing that kills a run.
        """
        try:
            line = json.dumps(json_safe(dict(record)), sort_keys=True,
                              ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.error("unserialisable trace record %s: %s", sorted(record)[:6], exc)
            return False
        try:
            self._handle.write(line)
            self._handle.write("\n")
            self._handle.flush()
        except OSError as exc:
            logger.error("write to %s failed: %s", self._path, exc)
            return False
        self._bytes_written += len(line) + 1
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<JsonlSpanExporter {self._path.name} spans={self._span_count} "
                f"events={self._event_count} closed={self._closed}>")
