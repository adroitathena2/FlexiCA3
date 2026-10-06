"""Paytriq observability: the trace, and the two checks that make it trustworthy.

Three things live here, and they exist in this order of importance:

1. **The trace itself** (:class:`OtelTracer`, :class:`JsonlSpanExporter`). A
   JSONL file, one record per line, written as the run happens.
2. **A summary of it** (:func:`compute_summary`). Counts, timings, and the
   ``distinct_gap_values`` figure that makes synthetic timing detectable by
   anyone who can read the file.
3. **A linter for it** (:func:`lint_trace`, :func:`assert_clean`). Internal
   consistency: valid lines, contiguous ``seq``, resolvable parents, an acyclic
   tree, and a named source on every decision.

Plus :class:`JsonlReplayStore`, which is not about the trace but is the reason a
trace can be trusted: replay returns *recorded bytes* and nothing else, so an
offline demo is a performance of a previous run rather than an improvisation.

Import direction, per ``core.protocols``: this package imports ``core`` and
nothing else in the project. ``agents``, ``decision``, ``blackboard`` and
``tools`` import from here; nothing here imports them.
"""
from __future__ import annotations

from .exporter import (
    ATTR_AGENT,
    ATTR_CODE_SHA256,
    ATTR_EVENT_ID,
    ATTR_GIT_COMMIT,
    ATTR_RUN_ID,
    ATTR_STATUS,
    ATTR_TRACE_KIND,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_NAME,
    HOST_ID_SALT,
    JsonlSpanExporter,
    TraceRecord,
    classify_record,
    iter_trace_records,
    json_safe,
    record_from_span,
)
from .lint import (
    PROBLEM_CODES,
    Problem,
    TraceLintError,
    assert_clean,
    lint_trace,
)
from .replay import JsonlReplayStore, canonical_json
from .summary import compute_summary, gap_statistics, host_id
from .tracer import OtelTracer

__all__ = [
    # tracer
    "OtelTracer",
    # exporter / file format
    "JsonlSpanExporter",
    "record_from_span",
    "iter_trace_records",
    "classify_record",
    "json_safe",
    "TraceRecord",
    # summary
    "compute_summary",
    "gap_statistics",
    "host_id",
    # lint
    "lint_trace",
    "assert_clean",
    "Problem",
    "TraceLintError",
    "PROBLEM_CODES",
    # replay
    "JsonlReplayStore",
    "canonical_json",
    # shared attribute names
    "ATTR_TRACE_KIND", "ATTR_AGENT", "ATTR_STATUS", "ATTR_RUN_ID",
    "ATTR_EVENT_ID", "ATTR_GIT_COMMIT", "ATTR_CODE_SHA256",
    "GEN_AI_OPERATION_NAME", "GEN_AI_AGENT_NAME", "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_MODEL", "GEN_AI_TOOL_NAME", "HOST_ID_SALT",
]
