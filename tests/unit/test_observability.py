"""Unit tests for the ``observability`` package.

Scope, and why it is this scope
-------------------------------
These tests cover the parts of this package whose *correctness is arithmetic*:
the on-disk record shape, the parent/child linkage, the gap computation, the
linter's checks, and replay key stability. That is the surface where a bug would
silently corrupt the evidence a run presents about itself.

What is deliberately **not** here: assertions that a specific number of spans
exists. Real timing is not reproducible, so a test pinning gap values would either
be flaky or would freeze the very jitter the gap metric exists to measure. The
tests instead assert properties that must hold for any genuine capture (gaps are
irregular) and for any hand-written one (a constant interval de-duplicates to
exactly one value).

No fixture data is hardcoded anywhere. Traces under test are produced by running
real spans through the real exporter, and the one synthetic file (the
constant-cadence trace) is built from arithmetic in the test body so its
provenance is visible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Self-sufficient path bootstrap: this file lives two levels below the repo root
# and there is no tests/conftest.py, so make the root importable whether pytest
# was started with `pytest` or `python -m pytest`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datetime import UTC  # noqa: E402

import pytest  # noqa: E402

from core.config import Settings  # noqa: E402
from core.errors import ReplayIntegrityError  # noqa: E402
from core.ids import new_id  # noqa: E402
from core.ids import span_id as new_span_id  # noqa: E402
from core.ids import trace_id as new_trace_id  # noqa: E402
from core.protocols import ReplayStore, Tracer  # noqa: E402
from core.schemas import (  # noqa: E402
    AgentId,
    Decision,
    DecisionSource,
    GateKind,
    GateOutcome,
    Handoff,
    ToolStatus,
    TraceKind,
)
from observability import (  # noqa: E402
    JsonlReplayStore,
    JsonlSpanExporter,
    OtelTracer,
    Problem,
    TraceLintError,
    assert_clean,
    classify_record,
    compute_summary,
    gap_statistics,
    host_id,
    json_safe,
    lint_trace,
)
from observability.exporter import (  # noqa: E402
    ATTR_STATUS,
    ATTR_TRACE_KIND,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_NAME,
)
from observability.summary import assert_ids_are_hex  # noqa: E402

RUN_ID = "run_unit_observability"


# ===================================================================== fixtures
@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely inside a temp dir. Touches no repo state."""
    return Settings(
        traces_dir=tmp_path / "traces",
        replay_path=tmp_path / "replay.jsonl",
        git_commit="deadbeefcafe",
        code_sha256="f" * 64,
    )


@pytest.fixture()
def replay_store(settings: Settings) -> JsonlReplayStore:
    return JsonlReplayStore(settings.replay_path)


@pytest.fixture()
def tracer(settings: Settings, replay_store: JsonlReplayStore) -> OtelTracer:
    """A configured tracer. Tests that need their own spans build their own."""
    made = OtelTracer(settings, replay_store=replay_store)
    made.configure(RUN_ID, service="paytriq", event_id=new_id("evt"))
    return made


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path: Path, records: list[Any]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _span_record(
    *, span_id: str, start_ns: int, end_ns: int,
    kind: str = "agent", parent: str | None = None,
    name: str = "invoke_agent A1.plan", attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One minimal, schema-valid ``SpanRecord`` payload.

    Built by this helper rather than pasted as JSON so the test's inputs are
    obviously synthetic and obviously minimal — no plausible-looking content.
    """
    return {
        "trace_id": new_trace_id(),
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "kind": kind,
        "agent": None,
        "start_unix_nano": start_ns,
        "end_unix_nano": end_ns,
        "status": "ok",
        "attributes": attributes or {ATTR_TRACE_KIND: kind},
    }


# ========================================================== exporter: output shape
def test_exporter_works_standalone_without_a_tracer(settings: Settings) -> None:
    """The exporter is a standalone OTel ``SpanExporter``, not tracer-coupled.

    Driven directly through ``TracerProvider`` + ``SimpleSpanProcessor`` so the
    test would fail if the class only worked when :class:`OtelTracer` happened to
    be the caller.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.trace import SpanKind

    path = settings.traces_dir / "standalone.jsonl"
    exporter = JsonlSpanExporter(path)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = provider.get_tracer("unit-test")
    with otel.start_span(
        "invoke_agent A1.plan",
        kind=SpanKind.INTERNAL,
        attributes={
            ATTR_TRACE_KIND: "agent",
            ATTR_STATUS: "ok",
            GEN_AI_OPERATION_NAME: "invoke_agent",
            GEN_AI_AGENT_NAME: "A1",
        },
    ):
        pass
    provider.shutdown()

    assert path.exists()
    lines = _read_lines(path)
    assert len(lines) == 1
    record = lines[0]
    assert set(record) == {
        "trace_id", "span_id", "parent_span_id", "name", "kind", "agent",
        "start_unix_nano", "end_unix_nano", "status", "attributes",
    }
    assert record["name"] == "invoke_agent A1.plan"
    assert record["kind"] == "agent"
    assert record["agent"] == "A1"
    assert record["status"] == "ok"
    assert record["attributes"][GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert record["attributes"][GEN_AI_AGENT_NAME] == "A1"
    assert exporter.span_count == 1
    assert exporter.closed is True


def test_exported_span_ids_are_correctly_sized_hex(tracer: OtelTracer) -> None:
    """32 hex for a trace id, 16 for a span id — pasteable into any OTel viewer."""
    from core.schemas import SpanRecord

    with tracer.agent(AgentId.A1_DISCOVERY, "A1.plan", step=0):
        pass
    spans = [line for line in _read_lines(Path(tracer.finish()["trace_file"]))
             if classify_record(line) == "span"]
    assert spans
    assert_ids_are_hex([SpanRecord(**line) for line in spans])


def test_exporter_output_is_strict_json_even_for_awkward_attributes(
        settings: Settings) -> None:
    """NaN, bytes and enums must not emit the invalid JSON tokens ``NaN``/``Infinity``."""
    from datetime import datetime
    from enum import StrEnum
    from pathlib import Path as P

    class Flavour(StrEnum):
        SALTY = "salty"

    payloads = [
        float("nan"), float("inf"), -float("inf"), 10 ** 30,
        b"\x00\xff", {"nested": [1, 2, {"deep": True}]}, ("t", 1),
        {1, 2, 3}, Flavour.SALTY, datetime(2026, 1, 1, tzinfo=UTC),
        P("a/b"), ValueError("boom"),
    ]
    safe = json_safe(payloads)
    # Round-trips through a strict parser: json.loads by default *accepts* NaN,
    # so parse_constant is used to make its acceptance an error.
    json.loads(json.dumps(safe), parse_constant=_reject_constant)
    assert safe[0] == "nan"
    assert safe[1] == "inf"
    assert safe[2] == "-inf"
    assert safe[3] == 10 ** 30
    assert safe[4].startswith("base64:")
    assert safe[6] == ["t", 1]
    assert safe[7] == [1, 2, 3]
    assert safe[8] == "salty"
    # str(Path) is platform-native, so the expectation is platform-native too.
    assert safe[10] == str(P("a/b"))
    assert safe[11] == "ValueError: boom"


def _reject_constant(token: str) -> Any:
    raise AssertionError(f"non-JSON constant {token!r} reached the output")


def test_exporter_sorts_a_batch_by_start_time(settings: Settings) -> None:
    """A batch is written in start order even when it arrives in end order.

    This is the ``SimpleSpanProcessor`` problem: spans are handed over when they
    *end*, so a batch can arrive backwards. Without the sort the trace file reads
    in completion order and any reader who assumes chronological order is misled.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import SpanKind

    base = 1_700_000_000_000_000_000
    path = settings.traces_dir / "batch.jsonl"
    exporter = JsonlSpanExporter(path)
    otel = TracerProvider().get_tracer("unit-test")

    # s3 starts last and ends first; export order is the reverse of start order.
    spans = []
    for index in (1, 2, 3):
        span = otel.start_span(
            f"s{index}", kind=SpanKind.INTERNAL,
            attributes={ATTR_TRACE_KIND: "agent"},
            start_time=base + index * 1_000_000,
        )
        span.end(end_time=base + 10_000_000 - index * 1_000_000)
        spans.append(span)

    exporter.export(list(reversed(spans)))
    exporter.shutdown()

    written = [line["name"] for line in _read_lines(path)]
    assert written == ["s1", "s2", "s3"]
    starts = [line["start_unix_nano"] for line in _read_lines(path)]
    assert starts == sorted(starts)
    assert exporter.span_count == 3


# ==================================================== tracer: semconv and linkage
def test_span_kinds_follow_genai_semconv(tracer: OtelTracer) -> None:
    """agent/tool are INTERNAL; llm is CLIENT. This distinction is load-bearing."""
    captured: list[tuple[str, Any, dict[str, Any]]] = []

    with tracer.agent(AgentId.A1_DISCOVERY, "A1.plan") as agent_span:
        captured.append(("agent", agent_span, dict(agent_span.attributes)))
        with tracer.tool(AgentId.A1_DISCOVERY, "overpass_nearby") as tool_span:
            captured.append(("tool", tool_span, dict(tool_span.attributes)))
            with tracer.llm(AgentId.A1_DISCOVERY, "gemini-3-flash") as llm_span:
                captured.append(("llm", llm_span, dict(llm_span.attributes)))

    by_name = {name: (span, attrs) for name, span, attrs in captured}
    agent, agent_attrs = by_name["agent"]
    tool, tool_attrs = by_name["tool"]
    llm, llm_attrs = by_name["llm"]

    assert agent.kind.name == "INTERNAL"
    assert tool.kind.name == "INTERNAL"
    assert llm.kind.name == "CLIENT"

    assert agent.name == "invoke_agent A1.plan"
    assert tool.name == "execute_tool overpass_nearby"
    assert llm.name == "chat gemini-3-flash"

    assert agent_attrs[GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert tool_attrs[GEN_AI_TOOL_NAME] == "overpass_nearby"
    assert llm_attrs[GEN_AI_PROVIDER_NAME] == "gemini"
    assert llm_attrs[GEN_AI_REQUEST_MODEL] == "gemini-3-flash"


def test_handoff_attributes_live_outside_the_reserved_genai_namespace(
        tracer: OtelTracer) -> None:
    """Handoffs have no standard OTel attribute, so they must not squat in gen_ai.*."""
    handoff = Handoff(
        handoff_id=new_id("hnd"), event_id=tracer.event_id, run_id=tracer.run_id,
        from_agent=AgentId.A1_DISCOVERY, to_agent=AgentId.A2_PRICING,
        reason="leads found", decision_source=DecisionSource.CLEF, confidence=0.81,
        summary="three leads within 3 km", payload_refs=["brd_1", "brd_2"],
    )
    tracer.handoff(handoff)
    spans = [line for line in _read_lines(Path(tracer.finish()["trace_file"]))
             if classify_record(line) == "span"]
    handoff_spans = [line for line in spans if line["kind"] == "handoff"]
    assert len(handoff_spans) == 1
    attrs = handoff_spans[0]["attributes"]

    assert attrs["handoff.from"] == "A1"
    assert attrs["handoff.to"] == "A2"
    assert attrs["handoff.reason"] == "leads found"
    assert attrs["handoff.decision_source"] == "clef"
    assert attrs["handoff.confidence"] == 0.81
    # Nothing handoff-shaped may appear inside the specification-reserved space.
    assert not [key for key in attrs if key.startswith("gen_ai.handoff")]
    assert not [key for key in attrs if key == "gen_ai.handoff.to"]


def test_handoff_appears_as_both_an_event_and_a_span(tracer: OtelTracer) -> None:
    """Both views, so a handoff is visible in the stream *and* the span tree."""
    tracer.handoff(Handoff(
        handoff_id=new_id("hnd"), event_id=tracer.event_id, run_id=tracer.run_id,
        from_agent=AgentId.A2_PRICING, to_agent=AgentId.A3_OUTREACH,
        reason="offer drafted", decision_source=DecisionSource.RULES, confidence=0.9,
    ))
    lines = _read_lines(Path(tracer.finish()["trace_file"]))
    events = [line for line in lines if classify_record(line) == "event"]
    spans = [line for line in lines if classify_record(line) == "span"]
    handoff_events = [line for line in events if line["kind"] == "handoff"]
    handoff_spans = [line for line in spans if line["kind"] == "handoff"]

    assert len(handoff_events) == 1 and len(handoff_spans) == 1
    # The event joins to the span it was emitted inside.
    assert handoff_events[0]["span_id"] == handoff_spans[0]["span_id"]
    assert handoff_events[0]["attributes"]["to_agent"] == "A3"
    # Counted once, not twice: summary reads events for semantic counts.
    assert compute_summary(tracer.trace_path, tracer.run_id).handoff_count == 1


def test_message_appears_as_both_an_event_and_a_span(tracer: OtelTracer) -> None:
    """Track 3: Message kind mirrors handoff — one event plus one span."""
    from core.schemas import Message, MessageKind

    tracer.message(Message(
        message_id="msg_1", event_id=tracer.event_id, run_id=tracer.run_id,
        from_agent=AgentId.A1_DISCOVERY, to_agent=AgentId.A2_PRICING,
        kind=MessageKind.PROPOSAL, body="proposing 45k silver tier",
        refs=["be_1"], decision_source=DecisionSource.CLEF, confidence=0.77,
        summary="opening proposal",
    ))
    lines = _read_lines(Path(tracer.finish()["trace_file"]))
    events = [line for line in lines if classify_record(line) == "event"]
    spans = [line for line in lines if classify_record(line) == "span"]
    message_events = [line for line in events if line["kind"] == "message"]
    message_spans = [line for line in spans if line["kind"] == "message"]

    assert len(message_events) == 1 and len(message_spans) == 1
    # The event joins to the span it was emitted inside.
    assert message_events[0]["span_id"] == message_spans[0]["span_id"]
    assert message_spans[0]["name"] == "message A1->A2"
    attrs = message_spans[0]["attributes"]
    assert attrs["message.from"] == "A1"
    assert attrs["message.to"] == "A2"
    assert attrs["message.decision_source"] == "clef"
    assert attrs["message.confidence"] == 0.77
    assert attrs["message.id"] == "msg_1"
    assert not [key for key in attrs if key.startswith("gen_ai.message")]
    assert message_events[0]["attributes"]["to_agent"] == "A2"
    # Counted once, not twice: summary reads events for semantic counts.
    assert compute_summary(tracer.trace_path, tracer.run_id).message_count == 1
    assert_clean(tracer.trace_path)


def test_parent_child_linkage_across_nested_operations(tracer: OtelTracer) -> None:
    """A tool inside an agent gets the agent's span id as its parent."""
    with tracer.agent(AgentId.A1_DISCOVERY, "A1.act", step=1) as outer:
        outer_id = f"{outer.get_span_context().span_id:016x}"
        with tracer.tool(AgentId.A1_DISCOVERY, "overpass_nearby") as middle:
            middle_id = f"{middle.get_span_context().span_id:016x}"
            with tracer.llm(AgentId.A1_DISCOVERY, "gemini-3-flash") as inner:
                inner_id = f"{inner.get_span_context().span_id:016x}"
        assert middle.parent.span_id == outer.get_span_context().span_id
        assert inner.parent.span_id == middle.get_span_context().span_id

    lines = _read_lines(Path(tracer.finish()["trace_file"]))
    spans = {line["span_id"]: line for line in lines
             if classify_record(line) == "span"}
    assert spans[middle_id]["parent_span_id"] == outer_id
    assert spans[inner_id]["parent_span_id"] == middle_id
    # The outermost span is the only root.
    assert spans[outer_id]["parent_span_id"] is None
    assert compute_summary(tracer.trace_path, tracer.run_id).root_span_count == 1


def test_events_are_sequenced_and_monotonic(tracer: OtelTracer) -> None:
    """seq starts at 0, increments by exactly one, and ts never goes backwards."""
    for index in range(3):
        with tracer.agent(AgentId.A1_DISCOVERY, f"A1.step{index}"):
            time.sleep(0.001 * (index + 1))

    events = tracer.events
    assert [event.seq for event in events] == list(range(len(events)))
    timestamps = [event.ts for event in events]
    assert timestamps == sorted(timestamps)
    assert len(tracer.events) == len(events)  # reading events copies the list
    assert_clean(tracer.finish()["trace_file"])


def test_decision_records_its_full_provenance(tracer: OtelTracer) -> None:
    """source/confidence/degraded/model/latency_ms, in both naming conventions."""
    decision = Decision(
        request_id=new_id("dec"), question="escalate?", choice="no",
        probabilities={"yes": 0.31, "no": 0.69}, confidence=0.69,
        source=DecisionSource.RULES, model="rules-engine", latency_ms=1.5,
        degraded=True,
    )
    with tracer.decision(AgentId.A7_ARBITER, decision):
        pass

    lines = _read_lines(Path(tracer.finish()["trace_file"]))
    span = next(line for line in lines
                if classify_record(line) == "span" and line["kind"] == "decision")
    assert span["attributes"]["decision.source"] == "rules"
    assert span["attributes"]["decision.confidence"] == 0.69
    assert span["attributes"]["decision.degraded"] is True
    assert span["attributes"]["decision.model"] == "rules-engine"
    assert span["attributes"]["decision.latency_ms"] == 1.5

    event = next(line for line in lines
                 if classify_record(line) == "event" and line["kind"] == "decision")
    assert event["attributes"]["source"] == "rules"
    assert event["attributes"]["degraded"] is True
    assert ATTR_STATUS in span["attributes"]


def test_exception_inside_a_span_is_recorded_as_error(tracer: OtelTracer) -> None:
    """The tracer observes failures; it never swallows them."""
    with pytest.raises(RuntimeError, match="boom"), tracer.tool(AgentId.A2_PRICING, "maps_matrix"):
        raise RuntimeError("boom")
    lines = _read_lines(Path(tracer.finish()["trace_file"]))
    spans = [line for line in lines if classify_record(line) == "span"]
    failed = [line for line in spans if line["name"] == "execute_tool maps_matrix"]
    assert len(failed) == 1
    assert failed[0]["status"] == "error"


def test_finish_writes_a_summary_sidecar_and_is_idempotent(
        tracer: OtelTracer) -> None:
    with tracer.agent(AgentId.A6_AUDIT, "A6.observe"):
        pass
    first = tracer.finish()
    second = tracer.finish()
    assert first == second
    sidecar = Path(first["summary_file"])
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["run_id"] == tracer.run_id
    assert payload["span_count"] == first["span_count"]
    assert first["integrity_ok"] is True


def test_tracer_satisfies_the_tracer_protocol(tracer: OtelTracer) -> None:
    assert isinstance(tracer, Tracer)
    assert isinstance(JsonlReplayStore(Path("unused.jsonl"), read_only=True),
                      ReplayStore)


# ======================================================================= summary
def _irregular_run(tracer: OtelTracer) -> None:
    """A run whose operations genuinely take different amounts of time."""
    with tracer.agent(AgentId.A1_DISCOVERY, "A1.plan"):
        time.sleep(0.0012)
        with tracer.tool(AgentId.A1_DISCOVERY, "overpass_nearby"):
            time.sleep(0.0061)
            with tracer.llm(AgentId.A1_DISCOVERY, "gemini-3-flash"):
                time.sleep(0.0027)
        with tracer.tool(AgentId.A1_DISCOVERY, "nominatim_geocode"):
            time.sleep(0.0043)
    with tracer.agent(AgentId.A2_PRICING, "A2.plan"):
        time.sleep(0.0009)


def test_real_run_has_many_distinct_gap_values(tracer: OtelTracer) -> None:
    """The anti-fabrication property: genuine timing is irregular.

    This is the property the project publishes. A hand-written trace has a
    constant cadence and scores 1 (see
    :func:`test_hand_written_constant_cadence_yields_exactly_one`); a real
    capture scores many.
    """
    _irregular_run(tracer)
    summary = compute_summary(tracer.finish()["trace_file"], tracer.run_id)

    assert summary.span_count == 5
    # +1 for the run.configure event configure() emits to make the file
    # self-describing; it is not an operation, so it contributes no span.
    assert summary.event_count == 6
    assert len(summary.consecutive_start_gaps_ms) == 4
    assert summary.distinct_gap_values == len(
        set(summary.consecutive_start_gaps_ms))
    assert summary.distinct_gap_values >= 3, (
        "a genuine capture must show irregular inter-arrival timing; got "
        f"{summary.consecutive_start_gaps_ms}"
    )
    assert summary.wall_clock_ms > 0
    assert summary.root_span_count == 2


def test_hand_written_constant_cadence_yields_exactly_one(
        settings: Settings, tmp_path: Path) -> None:
    """A constant interval must de-duplicate to exactly 1.

    Built arithmetically rather than pasted as JSON: the point of the fixture is
    that its times are *synthetic by construction* — every start is exactly
    ``base + i * 100ms`` — and that must be obvious from the test source.
    """
    base = 1_700_000_000_000_000_000
    step = 100_000_000  # exactly 100 ms
    records = [
        _span_record(span_id=new_span_id(),
                     start_ns=base + index * step,
                     end_ns=base + index * step + 1_000_000)
        for index in range(6)
    ]
    path = _write_lines(tmp_path / "constant.jsonl", records)
    summary = compute_summary(path, "run_handwritten")

    assert summary.consecutive_start_gaps_ms == [100.0] * 5
    assert summary.distinct_gap_values == 1, (
        "a constant interval MUST collapse to exactly one distinct gap value; "
        f"got {summary.distinct_gap_values}"
    )
    assert summary.event_count == 0 and summary.span_count == 6

    # And the linter must *mention* it, as a warning rather than an error: this
    # file is internally consistent, which is exactly the point.
    codes = {problem.code for problem in lint_trace(path)}
    assert codes == {"constant_cadence"}
    warnings = [problem for problem in lint_trace(path)
                if problem.code == "constant_cadence"]
    assert warnings[0].severity == "warning"
    assert_clean(path)                      # warnings do not fail by default
    with pytest.raises(TraceLintError):
        assert_clean(path, strict=True)


def test_gap_statistics_is_pure_and_order_independent() -> None:
    """Inputs are nanoseconds and may arrive in any order."""
    starts = [5_000_000, 1_000_000, 9_000_000, 3_000_000]
    gaps, distinct = gap_statistics(starts)
    assert gaps == [2.0, 2.0, 4.0]
    assert distinct == 2
    assert gap_statistics(list(reversed(starts))) == (gaps, distinct)
    # Zero and one operation cannot produce a gap; say so rather than invent one.
    assert gap_statistics([]) == ([], 0)
    assert gap_statistics([7]) == ([], 0)


def test_summary_counts_every_documented_quantity(
        tracer: OtelTracer, replay_store: JsonlReplayStore) -> None:
    """handoffs, decisions by source, degraded decisions, tools, gates, wall clock."""
    with tracer.agent(AgentId.A1_DISCOVERY, "A1.plan"):
        with tracer.tool(AgentId.A1_DISCOVERY, "overpass_nearby",
                         status=ToolStatus.CACHED, degraded=True):
            time.sleep(0.002)
        with tracer.tool(AgentId.A2_PRICING, "maps_matrix",
                         status=ToolStatus.UNAVAILABLE):
            time.sleep(0.003)
        with tracer.llm(AgentId.A1_DISCOVERY, "gemini-3-flash"):
            time.sleep(0.001)

    for source, degraded in ((DecisionSource.CLEF, False),
                             (DecisionSource.RULES, True),
                             (DecisionSource.RULES, False)):
        with tracer.decision(AgentId.A7_ARBITER, Decision(
            request_id=new_id("dec"), question="continue?", choice="yes",
            probabilities={"yes": 0.6, "no": 0.4}, confidence=0.6,
            source=source, model=f"m-{source.value}", degraded=degraded,
        )):
            pass

    tracer.handoff(Handoff(
        handoff_id=new_id("hnd"), event_id=tracer.event_id, run_id=tracer.run_id,
        from_agent=AgentId.A1_DISCOVERY, to_agent=AgentId.A2_PRICING,
        reason="leads", decision_source=DecisionSource.CLEF, confidence=0.7,
    ))
    tracer.event(TraceKind.HUMAN, "gate.send", agent=AgentId.A3_OUTREACH,
                 gate_kind=GateKind.SEND.value, outcome=GateOutcome.APPROVE.value)
    tracer.event(TraceKind.HUMAN, "gate.mou", agent=AgentId.A4_CONTRACT,
                 gate_kind=GateKind.MOU.value, outcome=GateOutcome.APPROVE.value)

    summary = compute_summary(tracer.finish()["trace_file"], tracer.run_id)
    assert summary.handoff_count == 1
    assert summary.llm_call_count == 1
    assert summary.tool_call_count == 2
    assert summary.tool_status_counts == {"cached": 1, "unavailable": 1}
    assert summary.human_gate_count == 2
    assert summary.decision_counts == {"clef": 1, "rules": 2}
    assert summary.degraded_decision_count == 1
    assert summary.wall_clock_ms > 0
    # Provenance survives into the summary, so a trace identifies its own build.
    assert summary.git_commit == "deadbeefcafe"
    assert summary.code_sha256 == "f" * 64
    assert summary.run_id == tracer.run_id
    assert summary.trace_file.endswith(".jsonl")


# ========================================================================= lint
def test_lint_catches_planted_problems(tmp_path: Path) -> None:
    """One problem per check, planted in a single file so nothing hides."""
    base = 1_700_000_000_000_000_000
    good = new_span_id()
    records: list[Any] = [
        # invalid JSON, spliced in as raw text below
        _span_record(span_id=good, start_ns=base, end_ns=base + 5_000_000),
        # end before start
        _span_record(span_id=new_span_id(), start_ns=base + 10_000_000,
                     end_ns=base + 1_000_000),
        # parent that is not in the file
        _span_record(span_id=new_span_id(), start_ns=base + 20_000_000,
                     end_ns=base + 21_000_000, parent=new_span_id()),
        # a two-node parent cycle
        _span_record(span_id="a" * 16, start_ns=base + 30_000_000,
                     end_ns=base + 31_000_000, parent="b" * 16),
        _span_record(span_id="b" * 16, start_ns=base + 31_000_000,
                     end_ns=base + 32_000_000, parent="a" * 16),
        # decision with no source
        {"seq": 0, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
         "span_id": new_span_id(), "parent_span_id": None, "kind": "decision",
         "name": "decide:dec_x", "agent": None, "attributes": {}},
        # llm record naming no model
        {"seq": 1, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
         "span_id": new_span_id(), "parent_span_id": None, "kind": "llm",
         "name": "chat", "agent": None, "attributes": {}},
        # seq jumps from 1 to 3
        {"seq": 3, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
         "span_id": new_span_id(), "parent_span_id": None, "kind": "agent",
         "name": "A3.plan", "agent": None, "attributes": {}},
        # ENVIRONMENT presented as a reasoning agent
        {"seq": 4, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
         "span_id": new_span_id(), "parent_span_id": None, "kind": "agent",
         "name": "sponsor", "agent": "ENV", "attributes": {}},
        # not a record of either type at all
        {"hello": "world"},
        # schema violation: seq must be >= 0
        {"seq": -1, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
         "span_id": new_span_id(), "parent_span_id": None, "kind": "agent",
         "name": "A1.plan", "agent": None, "attributes": {}},
        # confidence out of range on a decision
        {"seq": 5, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
         "span_id": new_span_id(), "parent_span_id": None, "kind": "decision",
         "name": "decide:dec_y", "agent": None,
         "attributes": {"source": "clef", "confidence": 1.4}},
    ]
    path = tmp_path / "planted.jsonl"
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    body += "{not json at all\n"
    path.write_text(body, encoding="utf-8")

    codes = {problem.code for problem in lint_trace(path)}
    for expected in ("invalid_json", "end_before_start", "orphan_parent", "cycle",
                     "decision_missing_source", "llm_missing_model", "seq_gap",
                     "environment_as_agent", "unknown_record", "schema_invalid",
                     "decision_confidence_range"):
        assert expected in codes, f"lint missed {expected}; found {sorted(codes)}"

    with pytest.raises(TraceLintError) as excinfo:
        assert_clean(path)
    # Every problem is reported, not just the first.
    assert "decision_missing_source" in str(excinfo.value)
    assert "llm_missing_model" in str(excinfo.value)


def test_lint_flags_a_non_zero_based_sequence(tmp_path: Path) -> None:
    record = {"seq": 5, "run_id": RUN_ID, "event_id": "", "trace_id": new_trace_id(),
              "span_id": new_span_id(), "parent_span_id": None, "kind": "agent",
              "name": "A1.plan", "agent": None, "attributes": {}}
    path = _write_lines(tmp_path / "seq.jsonl", [record])
    codes = {problem.code for problem in lint_trace(path)}
    assert "seq_not_zero_based" in codes
    with pytest.raises(TraceLintError):
        assert_clean(path)


def test_lint_flags_a_timestamp_regression(tmp_path: Path) -> None:
    """An event whose ts goes backwards means a record was reordered."""
    base = 1_700_000_000_000_000_000
    first = _span_record(span_id=new_span_id(), start_ns=base,
                         end_ns=base + 9_000_000)
    second = _span_record(span_id=new_span_id(), start_ns=base + 20_000_000,
                          end_ns=base + 21_000_000)
    path = _write_lines(tmp_path / "regression.jsonl", [second, first])
    codes = {problem.code for problem in lint_trace(path)}
    assert "timestamp_regression" in codes


def test_lint_accepts_span_start_times_out_of_order(tmp_path: Path) -> None:
    """Child-before-parent is the *expected* write order and must not be flagged.

    ``SimpleSpanProcessor`` exports on span end, so a child is always written
    first. A linter that flagged this would fail every genuine trace.
    """
    base = 1_700_000_000_000_000_000
    parent = new_span_id()
    child = new_span_id()
    path = _write_lines(tmp_path / "completion_order.jsonl", [
        _span_record(span_id=child, start_ns=base + 5_000_000,
                     end_ns=base + 9_000_000, parent=parent, kind="tool"),
        _span_record(span_id=parent, start_ns=base, end_ns=base + 10_000_000),
    ])
    assert assert_clean(path).span_count == 2


def test_restarting_a_run_appends_without_breaking_the_sequence(
        settings: Settings, replay_store: JsonlReplayStore) -> None:
    """A retried run joins the same file with a contiguous ``seq``.

    The exporter never truncates, so a second attempt at the same ``run_id``
    appends. ``seq`` is resumed from disk rather than reset, which is what keeps
    the merged file internally consistent instead of producing two runs' worth of
    events with a repeating sequence.
    """
    first = OtelTracer(settings, replay_store=replay_store)
    first.configure(RUN_ID, event_id=new_id("evt"))
    with first.agent(AgentId.A1_DISCOVERY, "A1.plan"):
        pass
    first_events = len(first.events)
    first.finish()

    second = OtelTracer(settings, replay_store=replay_store)
    second.configure(RUN_ID, event_id=new_id("evt"))
    with second.agent(AgentId.A2_PRICING, "A2.plan"):
        pass
    path = second.finish()["trace_file"]

    seqs = [line["seq"] for line in _read_lines(Path(path))
            if classify_record(line) == "event"]
    # First attempt: run.configure (0) + agent (1). Second attempt resumes at 2.
    assert first_events == 2
    assert seqs == [0, 1, 2, 3], "merged file must keep one contiguous sequence"
    assert_clean(path)


def test_problem_is_structured_and_printable() -> None:
    problem = Problem("cycle", "a -> b -> a", line=7, record="a" * 16)
    assert problem.is_error
    assert problem.to_dict() == {
        "code": "cycle", "message": "a -> b -> a", "line": 7,
        "severity": "error", "record": "a" * 16,
    }
    assert "ERROR cycle line 7" in str(problem)
    assert Problem("constant_cadence", "x", severity="warning").is_error is False


def test_lint_rejects_a_missing_or_wrong_kind_of_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lint_trace(tmp_path / "nope.jsonl")
    with pytest.raises(IsADirectoryError):
        lint_trace(tmp_path)
    with pytest.raises(FileNotFoundError):
        compute_summary(tmp_path / "nope.jsonl", RUN_ID)


# ====================================================================== replay
def test_replay_key_is_stable_and_namespaced(settings: Settings) -> None:
    """Same request -> same key, regardless of dict ordering or namespace."""
    payload = {"question": "escalate?", "options": ["yes", "no"], "temperature": 0.0}
    reordered = {"temperature": 0.0, "options": ["yes", "no"],
                 "question": "escalate?"}
    key = JsonlReplayStore.key("clef.decide", payload)

    assert key == JsonlReplayStore.key("clef.decide", reordered), (
        "canonical JSON must be key-order independent"
    )
    assert len(key) == 64 and int(key, 16) >= 0
    assert key != JsonlReplayStore.key("gemini.decide", payload), (
        "two namespaces asking the same question must not collide"
    )
    assert key != JsonlReplayStore.key("clef.decide", {**payload, "temperature": 0.1})

    # Stable across processes and instances, which is the whole point of a key.
    store = JsonlReplayStore(settings.replay_path)
    assert store.key("clef.decide", payload) == key


def test_replay_returns_recorded_bytes_verbatim(
        settings: Settings, replay_store: JsonlReplayStore) -> None:
    """A recording round-trips unchanged, and a miss is ``None`` — never a guess."""
    payload = {"question": "who arbitrates?", "options": ["A7", "A5"]}
    output = {"choice": "A7", "probabilities": {"A7": 0.74, "A5": 0.26},
              "raw_text": "the arbiter A7 decides disputes"}
    key = replay_store.record("clef.decide", payload, output, model="clef-flash",
                             source="live", run_id=RUN_ID)

    replayed = replay_store.get(key)
    assert replayed == output
    assert replayed["raw_text"] == output["raw_text"]
    # The stored bytes are auditable against what the provider returned.
    raw = replay_store.get_raw(key)
    assert raw is not None
    assert json.loads(raw)["output"] == output

    # A miss is None. Never a plausible default.
    miss = replay_store.get(JsonlReplayStore.key("clef.decide", {"other": "request"}))
    assert miss is None
    assert replay_store.stats()["misses"] == 1
    assert replay_store.stats()["hits"] == 1

    # Mutating the returned object must not corrupt the recording.
    replayed["choice"] = "A5"
    assert replay_store.get(key)["choice"] == "A7"


def test_replay_survives_a_new_store_instance(settings: Settings) -> None:
    payload = {"prompt": "x"}
    output = {"text": "y"}
    first = JsonlReplayStore(settings.replay_path)
    key = first.record("gemini.chat", payload, output, model="m", source="live")
    first.close()

    second = JsonlReplayStore(settings.replay_path)
    assert second.get(key) == output
    assert second.stats()["entry_count"] == 1
    assert second.stats()["namespaces"] == ["gemini.chat"]


def test_replay_refuses_a_corrupt_file(tmp_path: Path) -> None:
    """A recording that cannot be parsed must not be presented as real."""
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"key": "k", "output": {}}\nnot json\n', encoding="utf-8")
    with pytest.raises(ReplayIntegrityError):
        JsonlReplayStore(corrupt).entries()

    wrong_shape = tmp_path / "wrong.jsonl"
    wrong_shape.write_text(json.dumps({"key": "k", "output": ["not", "an", "object"]})
                           + "\n", encoding="utf-8")
    store = JsonlReplayStore(wrong_shape)
    with pytest.raises(ReplayIntegrityError):
        store.get("k")

    readonly = JsonlReplayStore(tmp_path / "ro.jsonl", read_only=True)
    with pytest.raises(ReplayIntegrityError):
        readonly.put("k", {"a": 1})


def test_replay_stats_report_conflicts_and_unverifiable_entries(
        settings: Settings, replay_store: JsonlReplayStore) -> None:
    payload = {"q": 1}
    output = {"a": 1}
    replay_store.record("ns", payload, output, model="m", source="live")
    replay_store.record("ns", payload, {**output, "a": 2}, model="m", source="live")
    replay_store.put("deadbeef", {"b": 2})

    stats = replay_store.stats()
    assert stats["entry_count"] == 2
    assert stats["conflicts"] == 1
    assert stats["unverifiable_entries"] == 1
    assert stats["namespaces"] == ["ns"]
    assert stats["bytes"] > 0
    # Last write wins, and the key still resolves to a recorded answer.
    key = replay_store.key("ns", payload)
    assert replay_store.get(key) == {"a": 2}


def test_tracer_replay_log_returns_recordings_and_nothing_else(
        tracer: OtelTracer, replay_store: JsonlReplayStore) -> None:
    """``replay_log()`` reads recordings; it never generates one."""
    assert tracer.replay_log() == []
    tracer.record_output("gemini.chat", {"prompt": "p"},
                         {"text": "recorded"}, model="gemini-3-flash",
                         source="live", run_id=tracer.run_id)
    log = tracer.replay_log()
    assert [entry["output"] for entry in log] == [{"text": "recorded"}]


def test_tracer_works_with_no_configuration_or_network(tmp_path: Path) -> None:
    """No env vars, no credentials, no collector: the tracer still runs.

    ``get_settings()`` reads whatever the environment happens to hold, so this
    asserts the *capability* — that nothing in the observability path needs a
    network, a credential, or a config file — rather than pretending to unset an
    environment it cannot control.
    """
    made = OtelTracer(Settings(traces_dir=tmp_path / "t",
                               replay_path=tmp_path / "r.jsonl"))
    made.configure(RUN_ID, event_id=new_id("evt"))
    with made.agent(AgentId.A6_AUDIT, "A6.observe"), made.tool(AgentId.A6_AUDIT, "artifact_writer"):
        pass
    result = made.finish()
    assert result["integrity_ok"] is True
    assert result["span_count"] == 2
    # No store attached and no file on disk: an empty answer, not a fabricated one.
    assert made.replay_log() == []


def test_host_id_is_salted_and_deterministic() -> None:
    first = host_id("some-machine")
    assert first == host_id("some-machine")
    assert first != host_id("other-machine")
    assert len(first) == 64
    assert "some-machine" not in first
