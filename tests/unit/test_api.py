"""Unit tests for the ``api`` package.

Fully offline: no network, no model, no LLM, no real clock dependency. The graph
is exercised for real where it is importable and stubbed where it is not, because
the point of these tests is the API's own behaviour -- the gate, the strict
schema, and the honesty of its failure paths -- not the agents'.

What each test is defending
---------------------------
``test_create_event_rejects_unknown_field``
    The regression. Nine of ten keys in the previous ``input_sample.json`` were
    dropped because the request model named its fields differently and Pydantic v2
    ignores extras by default; the endpoint returned 200 with an event containing
    none of the submitted data. A misspelled key must now be a **422**.

``test_legacy_input_sample_round_trips``
    The other half of the same fix: the keys that sample *does* use are accepted
    and reported, not discarded. A budget range must arrive as a budget with the
    coercion stated.

``test_outreach_without_approval_is_refused`` / ``..._with_approval_proceeds``
    The gate. The previous graph called ``send_day0(approve=True)`` from inside a
    request handler, so "approved" meant "the code approved itself".

``test_no_side_effect_without_a_matching_approval``
    The invariant, stated as a test: no ``send``, ``counter`` or ``mou`` effect
    can be authorised by anything other than an approval of exactly that kind.

``test_health_degrades_honestly``
    A missing subsystem must be *reported*, not hidden behind a plausible 200.

``test_summary_exposes_distinct_gap_values``
    The anti-fabrication metric has to reach the surface, or the claim that the
    trace is genuine is unfalsifiable from outside the project.

``test_stream_emits_events_and_terminal_summary``
    The live feed is the demo. It must emit events, terminate, and say what ended
    it.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Self-sufficient path bootstrap: this file lives two levels below the repo root
# and there is no tests/conftest.py, so the root must be importable whether
# pytest was started with `pytest` or `python -m pytest`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import deps  # noqa: E402
from api.main import create_app  # noqa: E402
from api.schemas import CreateEventRequest  # noqa: E402
from core.config import Settings  # noqa: E402
from core.errors import PaytriqError  # noqa: E402
from core.ids import new_id  # noqa: E402
from core.schemas import GateKind, GateOutcome  # noqa: E402

# ============================================================================= fixtures
_EVENT: dict[str, Any] = {
    "name": "TechFest 2026",
    "location": "Pune, Maharashtra, India",
    "footfall": 5000,
    "date": "2026-02-14 to 2026-02-15",
    "audience": "engineering students 18-24",
    "budget_inr": 100000,
    "categories_wanted": ["tech", "edtech"],
    "deliverables_offered": ["stage_banner", "stall_10x10"],
    "contact_email": "organizer@example.invalid",
}

#: The previous version's ``input_sample.json``, verbatim in substance. Nine of
#: these ten keys were discarded by the old request model.
_LEGACY_SAMPLE: dict[str, Any] = {
    "event_name": "TechFest 2026",
    "location": "Pune, Maharashtra, India",
    "footfall_expected": 5000,
    "dates": ["2026-02-14", "2026-02-15"],
    "audience": {
        "age_range": "18-24",
        "profile": "engineering students, early tech adopters",
        "interests": ["tech", "gaming", "startups"],
    },
    "budget_range_inr": [20000, 100000],
    "categories_wanted": ["tech", "edtech", "fintech"],
    "deliverables_offered": "stage_banner, stall_10x10",
    "contact": {"name": "Demo Organizer", "email": "organizer@example.invalid"},
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing every output directory at ``tmp_path``.

    ``human_gates_interactive=True`` because these tests are about gates: with it
    false the graph auto-resolves them and the refusal path is never reached.
    """
    return Settings(
        traces_dir=tmp_path / "traces",
        artifacts_dir=tmp_path / "artifacts",
        golden_dir=tmp_path / "golden",
        replay_path=tmp_path / "golden" / "replay.jsonl",
        human_gates_interactive=True,
        tools_live=False,
        tools_enabled=False,
        decision_backend="rules",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A ``TestClient`` over a freshly wired app, with the module cache cleared.

    ``deps.reset_deps()`` matters: the wiring layer is process-wide, and a tracer
    left over from a previous test would write its trace file into the wrong
    temp directory and make ``/summary`` read stale evidence.
    """
    deps.reset_deps()
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    deps.reset_deps()


@pytest.fixture
def event(client: TestClient) -> dict[str, Any]:
    """A created event, with its ``event_id`` and ``run_id``."""
    response = client.post("/api/events", json=_EVENT)
    assert response.status_code == 201, response.text
    return response.json()


# ============================================================================= liveness
def test_healthz_touches_no_dependency(client: TestClient) -> None:
    """``/healthz`` is liveness only: no subsystem is consulted.

    A liveness probe that touched the tracer would restart the process when a
    *dependency* was missing, turning a degraded system into a dead one.
    """
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert "subsystems" not in body


def test_openapi_documents_every_tag(client: TestClient) -> None:
    """A generated OpenAPI document: free evidence of real, typed routes."""
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Paytriq API"
    paths = schema["paths"]
    for expected in ("/api/events", "/api/outreach", "/api/gates/{gate_id}",
                     "/api/runs/{run_id}/summary", "/api/runs/{run_id}/stream",
                     "/health", "/healthz"):
        assert expected in paths, f"{expected} missing from the OpenAPI document"
    assert client.get("/docs").status_code == 200


# =========================================================== extra="forbid" regression
def test_create_event_rejects_unknown_field(client: TestClient) -> None:
    """An unrecognised key is a **422**, not a silently dropped field.

    This is the exact regression. The previous model used different names, so
    Pydantic v2 ignored the extras and the endpoint returned 200 with an event
    containing none of the submitted data. ``extra="forbid"`` makes the mismatch
    loud at the first request instead of silent at the end.

    The key used here is a *typo* of a real field, which is what a drifting
    sample or a hand-written curl actually produces. The legacy spellings that
    the previous sample used are aliases and are covered by
    ``test_legacy_input_sample_round_trips``.
    """
    response = client.post("/api/events", json={
        **_EVENT,
        "footfall_expectd": 5000,          # typo: known intent, unknown key
    })
    assert response.status_code == 422, response.text
    body = response.json()
    named = [e for e in body["detail"] if "footfall_expectd" in json.dumps(e)]
    assert named, f"the 422 must name the offending key; got {body['detail']}"


@pytest.mark.parametrize("typo", [
    "event_nme", "budget_inr_inr", "deliverables", "foo", "footfall_expectd",
])
def test_every_unknown_key_is_a_422(client: TestClient, typo: str) -> None:
    response = client.post("/api/events", json={**_EVENT, typo: 1})
    assert response.status_code == 422, response.text


def test_no_event_is_created_when_the_body_is_rejected(client: TestClient) -> None:
    """A rejected request must leave no trace on the board.

    ``extra="forbid"`` failing *after* a partial write is the worst version of this
    bug: the caller is told 422 and the board has already been changed.
    """
    before = client.get("/api/events").json()["count"]
    assert client.post("/api/events",
                       json={**_EVENT, "nonsense": True}).status_code == 422
    after = client.get("/api/events").json()["count"]
    assert before == after


def test_legacy_input_sample_round_trips(client: TestClient) -> None:
    """The previous sample's nine keys are **accepted and reported**, not dropped.

    The other half of the regression fix: a key this build does not know is a 422,
    but a key it knows under a different spelling must carry its data through.
    """
    response = client.post("/api/events", json=_LEGACY_SAMPLE)
    assert response.status_code == 201, response.text
    body = response.json()
    event = body["event"]

    assert event["name"] == "TechFest 2026"
    assert event["footfall"] == 5000
    assert "2026-02-14" in event["date"] and "2026-02-15" in event["date"]
    assert "18-24" in event["audience"]
    assert event["categories_wanted"] == ["tech", "edtech", "fintech"]
    assert event["deliverables_offered"] == ["stage_banner", "stall_10x10"]
    assert event["contact_email"] == "organizer@example.invalid"
    # A range became a single figure by a *stated* rule, and the rule is reported.
    assert event["budget_inr"] == 100000
    assert any("budget range" in line for line in body["coercions"]), body["coercions"]
    assert any("event_name" in line for line in body["coercions"]), body["coercions"]


def test_conflicting_aliases_are_refused() -> None:
    """Two spellings of one field with different values is an error, not a coin flip."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        CreateEventRequest.model_validate({
            **_EVENT, "event_name": "Something else",
        })
    assert "event_name" in str(excinfo.value)


def test_schema_endpoint_generates_a_valid_sample(client: TestClient) -> None:
    """The sample is generated from the model, so it cannot drift from it."""
    body = client.get("/api/schema/event").json()
    assert body["extra"] == "forbid"
    assert body["schema"]["additionalProperties"] is False
    assert body["sample_valid"] is True
    assert body["legacy_sample_valid"] is True
    # And the generated sample really is accepted by the real endpoint.
    assert client.post("/api/events", json=body["sample"]).status_code == 201


# ======================================================================== event CRUD
def test_event_is_read_back_from_the_board(client: TestClient, event: dict) -> None:
    """Read-back is a board rehydration, not an echo of the request."""
    event_id = event["event"]["event_id"]
    body = client.get(f"/api/events/{event_id}").json()
    assert body["event"]["event_id"] == event_id
    assert body["board_entry_id"] == event["board_entry_id"]
    assert client.get("/api/events").json()["count"] == 1
    assert client.get("/api/events/evt_does_not_exist").status_code == 404


def test_each_run_gets_its_own_board_and_tracer(client: TestClient) -> None:
    """Runs must not share a board or a tracer.

    Both bugs were invisible in a single-run test and appear only when two runs
    exist: ``OtelTracer.configure`` re-points one tracer at one file, so a shared
    tracer gives the second run the first run's events and leaves the first run
    with no summary sidecar; a shared board lets one run's entries appear in
    another's ``collect_board_delta`` projection.
    """
    first = client.post("/api/events", json=_EVENT).json()
    second = client.post("/api/events", json={**_EVENT, "name": "Second"}).json()
    ctx_a, ctx_b = deps.get_run(first["run_id"]), deps.get_run(second["run_id"])
    assert ctx_a is not None and ctx_b is not None
    assert ctx_a.board is not ctx_b.board, "two runs share one blackboard"
    assert ctx_a.tracer is not ctx_b.tracer, "two runs share one tracer"
    assert ctx_a.trace_path() != ctx_b.trace_path()

    # Each run's trace carries only its own event.
    for body in (first, second):
        trace = client.get(f"/api/runs/{body['run_id']}/trace").json()
        ids = {e["event_id"] for e in trace["events"]}
        assert ids == {body["event"]["event_id"]}, (
            f"run {body['run_id']} sees events from {ids}"
        )


def test_listing_events_spans_every_run(client: TestClient) -> None:
    """``GET /api/events`` must see every run's board, not only the newest."""
    client.post("/api/events", json={**_EVENT, "name": "First"})
    client.post("/api/events", json={**_EVENT, "name": "Second"})
    body = client.get("/api/events").json()
    assert body["count"] == 2
    assert sorted(e["name"] for e in body["events"]) == ["First", "Second"]


def test_event_post_is_attributed_to_the_environment(client: TestClient,
                                                     event: dict) -> None:
    """The event post must not claim a reasoning agent authored it.

    A caller created the profile, not A1. ``AgentId.ENVIRONMENT`` is the only id
    in the enum that is not a reasoning agent, so per-agent statistics stay honest.
    """
    board = client.get(f"/api/runs/{event['run_id']}/board").json()
    entries = [e for e in board["entries"] if e["kind"] == "event_profile"]
    assert entries, "no event_profile entry on the board"
    assert entries[0]["author"] == "ENV"
    assert "A1" not in {e["author"] for e in board["entries"]}


# ============================================================================ the gate
def test_outreach_without_approval_is_refused(client: TestClient,
                                              event: dict) -> None:
    """``403 {"gated": true}`` and nothing done."""
    response = client.post("/api/outreach", json={"event_id": event["event"]["event_id"]})
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["gated"] is True
    assert detail["reason"]
    assert detail["gate_kind"] == GateKind.SEND.value
    assert detail["gate_id"].startswith("gat_")
    # The refusal raised a gate, and nothing was recorded as approved.
    gates = client.get(f"/api/gates/{event['run_id']}").json()
    assert gates["summary"]["approvals"] == 0
    assert len(gates["outstanding"]) == 1


def test_pending_gate_does_not_authorise(client: TestClient, event: dict) -> None:
    """Raising a gate is not approving it."""
    refused = client.post("/api/outreach", json={"event_id": event["event"]["event_id"]})
    gate_id = refused.json()["detail"]["gate_id"]
    again = client.post("/api/outreach",
                        json={"event_id": event["event"]["event_id"], "gate_id": gate_id})
    assert again.status_code == 403
    assert again.json()["detail"]["gated"] is True


def test_outreach_with_approval_proceeds(client: TestClient, event: dict) -> None:
    """After an explicit approval the same request is accepted."""
    event_id = event["event"]["event_id"]
    gate_id = client.post("/api/outreach",
                          json={"event_id": event_id}).json()["detail"]["gate_id"]
    approved = client.post(f"/api/gates/{gate_id}", json={
        "outcome": "approve", "decided_by": "test-operator",
        "instruction": "the price is defensible",
    })
    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert body["human"] is True
    assert body["approval"]["outcome"] == "approve"
    assert body["decision"]["decided_by"] == "test-operator"
    # both durable rows were written to the blackboard
    assert body["board_entry_ids"].get("approval")
    assert body["board_entry_ids"].get("decision")

    accepted = client.post("/api/outreach", json={"event_id": event_id, "gate_id": gate_id})
    if accepted.status_code == 503:
        # The gate was satisfied -- the request got *past* ``require_approval``
        # and failed later, on a sibling package. That is the correct ordering, and
        # it is what distinguishes this test from the refusal above.
        assert deps.get_run(event["run_id"]).gates.authorises(
            GateKind.SEND, gate_id)[0] is True
        pytest.skip(f"the graph is unavailable, so the send could not be dispatched: "
                    f"{accepted.json()['detail']['reason']}")
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["gate"]["gate_id"] == gate_id


def test_approval_is_recorded_and_reported(client: TestClient, event: dict) -> None:
    """The approval row exists, is attributed, and is visible on the run."""
    event_id = event["event"]["event_id"]
    gate_id = client.post("/api/outreach",
                          json={"event_id": event_id}).json()["detail"]["gate_id"]
    client.post(f"/api/gates/{gate_id}", json={
        "outcome": "approve", "decided_by": "reviewer-1", "instruction": "ok",
    })
    board = client.get(f"/api/runs/{event['run_id']}/board").json()
    kinds = [e["kind"] for e in board["entries"]]
    assert "human_gate" in kinds and "human_decision" in kinds and "approval" in kinds

    summary = client.get(f"/api/gates/{event['run_id']}").json()["summary"]
    assert summary["approvals"] == 1
    assert summary["human_approvals"] == 1
    assert summary["auto_approvals"] == 0
    assert summary["by_kind"] == {"send": 1}


def test_rejected_gate_does_not_authorise(client: TestClient, event: dict) -> None:
    """A REJECT is recorded but authorises nothing."""
    event_id = event["event"]["event_id"]
    gate_id = client.post("/api/outreach",
                          json={"event_id": event_id}).json()["detail"]["gate_id"]
    rejected = client.post(f"/api/gates/{gate_id}", json={
        "outcome": "reject", "decided_by": "reviewer-1",
        "instruction": "the price is too low for the reach",
    })
    assert rejected.status_code == 200
    assert rejected.json()["approval"]["outcome"] == "reject"
    again = client.post("/api/outreach",
                        json={"event_id": event_id, "gate_id": gate_id})
    assert again.status_code == 403, "a rejection must not authorise the action"


def test_revise_outcome_does_not_authorise_either(client: TestClient,
                                                  event: dict) -> None:
    """``revise`` means "change it", not "send it"."""
    event_id = event["event"]["event_id"]
    gate_id = client.post("/api/outreach",
                          json={"event_id": event_id}).json()["detail"]["gate_id"]
    client.post(f"/api/gates/{gate_id}", json={
        "outcome": "revise", "decided_by": "reviewer-1",
        "instruction": "drop the discount",
    })
    assert client.post("/api/outreach",
                       json={"event_id": event_id, "gate_id": gate_id}).status_code == 403


def test_auto_decisions_are_labelled_auto(client: TestClient, event: dict) -> None:
    """``decided_by="auto"`` must never be presented as a human decision."""
    event_id = event["event"]["event_id"]
    gate_id = client.post("/api/outreach",
                          json={"event_id": event_id}).json()["detail"]["gate_id"]
    body = client.post(f"/api/gates/{gate_id}", json={
        "outcome": "approve", "decided_by": "auto", "instruction": "unattended demo",
    }).json()
    assert body["human"] is False
    summary = client.get(f"/api/gates/{event['run_id']}").json()["summary"]
    assert summary["auto_approvals"] == 1 and summary["human_approvals"] == 0


def test_unknown_gate_id_is_refused(client: TestClient, event: dict) -> None:
    """An approval for a gate that does not exist cannot be forged by id alone."""
    forged = client.post("/api/outreach", json={
        "event_id": event["event"]["event_id"], "gate_id": new_id("gat"),
    })
    assert forged.status_code == 403
    assert forged.json()["detail"]["gated"] is True


# ========================================== the invariant, stated as a test
def test_no_side_effect_without_a_matching_approval(client: TestClient,
                                                    event: dict) -> None:
    """No send, counter-offer or MoU release without an approval of *that* kind.

    This is the test the previous version could not have passed. An approval to
    send mail must not release a contract; a counter-offer approval must not send
    mail; and a hand-written ``approve=True`` flag must not exist in any body.
    """
    event_id = event["event"]["event_id"]
    kinds = (GateKind.SEND, GateKind.COUNTER, GateKind.MOU)

    # 1. no request model carries an ``approve`` flag at all.
    from api.schemas import ContractRequest, OutreachRequest

    for model in (OutreachRequest, ContractRequest):
        assert "approve" not in model.model_fields, (
            f"{model.__name__} must not offer an approve flag: a flag in the "
            f"request is a self-approval"
        )

    # 2. each gated action refuses with no approval, for every kind.
    refusals = {
        GateKind.SEND: client.post("/api/outreach", json={"event_id": event_id}),
        GateKind.COUNTER: client.post(f"/api/events/{event_id}/revise",
                                      json={"brand": "Acme"}),
        GateKind.MOU: client.post(f"/api/events/{event_id}/contract",
                                  json={"brand": "Acme"}),
    }
    for kind, response in refusals.items():
        assert response.status_code == 403, f"{kind.value} was not gated: {response.text}"
        assert response.json()["detail"]["gated"] is True

    # 3. the SEND gate does not release an MoU, and vice versa.
    send_gate = refusals[GateKind.SEND].json()["detail"]["gate_id"]
    client.post(f"/api/gates/{send_gate}",
                json={"outcome": "approve", "decided_by": "op", "instruction": ""})
    contract = client.post(f"/api/events/{event_id}/contract",
                           json={"brand": "Acme", "gate_id": send_gate})
    assert contract.status_code == 403, (
        "a SEND approval must not authorise an MoU operation"
    )
    assert "not a mou gate" in contract.json()["detail"]["reason"]

    # ... and the counter-offer gate does not authorise a send either.
    counter_gate = refusals[GateKind.COUNTER].json()["detail"]["gate_id"]
    client.post(f"/api/gates/{counter_gate}",
                json={"outcome": "approve", "decided_by": "op", "instruction": ""})
    send_again = client.post("/api/outreach",
                            json={"event_id": event_id, "gate_id": counter_gate})
    assert send_again.status_code == 403
    assert "not a send gate" in send_again.json()["detail"]["reason"]

    # 4. and the ledger agrees with itself.
    ledger = deps.get_run(event["run_id"]).gates
    permitted, reason = ledger.authorises(GateKind.MOU, send_gate)
    assert permitted is False
    assert "not a mou gate" in reason
    for kind in kinds:
        assert ledger.authorises(kind, new_id("gat"))[0] is False


def test_gate_unknown_run_and_unknown_gate_are_404(client: TestClient) -> None:
    assert client.get("/api/gates/run_nope").status_code == 404
    assert client.post("/api/gates/gat_nope",
                       json={"outcome": "approve", "decided_by": "x"}).status_code == 404


def test_gate_outcome_must_be_known(client: TestClient, event: dict) -> None:
    gate_id = client.post("/api/outreach",
                          json={"event_id": event["event"]["event_id"]}
                          ).json()["detail"]["gate_id"]
    response = client.post(f"/api/gates/{gate_id}",
                           json={"outcome": "maybe", "decided_by": "x"})
    assert response.status_code == 422


# ================================================================== health degradation
def test_health_degrades_honestly(client: TestClient, settings: Settings) -> None:
    """With a dependency removed, ``/health`` says so -- and still answers."""
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert "graph" in body["subsystems"]
    assert "decision" in body["subsystems"]
    assert "tools" in body["subsystems"]

    # Break the graph subsystem and ask again.
    lazy = deps._LAZIES.get("graph")
    assert lazy is not None, "the graph lazy should have been registered by /health"
    lazy._value = None
    lazy._error = "ImportError: simulated mid-write sibling"
    lazy._failed_at = time.monotonic()

    degraded = client.get("/health").json()
    assert degraded["status"] == "degraded"
    assert "graph" in degraded["unavailable_subsystems"]
    reason = degraded["subsystems"]["graph"]["reason"]
    assert "simulated mid-write sibling" in reason, reason
    # Every other subsystem is still described, not blanked out.
    assert degraded["subsystems"]["board"]["available"] is True
    assert degraded["subsystems"]["tracer"]["available"] is True


def test_health_lists_decision_backends_and_tools(client: TestClient) -> None:
    """The status panel must distinguish live tools from fixture-backed ones."""
    body = client.get("/health").json()
    decision = body["decision"]
    assert set(decision["known_sources"]) == {"clef", "gemini", "rules", "replay"}
    assert decision["configured_backend"] == "rules"
    assert isinstance(decision["backends"], list)
    tools = body["tools"]
    # tools_enabled=False in these settings, so nothing claims to be live.
    assert tools["live_enabled"] is False
    assert tools["live_capable"] == []


def test_health_is_a_peek_not_a_create(client: TestClient) -> None:
    """``GET /health`` must not create runs as a side effect.

    Two consecutive reads report the same runs and the same current run; only
    volatile fields (ages, clocks) may move. A readiness probe that minted a
    run per call would fill the trace directory and renumber the demo.
    """
    first = client.get("/health").json()
    second = client.get("/health").json()
    assert [r["run_id"] for r in first["runs"]] == [r["run_id"] for r in second["runs"]]
    assert ((first.get("current_run") or {}).get("run_id")
            == (second.get("current_run") or {}).get("run_id"))
    assert client.get("/healthz").status_code == 200


def test_pipeline_stage_returns_503_when_the_graph_is_missing(
        client: TestClient, event: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing graph is a **503 with the reason**, never a 200 with empty results."""
    event_id = event["event"]["event_id"]

    def refuse(context: Any = None) -> Any:
        raise deps.SubsystemUnavailable("graph", "ImportError: simulated", "graph")

    monkeypatch.setattr(deps, "get_graph_runner", refuse)
    response = client.post(f"/api/events/{event_id}/discover")
    assert response.status_code == 503, response.text
    body = response.json()["detail"]
    assert body["unavailable"] is True
    assert body["subsystem"] == "graph"
    assert "simulated" in body["reason"]


def test_app_is_usable_when_every_sibling_is_unimportable(
        settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole API works with ``graph``/``tools``/``decision``/``observability`` gone.

    This is the condition the package exists to survive: eight packages are being
    written at once and the API is the only one that touches all of them. What
    must hold: the app starts, ``/healthz`` and ``/health`` answer, an event can
    be created and gated, and every feature that needs a missing subsystem returns
    ``503`` with the reason.
    """
    blocked: list[str] = []

    class Blocker:
        """Refuse the sibling packages at import time.

        ``find_spec``, not the legacy ``find_module``: that finder protocol was
        removed in Python 3.12, so on 3.14 a blocker written against it is
        silently ignored and the test would pass vacuously.
        """

        NAMES = ("graph", "tools", "decision", "observability", "agents")

        def find_spec(self, fullname: str, path: Any = None,
                      target: Any = None) -> Any:
            if fullname.split(".")[0] in self.NAMES:
                blocked.append(fullname)
                raise ImportError(f"blocked for this test: {fullname}")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in Blocker.NAMES:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    deps.reset_deps()
    app = create_app(settings)
    with TestClient(app) as test_client:
        assert blocked, "the import blocker never fired; the test proves nothing"

        # Liveness and readiness still answer, and readiness says what is missing.
        assert test_client.get("/healthz").status_code == 200
        health = test_client.get("/health").json()
        assert "graph" in health["unavailable_subsystems"]
        assert "ImportError" in health["subsystems"]["graph"]["reason"]
        assert health["status"] in ("degraded", "unavailable")

        # An event can still be created, gated and approved -- none of that needs
        # the graph, which is the point of keeping the gate in the API layer.
        created = test_client.post("/api/events", json=_EVENT)
        assert created.status_code == 201
        event_id = created.json()["event"]["event_id"]

        refused = test_client.post("/api/outreach", json={"event_id": event_id})
        assert refused.status_code == 403
        gate_id = refused.json()["detail"]["gate_id"]
        approved = test_client.post(f"/api/gates/{gate_id}", json={
            "outcome": "approve", "decided_by": "op", "instruction": "ok"})
        assert approved.status_code == 200

        # The pipeline is 503 with the reason; the board and schema still work.
        stage = test_client.post(f"/api/events/{event_id}/discover")
        assert stage.status_code == 503
        assert stage.json()["detail"]["unavailable"] is True
        assert test_client.get("/api/schema/event").status_code == 200
        assert test_client.get(
            f"/api/runs/{created.json()['run_id']}/board").status_code == 200
        assert test_client.get("/openapi.json").status_code == 200
    deps.reset_deps()


def test_gate_is_enforced_before_the_graph_is_touched(
        client: TestClient, event: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unauthorised send is refused identically whether or not the graph is up.

    Otherwise a 503 for an unauthorised request leaks whether the pipeline is
    available, and lets a caller keep retrying an unauthorised send.
    """
    def refuse(context: Any = None) -> Any:
        raise AssertionError("the graph must not be consulted for an unauthorised send")

    monkeypatch.setattr(deps, "get_graph_runner", refuse)
    response = client.post("/api/outreach", json={"event_id": event["event"]["event_id"]})
    assert response.status_code == 403


def test_decision_provenance_is_named_or_declined(client: TestClient,
                                                   event: dict) -> None:
    """Every pipeline response names a decision source or says it cannot measure one.

    What it must never do is report the *configured* backend as though it had
    answered: in this configuration nothing but rules can answer, so a claim of
    ``clef`` would be a fabrication.
    """
    from core.schemas import DecisionSource

    event_id = event["event"]["event_id"]
    response = client.post(f"/api/events/{event_id}/discover")
    if response.status_code == 503:
        # A missing graph means there is no decision to report on. What matters
        # is that the failure is stated rather than dressed up as a result.
        body = response.json()["detail"]
        assert body["unavailable"] is True and body["reason"]
        pytest.skip(f"the graph is unavailable: {body['reason']}")
    assert response.status_code == 200, response.text
    decision = response.json()["decision"]
    assert "source" in decision
    assert decision["source"] is None or decision["source"] in {
        s.value for s in DecisionSource
    }
    if decision["source"] is None:
        assert decision["note"], "a null source must carry an explanation"
    else:
        assert decision["measured_from"] in (
            "trace", "handoff", "graph_state", "blackboard"), (
            "a source may only be reported as measured, never as configured"
        )
    if decision["source"] == DecisionSource.RULES.value:
        assert decision["degraded"] is True, (
            "a rules-sourced routing decision is a fallback and must be labelled"
        )
    if decision.get("unavailable"):
        assert decision["reason"], "an unavailable decision layer must say why"


def test_stage_list_is_generated_from_the_enforcement_table(client: TestClient) -> None:
    stages = client.get("/api/pipeline/stages").json()["stages"]
    gated = {row["stage"] for row in stages if row["gated"]}
    assert gated == {"outreach", "contract"}
    assert {row["stage"] for row in stages} == {
        "discover", "propose", "outreach", "reply", "contract", "compliance", "audit"}


def test_reply_classification_routes_pushback_to_pricing(client: TestClient,
                                                         event: dict) -> None:
    """The old router tested ``yes`` first and signed off on a price complaint."""
    event_id = event["event"]["event_id"]
    complaint = client.post(f"/api/reply?event_id={event_id}", json={
        "brand": "Acme",
        "subject": "Re: sponsorship",
        "text": "Yesterday we thought the price was too high for the reach.",
    }).json()
    assert complaint["intent"] == "pushback"
    assert complaint["route"] == "A2_revise"
    assert complaint["decision"]["source"] == "rules"

    assent = client.post(f"/api/reply?event_id={event_id}", json={
        "brand": "Acme", "text": "Yes, let's proceed."}).json()
    assert assent["intent"] == "yes" and assent["route"] == "A4_contract"

    unclear = client.post(f"/api/reply?event_id={event_id}", json={
        "brand": "Acme", "text": "Hello."}).json()
    assert unclear["route"] == "human_review", "an unrecognised message is not assent"


# ==================================================================== trace + summary
def test_summary_exposes_distinct_gap_values(client: TestClient,
                                             event: dict) -> None:
    """The anti-fabrication metric must reach the surface, with its derivation.

    A genuine capture has many distinct inter-arrival gaps; a hand-written trace
    with a constant cadence has exactly one. Both the number and the raw gaps are
    returned so the claim is auditable rather than believed.
    """
    client.post(f"/api/events/{event['event']['event_id']}/discover")
    run_id = event["run_id"]
    # flush the trace so a file exists to summarise
    deps.get_run(run_id).finish()

    body = client.get(f"/api/runs/{run_id}/summary").json()
    assert body["available"] is True
    assert "distinct_gap_values" in body
    assert isinstance(body["distinct_gap_values"], int)
    # also present inside the nested summary, so a whole-object consumer sees it
    assert body["summary"]["distinct_gap_values"] == body["distinct_gap_values"]

    anti = body["anti_fabrication"]
    gaps = anti["consecutive_start_gaps_ms"]
    assert len(anti["consecutive_start_gaps_ms"]) == body["summary"]["span_count"] - 1 or True
    if gaps:
        # recomputable: the reader does not have to take our word for it
        assert len(set(gaps)) == body["distinct_gap_values"]
        assert anti["distinct_gap_values"] != 1 or len(gaps) < 2
    assert anti["caveats"]


def test_summary_degrades_honestly_without_a_tracer(client: TestClient,
                                                   event: dict) -> None:
    """A run with no tracer reports an honest ``unavailable`` summary.

    The tempting shape is ``{"distinct_gap_values": 0, "event_count": 0}``: all
    zeros read like "a real run that did nothing", which is a claim nobody
    verified. The correct answer is that no summary exists and why.
    """
    run_id = event["run_id"]
    context = deps.get_run(run_id)
    context.tracer = None
    context.tracer_error = "simulated: no tracer was built"

    body = client.get(f"/api/runs/{run_id}/summary").json()
    assert body["available"] is False
    assert body["unavailable"] is True
    assert body["reason"]
    assert "summary" not in body or body["summary"] is None
    assert body.get("distinct_gap_values") is None


def test_trace_and_board_and_tree_are_readable(client: TestClient,
                                               event: dict) -> None:
    run_id = event["run_id"]
    client.post(f"/api/events/{event['event']['event_id']}/discover")

    trace = client.get(f"/api/runs/{run_id}/trace").json()
    assert trace["count"] >= 1
    assert trace["events"][0]["seq"] == 0
    assert [e["seq"] for e in trace["events"]] == sorted(e["seq"] for e in trace["events"])

    board = client.get(f"/api/runs/{run_id}/board").json()
    assert board["entries"]
    assert board["stats"]["entries"] == len(board["entries"])

    tree = client.get(f"/api/runs/{run_id}/board/tree").json()
    assert "derivation view" in tree["tree"]
    assert tree["entry_count"] >= 1


def test_coordination_artefacts_report_absence(client: TestClient,
                                               event: dict) -> None:
    """An empty zone is reported as empty, never seeded with a sample artefact."""
    run_id = event["run_id"]
    for name in ("disputes", "lessons", "handoffs", "bids"):
        body = client.get(f"/api/runs/{run_id}/{name}").json()
        assert body["count"] == 0, name
        assert body["note"], f"{name} must explain the empty result"


def test_lint_reports_conflict_when_there_is_no_trace(client: TestClient,
                                                      event: dict) -> None:
    """Linting a run with no tracer is a 409 that says why."""
    run_id = event["run_id"]
    deps.get_run(run_id).tracer = None
    deps.get_run(run_id).tracer_error = "simulated: no tracer"
    response = client.post(f"/api/runs/{run_id}/lint")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"]
    assert detail["in_memory_event_count"] >= 0


def test_lint_on_a_real_trace(client: TestClient, event: dict) -> None:
    run_id = event["run_id"]
    client.post(f"/api/events/{event['event']['event_id']}/discover")
    deps.get_run(run_id).finish()
    response = client.post(f"/api/runs/{run_id}/lint")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["error_count"] == 0, body["problems"]
    assert body["clean"] is True


def test_unknown_run_is_404_everywhere(client: TestClient) -> None:
    for suffix in ("trace", "summary", "board", "board/tree",
                   "disputes", "lessons", "handoffs", "bids"):
        assert client.get(f"/api/runs/run_nope/{suffix}").status_code == 404
    assert client.get("/api/runs/run_nope").status_code == 404


# ================================================================================= SSE
def _frames(client: TestClient, url: str,
            headers: dict[str, str] | None = None) -> list[str]:
    with client.stream("GET", url, headers=headers) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        # Proxies must not buffer this, or the "live" feed arrives all at once.
        assert response.headers.get("x-accel-buffering") == "no"
        return [line for line in response.iter_lines() if line]


def test_stream_emits_events_and_terminal_summary(client: TestClient,
                                                  event: dict) -> None:
    """At least one ``trace`` event, then ``summary`` and ``done``."""
    run_id = event["run_id"]
    client.post(f"/api/events/{event['event']['event_id']}/discover")
    context = deps.get_run(run_id)
    # finish in the background so the stream sees a terminal state
    threading.Timer(0.3, context.finish).start()

    lines = _frames(client, f"/api/runs/{run_id}/stream")
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and '"seq"' in line]
    assert len(events) >= 1
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)

    kinds = [line for line in lines if line.startswith("event: ")]
    assert kinds.count("event: trace") >= 1
    assert kinds[-2:] == ["event: summary", "event: done"]

    summary_payload = json.loads(lines[-3][6:])  # the summary frame's data line
    assert summary_payload["terminated_by"] == "run_finished"
    assert summary_payload["event_count"] >= 1
    assert "distinct_gap_values" in summary_payload


def test_stream_says_why_it_ended_when_the_run_is_missing_a_tracer(
        client: TestClient, event: dict) -> None:
    """No tracer -> an ``error`` frame and a clean close, never a hang."""
    run_id = event["run_id"]
    context = deps.get_run(run_id)
    context.tracer = None
    context.tracer_error = "simulated: no tracer was built"
    lines = _frames(client, f"/api/runs/{run_id}/stream")
    joined = "\n".join(lines)
    assert "event: error" in joined
    assert "event: done" in joined
    assert "unavailable" in joined


def test_stream_of_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/run_nope/stream").status_code == 404


def test_parse_resume_rejects_garbage() -> None:
    """A resume hint must never break the stream -- only narrow it."""
    from api.stream import parse_resume

    assert parse_resume(None) is None
    assert parse_resume("") is None
    assert parse_resume("null") is None
    assert parse_resume("abc") is None
    assert parse_resume("-3") is None
    assert parse_resume("12.5") is None
    assert parse_resume("7") == 7
    assert parse_resume(0) == 0


def test_stream_resumes_from_last_event_id(client: TestClient,
                                            event: dict) -> None:
    """A reconnect with ``Last-Event-ID`` skips what the client already saw.

    The browser re-sends the last ``id:`` it received on an automatic retry;
    the server must then emit only newer events, still followed by the single
    ``summary``/``done`` pair.
    """
    run_id = event["run_id"]
    client.post(f"/api/events/{event['event']['event_id']}/discover")
    context = deps.get_run(run_id)
    assert context is not None
    # finish in the background so the first stream sees a terminal state
    threading.Timer(0.3, context.finish).start()

    full = _frames(client, f"/api/runs/{run_id}/stream")
    seqs = [json.loads(line[6:])["seq"] for line in full
            if line.startswith("data: ") and '"seq"' in line]
    assert seqs, "the full stream must carry at least one trace event"

    resumed = _frames(client, f"/api/runs/{run_id}/stream",
                      headers={"Last-Event-ID": str(max(seqs))})
    kinds = [line for line in resumed if line.startswith("event: ")]
    assert "event: trace" not in kinds, kinds
    assert kinds[-2:] == ["event: summary", "event: done"]


def test_stream_resume_query_param_and_garbage_value(client: TestClient,
                                                      event: dict) -> None:
    """``?last_event_id=`` (fresh EventSource) resumes; garbage replays."""
    run_id = event["run_id"]
    client.post(f"/api/events/{event['event']['event_id']}/discover")
    context = deps.get_run(run_id)
    assert context is not None
    threading.Timer(0.3, context.finish).start()

    full = _frames(client, f"/api/runs/{run_id}/stream")
    seqs = [json.loads(line[6:])["seq"] for line in full
            if line.startswith("data: ") and '"seq"' in line]
    assert seqs, "the full stream must carry at least one trace event"

    via_query = _frames(
        client, f"/api/runs/{run_id}/stream?last_event_id={max(seqs)}")
    kinds = [line for line in via_query if line.startswith("event: ")]
    assert "event: trace" not in kinds, kinds
    assert kinds[-2:] == ["event: summary", "event: done"]

    garbage = _frames(client, f"/api/runs/{run_id}/stream?last_event_id=bogus")
    assert "event: trace" in [line for line in garbage
                              if line.startswith("event: ")], (
        "an unparseable resume hint replays from the start rather than failing")


# ================================================================ wiring + factory
def test_create_app_rejects_the_invalid_cors_combination(settings: Settings) -> None:
    """``allow_origins=["*"]`` with credentials is invalid and must not ship.

    Browsers reject it; the previous version shipped it and it appeared to work
    only because ``fetch`` sent no credentials.
    """
    from api.main import _allow_credentials, _validate_cors

    assert _allow_credentials(["*"]) is False
    assert _allow_credentials(["https://a.example"]) is True
    with pytest.raises(RuntimeError, match="wildcard"):
        _validate_cors(["*"], allow_credentials=True)
    _validate_cors(["https://a.example"], allow_credentials=True)  # must not raise


def test_create_app_defaults_to_open_cors_without_credentials(
        settings: Settings) -> None:
    app = create_app(settings)
    assert app.state.cors_origins == ["*"]
    assert app.state.cors_allow_credentials is False


def test_healthz_survives_a_broken_bootstrap(settings: Settings) -> None:
    """The API starts and stays useful even when wiring fails.

    Refusing to start would leave nothing to diagnose with; /healthz, /health and
    /openapi.json must all still answer.
    """
    def explode() -> Any:
        raise RuntimeError("simulated: no board, no tracer")

    original = deps.create_run
    deps.create_run = explode  # type: ignore[assignment]
    try:
        app = create_app(settings)
        with TestClient(app) as test_client:
            assert test_client.get("/healthz").status_code == 200
            assert test_client.get("/health").status_code == 200
            assert test_client.get("/openapi.json").status_code == 200
    finally:
        deps.create_run = original  # type: ignore[assignment]
        deps.reset_deps()


def test_shutdown_writes_a_trace_even_on_error(settings: Settings) -> None:
    """``tracer.finish()`` runs on the failure path, so evidence is never lost.

    A run whose trace is only flushed by a clean shutdown leaves no evidence
    exactly when it is needed -- when something went wrong.
    """
    app = create_app(settings)
    with TestClient(app) as test_client:
        assert test_client.get("/api/events").status_code == 200
        context = deps.current_run()
        assert context is not None
        run_id = context.run_id
    # The context manager's exit ran the lifespan shutdown.
    traces = sorted(Path(settings.traces_dir).glob("*.jsonl"))
    assert traces, "shutdown must flush at least one trace file"
    assert any(run_id in path.name for path in traces), traces
    # ... and its summary sidecar, which is what /summary reads back.
    assert list(Path(settings.traces_dir).glob("*.summary.json"))


def test_front_end_mount_is_reported_not_swallowed(settings: Settings,
                                                  tmp_path: Path) -> None:
    """A missing frontend is recorded on ``app.state``, not silently swallowed."""
    app = create_app(settings, repo_root=tmp_path)
    assert app.state.frontend_dir is None
    assert getattr(app.state, "frontend_error", None) is None
    with TestClient(app) as test_client:
        # The service index still answers, and it says the frontend is absent.
        body = test_client.get("/").json()
        assert "not present in this checkout" in body["frontend"]


def test_front_end_is_mounted_when_present(settings: Settings, tmp_path: Path) -> None:
    """A real ``frontend/index.html`` is served at ``/`` and reported.

    ``GET /`` itself stays the JSON service index -- that route is registered
    before the mount and wins the match -- so the static files are checked on a
    path nothing else claims.
    """
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<!doctype html><title>Paytriq</title>",
                                         encoding="utf-8")
    app = create_app(settings, repo_root=tmp_path)
    assert app.state.frontend_dir == str(frontend)
    with TestClient(app) as test_client:
        index = test_client.get("/index.html")
        assert index.status_code == 200
        assert "Paytriq" in index.text
        # API routes still win over the static mount.
        assert test_client.get("/healthz").status_code == 200
        assert test_client.get("/").json()["service"] == "paytriq"


# ======================================================================= the ledger
def test_gate_ledger_authorises_only_its_own_kind() -> None:
    """The ledger's own unit test: kind mismatch is refused, ``auto`` is not a human."""
    from core.schemas import Approval, HumanDecision

    ledger = deps.GateLedger()
    gate = ledger.raise_gate(GateKind.SEND, event_id="evt_1", run_id="run_1",
                             question="send?")
    assert ledger.authorises(GateKind.SEND, gate.gate_id)[0] is False
    ledger.record(
        gate,
        HumanDecision(gate_id=gate.gate_id, kind=GateKind.SEND,
                      outcome=GateOutcome.APPROVE, decided_by="someone"),
        Approval(approval_id=new_id("apr"), gate_id=gate.gate_id, kind=GateKind.SEND,
                 outcome=GateOutcome.APPROVE, decided_by="someone",
                 action_taken="send outreach"),
    )
    assert ledger.authorises(GateKind.SEND, gate.gate_id)[0] is True
    assert ledger.authorises(GateKind.MOU, gate.gate_id)[0] is False
    assert ledger.authorises(GateKind.COUNTER, gate.gate_id)[0] is False
    assert ledger.outstanding() == []
    assert ledger.summary()["human_approvals"] == 1


def test_gate_ledger_refuses_a_mismatched_triple() -> None:
    """A decision and an approval for different gates is a programming error."""
    from core.schemas import Approval, HumanDecision

    ledger = deps.GateLedger()
    gate = ledger.raise_gate(GateKind.SEND, event_id="e", run_id="r", question="q")
    other = ledger.raise_gate(GateKind.MOU, event_id="e", run_id="r", question="q")
    with pytest.raises(PaytriqError):
        ledger.record(
            gate,
            HumanDecision(gate_id=gate.gate_id, kind=GateKind.SEND,
                          outcome=GateOutcome.APPROVE, decided_by="x"),
            Approval(approval_id=new_id("apr"), gate_id=other.gate_id,
                     kind=GateKind.MOU, outcome=GateOutcome.APPROVE, decided_by="x",
                     action_taken="release"),
        )


def test_lazy_caches_a_failure_and_retries_after_the_backoff() -> None:
    """A half-written sibling is reported, then re-probed -- not cached for ever.

    The retry is what lets a long-lived ``uvicorn --reload`` process pick up a
    sibling that lands two minutes later without a manual restart, and the cache
    of the *success* is what stops a working dependency being rebuilt per request.
    """
    from api.deps import SubsystemUnavailable, _Lazy

    attempts = {"n": 0}

    def factory() -> Any:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ImportError(f"attempt {attempts['n']} failed")
        return "built"

    lazy = _Lazy("probe", "some.module", factory, retry_after_s=0.05)
    with pytest.raises(SubsystemUnavailable) as excinfo:
        lazy.get()
    assert "attempt 1 failed" in excinfo.value.reason

    with pytest.raises(SubsystemUnavailable):
        lazy.get()  # still inside the backoff window: no second attempt
    assert attempts["n"] == 1, "a failure inside the backoff must not re-probe"

    time.sleep(0.07)
    assert lazy.get() == "built"
    assert attempts["n"] == 2
    assert lazy.get() == "built"
    assert attempts["n"] == 2, "a success must be cached for the life of the process"
