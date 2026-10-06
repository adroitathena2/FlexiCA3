"""Minimal round-trip tests for the first-class Message record (Track 1)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.schemas import AgentId, DecisionSource, Message, MessageKind


def make_message(**overrides) -> Message:
    base = {
        "message_id": "msg_1",
        "event_id": "evt_test_1",
        "run_id": "run_test",
        "from_agent": AgentId.A2_PRICING,
        "to_agent": AgentId.A3_OUTREACH,
        "kind": MessageKind.PROPOSAL,
        "body": "proposing 45k silver tier",
        "refs": ["be_1"],
        "decision_source": DecisionSource.CLEF,
        "confidence": 0.8,
        "summary": "opening proposal",
    }
    base.update(overrides)
    return Message(**base)


def test_message_kind_values() -> None:
    assert [k.value for k in MessageKind] == [
        "proposal", "critique", "counter_proposal", "revision_request", "verdict",
    ]


def test_message_round_trip_json() -> None:
    msg = make_message()
    restored = Message(**msg.model_dump(mode="json"))
    assert restored == msg
    assert restored.kind is MessageKind.PROPOSAL
    assert isinstance(restored.from_agent, AgentId)


def test_message_rejects_self_send() -> None:
    with pytest.raises(ValidationError, match="must change the active agent"):
        make_message(from_agent=AgentId.A2_PRICING, to_agent=AgentId.A2_PRICING)


def test_message_rejects_empty_body() -> None:
    with pytest.raises(ValidationError):
        make_message(body="")


def test_message_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        make_message(confidence=1.5)
    with pytest.raises(ValidationError):
        make_message(confidence=-0.1)
