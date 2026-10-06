"""Unit tests for the Paytriq blackboard.

What is being defended here, and why each test matters more than its line count
suggests:

* **Append-only is enforced, not documented.** A blackboard that can be edited
  cannot answer "why did we do this?", so the tests check that neither the
  caller's dict nor the entry handed back from ``read()`` can reach history.
* **Typed storage.** ``post_model`` / ``get_model`` / ``latest_model`` round-trip
  must be lossless, including enum and datetime fields. If this breaks, two
  agents can hold "an offer" that means different things.
* **Justification chains.** ``refs`` must be traversable, deduplicated and
  cycle-safe, and a snapshot must preserve them exactly.
* **Concurrency.** The API posts from request handlers while an SSE stream reads,
  so sequence numbers must stay gap-free under real threads.
* **Schema enforcement at the boundary.** A ``Dispute`` naming the same agent as
  claimant and opponent is rejected by ``core.schemas``; the board must not let
  one through, and must not write an entry when it does not.

No network, no model, no fixtures beyond what is written inline: this file is
part of what proves the package is testable in isolation.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from blackboard import (
    InMemoryBlackboard,
    entry_from_dict,
    from_jsonl,
    render_tree,
    restore,
    snapshot,
    to_jsonl,
)
from blackboard.zones import ZONE_NAMES, Zone, validate_post
from blackboard.zones import zone as resolve_zone
from core.errors import BlackboardError, SchemaError, ZoneNotFound
from core.protocols import Blackboard, BoardEntry
from core.schemas import (
    AgentId,
    Bid,
    BrandLead,
    DecisionSource,
    Dispute,
    DisputeZone,
    EventProfile,
    Offer,
    RiskFlag,
    ROIReport,
    Severity,
)


# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #
@pytest.fixture()
def board() -> InMemoryBlackboard:
    """A fresh board. Never shared between tests -- append-only state leaks."""
    return InMemoryBlackboard()


#: Fixed so that a model rebuilt after a snapshot/restore round trip is
#: comparable to the model that was posted, not merely similar.
FIXED_TS = datetime(2026, 10, 4, 9, 0, 0, tzinfo=UTC)


def make_event(event_id: str = "evt_test_1") -> EventProfile:
    return EventProfile(
        event_id=event_id,
        name="TechFest 2026",
        location="Pune",
        footfall=4000,
        date="2026-11-02",
        audience="engineering undergraduates",
        budget_inr=150_000.0,
        categories_wanted="electronics, food",
        deliverables_offered="banner, stall, social posts",
        contact_email="events@techfest.example",
        created_at=FIXED_TS,
    )


def make_lead(lead_id: str = "brd_1") -> BrandLead:
    return BrandLead(
        lead_id=lead_id,
        name="Acme Electronics",
        category="electronics",
        distance_km=1.2,
        contact_email="sponsors@acme.example",
        rating=4.3,
        fit_score=72.0,
        fit_breakdown={"distance": 40.0, "category": 32.0},
        fit_rationale="nearby, sells the category the event wants",
        source="rules",
        discovered_at=FIXED_TS,
    )


def make_offer(offer_id: str = "off_1", amount: float = 60_000.0,
               version: int = 1) -> Offer:
    return Offer(
        offer_id=offer_id,
        event_id="evt_test_1",
        brand="Acme Electronics",
        tier="gold",
        amount_inr=amount,
        deliverables=["main banner", "social media posts"],
        fit_score=72.0,
        version=version,
        created_at=FIXED_TS,
    )


# =========================================================================== #
# protocol + registry
# =========================================================================== #
def test_board_satisfies_the_frozen_protocol(board: InMemoryBlackboard) -> None:
    """The Protocol is the contract every other package is written against."""
    assert isinstance(board, Blackboard)
    for method in ("post", "read", "latest", "zones", "history"):
        assert callable(getattr(board, method))


def test_registry_exposes_every_required_zone() -> None:
    required = {
        "event", "opportunities", "offers", "threads", "contracts",
        "risk_flags", "audit", "disputes", "bids", "decisions", "lessons",
        "approvals", "handoffs",
    }
    assert required.issubset(set(ZONE_NAMES))
    assert len(set(ZONE_NAMES)) == len(ZONE_NAMES), "zone names must be unique"
    assert all(resolve_zone(name).allowed_kinds for name in ZONE_NAMES)
    assert all(resolve_zone(name).purpose for name in ZONE_NAMES)


def test_zones_declare_their_owner() -> None:
    owners = {
        "opportunities": AgentId.A1_DISCOVERY,
        "offers": AgentId.A2_PRICING,
        "threads": AgentId.A3_OUTREACH,
        "contracts": AgentId.A4_CONTRACT,
        "risk_flags": AgentId.A5_COMPLIANCE,
        "audit": AgentId.A6_AUDIT,
        "decisions": AgentId.A7_ARBITER,
    }
    for name, owner in owners.items():
        zone_def = resolve_zone(name)
        assert zone_def.authoritative_agent is owner
        assert zone_def.is_owned_by(owner)
    for name in ("event", "disputes", "bids", "lessons", "approvals", "handoffs"):
        assert resolve_zone(name).authoritative_agent is None
        assert resolve_zone(name).is_owned_by(AgentId.ENVIRONMENT)
    validate_post("risk_flags", "risk_flag", AgentId.A5_COMPLIANCE)
    validate_post("risk_flags", "risk_flag", AgentId.A6_AUDIT)  # reported, not refused


def test_foreign_author_posts_are_accepted_and_counted(
    board: InMemoryBlackboard,
) -> None:
    """Ownership is declared, not enforced; a non-owner post is still visible.

    A5 legitimately writes ``audit_finding`` entries into the audit zone and A6
    raises a ``risk_flag`` when it catches a compliance-summary mismatch, so
    refusing a non-owner would break real work. It must instead be countable.
    """
    board.post("risk_flags", "risk_flag", AgentId.A5_COMPLIANCE, {"flag_id": "r1"})
    board.post("risk_flags", "risk_flag", AgentId.A6_AUDIT, {"flag_id": "r2"})
    board.post("audit", "roi_report", AgentId.A6_AUDIT, {"report_id": "rpt"})
    board.post("audit", "audit_finding", AgentId.A5_COMPLIANCE, {"promise": "banner"})
    board.post("disputes", "dispute", AgentId.A2_PRICING, {"dispute_id": "d1"})

    stats = board.stats()
    assert stats["foreign_posts"] == {"risk_flags": 1, "audit": 1}
    assert stats["foreign_post_total"] == 2
    assert stats["entries"] == 5, "nothing was rejected"
    # The board still stores the declared owner's entries unchanged.
    assert board.read("risk_flags", kind="risk_flag")[0].author is AgentId.A5_COMPLIANCE


def test_zone_kinds_cover_what_the_pipeline_agents_post(
    board: InMemoryBlackboard,
) -> None:
    """Every (zone, kind) pair the rest of Paytriq posts must be accepted."""
    expected = {
        "event": ("event_profile",),
        "opportunities": ("brand_lead",),
        "offers": ("offer",),
        "threads": ("thread",),
        "contracts": ("mou",),
        "risk_flags": ("risk_flag",),
        "audit": ("roi_report", "audit_finding", "compliance_summary"),
        "disputes": ("dispute",),
        "bids": ("bid",),
        "decisions": ("arbitration_decision", "escalation"),
        "lessons": ("lesson",),
        "approvals": ("human_gate", "approval", "human_decision"),
        "handoffs": ("handoff",),
    }
    for zone_name, kinds in expected.items():
        assert resolve_zone(zone_name).allowed_kinds == kinds
    board.post("decisions", "arbitration_decision", AgentId.A7_ARBITER,
               {"decision_id": "dec_1", "outcome": "revise"})
    board.post("audit", "compliance_summary", AgentId.A5_COMPLIANCE, {"pct": 100})
    board.post("approvals", "human_gate", AgentId.A7_ARBITER,
               {"gate_id": "gat_1"})
    assert len(board.history()) == 3


def test_unknown_zone_raises_zone_not_found(board: InMemoryBlackboard) -> None:
    with pytest.raises(ZoneNotFound):
        validate_post("nowhere", "offer", AgentId.A2_PRICING)
    with pytest.raises(ZoneNotFound):
        board.post("nowhere", "offer", AgentId.A2_PRICING, {})
    with pytest.raises(ZoneNotFound):
        board.read("nowhere")
    with pytest.raises(ZoneNotFound):
        board.latest("nowhere")


def test_disallowed_kind_is_rejected_and_writes_nothing(
    board: InMemoryBlackboard,
) -> None:
    """An ``Offer`` in the ``disputes`` zone is a programming fault, not a fact."""
    with pytest.raises(BlackboardError, match="does not accept kind"):
        board.post("disputes", "offer", AgentId.A2_PRICING, {"offer_id": "x"})
    assert board.history() == []
    # The sequence counter must not have burned a number on the failed post.
    first = board.post("disputes", "dispute", AgentId.A2_PRICING, {"d": 1})
    assert first.seq == 1


def test_non_agent_author_is_rejected(
    board: InMemoryBlackboard,
) -> None:
    """``BoardEntry.author`` is an ``AgentId``; a string is not an agent."""
    with pytest.raises(BlackboardError, match="author must be"):
        board.post("contracts", "mou", "A4", {"mou_id": "mou_1"})


def test_bad_metadata_is_rejected_before_anything_is_written(
    board: InMemoryBlackboard,
) -> None:
    with pytest.raises(BlackboardError, match="payload must be a dict"):
        board.post("event", "event_profile", AgentId.A1_DISCOVERY, ["not", "a", "dict"])
    with pytest.raises(BlackboardError, match=r"confidence must be within"):
        board.post("event", "event_profile", AgentId.A1_DISCOVERY, {},
                   confidence=82.0)
    with pytest.raises(BlackboardError, match="unknown decision source"):
        board.post("event", "event_profile", AgentId.A1_DISCOVERY, {},
                   source="crystal_ball")
    with pytest.raises(BlackboardError, match="refs must be a list"):
        board.post("event", "event_profile", AgentId.A1_DISCOVERY, {},
                   refs="be_abc")
    assert board.history() == []


# =========================================================================== #
# append-only
# =========================================================================== #
def test_board_has_no_mutation_or_deletion_api(
    board: InMemoryBlackboard,
) -> None:
    """The class must not grow an update/delete/clear method behind our back."""
    forbidden = ("delete", "remove", "update", "clear", "pop", "set", "retract")
    public = [name for name in dir(board) if not name.startswith("_")]
    assert not [name for name in public if name in forbidden]


def test_history_only_grows_and_never_changes(board: InMemoryBlackboard) -> None:
    first = board.post("event", "event_profile", AgentId.A1_DISCOVERY,
                       {"event_id": "evt_test_1"})
    second = board.post("offers", "offer", AgentId.A2_PRICING, {"offer_id": "off_1"})
    history = board.history()
    assert [e.entry_id for e in history] == [first.entry_id, second.entry_id]

    third = board.post("audit", "roi_report", AgentId.A6_AUDIT, {"report_id": "r1"})
    assert [e.entry_id for e in board.history()] == [
        first.entry_id, second.entry_id, third.entry_id
    ]
    # The earlier snapshot is unchanged by the later post.
    assert len(history) == 2
    assert board.read("event")[0].entry_id == first.entry_id


def test_payload_is_copied_on_the_way_in(
    board: InMemoryBlackboard,
) -> None:
    """A caller reusing its payload dict must not be able to rewrite history."""
    payload: dict[str, Any] = {"offer_id": "off_1", "deliverables": ["banner"]}
    board.post("offers", "offer", AgentId.A2_PRICING, payload)

    payload["amount_inr"] = 999_999.0
    payload["deliverables"].append("fireworks")

    stored = board.latest("offers")
    assert stored is not None
    assert stored.payload == {"offer_id": "off_1", "deliverables": ["banner"]}
    assert "amount_inr" not in stored.payload


def test_reads_return_copies_so_history_cannot_be_edited(
    board: InMemoryBlackboard,
) -> None:
    entry = board.post("offers", "offer", AgentId.A2_PRICING,
                       {"offer_id": "off_1", "nested": {"amount": 1}})
    stolen = board.read("offers")[0]
    stolen.payload["nested"]["amount"] = -1
    stolen.payload["injected"] = True
    stolen.seq = 999
    stolen.entry_id = "be_forged"

    fresh = board.latest("offers")
    assert fresh is not None
    assert fresh.payload == {"offer_id": "off_1", "nested": {"amount": 1}}
    assert fresh.seq == entry.seq
    assert fresh.entry_id == entry.entry_id
    assert len(board.history()) == 1


def test_post_never_invents_payload_fields(board: InMemoryBlackboard) -> None:
    """Verbatim storage: what the agent said is what the board holds."""
    payload = {"brand": "Acme", "amount_inr": 60_000.0}
    entry = board.post("offers", "offer", AgentId.A2_PRICING, payload)
    assert entry.payload == payload
    assert set(entry.payload) == {"brand", "amount_inr"}


def test_sequence_numbers_are_monotonic_and_dense(
    board: InMemoryBlackboard,
) -> None:
    seqs = [
        board.post("bids", "bid", AgentId.A1_DISCOVERY,
                   {"bid_id": f"bid_{i}", "task_id": "task_1", "agent": "A1",
                    "feasibility": 0.9, "expected_value": 3.0, "effort": 2,
                    "rationale": "r"})
        .seq
        for i in range(5)
    ]
    assert seqs == [1, 2, 3, 4, 5]
    assert [e.seq for e in board.history()] == seqs


# =========================================================================== #
# reads
# =========================================================================== #
def test_read_filters_zone_kind_and_limit(board: InMemoryBlackboard) -> None:
    for i in range(3):
        board.post("offers", "offer", AgentId.A2_PRICING, {"offer_id": f"off_{i}"})
    board.post("approvals", "human_decision", AgentId.A3_OUTREACH, {"gate_id": "g1"})
    board.post("approvals", "approval", AgentId.A3_OUTREACH, {"approval_id": "a1"})

    assert len(board.read("offers")) == 3
    assert len(board.read("offers", kind="offer")) == 3
    assert len(board.read("approvals")) == 2
    assert len(board.read("approvals", kind="approval")) == 1
    # ``limit`` keeps the most recent entries, still chronological.
    assert [e.payload["offer_id"] for e in board.read("offers", limit=2)] == [
        "off_1", "off_2",
    ]
    assert board.read("offers", limit=0) == []
    with pytest.raises(BlackboardError, match="limit must be >= 0"):
        board.read("offers", limit=-1)


def test_latest_returns_none_for_an_empty_zone(board: InMemoryBlackboard) -> None:
    assert board.latest("offers") is None
    assert board.latest("offers", "offer") is None
    board.post("bids", "bid", AgentId.A3_OUTREACH,
               {"bid_id": "bid_1", "task_id": "task_1", "agent": "A3",
                "feasibility": 0.5, "expected_value": 1.0, "effort": 1,
                "rationale": "r"})
    assert board.latest("bids") is not None
    assert board.latest("bids", "offer") is None


def test_zones_lists_the_registry_including_empty_zones(
    board: InMemoryBlackboard,
) -> None:
    assert board.zones() == list(ZONE_NAMES)
    board.post("bids", "bid", AgentId.A1_DISCOVERY,
               {"bid_id": "bid_1", "task_id": "task_1", "agent": "A1",
                "feasibility": 0.5, "expected_value": 1.0, "effort": 1,
                "rationale": "r"})
    assert board.zones() == list(ZONE_NAMES)


def test_stats_counts_per_zone_author_and_kind(board: InMemoryBlackboard) -> None:
    board.post("opportunities", "brand_lead", AgentId.A1_DISCOVERY, {"lead_id": "b1"})
    board.post("offers", "offer", AgentId.A2_PRICING, {"offer_id": "o1"})
    board.post("offers", "offer", AgentId.A2_PRICING, {"offer_id": "o2"})
    stats = board.stats()
    assert stats["entries"] == 3
    assert stats["zones"]["offers"] == 2
    assert stats["zones"]["opportunities"] == 1
    assert stats["zones"]["audit"] == 0, "empty zones must still be reported"
    assert stats["authors"] == {"A1": 1, "A2": 2}
    assert stats["kinds"] == {"brand_lead": 1, "offer": 2}
    assert stats["ref_edges"] == 0
    assert stats["foreign_post_total"] == 0
    assert stats["foreign_posts"] == {}


# =========================================================================== #
# typed storage
# =========================================================================== #
def test_post_model_stores_both_model_and_dict(board: InMemoryBlackboard) -> None:
    offer = make_offer()
    entry = board.post_model("offers", offer)
    assert entry.zone == "offers"
    assert entry.kind == "offer"
    assert entry.author is AgentId.A2_PRICING, "zone's authoritative writer"
    assert entry.payload == offer.model_dump(mode="json")
    assert board.model_kind_of(entry.entry_id) == "offer"


def test_typed_round_trip_via_get_model_is_lossless(
    board: InMemoryBlackboard,
) -> None:
    """Enums, datetimes and optionals must survive model -> dict -> model."""
    flag = RiskFlag(
        flag_id="rsk_1",
        event_id="evt_test_1",
        brand="Acme Electronics",
        severity=Severity.BLOCKING,
        code="missing_insurance",
        message="No public liability cover on file for an on-site activation.",
        evidence=["rsk_1_doc_requested"],
    )
    entry = board.post_model("risk_flags", flag)
    restored = board.get_model(entry.entry_id, RiskFlag)
    assert restored == flag
    assert isinstance(restored.severity, Severity)
    assert restored.severity is Severity.BLOCKING
    assert isinstance(restored.raised_at, type(flag.raised_at))


def test_typed_round_trip_survives_snapshot_restore(
    board: InMemoryBlackboard,
) -> None:
    entry = board.post_model("opportunities", make_lead())
    revived = restore(snapshot(board))
    assert revived.get_model(entry.entry_id, BrandLead) == make_lead()
    assert revived.model_kind_of(entry.entry_id) == "brand_lead"


def test_latest_model_returns_the_most_recent(
    board: InMemoryBlackboard,
) -> None:
    board.post_model("offers", make_offer("off_1", 60_000.0, version=1))
    board.post_model("offers", make_offer("off_1", 45_000.0, version=2))
    latest = board.latest_model("offers", Offer)
    assert latest is not None
    assert latest.version == 2
    assert latest.amount_inr == 45_000.0
    assert latest.offer_id == "off_1"
    # v1 is still readable: revisions append, they never overwrite.
    assert len(board.read("offers", kind="offer")) == 2
    assert board.latest_model("offers", Offer, kind="offer") == latest
    assert board.latest_model("contracts", ROIReport) is None


def test_get_model_can_type_a_raw_payload(board: InMemoryBlackboard) -> None:
    """A dict posted raw is still provably *an Offer* when asked."""
    entry = board.post("offers", "offer", AgentId.A2_PRICING,
                       make_offer().model_dump(mode="json"))
    assert board.model_kind_of(entry.entry_id) is None
    assert board.get_model(entry.entry_id, Offer).amount_inr == 60_000.0

    with pytest.raises(SchemaError):
        board.get_model(entry.entry_id, RiskFlag)


def test_post_model_metadata_is_resolved_not_invented(
    board: InMemoryBlackboard,
) -> None:
    """Handoff carries its own author, source and confidence; they are copied."""
    from core.schemas import Handoff

    handoff = Handoff(
        handoff_id="hnd_1",
        event_id="evt_test_1",
        run_id="run_test",
        from_agent=AgentId.A2_PRICING,
        to_agent=AgentId.A3_OUTREACH,
        reason="offer priced and ready to send",
        decision_source=DecisionSource.CLEF,
        confidence=0.77,
    )
    entry = board.post_model("handoffs", handoff, refs=["be_1"])
    assert entry.author is AgentId.A2_PRICING
    assert entry.source is DecisionSource.CLEF
    assert entry.confidence == pytest.approx(0.77)
    assert entry.refs == ["be_1"]


def test_post_model_never_infers_refs_from_evidence(
    board: InMemoryBlackboard,
) -> None:
    """Evidence holds flag ids and URLs, not board entry ids."""
    dispute = Dispute(
        dispute_id="dsp_1",
        event_id="evt_test_1",
        zone=DisputeZone.PRICING,
        claimant=AgentId.A2_PRICING,
        opponent=AgentId.A5_COMPLIANCE,
        position="price is defensible",
        counter_position="price needs a cover note",
        evidence=["rsk_1", "https://example.invalid/policy"],
    )
    entry = board.post_model("disputes", dispute)
    assert entry.refs == []
    assert board.refs_of(entry.entry_id) == []


def test_post_model_rejects_an_unknown_model(board: InMemoryBlackboard) -> None:
    from core.schemas import TraceEvent, TraceKind

    trace = TraceEvent(
        seq=1, run_id="run_1", event_id="evt_1", trace_id="t" * 32,
        span_id="s" * 16, kind=TraceKind.AGENT, name="x",
    )
    with pytest.raises(BlackboardError, match="no blackboard kind"):
        board.post_model("event", trace, author=AgentId.A1_DISCOVERY)


def test_post_model_rejects_a_dict(board: InMemoryBlackboard) -> None:
    with pytest.raises(BlackboardError, match="model instance"):
        board.post_model("offers", make_offer().model_dump(mode="json"))


def test_post_model_requires_an_author_when_nothing_implies_one(
    board: InMemoryBlackboard,
) -> None:
    with pytest.raises(BlackboardError, match="author is required"):
        board.post_model("event", make_event())


def test_bid_is_typed_end_to_end(board: InMemoryBlackboard) -> None:
    bid = Bid(
        bid_id="bid_1",
        task_id="task_design_banner",
        agent=AgentId.A3_OUTREACH,
        feasibility=0.8,
        expected_value=120.0,
        effort=3,
        rationale="already produced banner copy this cycle",
    )
    entry = board.post_model("bids", bid)
    assert entry.author is AgentId.A3_OUTREACH
    assert bid.utility == pytest.approx(32.0)
    assert board.get_model(entry.entry_id, Bid).utility == pytest.approx(32.0)


# =========================================================================== #
# reference chains
# =========================================================================== #
def test_refs_of_walks_the_whole_chain_in_breadth_first_order(
    board: InMemoryBlackboard,
) -> None:
    event = board.post("event", "event_profile", AgentId.A1_DISCOVERY,
                       {"event_id": "evt_test_1"})
    lead = board.post("opportunities", "brand_lead", AgentId.A1_DISCOVERY,
                      {"lead_id": "brd_1"}, refs=[event.entry_id])
    offer = board.post("offers", "offer", AgentId.A2_PRICING, {"offer_id": "off_1"},
                       refs=[lead.entry_id])
    thread = board.post("threads", "thread", AgentId.A3_OUTREACH,
                        {"thread_id": "thr_1"}, refs=[offer.entry_id])

    assert board.refs_of(thread.entry_id) == [
        offer.entry_id, lead.entry_id, event.entry_id
    ]
    assert board.refs_of(thread.entry_id, transitive=False) == [offer.entry_id]
    assert board.refs_of(event.entry_id) == []
    # An entry is never part of its own ancestry.
    assert thread.entry_id not in board.refs_of(thread.entry_id)


def test_refs_of_deduplicates_a_diamond(
    board: InMemoryBlackboard,
) -> None:
    root = board.post("event", "event_profile", AgentId.A1_DISCOVERY, {"e": 1})
    left = board.post("bids", "bid", AgentId.A1_DISCOVERY, {"bid_id": "l"},
                      refs=[root.entry_id])
    right = board.post("bids", "bid", AgentId.A1_DISCOVERY, {"bid_id": "r"},
                       refs=[root.entry_id])
    merged = board.post("decisions", "arbitration_decision", AgentId.A7_ARBITER,
                        {"outcome": "award"},
                        refs=[left.entry_id, right.entry_id, root.entry_id])
    assert board.refs_of(merged.entry_id) == [left.entry_id, right.entry_id,
                                              root.entry_id]


def test_refs_of_keeps_unresolvable_citations_recorded_but_not_walked(
    board: InMemoryBlackboard,
) -> None:
    """A risk-flag id is a legitimate reference; there is just nothing to walk."""
    event = board.post("event", "event_profile", AgentId.A1_DISCOVERY, {"e": 1})
    dispute = board.post(
        "disputes", "dispute", AgentId.A2_PRICING,
        {"dispute_id": "dsp_1", "evidence": ["rsk_1", event.entry_id]},
        refs=["rsk_1", event.entry_id],
    )
    assert dispute.refs == ["rsk_1", event.entry_id], "stored verbatim"
    assert board.refs_of(dispute.entry_id) == [event.entry_id]


def test_refs_of_rejects_an_unknown_entry(board: InMemoryBlackboard) -> None:
    with pytest.raises(BlackboardError, match="unknown entry_id"):
        board.refs_of("be_does_not_exist")
    with pytest.raises(BlackboardError, match="unknown entry_id"):
        board.entry("be_does_not_exist")
    with pytest.raises(BlackboardError, match="unknown entry_id"):
        board.get_model("be_does_not_exist", Offer)
    with pytest.raises(BlackboardError, match="unknown entry_id"):
        board.model_kind_of("be_does_not_exist")


def test_a_forged_citation_chain_is_not_invented(
    board: InMemoryBlackboard,
) -> None:
    """``refs`` are stored as given; the board does not invent the justifications."""
    board.post("offers", "offer", AgentId.A2_PRICING, {"offer_id": "off_1"},
               refs=["be_nonexistent"])
    entry = board.latest("offers")
    assert entry is not None
    assert entry.refs == ["be_nonexistent"]
    assert board.refs_of(entry.entry_id) == []


# =========================================================================== #
# thread safety
# =========================================================================== #
def test_concurrent_posts_keep_sequence_numbers_dense_and_unique(
    board: InMemoryBlackboard,
) -> None:
    writers = 8
    per_writer = 40
    barrier = threading.Barrier(writers)
    errors: list[str] = []

    def write(worker: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(per_writer):
                board.post(
                    "bids", "bid", AgentId.A1_DISCOVERY,
                    {"bid_id": f"bid_{worker}_{i}", "task_id": f"task_{worker}",
                     "agent": "A1", "feasibility": 0.5, "expected_value": 1.0,
                     "effort": 1, "rationale": "concurrent"},
                )
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(f"writer {worker}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=write, args=(w,)) for w in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    history = board.history()
    assert len(history) == writers * per_writer
    assert [e.seq for e in history] == list(range(1, writers * per_writer + 1))
    assert len({e.entry_id for e in history}) == len(history)
    assert board.stats()["entries"] == len(history)


def test_reads_during_concurrent_posts_never_observe_a_torn_board(
    board: InMemoryBlackboard,
) -> None:
    """The API layer streams history while request handlers post into it."""
    total = 300
    stop = threading.Event()
    observed: list[int] = []
    errors: list[str] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                history = board.history()
                seqs = [e.seq for e in history]
                if seqs != sorted(seqs):
                    errors.append("history came back out of order")
                observed.append(len(seqs))
                board.zones()
                board.stats()
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(f"reader: {type(exc).__name__}: {exc}")

    def writer(worker: int) -> None:
        try:
            for i in range(total // 3):
                board.post("bids", "bid", AgentId.A2_PRICING,
                           {"bid_id": f"bid_{worker}_{i}", "task_id": "t",
                            "agent": "A2", "feasibility": 0.5,
                            "expected_value": 1.0, "effort": 1, "rationale": "r"})
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(f"writer {worker}: {type(exc).__name__}: {exc}")

    readers = [threading.Thread(target=reader) for _ in range(2)]
    writers = [threading.Thread(target=writer, args=(w,)) for w in range(3)]
    for thread in readers:
        thread.start()
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(timeout=30)
    stop.set()
    for thread in readers:
        thread.join(timeout=30)

    assert not errors, errors[:5]
    assert observed, "the reader thread never ran"
    assert board.stats()["entries"] == total
    assert max(observed) <= total


# =========================================================================== #
# serialization
# =========================================================================== #
def build_sample_board() -> tuple[InMemoryBlackboard, dict[str, str]]:
    """A miniature but realistic run: seed, discover, price, object, arbitrate."""
    board = InMemoryBlackboard()
    ids: dict[str, str] = {}

    ids["event"] = board.post_model(
        "event", make_event(), author=AgentId.A1_DISCOVERY).entry_id
    lead = make_lead()
    ids["lead"] = board.post_model(
        "opportunities", lead, refs=[ids["event"]], confidence=0.82).entry_id
    offer = make_offer()
    ids["offer"] = board.post_model(
        "offers", offer, refs=[ids["lead"]]).entry_id
    ids["flag"] = board.post_model(
        "risk_flags",
        RiskFlag(flag_id="rsk_1", event_id="evt_test_1",
                 brand="Acme Electronics", severity=Severity.BLOCKING,
                 code="missing_insurance", message="no cover note",
                 evidence=[ids["offer"]]),
        refs=[ids["offer"]],
    ).entry_id
    ids["dispute"] = board.post_model(
        "disputes",
        Dispute(dispute_id="dsp_1", event_id="evt_test_1",
                zone=DisputeZone.PRICING, claimant=AgentId.A2_PRICING,
                opponent=AgentId.A5_COMPLIANCE,
                position="price is defensible",
                counter_position="price needs a cover note",
                evidence=[ids["offer"], ids["flag"]]),
    ).entry_id
    ids["ruling"] = board.post(
        "decisions", "arbitration_decision", AgentId.A7_ARBITER,
        {"dispute_id": "dsp_1", "outcome": "revise",
         "rationale": "hold the offer pending the insurance certificate",
         "rounds": 1},
        refs=[ids["dispute"], ids["flag"], ids["offer"]],
        confidence=0.71,
        source=DecisionSource.CLEF,
    ).entry_id
    return board, ids


def test_snapshot_is_json_safe_and_complete() -> None:
    board, ids = build_sample_board()
    data = snapshot(board)
    assert json.loads(json.dumps(data)) == data, "must survive a JSON round trip"
    assert data["kind"] == "paytriq.blackboard.snapshot"
    assert data["zones"] == list(ZONE_NAMES)
    assert data["stats"]["entries"] == len(board.history())
    assert [e["entry_id"] for e in data["entries"]] == list(ids.values())
    assert data["entries"][-1]["refs"] == [
        ids["dispute"], ids["flag"], ids["offer"]
    ]
    assert data["entries"][-1]["source"] == DecisionSource.CLEF.value
    typed = [e for e in data["entries"] if "model_kind" in e]
    assert {e["model_kind"] for e in typed} == {
        "event_profile", "brand_lead", "offer", "risk_flag", "dispute"
    }


def test_snapshot_restore_is_lossless() -> None:
    board, _ = build_sample_board()
    revived = restore(snapshot(board))

    original, copy = board.history(), revived.history()
    assert len(copy) == len(original)
    for before, after in zip(original, copy, strict=False):
        assert (after.entry_id, after.zone, after.kind, after.author) == (
            before.entry_id, before.zone, before.kind, before.author
        )
        assert (after.seq, after.at, after.confidence, after.source) == (
            before.seq, before.at, before.confidence, before.source
        )
        assert after.payload == before.payload
        assert after.refs == before.refs

    assert revived.stats() == board.stats()
    assert revived.zones() == board.zones()
    assert revived.history()[0].payload["event_id"] == "evt_test_1"


def test_restored_board_continues_the_sequence() -> None:
    board, _ = build_sample_board()
    revived = restore(snapshot(board))
    highest = max(e.seq for e in revived.history())
    fresh = revived.post("bids", "bid", AgentId.A4_CONTRACT,
                         {"bid_id": "bid_after_restore", "task_id": "t",
                          "agent": "A4", "feasibility": 1.0, "expected_value": 1.0,
                          "effort": 1, "rationale": "r"})
    assert fresh.seq == highest + 1


def test_restored_board_keeps_typed_models_and_refs() -> None:
    board, ids = build_sample_board()
    revived = restore(snapshot(board))
    assert revived.get_model(ids["offer"], Offer) == make_offer()
    assert revived.get_model(ids["dispute"], Dispute).opponent is AgentId.A5_COMPLIANCE
    assert revived.refs_of(ids["ruling"]) == board.refs_of(ids["ruling"])


def test_jsonl_round_trip_is_one_line_per_entry() -> None:
    board, ids = build_sample_board()
    text = to_jsonl(board)
    lines = text.splitlines()
    assert len(lines) == len(board.history())
    assert text.endswith("\n")
    for line in lines:
        json.loads(line)  # exactly one JSON object per line

    revived = from_jsonl(text)
    assert [e.entry_id for e in revived.history()] == list(ids.values())
    assert revived.stats() == board.stats()
    assert revived.get_model(ids["offer"], Offer) == make_offer()


def test_jsonl_is_deterministic(board: InMemoryBlackboard) -> None:
    board.post_model("opportunities", make_lead(), confidence=0.82)
    assert to_jsonl(board) == to_jsonl(board)


def test_restore_rejects_a_corrupted_snapshot() -> None:
    board, _ = build_sample_board()
    data = snapshot(board)

    truncated = dict(data)
    truncated["entries"] = data["entries"][:-1]
    # Not corruption per se, but dropping entries must not corrupt the rest.
    assert len(restore(truncated).history()) == len(data["entries"]) - 1

    reordered = dict(data)
    reordered["entries"] = list(reversed(data["entries"]))
    with pytest.raises(BlackboardError, match="not strictly increasing"):
        restore(reordered)

    tampered = json.loads(json.dumps(data))
    tampered["entries"][0]["model_kind"] = "brand_lead"
    with pytest.raises(BlackboardError, match="does not validate"):
        restore(tampered)

    unknown_model = json.loads(json.dumps(data))
    unknown_model["entries"][0]["model_kind"] = "sponsorship"
    with pytest.raises(BlackboardError, match="unknown model_kind"):
        restore(unknown_model)

    with pytest.raises(BlackboardError, match=r"paytriq\.blackboard\.snapshot"):
        restore({"kind": "something.else", "entries": []})
    with pytest.raises(BlackboardError, match="unsupported snapshot version"):
        restore({"kind": "paytriq.blackboard.snapshot", "version": 99, "entries": []})
    with pytest.raises(BlackboardError, match="entries must be a list"):
        restore({"entries": "nope"})


def test_from_jsonl_reports_the_offending_line() -> None:
    board, _ = build_sample_board()
    text = to_jsonl(board)
    with pytest.raises(BlackboardError, match="line 2 is not valid JSON"):
        from_jsonl(text.splitlines()[0] + "\n{not json}\n")
    # Blank lines are tolerated; a truncated log must not silently look whole.
    assert len(from_jsonl("\n\n" + text + "\n").history()) == len(board.history())


def test_snapshot_refuses_unserialisable_payloads(
    board: InMemoryBlackboard,
) -> None:
    board.post("opportunities", "brand_lead", AgentId.A1_DISCOVERY,
               {"lead_id": "brd_1", "scratchpad": {1, 2, 3}})
    with pytest.raises(BlackboardError, match="cannot serialise set"):
        snapshot(board)


def test_entry_from_dict_round_trips_verbatim() -> None:
    entry = BoardEntry(
        entry_id="be_1", zone="offers", kind="offer",
        author=AgentId.A2_PRICING, payload={"offer_id": "off_1"},
        refs=["be_0"], confidence=0.5, source=DecisionSource.GEMINI,
        seq=2, at="2026-10-04T00:00:00+00:00",
    )
    parsed, model = entry_from_dict(
        {"entry_id": "be_1", "zone": "offers", "kind": "offer", "author": "A2",
         "seq": 2, "at": "2026-10-04T00:00:00+00:00",
         "confidence": 0.5, "source": "gemini", "refs": ["be_0"],
         "payload": {"offer_id": "off_1"}}
    )
    assert model is None
    assert parsed == entry
    with pytest.raises(BlackboardError, match="missing required field"):
        entry_from_dict({"entry_id": "be_1"})
    with pytest.raises(BlackboardError, match="unknown author"):
        entry_from_dict({"entry_id": "be_1", "zone": "offers", "kind": "offer",
                         "author": "A9", "seq": 1})


# =========================================================================== #
# render_tree
# =========================================================================== #
def test_render_tree_shows_the_derivation() -> None:
    board, _ = build_sample_board()
    tree = render_tree(board)
    assert "event" in tree and "decisions" in tree
    lines = [line for line in tree.splitlines() if "[event" in line or "[decisions" in line]
    assert lines[0].startswith("#1")
    assert not lines[0].startswith(" ")
    ruling_line = next(line for line in lines if "[decisions" in line)
    assert ruling_line.startswith(" "), "a cited entry must be indented"
    assert len(ruling_line) - len(ruling_line.lstrip()) > 3


def test_render_tree_prints_every_entry_exactly_once_with_its_citations(
    board: InMemoryBlackboard,
) -> None:
    """A drawn tree would redraw shared ancestors; the ladder must not."""
    root = board.post("event", "event_profile", AgentId.A1_DISCOVERY, {"e": 1})
    left = board.post("bids", "bid", AgentId.A1_DISCOVERY, {"bid_id": "l"},
                      refs=[root.entry_id])
    right = board.post("bids", "bid", AgentId.A3_OUTREACH, {"bid_id": "r"},
                       refs=[root.entry_id])
    merged = board.post("decisions", "arbitration_decision", AgentId.A7_ARBITER,
                        {"outcome": "award"},
                        refs=[left.entry_id, right.entry_id, root.entry_id])
    board.post("lessons", "lesson", AgentId.A7_ARBITER,
               {"lesson_id": "les_1", "event_id": "e", "trigger": "t",
                "correction": "c", "rule": "r"},
               refs=[merged.entry_id])

    tree = render_tree(board)
    body = [line for line in tree.splitlines() if line.startswith(("#", " ")) and "#" in line]
    assert len(body) == len(board.history()), "one printed line per entry"
    assert sum(1 for line in body if "[decisions" in line) == 1
    arbitration = next(line for line in body if "[decisions" in line)
    assert "cites #2 #3 #1" in arbitration, "citations are listed explicitly"
    # Indentation equals derivation depth: 0, 1, 1, 2, 3.
    depths = [len(line) - len(line.lstrip(" ")) for line in body]
    assert depths == [0, 2, 2, 4, 6]


def test_render_tree_handles_an_empty_board(board: InMemoryBlackboard) -> None:
    assert "empty" in render_tree(board).lower()


def test_render_tree_does_not_mutate_the_board() -> None:
    board, _ = build_sample_board()
    before = [e.entry_id for e in board.history()]
    render_tree(board)
    render_tree(board)
    assert [e.entry_id for e in board.history()] == before


# =========================================================================== #
# schema enforcement at the board boundary
# =========================================================================== #
def test_dispute_with_identical_claimant_and_opponent_is_rejected() -> None:
    """A self-dispute is not disagreement; ``core.schemas`` refuses to model it."""
    with pytest.raises(ValidationError, match="two distinct agents"):
        Dispute(
            dispute_id="dsp_1",
            event_id="evt_test_1",
            zone=DisputeZone.PRICING,
            claimant=AgentId.A2_PRICING,
            opponent=AgentId.A2_PRICING,
            position="price is defensible",
            counter_position="price is not defensible",
        )


def test_board_refuses_an_invalid_dispute_and_writes_nothing(
    board: InMemoryBlackboard,
) -> None:
    """``model_construct`` skips validation, so this exercises the board's own.

    An artefact that reached the board by any route that bypassed construction
    must not become a fact of record; ``post_model`` re-validates precisely so it
    cannot.
    """
    forged = Dispute.model_construct(
        dispute_id="dsp_1",
        event_id="evt_test_1",
        zone=DisputeZone.PRICING,
        claimant=AgentId.A2_PRICING,
        opponent=AgentId.A2_PRICING,
        position="price is defensible",
        counter_position="price is not defensible",
    )
    with pytest.raises(SchemaError, match="two distinct agents"):
        board.post_model("disputes", forged)
    assert board.history() == [], "a rejected post must leave no trace"


def test_a_self_dispute_posted_raw_cannot_be_read_back_as_a_dispute(
    board: InMemoryBlackboard,
) -> None:
    """The dict path is equally strict once you ask what the artefact *is*."""
    entry = board.post(
        "disputes", "dispute", AgentId.A2_PRICING,
        {"dispute_id": "dsp_1", "event_id": "evt_test_1",
         "zone": "pricing", "claimant": "A2", "opponent": "A2",
         "position": "p", "counter_position": "c"},
    )
    with pytest.raises(SchemaError, match="two distinct agents"):
        board.get_model(entry.entry_id, Dispute)


def test_dispute_parties_must_be_reasoning_agents(
    board: InMemoryBlackboard,
) -> None:
    with pytest.raises(ValidationError, match="reasoning agents only"):
        Dispute(
            dispute_id="dsp_2",
            event_id="evt_test_1",
            zone=DisputeZone.OUTREACH,
            claimant=AgentId.A3_OUTREACH,
            opponent=AgentId.ENVIRONMENT,
            position="the sponsor asked for a discount",
            counter_position="discounts need approval",
        )


def test_roi_report_without_assumptions_is_rejected() -> None:
    """Every derived figure needs a stated basis. An undocumented ROI is not a result."""
    with pytest.raises(ValidationError, match="assumptions must be non-empty"):
        ROIReport(report_id="rpt_1", event_id="evt_test_1", roi_multiple=3.2)


def test_zone_definition_rejects_a_kind_with_no_model() -> None:
    from blackboard.zones import register_zone

    with pytest.raises(BlackboardError, match="no canonical core model"):
        register_zone(Zone("bogus", "test zone", ("not_a_kind",)))
    with pytest.raises(BlackboardError, match="already registered"):
        register_zone(Zone("offers", "duplicate", ("offer",)))
    assert resolve_zone("offers").authoritative_agent is AgentId.A2_PRICING
