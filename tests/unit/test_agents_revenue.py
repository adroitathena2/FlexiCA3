"""Offline tests for A1–A4: the four revenue agents.

Everything here is offline and model-free by construction:

* ``decide`` is a stub callable that returns schema-valid ``Decision`` objects,
  so no decision backend, network or API key is involved;
* the tool registry is a set of stub ``Tool`` objects with explicit signatures,
  which also exercises the agents' signature-adaptive invocation;
* the blackboard is a strict in-test stub that mirrors the shipped zone
  registry — same allowed kinds, same single-writer rules — so an agent that
  posts to a zone it does not own fails here rather than in a demo. One test
  cross-checks the table below against ``blackboard.zones`` when it is importable.

The regressions pinned by this file
----------------------------------
* *"yesterday we thought the price was too high"* classifies as ``pushback``,
  not ``yes`` — the old router tested assent first and routed a price objection
  into contract signing.
* the send gate refuses to send with no approval record, and there is no default
  that skips it.
* A4 refuses to draft an MoU for an offer nobody accepted, and drafts from the
  accepted sponsor rather than the first one posted.
* ``BrandLead.fit_breakdown`` sums exactly to ``fit_score``.
* each agent degrades honestly when its tool is unavailable: nothing is
  fabricated, and the gap says which tool was missing.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from agents.a1_discovery import (  # noqa: E402
    COMPONENT_WEIGHTS,
    FIT_RUBRIC,
    DiscoveryAgent,
)
from agents.a2_pricing import (  # noqa: E402
    CONCESSION_OPTIONS,
    POSTURE_OPTIONS,
    RATE_PER_ATTENDEE_INR,
    RATE_PROVENANCE,
    TIER_OPTIONS,
    PricingAgent,
)
from agents.a3_outreach import INTENT_OPTIONS, OutreachAgent  # noqa: E402
from agents.a4_contract import ContractAgent  # noqa: E402
from core.errors import BlackboardError, HumanGateRequired  # noqa: E402
from core.ids import new_id, run_id  # noqa: E402
from core.protocols import (  # noqa: E402
    AgentContext,
    BoardEntry,
    ToolResult,
    classify_intent_fallback,
)
from core.schemas import (  # noqa: E402
    AgentId,
    BrandLead,
    Decision,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    GateKind,
    GateOutcome,
    HumanDecision,
    Intent,
    MoU,
    Offer,
    QuestionType,
    RunMode,
    Thread,
    ToolStatus,
    TraceKind,
)

# ============================================================================ board
#: Mirrors ``blackboard/zones.py``: ``zone -> (allowed kinds, authoritative
#: author or None)``. Kept local so these tests do not depend on another
#: package's import-time behaviour; the cross-check tests prove both that it has
#: not drifted and that every kind these agents post is really accepted.
ZONE_TABLE: dict[str, tuple[set[str], str | None]] = {
    "event": ({"event_profile"}, None),
    "opportunities": ({"brand_lead"}, "A1"),
    "offers": ({"offer"}, "A2"),
    "threads": ({"thread"}, "A3"),
    "contracts": ({"mou"}, "A4"),
    "risk_flags": ({"risk_flag"}, "A5"),
    "audit": ({"roi_report", "audit_finding", "compliance_summary"}, "A6"),
    "approvals": ({"human_gate", "approval", "human_decision"}, None),
    "handoffs": ({"handoff"}, None),
    "lessons": ({"lesson"}, None),
    "disputes": ({"dispute"}, None),
    "bids": ({"bid"}, None),
    "decisions": ({"arbitration", "escalation"}, "A7"),
}

#: Every ``(zone, kind, author)`` triple A1-A4 posts. Used by the integration
#: test to prove the real registry accepts all of them.
AGENT_POSTS: tuple[tuple[str, str, AgentId], ...] = (
    ("event", "event_profile", AgentId.A1_DISCOVERY),
    ("opportunities", "brand_lead", AgentId.A1_DISCOVERY),
    ("offers", "offer", AgentId.A2_PRICING),
    ("lessons", "lesson", AgentId.A2_PRICING),
    ("threads", "thread", AgentId.A3_OUTREACH),
    ("handoffs", "handoff", AgentId.A3_OUTREACH),
    ("approvals", "human_gate", AgentId.A3_OUTREACH),
    ("approvals", "human_decision", AgentId.A3_OUTREACH),
    ("approvals", "approval", AgentId.A3_OUTREACH),
    ("approvals", "human_gate", AgentId.A4_CONTRACT),
    ("approvals", "approval", AgentId.A4_CONTRACT),
    ("contracts", "mou", AgentId.A4_CONTRACT),
)


class StubBoard:
    """Strict append-only blackboard: same acceptance rules as the real one."""

    def __init__(self, zone_table: Mapping[str, Any] | None = None) -> None:
        self._entries: list[BoardEntry] = []
        self._zones: Mapping[str, Any] = zone_table or ZONE_TABLE

    # ------------------------------------------------------------------- write
    def post(self, zone: str, kind: str, author: AgentId, payload: dict[str, Any],
             *, refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> BoardEntry:
        key = (zone or "").strip().lower()
        if key not in self._zones:
            raise BlackboardError(f"unknown zone {zone!r}")
        kinds, owner = self._zones[key]
        if kind not in kinds:
            raise BlackboardError(
                f"zone {key!r} does not accept kind {kind!r}; allowed: {sorted(kinds)}")
        if owner is not None and author.value != owner:
            raise BlackboardError(
                f"zone {key!r} is authoritative for {owner}; {author.value} may "
                f"not post here")
        if not 0.0 <= confidence <= 1.0:
            raise BlackboardError(f"confidence out of range: {confidence}")
        entry = BoardEntry(
            entry_id=new_id("be"), zone=key, kind=kind, author=author,
            payload=dict(payload), refs=list(refs or ()),
            confidence=float(confidence), source=source, seq=len(self._entries) + 1,
            at="2026-10-04T00:00:00+00:00",
        )
        self._entries.append(entry)
        return entry

    # -------------------------------------------------------------------- read
    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[BoardEntry]:
        key = (zone or "").strip().lower()
        if key not in self._zones:
            raise BlackboardError(f"unknown zone {zone!r}")
        rows = [e for e in self._entries
                if e.zone == key and (kind is None or e.kind == kind)]
        if limit is not None:
            rows = rows[len(rows) - limit:] if limit else []
        return rows

    def latest(self, zone: str, kind: str | None = None) -> BoardEntry | None:
        rows = self.read(zone, kind=kind)
        return rows[-1] if rows else None

    def history(self) -> list[BoardEntry]:
        return list(self._entries)

    def latest_model(self, zone: str, model_cls: type, kind: str | None = None):
        rows = self.read(zone, kind=kind)
        return model_cls.model_validate(rows[-1].payload) if rows else None

    def get_model(self, entry_id: str, model_cls: type):
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return model_cls.model_validate(entry.payload)
        raise BlackboardError(f"unknown entry {entry_id!r}")

    def kinds(self, zone: str, kind: str) -> list[BoardEntry]:
        return self.read(zone, kind=kind)


# =========================================================================== tools
class StubTool:
    """A ``Tool`` with a real signature, so ``inspect``-based calls are tested."""

    def __init__(self, name: str, run: Callable[..., ToolResult],
                 *, available: tuple[bool, str] = (True, "")) -> None:
        self.name = name
        self.description = f"stub {name}"
        self.mode = "fixture"
        self.backend = "stub"
        self._run = run
        self._available = available
        self.calls: list[dict[str, Any]] = []

    def available(self) -> tuple[bool, str]:
        return self._available

    def run(self, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        return self._run(**kwargs)


def brand_rows(count: int = 4, *, with_email: bool = False) -> list[dict[str, Any]]:
    """Synthetic candidate rows. Clearly labelled, like ``tools/fixtures.py``."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append({
            "lead_id": new_id("brd"),
            "name": f"STUB_Brand_{index}",
            "category": "cafe" if index % 2 == 0 else "printing",
            "distance_km": 0.5 + index,
            "contact_email": f"team{index}@real-brand.example.org" if with_email else None,
            "phone": None,
            "rating": 4.0 + index * 0.1,
            "source": "stub_tool",
        })
    return rows


def make_search_tool(rows: Sequence[Mapping[str, Any]] | None = None,
                     *, result: ToolResult | None = None,
                     available: tuple[bool, str] = (True, "")) -> StubTool:
    payload = list(rows) if rows is not None else brand_rows()

    def run(**kwargs: Any) -> ToolResult:
        if result is not None:
            return result
        return ToolResult.success(payload, source="search_brands:stub")

    return StubTool("search_brands", run, available=available)


def make_send_tool(*, status: ToolStatus = ToolStatus.OK,
                   reason: str = "") -> StubTool:
    def run(**kwargs: Any) -> ToolResult:
        if status is ToolStatus.OK:
            return ToolResult.success({"status": "sent"}, source="send_email:stub",
                                      latency_ms=12.0)
        return ToolResult(ok=False, data=None, status=status, source="send_email:stub",
                          degraded=True, reason=reason or f"status={status.value}")

    return StubTool("send_email", run)


def make_render_tool(*, ok: bool = True) -> StubTool:
    def run(**kwargs: Any) -> ToolResult:
        if not ok:
            return ToolResult.failed("no renderer backend installed",
                                     source="render_pdf:stub")
        return ToolResult.success(f"out/{kwargs.get('out_name', 'mou.pdf')}",
                                  source="render_pdf:stub")

    return StubTool("render_pdf", run)


def make_text_tool(*, text: str | None = None,
                   source: str = "generate_text:gemini-flash",
                   ok: bool = True) -> StubTool:
    def run(**kwargs: Any) -> ToolResult:
        if not ok:
            return ToolResult.unavailable("no text backend", source="generate_text")
        return ToolResult.success(text or "A short, concrete pitch.", source=source)

    return StubTool("generate_text", run)


# ========================================================================== decide
def make_decide(*, script: Mapping[str, Any] | None = None,
                default_score: str = "good", default_choice_index: int = 0,
                confidence: float = 0.8,
                source: DecisionSource = DecisionSource.CLEF,
                degraded: bool = False) -> tuple[Callable[[DecisionRequest], Decision],
                                                list[DecisionRequest]]:
    """A stub decision backend. Records every request it is asked."""
    seen: list[DecisionRequest] = []
    table = dict(script or {})

    def decide(request: DecisionRequest) -> Decision:
        seen.append(request)
        override = table.get(request.decision_point)
        if request.question_type is QuestionType.SCORE:
            options = list(request.rubric) or list(FIT_RUBRIC)
            choice = str(override) if override is not None else default_score
        elif request.question_type is QuestionType.NOUL:
            options = list(request.options) or ["yes", "no"]
            choice = str(override) if override is not None else options[0]
        else:
            options = list(request.options)
            if len(options) < 2:  # pragma: no cover - DecisionRequest forbids this
                raise AssertionError("choice question needs two options")
            choice = (str(override) if override is not None
                      else options[default_choice_index])
        # A scripted off-option answer is returned verbatim rather than repaired:
        # a misbehaving backend is exactly what the agents must survive.
        weights = {choice: confidence}
        others = [o for o in options if o != choice]
        if others:
            share = max(0.0, 1.0 - confidence) / len(others)
            for other in others:
                weights[other] = share
        else:
            weights[choice] = 1.0
        return Decision(
            request_id=request.request_id,
            question=request.question,
            choice=choice,
            probabilities=weights,
            confidence=weights[choice],
            source=source,
            model=f"stub:{source.value}",
            latency_ms=1.0,
            degraded=degraded,
        )

    return decide, seen


def make_intent_oracle() -> tuple[Callable[[DecisionRequest], Decision],
                                  list[DecisionRequest]]:
    """A stub backend that classifies with ``core.protocols.classify_intent_fallback``.

    Used to prove A3's decision path *agrees* with the frozen correctly-ordered
    classifier, rather than agreeing with a stub written to match A3.
    """
    seen: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> Decision:
        seen.append(request)
        if request.decision_point == "a3.classify_intent":
            intent = classify_intent_fallback(str(request.state.get("reply_text", "")))
            weights = {intent.value: 0.85}
            for other in INTENT_OPTIONS:
                if other != intent.value:
                    weights[other] = 0.15 / (len(INTENT_OPTIONS) - 1)
            return Decision(request_id=request.request_id, question=request.question,
                            choice=intent.value, probabilities=weights,
                            confidence=weights[intent.value],
                            source=DecisionSource.RULES, model="stub:rules")
        weights = {request.options[0]: 0.7}
        for other in request.options[1:]:
            weights[other] = 0.3 / (len(request.options) - 1)
        return Decision(request_id=request.request_id, question=request.question,
                        choice=request.options[0], probabilities=weights,
                        confidence=weights[request.options[0]],
                        source=DecisionSource.RULES, model="stub:rules")

    return decide, seen


# ======================================================================= tracer
class StubTracer:
    """Records span names and handoffs; yields a null context manager."""

    def __init__(self) -> None:
        self.spans: list[str] = []
        self.handoffs: list[Any] = []
        self.events: list[Any] = []

    def configure(self, run_id: str, *, service: str = "paytriq") -> None:
        return None

    @contextmanager
    def _span(self, name: str) -> Iterator[None]:
        self.spans.append(name)
        yield

    def agent(self, agent: AgentId, name: str, **attrs: Any) -> Any:
        return self._span(name)

    def llm(self, agent: AgentId, model: str, *, provider: str = "gemini",
            **attrs: Any) -> Any:
        return self._span(f"llm:{model}")

    def tool(self, agent: AgentId, tool_name: str, **attrs: Any) -> Any:
        return self._span(f"tool:{tool_name}")

    def decision(self, agent: AgentId, decision: Decision, **attrs: Any) -> Any:
        self.spans.append(f"decision:{decision.choice}")
        return self._span("decision")

    def handoff(self, handoff: Any) -> None:
        self.handoffs.append(handoff)

    def event(self, kind: TraceKind, name: str, *, agent: AgentId | None = None,
              **attrs: Any) -> Any:
        self.events.append((kind, name))
        return None

    def finish(self) -> dict[str, Any]:
        return {"spans": len(self.spans)}

    def replay_log(self) -> list[dict[str, Any]]:
        return []


# ===================================================================== fixtures
def make_context(board: StubBoard, tools: Mapping[str, Any],
                 decide: Callable[[DecisionRequest], Decision], *,
                 event_id: str = "evt_test", tracer: StubTracer | None = None
                 ) -> tuple[AgentContext, StubTracer]:
    ctx_tracer = tracer or StubTracer()
    ctx = AgentContext(
        run_id=run_id(), event_id=event_id, agent=AgentId.A1_DISCOVERY, board=board,
        decide=decide, tracer=ctx_tracer, tools=dict(tools), mode=RunMode.OFFLINE,
        step_budget=4, deadline_s=30.0, scratch={},
    )
    return ctx, ctx_tracer


def seed_event(board: StubBoard, *, footfall: int = 5000,
               name: str = "STUB Tech Fest",
               categories: list[str] | None = None) -> EventProfile:
    event = EventProfile(
        event_id="evt_test", name=name, location="Shivajinagar, Pune",
        footfall=footfall, date="2026-11-20", audience="engineering students",
        budget_inr=None, categories_wanted=categories or ["cafe"],
        deliverables_offered=[], contact_email="fest@real-event.example.org",
    )
    board.post("event", "event_profile", AgentId.A1_DISCOVERY,
               event.model_dump(mode="json"))
    return event


def seed_leads(board: StubBoard, names: Sequence[str],
               *, with_email: bool = True) -> list[BrandLead]:
    leads: list[BrandLead] = []
    for index, name in enumerate(names):
        lead = BrandLead(
            lead_id=new_id("brd"), name=name,
            category="cafe" if index % 2 == 0 else "printing",
            distance_km=1.0 + index,
            contact_email=(f"partnerships@{name.lower()}.example.org"
                           if with_email else None),
            rating=4.2, fit_score=70.0 + index,
            fit_breakdown={"category_match": 30.0, "audience_overlap": 20.0},
            fit_rationale="seeded by the test, not by A1",
            source="stub_seed",
        )
        board.post("opportunities", "brand_lead", AgentId.A1_DISCOVERY,
                   lead.model_dump(mode="json"))
        leads.append(lead)
    return leads


def seed_offer(board: StubBoard, brand: str, *, amount: float = 40_000.0,
               tier: str = "Co-Sponsor", version: int = 1,
               revised: bool = False, revision_note: str = "",
               pitch: str = "") -> Offer:
    offer = Offer(
        offer_id=new_id("off"), event_id="evt_test", brand=brand, tier=tier,
        amount_inr=amount,
        deliverables=["Side-stage banner logo", "Sponsor stall (6ft x 6ft)"],
        pitch=pitch or f"{brand} x STUB Tech Fest: {tier} at INR {amount:,.0f}.",
        fit_score=72.0, version=version, revised=revised,
        revision_note=revision_note,
    )
    board.post("offers", "offer", AgentId.A2_PRICING,
               offer.model_dump(mode="json"))
    return offer


def seed_thread(board: StubBoard, brand: str, *, intent: Intent = Intent.UNKNOWN,
                status: str = "replied", reply_text: str = "",
                offer_id: str | None = None,
                email: str | None = None,
                thread_id: str | None = None) -> Thread:
    thread = Thread(
        thread_id=thread_id or new_id("thr"), event_id="evt_test", brand=brand,
        email=email or f"partnerships@{brand.lower()}.example.org", status=status,
        day=0, intent=intent, reply_text=reply_text, offer_id=offer_id,
        delivered=bool(reply_text),
    )
    board.post("threads", "thread", AgentId.A3_OUTREACH,
               thread.model_dump(mode="json"))
    return thread


def approve(board: StubBoard, gate_id: str, kind: GateKind, *,
            outcome: GateOutcome = GateOutcome.APPROVE,
            by: str = "test-human") -> HumanDecision:
    """Record the human's answer, the way the graph's gate provider would."""
    decision = HumanDecision(gate_id=gate_id, kind=kind, outcome=outcome,
                             decided_by=by, instruction="seeded by the test")
    board.post("approvals", "human_decision", AgentId.A3_OUTREACH,
               decision.model_dump(mode="json"))
    return decision


def auto_approver(board: StubBoard, kind: GateKind,
                  outcome: GateOutcome = GateOutcome.APPROVE) -> StubBoard:
    """Wrap ``board.post`` so every ``human_gate`` post is answered immediately.

    This is the seam a real deployment would wire to the API's interrupt
    endpoint: the agent raises a gate, something answers it, the agent resumes.

    The shipped zone registry has no ``human_gate`` kind, so the gate post itself
    is refused here exactly as it is in production; only the *answer* is injected.
    """
    original = board.post

    def post(zone: str, gate_kind: str, author: AgentId, payload: dict[str, Any],
             *, refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> BoardEntry:
        if zone == "approvals" and gate_kind == "human_gate":
            try:
                original(zone, gate_kind, author, payload, refs=refs,
                         confidence=confidence, source=source)
            except BlackboardError:
                pass  # the registry refuses this kind; the human still answers.
            answer = HumanDecision(gate_id=payload.get("gate_id", ""), kind=kind,
                                   outcome=outcome, decided_by="test-human",
                                   instruction="auto-answered by the test harness")
            return original("approvals", "human_decision", author,
                            answer.model_dump(mode="json"))
        return original(zone, gate_kind, author, payload, refs=refs,
                         confidence=confidence, source=source)

    board.post = post  # type: ignore[method-assign]
    return board


def threads_of(board: StubBoard) -> list[Thread]:
    return [Thread.model_validate(e.payload) for e in board.read("threads", kind="thread")]


def latest_thread(board: StubBoard, brand: str) -> Thread | None:
    rows = [t for t in threads_of(board) if t.brand == brand]
    return rows[-1] if rows else None


def handoffs_of(board: StubBoard) -> list[Any]:
    return [e.payload for e in board.read("handoffs", kind="handoff")]


# ============================================================================ A1
def test_a1_fit_breakdown_sums_exactly_to_fit_score() -> None:
    board = StubBoard()
    seed_event(board)
    decide, seen = make_decide()
    ctx, _ = make_context(board, {"search_brands": make_search_tool()}, decide)

    result = DiscoveryAgent().run(ctx)

    leads = board.read("opportunities", kind="brand_lead")
    assert leads, "A1 posted no leads at all"
    assert result.observation.sufficient is True
    assert result.degraded is False
    for entry in leads:
        lead = BrandLead.model_validate(entry.payload)
        assert lead.fit_breakdown, "a lead with no breakdown is not auditable"
        assert round(sum(lead.fit_breakdown.values()), 1) == pytest.approx(
            lead.fit_score, abs=1e-9), (
            f"breakdown {lead.fit_breakdown} does not sum to {lead.fit_score}")
        assert set(lead.fit_breakdown) == set(COMPONENT_WEIGHTS)
        assert "breakdown sum=" not in lead.fit_rationale, (
            "BrandLead flagged a breakdown/score mismatch in the rationale")
        assert lead.fit_score <= 100.0 and lead.fit_score >= 0.0
    # Four component decisions plus one pursuit decision per lead.
    points = [r.decision_point for r in seen if r.decision_point.startswith("a1.fit.")]
    assert len(points) == len(leads) * len(COMPONENT_WEIGHTS)
    assert all(p.split(".")[-1] in COMPONENT_WEIGHTS for p in points)


def test_a1_rationale_cites_the_components_that_actually_decided() -> None:
    board = StubBoard()
    seed_event(board, categories=["gym"])
    # Only the category mismatches; the lead still clears the bar overall, so it
    # is posted and its rationale has to admit the mismatch.
    decide, _ = make_decide(script={
        "a1.fit.category_match": "poor",
        "a1.fit.audience_overlap": "excellent",
        "a1.fit.distance": "excellent",
        "a1.fit.budget_capacity": "excellent",
    })
    ctx, _ = make_context(board, {"search_brands": make_search_tool()}, decide)

    DiscoveryAgent().run(ctx)

    entries = board.read("opportunities", kind="brand_lead")
    assert entries, "the lead should clear the bar despite the category mismatch"
    lead = BrandLead.model_validate(entries[0].payload)
    assert lead.fit_breakdown["category_match"] == pytest.approx(3.5)
    for component in COMPONENT_WEIGHTS:
        assert f"{component}=" in lead.fit_rationale
    assert "category_match=poor" in lead.fit_rationale
    assert "is not among the requested" in lead.fit_rationale
    assert "gym" not in lead.fit_breakdown
    assert "clef" in lead.fit_rationale or "rules" in lead.fit_rationale
    assert "decisions from" in lead.fit_rationale


def test_a1_degrades_honestly_without_a_search_tool() -> None:
    board = StubBoard()
    seed_event(board)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {}, decide)

    result = DiscoveryAgent().run(ctx)

    assert board.read("opportunities") == [], "A1 invented a business without a tool"
    assert result.degraded is True
    assert result.observation.sufficient is False
    joined = " ".join(result.observation.gaps)
    assert "search_brands" in joined
    assert "will not invent businesses" in joined


def test_a1_degrades_when_the_search_tool_is_unavailable() -> None:
    board = StubBoard()
    seed_event(board)
    tool = make_search_tool(available=(False, "no network egress in this environment"))
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"search_brands": tool}, decide)

    result = DiscoveryAgent().run(ctx)

    assert board.read("opportunities") == []
    assert result.degraded is True
    assert any("no network egress" in gap for gap in result.observation.gaps)


def test_a1_reports_a_gap_and_widens_the_radius_when_too_few_leads_clear() -> None:
    board = StubBoard()
    seed_event(board)
    decide, seen = make_decide(script={
        "a1.search_radius": "5",
        "a1.recovery_move": "widen_radius",
    })
    search = make_search_tool(brand_rows(1))
    ctx, _ = make_context(board, {"search_brands": search}, decide)

    result = DiscoveryAgent(step_budget=2).run(ctx)

    assert result.observation.sufficient is False
    assert result.observation.gaps, "a thin lead set must be reported as a gap"
    assert any("radius" in gap for gap in result.observation.gaps)
    assert "a1.recovery_move" in [r.decision_point for r in seen]
    assert result.observation.facts["attempts"] == 2
    assert result.observation.facts["radius_km"] == 15.0, (
        "the second pass must actually widen the radius")
    # One lead posted, and it is the one the tool really returned.
    assert [BrandLead.model_validate(e.payload).name
            for e in board.read("opportunities", kind="brand_lead")] == ["STUB_Brand_0"]


def test_a1_halts_instead_of_looping_when_the_tool_cannot_work() -> None:
    board = StubBoard()
    seed_event(board)
    decide, seen = make_decide()
    ctx, _ = make_context(board, {"search_brands": make_search_tool(result=ToolResult.unavailable("overpass down", source="search_brands:live"))}, decide)

    result = DiscoveryAgent(step_budget=5).run(ctx)

    assert result.observation.sufficient is False
    assert result.notes == [] or all("error" not in n for n in result.notes)
    assert "a1.recovery_move" not in [r.decision_point for r in seen], (
        "A1 asked how to recover from a missing tool; it should have halted")
    assert len([r for r in seen if r.decision_point == "a1.search_radius"]) == 1


def test_a1_never_posts_a_lead_it_could_not_score_from_a_decision() -> None:
    """A decision backend that returns an off-rubric label must not yield a lead."""
    board = StubBoard()
    seed_event(board)
    decide, _ = make_decide(script={"a1.fit.category_match": "out of this world"})
    ctx, _ = make_context(board, {"search_brands": make_search_tool()}, decide)

    result = DiscoveryAgent().run(ctx)

    assert board.read("opportunities") == []
    assert result.observation.sufficient is False
    assert any("not one of" in gap for gap in result.observation.gaps)


def test_a1_zone_table_matches_the_shipped_registry() -> None:
    """The local strict board must not be stricter than the real one.

    Two things matter and are checked separately: this table must not invent a
    kind the real registry does not have (or A1-A4 would post something a real
    board rejects), and the single-writer assignments for the zones these agents
    write to must match exactly (or this suite would pass while the real run
    failed with an authorisation error).
    """
    zones = pytest.importorskip("blackboard.zones", reason="blackboard not built yet")
    mine_only = ("event", "opportunities", "offers", "threads", "contracts",
                 "approvals", "handoffs", "lessons")
    for name in mine_only:
        registered = zones.ZONES.get(name)
        if registered is None:
            continue
        kinds, owner = ZONE_TABLE[name]
        assert kinds <= set(registered.allowed_kinds), (
            f"zone {name!r} does not allow {sorted(kinds - set(registered.allowed_kinds))}"
            f"; the test board would accept something the real board rejects")
        actual = (registered.authoritative_agent.value
                  if registered.authoritative_agent else None)
        assert actual == owner, f"zone {name!r} authoritative writer drifted"


def test_every_kind_these_agents_post_is_accepted_by_the_real_board() -> None:
    """The integration guarantee: no agent of mine can be rejected by a zone."""
    board_module = pytest.importorskip(
        "blackboard.board", reason="blackboard not built yet")
    board = board_module.InMemoryBlackboard()
    payload = {"stub": True}
    for zone, kind, author in AGENT_POSTS:
        try:
            board.post(zone, kind, author, dict(payload))
        except BlackboardError as exc:  # pragma: no cover - a real regression
            pytest.fail(f"the real board rejected {author.value} posting "
                        f"{kind!r} to {zone!r}: {exc}")


# ============================================================================ A2
def test_a2_price_scales_with_footfall_instead_of_a_flat_ladder() -> None:
    board = StubBoard()
    seed_event(board, footfall=500)
    seed_leads(board, ["STUB_Small"])
    decide, _ = make_decide(script={"a2.choose_tier": "Title Sponsor"})
    ctx, _ = make_context(board, {}, decide)

    small = PricingAgent().run(ctx)

    board2 = StubBoard()
    seed_event(board2, footfall=50_000)
    seed_leads(board2, ["STUB_Big"])
    ctx2, _ = make_context(board2, {}, decide)
    PricingAgent().run(ctx2)

    small_offer = Offer.model_validate(board.read("offers", kind="offer")[0].payload)
    big_offer = Offer.model_validate(board2.read("offers", kind="offer")[0].payload)

    assert small_offer.amount_inr != big_offer.amount_inr
    assert big_offer.amount_inr > small_offer.amount_inr * 2
    # Nothing may be quoted below the tier's delivery floor.
    floor = 12_000.0
    assert small_offer.amount_inr >= floor and big_offer.amount_inr >= floor
    assert small.observation.facts["assumptions"] == [RATE_PROVENANCE]
    assert "assumption" in RATE_PROVENANCE.lower()
    assert f"{RATE_PER_ATTENDEE_INR:.2f}" in small_offer.pitch


def test_a2_tier_comes_from_a_decision_over_the_tier_options() -> None:
    board = StubBoard()
    seed_event(board)
    seed_leads(board, ["STUB_One"])
    decide, seen = make_decide(script={"a2.choose_tier": "Stall Only"})
    ctx, _ = make_context(board, {}, decide)

    PricingAgent().run(ctx)

    tier_requests = [r for r in seen if r.decision_point == "a2.choose_tier"]
    assert tier_requests, "the tier was not chosen by a decision"
    assert tier_requests[0].options == list(TIER_OPTIONS)
    assert tier_requests[0].state["footfall"] == 5000
    assert tier_requests[0].state["tier_prices_inr"]
    offer = Offer.model_validate(board.read("offers", kind="offer")[0].payload)
    assert offer.tier == "Stall Only"
    assert offer.deliverables


def test_a2_composes_the_pitch_from_offer_fields_when_no_text_model_exists() -> None:
    board = StubBoard()
    seed_event(board)
    seed_leads(board, ["STUB_One"])
    decide, _ = make_decide(script={"a2.choose_tier": "Co-Sponsor"})
    ctx, _ = make_context(board, {}, decide)

    result = PricingAgent().run(ctx)

    entry = board.read("offers", kind="offer")[0]
    offer = Offer.model_validate(entry.payload)
    assert entry.source is DecisionSource.RULES
    assert offer.pitch and str(int(offer.amount_inr)) in offer.pitch.replace(",", "")
    assert offer.brand in offer.pitch
    for item in offer.deliverables:
        assert item in offer.pitch
    assert any("no text-generation tool" in note for note in result.notes), (
        "the absence of a text model must be stated, not silently absorbed")
    assert result.observation.sufficient is True
    assert result.observation.facts["assumptions"] == [RATE_PROVENANCE]


def test_a2_marks_the_pitch_as_generated_only_when_the_tool_names_a_model() -> None:
    board = StubBoard()
    seed_event(board)
    seed_leads(board, ["STUB_One"])
    decide, _ = make_decide(script={"a2.choose_tier": "Co-Sponsor"})
    tool = make_text_tool(text="A short pitch from a model.",
                          source="generate_text:gemini-3-flash")
    ctx, _ = make_context(board, {"generate_text": tool}, decide)

    PricingAgent().run(ctx)

    entry = board.read("offers", kind="offer")[0]
    assert entry.source is DecisionSource.GEMINI
    assert Offer.model_validate(entry.payload).pitch.startswith("A short pitch")
    assert tool.calls, "the text tool was never called"


def test_a2_degrades_when_the_text_tool_is_present_but_unusable() -> None:
    board = StubBoard()
    seed_event(board)
    seed_leads(board, ["STUB_One"])
    decide, _ = make_decide(script={"a2.choose_tier": "Co-Sponsor"})
    ctx, _ = make_context(board, {"generate_text": make_text_tool(ok=False)}, decide)

    result = PricingAgent().run(ctx)

    entry = board.read("offers", kind="offer")[0]
    offer = Offer.model_validate(entry.payload)
    assert entry.source is DecisionSource.RULES, (
        "text from an unusable backend must not be credited to a model")
    assert offer.pitch, "the offer must still carry a usable pitch"
    assert result.degraded is True


def test_a2_revision_uses_the_decided_concession_not_a_flat_halving() -> None:
    board = StubBoard()
    seed_event(board, footfall=20_000)
    lead_name = "STUB_Negotiator"
    seed_leads(board, [lead_name])
    base = seed_offer(board, lead_name, amount=80_000.0, tier="Title Sponsor")
    seed_thread(board, lead_name, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="The fee is too high for us this quarter.",
                offer_id=base.offer_id)
    decide, seen = make_decide(script={
        "a2.negotiation_posture": "reduce_price",
        "a2.concession_size": "reduce_15_percent",
    })
    ctx, _ = make_context(board, {}, decide)

    result = PricingAgent().run(ctx)

    offers = [Offer.model_validate(e.payload) for e in board.read("offers", kind="offer")]
    revised = offers[-1]
    assert len(offers) == 2
    assert revised.version == base.version + 1
    assert revised.revised is True
    assert revised.amount_inr == pytest.approx(base.amount_inr * 0.85, rel=0.01)
    assert revised.amount_inr != base.amount_inr / 2
    assert "applied -15" in revised.revision_note
    assert f"{base.amount_inr:,.0f}" in revised.revision_note
    assert f"{revised.amount_inr:,.0f}" in revised.revision_note
    assert "too high" in revised.revision_note
    assert revised.references_offer if hasattr(revised, "references_offer") else True
    entry = board.read("offers", kind="offer")[-1]
    assert entry.refs, "the revision must cite the offer it supersedes"
    assert result.observation.sufficient is True
    assert result.observation.facts["actual_change_pct"] == pytest.approx(15.0, abs=0.6)
    assert any(r.decision_point == "a2.concession_size" for r in seen)


def test_a2_concession_band_is_capped_and_never_breaches_the_floor() -> None:
    board = StubBoard()
    seed_event(board, footfall=1_000)
    lead_name = "STUB_Tight"
    seed_leads(board, [lead_name])
    # Sitting barely above the tier floor: a 25% cut must be clamped back to it.
    base = seed_offer(board, lead_name, amount=2_100.0, tier="Stall Only")
    seed_thread(board, lead_name, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="Can you do it cheaper?", offer_id=base.offer_id)
    decide, _ = make_decide(script={
        "a2.negotiation_posture": "reduce_price_and_add_deliverable",
        "a2.concession_size": "reduce_25_percent",
    })
    ctx, _ = make_context(board, {}, decide)

    result = PricingAgent().run(ctx)

    revised = Offer.model_validate(board.read("offers", kind="offer")[-1].payload)
    assert revised.amount_inr >= 2_000.0, "a revision went below the delivery floor"
    assert revised.amount_inr == 2_000.0
    assert "delivery floor" in revised.revision_note
    assert "requested -25%" in revised.revision_note
    assert "applied -4.8%" in revised.revision_note
    assert len(revised.deliverables) == len(base.deliverables) + 1
    assert revised.revised is True
    assert result.observation.facts["actual_change_pct"] <= 30.0
    assert "reduce_price" in POSTURE_OPTIONS
    assert set(CONCESSION_OPTIONS) >= {"reduce_25_percent"}


def test_a2_will_not_concede_below_a_quote_already_at_the_floor() -> None:
    board = StubBoard()
    seed_event(board, footfall=1_000)
    lead_name = "STUB_Brokeven"
    seed_leads(board, [lead_name])
    base = seed_offer(board, lead_name, amount=2_000.0, tier="Stall Only")
    seed_thread(board, lead_name, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="Reduce it please.", offer_id=base.offer_id)
    decide, seen = make_decide(script={
        "a2.negotiation_posture": "reduce_price",
        "a2.concession_size": "reduce_25_percent",
    })
    ctx, _ = make_context(board, {}, decide)

    PricingAgent().run(ctx)

    revised = Offer.model_validate(board.read("offers", kind="offer")[-1].payload)
    assert revised.amount_inr == base.amount_inr
    assert "hold_price" in revised.revision_note


def test_a2_holding_price_still_records_a_revision() -> None:
    board = StubBoard()
    seed_event(board)
    lead_name = "STUB_Firm"
    seed_leads(board, [lead_name])
    base = seed_offer(board, lead_name, amount=50_000.0)
    seed_thread(board, lead_name, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="Too expensive, and our brand guidelines forbid stalls.",
                offer_id=base.offer_id)
    decide, _ = make_decide(script={"a2.negotiation_posture": "hold_price",
                                    "a2.concession_size": "hold_price"})
    ctx, _ = make_context(board, {}, decide)

    PricingAgent().run(ctx)

    revised = Offer.model_validate(board.read("offers", kind="offer")[-1].payload)
    assert revised.amount_inr == base.amount_inr
    assert revised.version == base.version + 1
    assert revised.revised is True
    assert "posture=hold_price" in revised.revision_note


def test_a2_walk_away_produces_no_offer_and_hands_control_back() -> None:
    board = StubBoard()
    seed_event(board)
    lead_name = "STUB_Leave"
    seed_leads(board, [lead_name])
    base = seed_offer(board, lead_name, amount=60_000.0)
    seed_thread(board, lead_name, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="We only want a stall, nothing else.", offer_id=base.offer_id)
    decide, _ = make_decide(script={"a2.negotiation_posture": "walk_away"})
    ctx, _ = make_context(board, {}, decide)

    result = PricingAgent().run(ctx)

    assert len(board.read("offers", kind="offer")) == 1, "walk-away produced a new offer"
    assert result.observation.facts["outcome"] == "walk_away"
    targets = [h["to_agent"] for h in handoffs_of(board)]
    assert targets == [AgentId.A3_OUTREACH.value]


def test_a2_does_not_revise_the_same_pushback_twice() -> None:
    board = StubBoard()
    seed_event(board, footfall=10_000)
    lead_name = "STUB_Once"
    seed_leads(board, [lead_name])
    base = seed_offer(board, lead_name, amount=45_000.0)
    seed_thread(board, lead_name, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="Reduce the price please.", offer_id=base.offer_id)
    decide, _ = make_decide(script={"a2.negotiation_posture": "reduce_price",
                                    "a2.concession_size": "reduce_10_percent"})
    ctx, _ = make_context(board, {}, decide)

    PricingAgent(step_budget=4).run(ctx)

    assert len(board.read("offers", kind="offer")) == 2, (
        "the answered pushback was revised again")


# ============================================================================ A3
REGRESSION_REPLY = "yesterday we thought the price was too high"


def test_regression_price_objection_is_pushback_never_yes() -> None:
    """The sentence that the old router routed into contract signing."""
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Objector"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.UNKNOWN, status="replied",
                reply_text=REGRESSION_REPLY, offer_id=offer.offer_id)
    decide, seen = make_intent_oracle()
    ctx, _ = make_context(board, {}, decide)

    OutreachAgent().run(ctx)

    thread = latest_thread(board, brand)
    assert thread is not None
    assert thread.intent is Intent.PUSHBACK, (
        f"price objection classified as {thread.intent}; the old router tested "
        f"yes first and sent this to contract signing")
    assert thread.status == "negotiating"
    assert thread.reply_text == REGRESSION_REPLY
    # A3's decision path must agree with the frozen correctly-ordered classifier.
    assert classify_intent_fallback(REGRESSION_REPLY) is Intent.PUSHBACK
    request = next(r for r in seen if r.decision_point == "a3.classify_intent")
    assert request.state["reply_text"] == REGRESSION_REPLY
    assert set(request.options) == set(INTENT_OPTIONS)
    assert Intent.YES.value in request.options, (
        "the regression depends on yes being an available option")
    routed = [h["to_agent"] for h in handoffs_of(board)]
    assert routed == [AgentId.A2_PRICING.value], (
        "a price objection must reach A2, never A4")


def test_pushback_carries_the_exact_offer_the_sponsor_objected_to() -> None:
    """A second, unanswered objection must produce a second revision."""
    board = StubBoard()
    seed_event(board)
    brand = "STUB_TwoRounds"
    seed_offer(board, brand, amount=70_000.0, tier="Title Sponsor")
    seed_thread(board, brand, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="Too high.", thread_id="thr_round1")
    # A2 answered the first objection; A3 sent v2 and A3 classified the next
    # reply as pushback again, so a second revision is genuinely outstanding.
    second = seed_offer(board, brand, amount=60_000.0, tier="Title Sponsor",
                        version=2, revised=True, revision_note="v2: -14%")
    seed_thread(board, brand, intent=Intent.PUSHBACK, status="negotiating",
                reply_text="Still too high, can we meet halfway?",
                offer_id=second.offer_id, thread_id="thr_round2")
    decide, _ = make_decide(script={"a2.negotiation_posture": "reduce_price",
                                    "a2.concession_size": "reduce_10_percent"})
    ctx, _ = make_context(board, {}, decide)

    result = PricingAgent(step_budget=2).run(ctx)

    offers = [Offer.model_validate(e.payload)
              for e in board.read("offers", kind="offer")]
    assert len(offers) == 3, "the second, unanswered objection was not revised"
    newest = offers[-1]
    assert newest.version == second.version + 1
    assert newest.amount_inr == pytest.approx(second.amount_inr * 0.90, rel=0.01)
    assert "Still too high" in newest.revision_note
    assert result.observation.facts["version"] == 3


def test_send_gate_refuses_to_send_without_an_approval_record() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Gated"
    seed_offer(board, brand)
    seed_leads(board, [brand])
    seed_thread(board, brand, intent=Intent.UNKNOWN, status="draft")
    send = make_send_tool()
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    with pytest.raises(HumanGateRequired) as excinfo:
        OutreachAgent().run(ctx)

    assert "gate" in str(excinfo.value)
    assert send.calls == [], "a message was handed to the transport without approval"
    assert not any(t.status == "sent" for t in threads_of(board))
    assert not any(t.delivered for t in threads_of(board))
    parked = latest_thread(board, brand)
    assert parked is not None and parked.status == "pending_approval"
    assert parked.delivered is False
    assert board.read("approvals", kind="approval") == [], (
        "an approval row exists although no human decided anything")
    # The pending gate itself must be visible on the board, so a human can see
    # what they are being asked about.
    gates = board.read("approvals", kind="human_gate")
    assert len(gates) == 1
    assert gates[0].payload["kind"] == GateKind.SEND.value
    assert "STUB_Gated" in gates[0].payload["payload_preview"]
    assert gates[0].payload["question"]


def test_send_gate_still_refuses_when_the_board_cannot_record_the_gate() -> None:
    """A board variant with no ``human_gate`` kind must not weaken the gate."""
    narrow = {name: ({"approval", "human_decision"} if name == "approvals" else kinds,
                     owner)
              for name, (kinds, owner) in ZONE_TABLE.items()}
    board = StubBoard(narrow)
    seed_event(board)
    brand = "STUB_OldBoard"
    seed_offer(board, brand)
    seed_leads(board, [brand])
    send = make_send_tool()
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    with pytest.raises(HumanGateRequired):
        OutreachAgent().run(ctx)

    assert send.calls == []
    assert board.read("approvals", kind="approval") == []
    note = str(ctx.scratch.get("a3_gate_note"))
    assert "the board refused to store the gate itself" in note
    assert "does not accept kind 'human_gate'" in note


def test_send_gate_parking_mode_never_sends_either() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Parked"
    seed_offer(board, brand)
    seed_leads(board, [brand])
    send = make_send_tool()
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    agent = OutreachAgent()
    agent.propagate_gate = False
    result = agent.run(ctx)

    assert send.calls == []
    assert result.observation.facts.get("blocked") is True
    assert latest_thread(board, brand).status == "pending_approval"


def test_send_records_delivery_only_when_the_transport_says_ok() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.SEND)
    seed_event(board)
    brand = "STUB_Sent"
    offer = seed_offer(board, brand)
    seed_leads(board, [brand])
    send = make_send_tool(status=ToolStatus.OK)
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    result = OutreachAgent().run(ctx)

    thread = latest_thread(board, brand)
    assert thread.status == "sent"
    assert thread.delivered is True
    assert thread.offer_id == offer.offer_id
    assert send.calls, "an approved send did not reach the transport"
    assert send.calls[0]["to"] == thread.email
    approvals = board.read("approvals", kind="approval")
    assert approvals, "no durable Approval row was written for the send"
    assert approvals[-1].payload["kind"] == GateKind.SEND.value
    assert result.observation.sufficient is True


def test_a_failed_transport_is_never_reported_as_delivered() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.SEND)
    seed_event(board)
    brand = "STUB_Bounced"
    seed_offer(board, brand)
    seed_leads(board, [brand])
    send = make_send_tool(status=ToolStatus.UNAVAILABLE, reason="resend 429")
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    result = OutreachAgent().run(ctx)

    thread = latest_thread(board, brand)
    assert thread.status != "sent"
    assert thread.delivered is False
    assert result.degraded is True
    assert any("not delivered" in note.lower() or "429" in note
               for note in result.notes + result.observation.gaps)


def test_yes_raises_the_mou_need_and_never_touches_a2() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Accepted"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.UNKNOWN, status="replied",
                reply_text="Yes, we agree — please send the MoU.",
                offer_id=offer.offer_id)
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {}, decide)

    OutreachAgent().run(ctx)

    thread = latest_thread(board, brand)
    assert thread.intent is Intent.YES
    assert thread.status == "closed_won"
    handoffs = handoffs_of(board)
    assert [h["to_agent"] for h in handoffs] == [AgentId.A4_CONTRACT.value]
    assert handoffs[0]["decision_source"]
    assert handoffs[0]["confidence"] > 0


def test_neutral_reply_is_neutral_and_triggers_nothing() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Unclear"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.UNKNOWN, status="replied",
                reply_text="Attaching our brand guidelines, will revert next week.",
                offer_id=offer.offer_id)
    send = make_send_tool()
    decide, _ = make_decide(script={"a3.classify_intent": "neutral"})
    ctx, _ = make_context(board, {"send_email": send}, decide)

    result = OutreachAgent().run(ctx)

    thread = latest_thread(board, brand)
    assert thread.intent is Intent.NEUTRAL
    assert thread.status not in ("closed_won", "closed_lost")
    assert handoffs_of(board) == []
    assert send.calls == []
    # The classification is recorded with the source that produced it, so a
    # reader can tell a model answer from a rules answer.
    classified = board.read("threads", kind="thread")[-1]
    assert classified.source is DecisionSource.CLEF
    assert classified.payload["intent"] == Intent.NEUTRAL.value
    assert classified.payload["status"] == "negotiating"
    assert result.observation.sufficient is True


def test_refusal_closes_the_thread_without_a_handoff() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Declined"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.UNKNOWN, status="replied",
                reply_text="No thanks, we are not interested this year.",
                offer_id=offer.offer_id)
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {}, decide)

    OutreachAgent().run(ctx)

    thread = latest_thread(board, brand)
    assert thread.intent is Intent.NO
    assert thread.status == "closed_lost"
    assert handoffs_of(board) == []


def test_a3_refuses_to_invent_a_contact_address() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_NoAddress"
    seed_offer(board, brand)
    seed_leads(board, [brand], with_email=False)
    send = make_send_tool()
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    result = OutreachAgent().run(ctx)

    assert send.calls == []
    assert board.read("threads", kind="thread") == []
    assert brand in " ".join(result.observation.gaps)


def test_a3_degrades_honestly_without_a_send_transport() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.SEND)
    seed_event(board)
    brand = "STUB_NoTransport"
    seed_offer(board, brand)
    seed_leads(board, [brand])
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {}, decide)

    result = OutreachAgent().run(ctx)

    assert result.degraded is True
    assert any("send_email" in gap for gap in result.observation.gaps)
    assert not any(t.status == "sent" for t in threads_of(board))


def test_a3_does_not_resend_terms_the_sponsor_already_has() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.SEND)
    seed_event(board)
    brand = "STUB_SentOnce"
    offer = seed_offer(board, brand)
    seed_leads(board, [brand])
    send = make_send_tool()
    decide, _ = make_intent_oracle()
    ctx, _ = make_context(board, {"send_email": send}, decide)

    result = OutreachAgent(step_budget=4).run(ctx)

    assert len(send.calls) == 1, "the same offer was sent more than once"
    assert latest_thread(board, brand).offer_id == offer.offer_id
    assert result.observation.sufficient is True


# ============================================================================ A4
def test_a4_refuses_to_draft_for_an_unaccepted_offer() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_NotYet"
    seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.INTERESTED, status="negotiating",
                reply_text="Tell me more about footfall.", offer_id=None)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    result = ContractAgent().run(ctx)

    assert board.read("contracts") == [], (
        "an MoU was drafted for a sponsor who never agreed")
    assert result.observation.sufficient is False
    assert any("no thread" in gap or "Intent.YES" in gap
               for gap in result.observation.gaps)


def test_a4_refuses_when_there_are_no_offers_at_all() -> None:
    board = StubBoard()
    seed_event(board)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {}, decide)

    result = ContractAgent().run(ctx)

    assert board.read("contracts") == []
    assert result.observation.sufficient is False


def test_a4_drafts_from_the_accepted_offer_not_the_first_posted() -> None:
    board = StubBoard()
    seed_event(board)
    first_brand, second_brand = "STUB_FirstPriced", "STUB_ActuallyAccepted"
    first = seed_offer(board, first_brand, amount=30_000.0)
    second = seed_offer(board, second_brand, amount=55_000.0, tier="Title Sponsor")
    # The sponsor who said yes is the SECOND offer — the old code used
    # proposals[0] and would have contracted the first brand.
    seed_thread(board, second_brand, intent=Intent.YES, status="closed_won",
                reply_text="Yes, deal confirmed.", offer_id=second.offer_id)
    auto_approver(board, GateKind.MOU)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    result = ContractAgent().run(ctx)

    mouses = [MoU.model_validate(e.payload)
              for e in board.read("contracts", kind="mou")]
    assert mouses, "no MoU was drafted for the accepted sponsor"
    approved = [m for m in mouses if m.status == "approved"]
    assert approved, "the MoU was never approved by a human"
    assert approved[0].brand == second_brand
    assert first_brand not in approved[0].terms
    assert approved[0].amount_inr == second.amount_inr
    assert f"{second.amount_inr:,.0f}" in approved[0].terms
    assert first.offer_id not in [r for e in board.read("contracts", kind="mou")
                                  for r in e.refs]
    assert result.observation.sufficient is True


def test_a4_terms_contain_real_clauses_from_the_offer() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_Clauses"
    offer = seed_offer(board, brand, amount=42_000.0, tier="Co-Sponsor")
    seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                reply_text="Confirmed.", offer_id=offer.offer_id)
    auto_approver(board, GateKind.MOU)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    ContractAgent().run(ctx)

    mou = MoU.model_validate(board.read("contracts", kind="mou")[-1].payload)
    terms = mou.terms
    assert "PAYMENT SCHEDULE" in terms
    assert "LOGO USAGE RIGHTS" in terms
    assert "CANCELLATION AND NOTICE" in terms
    assert "14 days' written notice" in terms
    assert "DELIVERABLES" in terms
    assert "SIGNATURES" in terms
    assert "Shivajinagar, Pune" in terms and "2026-11-20" in terms
    assert offer.offer_id in terms and f"version {offer.version}" in terms
    for item in offer.deliverables:
        assert item in terms
    assert mou.deliverables == offer.deliverables
    assert mou.version == offer.version


def test_a4_mou_is_not_approved_without_a_human_decision() -> None:
    board = StubBoard()
    seed_event(board)
    brand = "STUB_NoSignoff"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                reply_text="Yes.", offer_id=offer.offer_id)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    with pytest.raises(HumanGateRequired):
        ContractAgent().run(ctx)

    statuses = [MoU.model_validate(e.payload).status
                for e in board.read("contracts", kind="mou")]
    assert statuses, "the draft itself was never posted"
    assert set(statuses) == {"pending_approval"}, (
        "an MoU reached a released state without a HumanDecision")


def test_a4_records_a_rejected_mou_when_the_human_says_no() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.MOU, outcome=GateOutcome.REJECT)
    seed_event(board)
    brand = "STUB_Rejected"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                reply_text="Yes.", offer_id=offer.offer_id)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    ContractAgent().run(ctx)

    statuses = [MoU.model_validate(e.payload).status
                for e in board.read("contracts", kind="mou")]
    assert statuses[-1] == "rejected"
    assert board.read("approvals", kind="approval") == [], (
        "a rejected gate must not be recorded as an approval")


def test_a4_degrades_honestly_when_the_render_is_unavailable() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.MOU)
    seed_event(board)
    brand = "STUB_NoPdf"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                reply_text="Yes.", offer_id=offer.offer_id)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {}, decide)

    result = ContractAgent().run(ctx)

    mou = MoU.model_validate(board.read("contracts", kind="mou")[-1].payload)
    assert mou.status == "approved", "the text artefact is still real"
    assert mou.document_path is None, (
        "a document path was claimed although nothing rendered")
    assert result.degraded is True
    assert "renderer" in " ".join(result.observation.gaps) or \
        "renderer" in " ".join(result.notes)


def test_a4_records_a_failed_render_without_claiming_a_path() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.MOU)
    seed_event(board)
    brand = "STUB_BrokenPdf"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                reply_text="Yes.", offer_id=offer.offer_id)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool(ok=False)}, decide)

    result = ContractAgent().run(ctx)

    mou = MoU.model_validate(board.read("contracts", kind="mou")[-1].payload)
    assert mou.document_path is None
    assert result.degraded is True


def test_a4_asks_a_decision_when_several_sponsors_accept() -> None:
    board = StubBoard()
    seed_event(board)
    for brand in ("STUB_AcceptA", "STUB_AcceptB"):
        offer = seed_offer(board, brand)
        seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                    reply_text="Yes, agreed.", offer_id=offer.offer_id)
    auto_approver(board, GateKind.MOU)
    decide, seen = make_decide(script={"a4.choose_accepted": "STUB_AcceptB"})
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    result = ContractAgent(step_budget=4).run(ctx)

    request = next(r for r in seen if r.decision_point == "a4.choose_accepted")
    assert sorted(request.options) == ["STUB_AcceptA", "STUB_AcceptB"]
    brands = {MoU.model_validate(e.payload).brand
              for e in board.read("contracts", kind="mou")}
    assert brands == {"STUB_AcceptB"}, "A4 drafted for more than the chosen sponsor"
    assert result.observation.sufficient is True


def test_a4_does_not_redraft_an_mou_it_already_wrote() -> None:
    board = StubBoard()
    auto_approver(board, GateKind.MOU)
    seed_event(board)
    brand = "STUB_Once"
    offer = seed_offer(board, brand)
    seed_thread(board, brand, intent=Intent.YES, status="closed_won",
                reply_text="Yes.", offer_id=offer.offer_id)
    decide, _ = make_decide()
    ctx, _ = make_context(board, {"render_pdf": make_render_tool()}, decide)

    result = ContractAgent(step_budget=4).run(ctx)

    drafts = [e for e in board.read("contracts", kind="mou")
              if MoU.model_validate(e.payload).status == "pending_approval"]
    assert len(drafts) == 1, "the MoU was drafted twice"
    assert result.observation.sufficient is True
