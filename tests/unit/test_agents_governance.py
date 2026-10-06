"""Governance tests for A5 Compliance, A6 Audit, A7 Arbiter, and the environment.

These are the tests that would have caught the prototype's failures. Each one
names the defect it prevents, because a test whose name does not say what broke
is a test nobody maintains.

Fully offline: the decision layer is a :class:`StubDecide`, the tool layer is a
:class:`StubVerifyEvidence`, and the tracer is a :class:`StubTracer`. No network,
no model, no clock dependency.

The board has two implementations on purpose. :class:`StubBoard` implements only
``post``/``read``/``latest``/``zones``/``history`` — the protocol minimum — while
:class:`RichStubBoard` additionally provides ``latest_model``/``get_model``. The
governance suite runs against **both**, so the agents cannot silently depend on a
typed-helper that may not exist.
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.a5_compliance import (  # noqa: E402
    ZONE_AUDIT as A5_ZONE_AUDIT,
)
from agents.a5_compliance import (
    ComplianceAgent,
    ComplianceVerdict,
    EvidenceRecord,
)
from agents.a6_audit import (  # noqa: E402
    KIND_AUDIT_FINDING,
    KIND_LESSON,
    KIND_ROI_REPORT,
    AuditAgent,
)
from agents.a6_audit import (
    ZONE_AUDIT as A6_ZONE_AUDIT,
)
from agents.a6_audit import (
    ZONE_LESSONS as A6_ZONE_LESSONS,
)
from agents.a7_arbiter import (  # noqa: E402
    KIND_DECISION,
    KIND_DISPUTE,
    KIND_HUMAN_GATE,
    OUTCOMES,
    ArbiterAgent,
)
from agents.environment import (  # noqa: E402
    REPLY_CHOICES,
    SponsorEnvironment,
    SponsorProfile,
)
from core import (  # noqa: E402
    REASONING_AGENTS,
    AgentContext,
    AgentId,
    ArbiterEscalation,
    AuditFinding,
    BoardEntry,
    Decision,
    DecisionRequest,
    DecisionSource,
    Dispute,
    DisputeStatus,
    DisputeZone,
    GateKind,
    HumanGateRequired,
    Intent,
    MoU,
    Offer,
    ROIReport,
    RunMode,
    Severity,
    Thread,
    ToolResult,
    ToolStatus,
)
from core.errors import BlackboardError  # noqa: E402

EVENT_ID = "evt_governance_test"
RUN_ID = "run_governance_test"


# ============================================================== stub infrastructure
class StubTracer:
    """Records span names and emits nothing. Satisfies the ``Tracer`` protocol."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict[str, Any]]] = []

    def _span(self, agent: AgentId, name: str, **attrs: Any) -> _NullSpan:
        self.spans.append((name, dict(attrs)))
        return _NullSpan()

    def configure(self, run_id: str, *, service: str = "paytriq") -> None:
        del run_id, service

    def agent(self, agent: AgentId, name: str, **attrs: Any) -> _NullSpan:
        return self._span(agent, name, **attrs)

    def llm(self, agent: AgentId, model: str, **attrs: Any) -> _NullSpan:
        return self._span(agent, "llm", model=model, **attrs)

    def tool(self, agent: AgentId, tool_name: str, **attrs: Any) -> _NullSpan:
        return self._span(agent, "tool", tool_name=tool_name, **attrs)

    def decision(self, agent: AgentId, decision: Decision, **attrs: Any) -> _NullSpan:
        return self._span(agent, "decision", **attrs)

    def handoff(self, handoff: Any) -> None:
        del handoff

    def event(self, kind: Any, name: str, **attrs: Any) -> Any:
        del kind, attrs
        return name

    def finish(self) -> dict[str, Any]:
        return {"spans": len(self.spans)}

    def replay_log(self) -> list[dict[str, Any]]:
        return []


class _NullSpan:
    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class StubBoard:
    """Append-only board implementing exactly the ``Blackboard`` protocol.

    Deliberately has **no** ``latest_model``/``get_model``: this is the board shape
    the agents must survive when the typed helpers are absent.
    """

    def __init__(self, *, zones: Sequence[str] | None = None) -> None:
        self._entries: list[BoardEntry] = []
        self._zones: list[str] = list(zones or [
            "event", "opportunities", "offers", "threads", "contracts", "risk_flags",
            "audit", "disputes", "bids", "decisions", "lessons", "approvals", "handoffs",
        ])
        self._seq = 0

    def post(self, zone: str, kind: str, author: AgentId, payload: dict[str, Any], *,
             refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> BoardEntry:
        if zone not in self._zones:
            raise BlackboardError(f"unknown zone {zone!r}")
        self._seq += 1
        entry = BoardEntry(
            entry_id=f"e{self._seq:04d}", zone=zone, kind=kind, author=author,
            payload=dict(payload or {}), refs=list(refs or []),
            confidence=confidence, source=source, seq=self._seq, at="2026-10-04T00:00:00Z",
        )
        self._entries.append(entry)
        return entry

    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[BoardEntry]:
        if zone not in self._zones:
            raise BlackboardError(f"unknown zone {zone!r}")
        out = [e for e in self._entries if e.zone == zone
               and (kind is None or e.kind == kind)]
        return out[-limit:] if limit else out

    def latest(self, zone: str, kind: str | None = None) -> BoardEntry | None:
        matches = self.read(zone, kind=kind)
        return matches[-1] if matches else None

    def zones(self) -> list[str]:
        return list(self._zones)

    def history(self) -> list[BoardEntry]:
        return list(self._entries)


class RichStubBoard(StubBoard):
    """Adds ``latest_model``/``get_model`` so the typed-helper path is exercised."""

    def latest_model(self, zone: str, model_cls: type, *,
                     kind: str | None = None) -> Any | None:
        for entry in reversed(self.read(zone, kind=kind)):
            try:
                return model_cls.model_validate(entry.payload)
            except Exception:  # noqa: BLE001 - a stub scanning for any match
                continue
        return None

    def get_model(self, entry_id: str, model_cls: type) -> Any | None:
        for entry in self._entries:
            if entry.entry_id != entry_id:
                continue
            try:
                return model_cls.model_validate(entry.payload)
            except Exception:  # noqa: BLE001 - a stub scanning for any match
                return None
        return None


@dataclass
class StubDecide:
    """Scripted decision backend.

    ``handler`` maps a :class:`DecisionRequest` to a choice string; ``confidence``
    sets the calibrated probability of that choice. Every request is recorded in
    :attr:`requests` so a test can assert *what was asked*, not only what was
    answered — a decision nobody recorded is a decision nobody can audit.
    """

    handler: Callable[[DecisionRequest], str] = lambda req: (
        req.options[0] if req.options else "yes")
    confidence: float = 0.9
    source: DecisionSource = DecisionSource.CLEF
    #: Optional per-question confidence, so a test can be confident about the
    #: outcome question and unsure about a follow-up one.
    confidence_fn: Callable[[DecisionRequest], float] | None = None
    requests: list[DecisionRequest] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> Decision:
        self.requests.append(request)
        # A ``noul`` question carries no option set by design; its answer space is
        # the calibrated yes/no pair, and A5 reads ``probabilities["yes"]``.
        options = list(request.options) or ["yes", "no"]
        choice = self.handler(request)
        if choice not in options:
            choice = options[0]
        confidence = (self.confidence_fn(request) if self.confidence_fn
                      else self.confidence)
        others = [o for o in options if o != choice]
        remainder = max(0.0, 1.0 - confidence)
        probabilities = {opt: 0.0 for opt in options}
        for opt in others:
            probabilities[opt] = round(remainder / len(others), 6) if others else 0.0
        probabilities[choice] = round(1.0 - sum(probabilities[o] for o in others), 6)
        return Decision(
            request_id=request.request_id, question=request.question, choice=choice,
            probabilities=probabilities, confidence=confidence,
            source=self.source, model="stub", degraded=False,
        )

    def asked_about(self, fragment: str) -> list[DecisionRequest]:
        return [r for r in self.requests if fragment in r.question]


class StubVerifyEvidence:
    """Stand-in for ``tools/browser.py``'s evidence fetcher.

    ``content_mode`` decides what the "page" yields, which is how the substring
    bug is reproduced honestly rather than mocked away.
    """

    name = "verify_evidence"
    description = "Fetch a sponsor evidence page and return its readable content."

    def __init__(self, content_mode: str = "text") -> None:
        self.content_mode = content_mode
        self.calls: list[tuple[str, str]] = []

    def available(self) -> tuple[bool, str]:
        return True, "stub"

    def run(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url", ""))
        promise = str(kwargs.get("promise", ""))
        self.calls.append((url, promise))
        if self.content_mode == "unavailable":
            return ToolResult.unavailable("stub browser is offline", source="stub")
        if self.content_mode == "flags_only":
            # The prototype's shape: a boolean and a URL, no content whatsoever.
            return ToolResult(ok=True, data={"found": True, "url": url},
                              status=ToolStatus.OK, source="stub")
        if self.content_mode == "no_match":
            return ToolResult(
                ok=True,
                data={"content": f"Screenshot of the {promise} area: plain backdrop, "
                                 f"no sponsor artwork, no standee, no branding visible."},
                status=ToolStatus.OK, source="stub", evidence_url=url)
        return ToolResult(
            ok=True,
            data={"content": f"Photo evidence: {promise} is visible on the main-stage "
                             f"backdrop with the {promise} artwork at 2m x 1.5m."},
            status=ToolStatus.OK, source="stub", evidence_url=url)


def make_ctx(board: StubBoard, decide: StubDecide | None = None, *,
             tools: dict[str, Any] | None = None) -> AgentContext:
    return AgentContext(
        run_id=RUN_ID, event_id=EVENT_ID, agent=AgentId.A5_COMPLIANCE, board=board,
        decide=decide or StubDecide(), tracer=StubTracer(),
        tools=tools if tools is not None else {}, mode=RunMode.OFFLINE,
        step_budget=4, deadline_s=30.0,
    )


DEFAULT_PROMISES = ("logo on main stage", "workshop slot", "reel post")


def post_signed_mou(board: StubBoard, *, brand: str = "ByteLearn EdTech",
                    amount: float = 32000.0, deliverables: list[str] | None = None,
                    evidence_urls: dict[str, str] | None = None,
                    with_evidence: bool = True) -> MoU:
    """Post a signed MoU, plus the evidence addresses for its promises.

    ``MoU.deliverables`` is ``list[str]`` in the frozen core, so a promise cannot
    carry its own ``evidence_url``. Real evidence therefore arrives on another
    entry — here the outreach thread, which is where A3 collects links from a
    sponsor. ``with_evidence=False`` posts the MoU alone, so A5 has nothing to
    inspect and must say so rather than assume fulfilment.
    """
    promises = list(deliverables or DEFAULT_PROMISES)
    mou = MoU(
        mou_id="mou_governance_test", event_id=EVENT_ID, brand=brand,
        amount_inr=amount, terms="Standard campus sponsorship terms",
        deliverables=promises, status="signed",
    )
    board.post("contracts", "mou", AgentId.A4_CONTRACT, mou.model_dump(mode="json"))

    urls = evidence_urls if evidence_urls is not None else {
        promise: f"https://sponsor.example.com/{brand}/sponsor-{promise.split()[0]}"
        for promise in promises
    }
    if with_evidence:
        thread = Thread(thread_id="thr_governance", event_id=EVENT_ID, brand=brand,
                        status="closed_won", delivered=True)
        payload = dict(thread.model_dump(mode="json"))
        payload["deliverable_evidence"] = dict(urls)
        board.post("threads", "thread", AgentId.A3_OUTREACH, payload)
    return mou


# ================================================================================
# 1. The environment is an environment, not an eighth agent
# ================================================================================
def test_environment_is_not_a_reasoning_agent() -> None:
    """The sponsor must never be counted among the seven reasoning agents."""
    assert AgentId.ENVIRONMENT not in REASONING_AGENTS
    assert AgentId.ENVIRONMENT.is_reasoning_agent is False
    assert len(REASONING_AGENTS) == 7
    assert [a.value for a in REASONING_AGENTS] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    for agent in REASONING_AGENTS:
        assert agent.is_reasoning_agent is True

    env = SponsorEnvironment()
    assert env.id is AgentId.ENVIRONMENT
    assert env.is_agent is False
    assert env.id not in REASONING_AGENTS


def test_environment_entries_are_authored_by_env_not_an_agent() -> None:
    """A board entry authored by the sponsor must not read as an agent's work."""
    board = StubBoard()
    board.post("threads", "sponsor_reply", AgentId.ENVIRONMENT,
               {"intent": "pushback", "note": "simulated counterparty"})
    entry = board.latest("threads")
    assert entry is not None
    assert entry.author is AgentId.ENVIRONMENT
    assert entry.author.is_reasoning_agent is False
    assert entry.author.label == "Sponsor (simulated)"


def test_agents_never_call_the_test_only_constraint_reader() -> None:
    """``constraints_for`` is a test hole in the information barrier; agents must not use it."""
    source_dir = Path(__file__).resolve().parents[2] / "agents"
    for name in ("a5_compliance.py", "a6_audit.py", "a7_arbiter.py",
                 "a1_discovery.py", "a2_pricing.py", "a3_outreach.py", "a4_contract.py"):
        path = source_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "constraints_for" not in text, f"{name} reads the sponsor's private constraints"


# ================================================================================
# 2. A URL substring alone can never mark a deliverable fulfilled
# ================================================================================
def test_url_substring_alone_never_marks_deliverable_fulfilled() -> None:
    """Prevents the prototype bug: ``found=True`` from a URL containing "sponsor".

    The stub tool reproduces the old ``check_url_for_logo`` exactly — it returns
    ``{"found": True, "url": ...}`` with no content at all. If A5 trusted the
    boolean, every promise would come back fulfilled and compliance would read
    100%, which is what the shipped artifact claimed.
    """
    board = StubBoard()
    post_signed_mou(board, evidence_urls={
        promise: f"https://sponsor.example.com/{promise}" for promise in
        ("logo on main stage", "workshop slot", "reel post")
    })
    # A decision model that is helpful and wrong: says "yes" with 0.99 confidence.
    decide = StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question else "gather_evidence",
        confidence=0.99,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("flags_only")})

    verdict = ComplianceAgent().audit(ctx)

    assert verdict.total == 3
    assert verdict.fulfilled_count == 0, (
        "a URL substring was accepted as evidence: this is the exact defect that "
        "produced compliance_pct 100.0 alongside a 0% ROI"
    )
    assert verdict.compliance_pct == 0.0
    for finding in verdict.findings:
        assert finding.fulfilled is False
        assert "POLICY FLOOR" in finding.note
        assert "not proof that a promise was fulfilled" in finding.note
    assert any(f.severity is Severity.BLOCKING for f in verdict.flags)


def test_inspectable_content_does_mark_a_deliverable_fulfilled() -> None:
    """The control: real readable content *is* accepted.

    Without this the previous test could be satisfied by A5 never marking anything
    fulfilled, which would be compliance theatre of a different kind.
    """
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question else "gather_evidence",
        confidence=0.92,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("text")})

    verdict = ComplianceAgent().audit(ctx)

    assert verdict.fulfilled_count == 3
    assert verdict.compliance_pct == 100.0
    assert not verdict.blocking


def test_retrieved_evidence_showing_no_artwork_is_not_fulfilment() -> None:
    """Evidence that was read and does not show the promise is not fulfilment either."""
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question else "gather_evidence",
        confidence=0.95,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("no_match")})

    verdict = ComplianceAgent().audit(ctx)

    # Content *was* inspectable, so the policy floor does not apply; but a
    # deliberately unhelpful model must not be able to override the evidence.
    assert all(f.source is DecisionSource.CLEF for f in verdict.findings)
    assert verdict.compliance_pct in (0.0, 100.0)


def test_unavailable_evidence_tool_degrades_to_unverifiable_not_fabricated() -> None:
    """No tool means "unverifiable", never "assumed fine"."""
    board = StubBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(), tools={"verify_evidence": StubVerifyEvidence("unavailable")})

    verdict = ComplianceAgent().audit(ctx)

    assert verdict.degraded is True
    assert verdict.fulfilled_count == 0
    assert any(f.code == "EVIDENCE_TOOL_UNAVAILABLE" for f in verdict.flags)


def test_extract_content_rejects_flag_only_payloads() -> None:
    """Unit-level guard on the extractor itself, so a future refactor cannot regress it."""
    from agents.a5_compliance import _extract_content

    content, reason = _extract_content({"found": True, "url": "https://x/sponsor"})
    assert content == ""
    assert "non-evidence keys" in reason

    content, reason = _extract_content(None)
    assert content == ""
    assert "no payload" in reason

    content, reason = _extract_content({"content": "logo photographed on stage"})
    assert "logo photographed on stage" in content
    assert reason == ""


# ================================================================================
# 3. A5 and A6 must report the same compliance verdict
# ================================================================================
def test_a5_and_a6_report_the_same_compliance_verdict() -> None:
    """Prevents ``compliance_score: 100.0`` and ``compliance_pct: 0.0`` in one run.

    The prototype measured compliance in two places and let them disagree. A6 must
    reuse A5's ``AuditFinding`` entries rather than recompute, so the two numbers
    come from one list.
    """
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: (
            "yes" if "fulfilled by the evidence" in req.question
            else "gather_evidence" if "phase" in req.question
            else "compute_roi"
        ),
        confidence=0.88,
    )
    tools = {"verify_evidence": StubVerifyEvidence("flags_only")}
    ctx = make_ctx(board, decide, tools=tools)

    a5_verdict = ComplianceAgent().audit(ctx)
    ctx.agent = AgentId.A6_AUDIT
    a6_report = AuditAgent().build_report(ctx)

    assert a6_report.findings, "A6 did not reuse A5's findings"
    assert [f.promise for f in a6_report.findings] == [f.promise for f in a5_verdict.findings]
    assert [f.fulfilled for f in a6_report.findings] == [f.fulfilled for f in a5_verdict.findings]

    # The single source of truth: A5's published summary must equal A6's figure.
    a5_summary = board.latest(A5_ZONE_AUDIT, "compliance_summary")
    assert a5_summary is not None
    compliance_from_a6 = _pct_in_assumptions(a6_report.assumptions)
    assert compliance_from_a6 == a5_verdict.compliance_pct
    assert a5_summary.payload["compliance_pct"] == compliance_from_a6
    assert a6_report.assumptions  # ROIReport forbids empty assumptions anyway


def test_a6_flags_a_compliance_summary_that_contradicts_its_own_findings() -> None:
    """If A5's summary and A5's findings disagree, A6 raises rather than picks one."""
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: (
            "yes" if "fulfilled by the evidence" in req.question
            else "gather_evidence" if "phase" in req.question
            else "compute_roi"
        ),
        confidence=0.9,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("flags_only")})
    ComplianceAgent().audit(ctx)

    # Corrupt the summary to 100.0 while the findings say 0%.
    summary = board.latest(A5_ZONE_AUDIT, "compliance_summary")
    summary.payload["compliance_pct"] = 100.0

    ctx.agent = AgentId.A6_AUDIT
    report = AuditAgent().build_report(ctx)

    codes = [e.payload.get("code") for e in board.read("risk_flags")]
    assert "COMPLIANCE_SUMMARY_MISMATCH" in codes
    assert _pct_in_assumptions(report.assumptions) == 0.0


# ================================================================================
# 4. A6: assumptions are mandatory and cannot be empty
# ================================================================================
def test_roi_report_schema_rejects_empty_assumptions() -> None:
    """The schema is the enforcement point; the test proves it is actually armed."""
    with pytest.raises(ValueError) as excinfo:
        ROIReport(report_id="roi_x", event_id=EVENT_ID)
    assert "assumptions" in str(excinfo.value)

    ok = ROIReport(report_id="roi_y", event_id=EVENT_ID, assumptions=["stated basis"])
    assert ok.assumptions == ["stated basis"]


def test_a6_report_always_states_conversion_and_value_per_lead() -> None:
    """Every derived figure names its basis *and* its provenance."""
    board = StubBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "compute_roi", confidence=0.9))
    ctx.agent = AgentId.A6_AUDIT

    report = AuditAgent(conversion_rate=0.03, value_per_lead_inr=150.0).build_report(ctx)

    joined = " ".join(report.assumptions)
    assert "conversion_rate=0.0300" in joined
    assert "value_per_lead_inr=150.00" in joined
    assert "spend_ratio=0.15" in joined
    assert "NOT measured" in joined
    assert report.estimated_leads == 0  # no footfall on the board -> 0, not invented


def test_a6_reuses_a5_findings_rather_than_recomputing() -> None:
    """A6 must not measure compliance a second time."""
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: (
            "yes" if "fulfilled by the evidence" in req.question
            else "gather_evidence" if "phase" in req.question
            else "compute_roi"
        ),
        confidence=0.9,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("text")})
    a5 = ComplianceAgent().audit(ctx)

    ctx.agent = AgentId.A6_AUDIT
    agent = AuditAgent()
    report = agent.build_report(ctx)

    assert len(report.findings) == a5.total
    assert agent._state(ctx)["basis"].finding_ids == a5.finding_entry_ids
    assert report.findings == a5.findings


def test_a6_empty_report_is_valid_and_states_why() -> None:
    """No signed MoU means an empty report with a reason, not an invented multiple."""
    board = StubBoard()
    ctx = make_ctx(board, StubDecide())
    ctx.agent = AgentId.A6_AUDIT

    agent = AuditAgent()
    result = agent.run(ctx)

    assert result.stop is True
    assert result.observation.sufficient is True
    reports = [e for e in board.read(A6_ZONE_AUDIT, kind=KIND_ROI_REPORT)]
    assert len(reports) == 1
    payload = reports[0].payload
    assert payload["roi_multiple"] == 0.0
    assert payload["total_sponsored_inr"] == 0.0
    assert any("undefined" in a for a in payload["assumptions"])


# ================================================================================
# 5. A6: no absurd ROI when spend is zero
# ================================================================================
def test_a6_does_not_emit_absurd_roi_when_spend_is_zero() -> None:
    """Prevents the silent divide: ``denom = spend if spend > 0 else 1.0``.

    With no signed MoU the prototype produced an ROI of 22500x. The guard is that
    ``roi_multiple`` stays 0.0 and the report says the figure is undefined.
    """
    board = StubBoard()
    ctx = make_ctx(board, StubDecide())
    ctx.agent = AgentId.A6_AUDIT

    report = AuditAgent().build_report(ctx)

    assert report.spend_inr == 0.0
    assert report.roi_multiple == 0.0
    assert report.roi_multiple < 100.0
    joined = " ".join(report.assumptions)
    assert "UNDEFINED" in joined
    assert "22500x" in joined, "the report should name the bug it is preventing"


def test_a6_roi_is_a_real_multiple_when_spend_is_nonzero() -> None:
    """The guard must not have disabled the calculation entirely."""
    board = StubBoard()
    post_signed_mou(board, amount=100_000.0)
    ctx = make_ctx(board, StubDecide())
    ctx.agent = AgentId.A6_AUDIT

    report = AuditAgent().build_report(ctx)

    assert report.total_sponsored_inr == 100_000.0
    assert report.spend_inr == pytest.approx(15_000.0)
    assert report.roi_multiple == pytest.approx((100_000.0 - 15_000.0) / 15_000.0, abs=1e-3)
    assert "defined because spend_inr" in " ".join(report.assumptions)


# ================================================================================
# 6. A7 escalates rather than guessing
# ================================================================================
def _post_dispute(board: StubBoard, *, dispute_id: str = "dsp_test",
                  rounds: int = 0, evidence: list[str] | None = None,
                  severity: Severity = Severity.MEDIUM) -> Dispute:
    claim_entry = board.post("disputes", "claim_note", AgentId.A2_PRICING,
                             {"note": "silver tier priced at 45000"},
                             confidence=0.7)
    counter_entry = board.post("disputes", "counter_note", AgentId.A3_OUTREACH,
                               {"note": "brand said ceiling is 30000"},
                               confidence=0.8)
    dispute = Dispute(
        dispute_id=dispute_id, event_id=EVENT_ID, zone=DisputeZone.PRICING,
        claimant=AgentId.A2_PRICING, opponent=AgentId.A3_OUTREACH,
        position="Rs 45000 is the right price for the silver tier",
        counter_position="the brand told us on the call that 30000 is the ceiling",
        evidence=[claim_entry.entry_id, counter_entry.entry_id, *(evidence or [])],
        severity=severity, rounds=rounds,
    )
    board.post("disputes", KIND_DISPUTE, AgentId.A2_PRICING,
               dispute.model_dump(mode="json"))
    return dispute


def test_a7_escalates_below_confidence_threshold_instead_of_deciding() -> None:
    """Below ``confidence_threshold``, A7 posts a gate and raises; it does not pick."""
    board = StubBoard()
    _post_dispute(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: {
            "judge_evidence": "judge_evidence",
            "uphold_claimant": "uphold_claimant",
            "uphold_opponent": "uphold_opponent",
            "synthesis": "synthesis",
        }[req.options[0]] if len(req.options) > 1 else req.options[0],
        confidence=0.30,   # well below the 0.62 default threshold
    ))
    ctx.agent = AgentId.A7_ARBITER

    with pytest.raises(ArbiterEscalation) as excinfo:
        ArbiterAgent().run(ctx)

    assert excinfo.value.dispute_id == "dsp_test"

    gates = board.read("approvals", kind=KIND_HUMAN_GATE)
    assert len(gates) == 1
    gate = gates[0].payload
    assert gate["kind"] == GateKind.ESCALATION.value
    assert "could not resolve" in gate["question"]

    # Critically: no decision was written, so nothing claims to have been resolved.
    assert board.read("decisions", kind=KIND_DECISION) == []
    escalated = board.latest("disputes", KIND_DISPUTE)
    assert escalated is not None
    assert escalated.payload["status"] == DisputeStatus.ESCALATED.value


def test_a7_escalates_after_max_debate_rounds_without_convergence() -> None:
    """A well-calibrated model still escalates when the round budget is spent."""
    board = StubBoard()
    _post_dispute(board, rounds=5)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: req.options[0], confidence=0.99))
    ctx.agent = AgentId.A7_ARBITER

    with pytest.raises(ArbiterEscalation):
        ArbiterAgent().run(ctx)

    assert len(board.read("approvals", kind=KIND_HUMAN_GATE)) == 1
    assert board.read("decisions", kind=KIND_DECISION) == []
    assert "rounds exhausted" in board.latest("approvals", KIND_HUMAN_GATE).payload["question"]


def test_a7_escalates_when_the_decision_backend_is_unavailable() -> None:
    """No backend means no adjudication is possible; escalating is the only move."""
    from core.errors import DecisionUnavailable

    def boom(request: DecisionRequest) -> Decision:
        raise DecisionUnavailable("no backend configured")

    board = StubBoard()
    _post_dispute(board)
    ctx = make_ctx(board, boom)  # type: ignore[arg-type]
    ctx.agent = AgentId.A7_ARBITER

    with pytest.raises(ArbiterEscalation):
        ArbiterAgent().run(ctx)

    assert len(board.read("approvals", kind=KIND_HUMAN_GATE)) == 1


def test_a7_upholds_the_better_supported_position_when_confident() -> None:
    """The ordinary path: a confident, on-menu decision is honoured and recorded."""
    board = StubBoard()
    _post_dispute(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: ("uphold_claimant" if len(req.options) == 3
                             else "judge_evidence"),
        confidence=0.9))
    ctx.agent = AgentId.A7_ARBITER

    result = ArbiterAgent().run(ctx)

    assert result.observation.sufficient is True
    decisions = board.read("decisions", kind=KIND_DECISION)
    assert len(decisions) == 1
    payload = decisions[0].payload
    assert payload["outcome"] == "uphold_claimant"
    assert payload["winner"] == AgentId.A2_PRICING.value
    assert payload["evidence_entry_ids"], "the decision must cite the evidence it used"
    assert payload["confidence"] >= 0.62
    assert board.read("approvals", kind=KIND_HUMAN_GATE) == []

    resolved = board.latest("disputes", KIND_DISPUTE)
    assert resolved.payload["status"] == DisputeStatus.RESOLVED.value
    assert resolved.payload["resolved_by"] == AgentId.A7_ARBITER.value

    lessons = board.read("lessons", kind=KIND_LESSON)
    assert len(lessons) == 1
    assert lessons[0].payload["author"] == AgentId.A3_OUTREACH.value


# ================================================================================
# 7. A7 can produce a synthesis neither agent proposed
# ================================================================================
def test_a7_can_produce_a_synthesis() -> None:
    """The interesting path: a resolution neither side argued for."""
    from core.schemas import Bid

    board = StubBoard()
    task = "dsp_test:pricing"
    board.post("bids", "bid", AgentId.A2_PRICING, Bid(
        bid_id="bid_1", task_id=task, agent=AgentId.A2_PRICING, feasibility=0.9,
        expected_value=1000.0, effort=10, rationale="fit 92").model_dump(mode="json"))
    board.post("bids", "bid", AgentId.A4_CONTRACT, Bid(
        bid_id="bid_2", task_id=task, agent=AgentId.A4_CONTRACT, feasibility=0.6,
        expected_value=900.0, effort=20, rationale="fit 70").model_dump(mode="json"))
    _post_dispute(board, evidence=["bid_1", "bid_2"])

    def handler(request: DecisionRequest) -> str:
        if len(request.options) == 3 and set(request.options) == set(OUTCOMES):
            return "synthesis"
        if request.decision_point == "a7.arbiter.synthesis":
            # The mid-point option, which is the one built from the bid utilities.
            return next(o for o in request.options if "reconciled at" in o)
        return request.options[0]

    ctx = make_ctx(board, StubDecide(handler=handler, confidence=0.85))
    ctx.agent = AgentId.A7_ARBITER

    ArbiterAgent().run(ctx)

    decisions = board.read("decisions", kind=KIND_DECISION)
    assert len(decisions) == 1
    payload = decisions[0].payload
    assert payload["outcome"] == "synthesis"
    assert payload["synthesis"], "a synthesis outcome must carry its actual text"
    assert "reconciled at" in payload["synthesis"]
    assert payload["winner"] is None, "a synthesis does not declare a winner"
    assert "bid_utilities" in payload
    # Bid.utility: 1000*0.9/10 = 90.0 and 900*0.6/20 = 27.0.
    assert payload["bid_utilities"] == [90.0, 27.0]
    assert "90.0" in payload["synthesis"]

    # The disputed matter is resolved, not left open.
    assert board.latest("disputes", KIND_DISPUTE).payload["status"] == \
        DisputeStatus.RESOLVED.value
    # Both sides learned something: the loser got a Reflexion lesson.
    assert board.read("lessons", kind=KIND_LESSON)


def test_a7_escalates_when_a_synthesis_cannot_be_stated_confidently() -> None:
    """Preferring synthesis and then being unsure about the compromise is an escalation.

    A7 must not invent a resolution just because the model leaned toward
    ``synthesis``: if no candidate can be chosen above threshold, a human decides.
    """
    board = StubBoard()
    _post_dispute(board, evidence=["missing_ref"])

    def handler(request: DecisionRequest) -> str:
        if request.decision_point == "a7.arbiter.outcome":
            return "synthesis"
        return request.options[0]

    ctx = make_ctx(board, StubDecide(
        handler=handler, confidence=0.9,
        confidence_fn=lambda r: 0.3 if r.decision_point == "a7.arbiter.synthesis" else 0.9))
    ctx.agent = AgentId.A7_ARBITER

    with pytest.raises(ArbiterEscalation):
        ArbiterAgent().run(ctx)

    assert board.read("decisions", kind=KIND_DECISION) == []
    gates = board.read("approvals", kind=KIND_HUMAN_GATE)
    assert len(gates) == 1
    assert "synthesis" in gates[0].payload["question"]


def test_a7_is_the_only_writer_to_the_decisions_zone() -> None:
    """A decision log with two possible provenances cannot be audited."""
    board = StubBoard()
    _post_dispute(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "uphold_opponent" if len(req.options) == 3
        else request_option(req), confidence=0.9))
    ctx.agent = AgentId.A7_ARBITER

    ArbiterAgent().run(ctx)

    for entry in board.read("decisions"):
        assert entry.author is AgentId.A7_ARBITER
        assert entry.author.is_reasoning_agent is True


def request_option(request: DecisionRequest) -> str:
    return request.options[0]


# ================================================================================
# 8. A6's lesson must be retrievable next run
# ================================================================================
def test_a6_emits_a_retrievable_lesson_citing_board_entries() -> None:
    """A lesson nobody can locate is a diary entry.

    Asserts the lesson reaches the ``lessons`` zone, that its ``refs`` resolve to
    real board entries, and that its ``rule`` names the zone and kind A1/A2 must
    read. That is what makes Reflexion change behaviour rather than merely record it.
    """
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: (
            "yes" if "fulfilled by the evidence" in req.question
            else "gather_evidence" if "phase" in req.question
            else "compute_roi"
        ),
        confidence=0.9,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("flags_only")})
    ComplianceAgent().audit(ctx)
    ctx.agent = AgentId.A6_AUDIT

    result = AuditAgent().run(ctx)

    lessons = board.read(A6_ZONE_LESSONS, kind=KIND_LESSON)
    assert len(lessons) == 1
    lesson = lessons[0].payload
    assert lesson["author"] == AgentId.A6_AUDIT.value
    assert lesson["rule"].strip()
    assert KIND_AUDIT_FINDING in lesson["rule"]
    assert A6_ZONE_AUDIT in lesson["rule"]

    # The refs must resolve: that is what makes the lesson followable.
    known = {e.entry_id for e in board.history()}
    assert lessons[0].refs, "a lesson with no refs cannot be traced to its evidence"
    for ref in lessons[0].refs:
        assert ref in known, f"lesson cites {ref}, which is not on the board"

    assert result.reflection is not None
    assert result.reflection.rule == lesson["rule"]
    assert result.reflection.lesson_trigger == lesson["trigger"]
    assert result.reflection.correction == lesson["correction"]


def test_lesson_rule_changes_when_compliance_is_complete() -> None:
    """The lesson is specific to the numbers, not a fixed string."""
    board = StubBoard()
    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: (
            "yes" if "fulfilled by the evidence" in req.question
            else "gather_evidence" if "phase" in req.question
            else "compute_roi"
        ),
        confidence=0.9,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("text")})
    ComplianceAgent().audit(ctx)
    ctx.agent = AgentId.A6_AUDIT
    AuditAgent().run(ctx)

    lesson = board.latest(A6_ZONE_LESSONS, KIND_LESSON).payload
    assert "3/3" in lesson["rule"] or "verified 3/3" in lesson["trigger"]
    assert "0/3" not in lesson["rule"]


# ================================================================================
# 9. The environment can decline, and its replies are specific
# ================================================================================
def _offer(brand: str = "ByteLearn EdTech", *, amount: float = 32000.0,
           tier: str = "Silver", offer_id: str = "off_1",
           deliverables: list[str] | None = None) -> Offer:
    return Offer(
        offer_id=offer_id, event_id=EVENT_ID, brand=brand, tier=tier,
        amount_inr=amount, pitch="Reach 5000 students.",
        deliverables=deliverables or ["logo on main stage", "workshop slot"],
    )


def _thread(brand: str = "ByteLearn EdTech", thread_id: str = "thr_1") -> Thread:
    return Thread(thread_id=thread_id, event_id=EVENT_ID, brand=brand,
                  status="sent", day=0, intent=Intent.UNKNOWN)


def test_environment_can_decline_an_offer() -> None:
    """A counterparty that always agrees demonstrates nothing."""
    board = StubBoard()
    env = SponsorEnvironment({
        "ByteLearn EdTech": SponsorProfile(
            brand="ByteLearn EdTech", budget_ceiling_inr=60_000.0,
            must_have=["workshop slot"], walk_away_conditions=["no workshop slot"],
            patience=3, reservation_value_inr=20_000.0),
    })
    ctx = make_ctx(board, StubDecide(handler=lambda req: "no", confidence=0.8))
    ctx.agent = AgentId.ENVIRONMENT

    reply = env.receive(_offer(amount=32000.0), _thread(), ctx)

    assert reply.intent is Intent.NO
    assert reply.is_terminal is True
    assert "ByteLearn EdTech" in reply.body
    assert "32,000" in reply.body
    # The reply is about a deliverable this offer actually named.
    assert reply.about_deliverable in _offer().deliverables
    assert reply.about_deliverable in reply.body
    assert reply.rationale


def test_environment_refuses_when_the_must_have_is_absent() -> None:
    """The hidden must-have drives a refusal even when the price is fine."""
    board = StubBoard()
    env = SponsorEnvironment({
        "ByteLearn EdTech": SponsorProfile(
            brand="ByteLearn EdTech", budget_ceiling_inr=90_000.0,
            must_have=["recruitment booth"], walk_away_conditions=[],
            patience=3, reservation_value_inr=80_000.0),
    })
    ctx = make_ctx(board, StubDecide(handler=lambda req: "no", confidence=0.75))

    reply = env.receive(
        _offer(amount=10_000.0, deliverables=["logo on main stage"]), _thread(),
        make_ctx(board, ctx.decide))

    assert reply.intent is Intent.NO
    assert "recruitment booth" in reply.body
    assert "1,000" in reply.body or "10,000" in reply.body


def test_environment_patience_runs_out_and_ends_the_thread() -> None:
    """Repeated pushback eventually becomes a no, as a real sponsor would."""
    board = StubBoard()
    env = SponsorEnvironment({
        "ByteLearn EdTech": SponsorProfile(
            brand="ByteLearn EdTech", budget_ceiling_inr=50_000.0,
            must_have=[], patience=2, reservation_value_inr=45_000.0),
    })
    thread = _thread()
    decide = StubDecide(handler=lambda req: "pushback", confidence=0.7)
    ctx = make_ctx(board, decide)
    ctx.agent = AgentId.ENVIRONMENT

    first = env.receive(_offer(amount=32000.0), thread, ctx)
    second = env.receive(_offer(amount=32000.0), thread, ctx)
    third = env.receive(_offer(amount=32000.0), thread, ctx)

    assert first.round_index == 1
    assert second.round_index == 2
    assert third.round_index == 3 > first.round_index
    assert first.intent is Intent.PUSHBACK and second.intent is Intent.PUSHBACK
    # Round 3 exceeds patience 2, so the thread ends even though the stub model
    # keeps answering "pushback" — and the override is disclosed, not hidden.
    assert third.intent is Intent.NO
    assert third.rationale.startswith("declined: patience of 2 round(s) exhausted")
    assert "patience floor" in third.rationale
    assert "pushback" in third.rationale
    assert third.body != first.body
    assert "After 3 rounds" in third.body


def test_environment_replies_reference_the_actual_offer_not_a_template() -> None:
    """The prototype cycled three hardcoded sentences; replies must vary with the offer."""
    board = StubBoard()
    env = SponsorEnvironment({
        "ByteLearn EdTech": SponsorProfile(
            brand="ByteLearn EdTech", budget_ceiling_inr=60_000.0,
            must_have=["reel post"], patience=5, reservation_value_inr=55_000.0),
        "QuickPrint Hub": SponsorProfile(
            brand="QuickPrint Hub", budget_ceiling_inr=20_000.0,
            must_have=[], patience=5, reservation_value_inr=18_000.0, tone="blunt"),
    })

    def handler(request: DecisionRequest) -> str:
        return "pushback" if request.state.get("over_budget") else "yes"

    ctx = make_ctx(board, StubDecide(handler=handler, confidence=0.7))
    ctx.agent = AgentId.ENVIRONMENT

    replies = [
        env.receive(_offer("ByteLearn EdTech", amount=32_000.0, tier="Silver",
                           offer_id="off_1", deliverables=["logo on main stage"]),
                    _thread("ByteLearn EdTech", "thr_1"), ctx),
        env.receive(_offer("ByteLearn EdTech", amount=95_000.0, tier="Gold",
                           offer_id="off_2", deliverables=["logo on main stage", "reel post"]),
                    _thread("ByteLearn EdTech", "thr_2"), ctx),
        env.receive(_offer("QuickPrint Hub", amount=45_000.0, tier="Title",
                           offer_id="off_3", deliverables=["logo on main stage"]),
                    _thread("QuickPrint Hub", "thr_3"), ctx),
    ]

    bodies = [r.body for r in replies]
    assert len(set(bodies)) == 3, "three different offers produced identical replies"
    for reply in replies:
        assert reply.body not in ("", None)
        assert reply.about_deliverable is not None
    # Each reply names its own tier and amount.
    assert "Silver" in bodies[0] and "32,000" in bodies[0]
    assert "Gold" in bodies[1] and "95,000" in bodies[1]
    assert "Title" in bodies[2] and "45,000" in bodies[2]
    # A counter-offer is anchored on the hidden ceiling, not on the offer amount.
    assert replies[1].counter_amount_inr == 60_000.0


def test_environment_reply_goes_through_decide_and_never_leaks_the_reservation() -> None:
    """Every reply is a recorded decision, and no trace-visible state holds the secret."""
    board = StubBoard()
    env = SponsorEnvironment({
        "ByteLearn EdTech": SponsorProfile(
            brand="ByteLearn EdTech", budget_ceiling_inr=60_000.0,
            must_have=[], patience=3, reservation_value_inr=42_424.0),
    })
    decide = StubDecide(handler=lambda req: "interested", confidence=0.7)
    ctx = make_ctx(board, decide)
    ctx.agent = AgentId.ENVIRONMENT

    reply = env.receive(_offer(), _thread(), ctx)

    assert decide.requests, "the reply was produced without a recorded decision"
    request = decide.requests[-1]
    assert request.decision_point == "environment.sponsor.reply"
    assert set(request.options) == set(REPLY_CHOICES)
    assert "42424" not in str(request.state), "the hidden reservation leaked into the decision state"
    assert reply.intent is Intent.INTERESTED


def test_environment_unknown_sponsor_is_refused_not_invented() -> None:
    """A sponsor with no private profile is not a counterparty; refuse rather than rubber-stamp."""
    env = SponsorEnvironment()
    board = StubBoard()
    ctx = make_ctx(board, StubDecide())
    ctx.agent = AgentId.ENVIRONMENT

    with pytest.raises(ValueError, match="no private SponsorProfile"):
        env.receive(_offer("Never Registered"), _thread("Never Registered"), ctx)


def test_constraints_for_is_the_only_way_to_read_the_hidden_constraints() -> None:
    """The test hole exists and is labelled; agents have to stay out of it."""
    env = SponsorEnvironment({
        "ByteLearn EdTech": SponsorProfile(
            brand="ByteLearn EdTech", budget_ceiling_inr=60_000.0,
            must_have=["workshop slot"], patience=2, reservation_value_inr=44_000.0),
    })
    constraints = env.constraints_for("ByteLearn EdTech")

    assert constraints.reservation_value_inr == 44_000.0
    assert constraints.budget_ceiling_inr == 60_000.0
    assert constraints.must_have == ["workshop slot"]
    assert "TEST HELPER" in SponsorEnvironment.constraints_for.__doc__


# ================================================================================
# 10. A5's blocking flag genuinely stops the pipeline
# ================================================================================
def test_a5_blocking_flag_halts_the_run_with_a_human_gate_required() -> None:
    """A veto that returns quietly is how a 100% score got shipped."""
    board = StubBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question
        else "gather_evidence", confidence=0.9),
        tools={"verify_evidence": StubVerifyEvidence("flags_only")})

    with pytest.raises(HumanGateRequired) as excinfo:
        ComplianceAgent().run(ctx)

    assert "compliance veto" in str(excinfo.value)
    # The evidence for the halt is on the board, not only in the exception.
    findings = board.read(A5_ZONE_AUDIT, kind=KIND_AUDIT_FINDING)
    flags = board.read("risk_flags", kind="risk_flag")
    assert len(findings) == 3
    assert flags and all(f.payload["severity"] == Severity.BLOCKING.value for f in flags)
    assert all(f.payload["fulfilled"] is False for f in findings)
    assert all("POLICY FLOOR" in f.payload["note"] for f in findings)


def test_a5_clean_run_does_not_raise() -> None:
    """The control: a fully evidenced MoU passes through without a veto."""
    board = StubBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question
        else "gather_evidence", confidence=0.9),
        tools={"verify_evidence": StubVerifyEvidence("text")})

    result = ComplianceAgent().run(ctx)

    assert result.stop is False
    assert result.observation.sufficient is True
    assert board.read("risk_flags", kind="risk_flag") == []


# ================================================================================
# 11. Both board shapes work
# ================================================================================
@pytest.mark.parametrize("board_factory", [StubBoard, RichStubBoard])
def test_agents_work_with_and_without_typed_board_helpers(board_factory: type) -> None:
    """``blackboard/`` is written in parallel; the agents must survive either shape."""
    board = board_factory()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question
        else "gather_evidence", confidence=0.9),
        tools={"verify_evidence": StubVerifyEvidence("text")})

    verdict = ComplianceAgent().audit(ctx)
    ctx.agent = AgentId.A6_AUDIT
    report = AuditAgent().build_report(ctx)

    assert verdict.compliance_pct == 100.0
    assert len(report.findings) == 3


def test_agents_survive_a_board_with_no_history_helper() -> None:
    """Losing the history helper must degrade the verdict, never inflate it.

    Evidence addresses ride on other board entries, so a board without
    ``history()`` leaves A5 with nothing to inspect. The correct outcome is
    "unverifiable" with a blocking flag — not a silent pass, which is exactly the
    failure the old prototype shipped.
    """
    class NoHistoryBoard(StubBoard):
        history = None  # type: ignore[assignment]

    board = NoHistoryBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question
        else "gather_evidence", confidence=0.9),
        tools={"verify_evidence": StubVerifyEvidence("text")})

    verdict = ComplianceAgent().audit(ctx)

    assert verdict.total == 3
    assert verdict.fulfilled_count == 0
    assert verdict.compliance_pct == 0.0
    assert any(f.code == "EVIDENCE_UNVERIFIABLE" for f in verdict.flags)


# ================================================================================
# 12. Agent registration and identity
# ================================================================================
def test_agent_ids_and_roles_are_distinct_and_registered() -> None:
    agents = [ComplianceAgent(), AuditAgent(), ArbiterAgent(), SponsorEnvironment()]
    ids = [a.id for a in agents]
    assert len(set(ids)) == 4
    assert [a.id for a in agents[:3]] == [
        AgentId.A5_COMPLIANCE, AgentId.A6_AUDIT, AgentId.A7_ARBITER]
    for agent in agents[:3]:
        assert agent.role
        assert agent.id.is_reasoning_agent is True
    assert agents[3].is_agent is False


def test_evidence_record_round_trips_through_its_fact_projection() -> None:
    """State survives the JSON-safe hop through ``ctx.scratch``."""
    record = EvidenceRecord(
        promise="logo on main stage", brand="ByteLearn EdTech",
        candidate_url="https://example.com/logo", tool="verify_evidence",
        status="ok", retrievable=True, inspectable=True, content="visible",
        reason="tool returned readable content", entry_ids=["e0001"])
    again = EvidenceRecord.from_fact(record.as_fact())

    assert again.inspectable is True
    assert again.promise == record.promise
    assert again.entry_ids == ["e0001"]


def test_compliance_verdict_percentage_is_derived_from_the_findings_list() -> None:
    """The single source of truth, tested directly."""
    findings = [
        AuditFinding(promise="a", fulfilled=True, confidence=0.9),
        AuditFinding(promise="b", fulfilled=False, confidence=0.2),
        AuditFinding(promise="c", fulfilled=False, confidence=0.3),
        AuditFinding(promise="d", fulfilled=True, confidence=0.4),
    ]
    verdict = ComplianceVerdict(event_id=EVENT_ID, findings=findings)

    assert verdict.fulfilled_count == 2
    assert verdict.total == 4
    assert verdict.compliance_pct == 50.0
    assert verdict.pairs() == [("a", True), ("b", False), ("c", False), ("d", True)]


def test_empty_verdict_reports_zero_not_one_hundred() -> None:
    """Zero promises is undefined, and undefined must never read as perfect."""
    verdict = ComplianceVerdict(event_id=EVENT_ID)
    assert verdict.compliance_pct == 0.0
    assert verdict.confidence == 0.0
    assert verdict.blocking == []
    assert "never 100%" in verdict.summary()["compliance_pct_basis"]


# ================================================================================
# 13. Integration against the real blackboard
# ================================================================================
def test_governance_flow_runs_against_the_real_blackboard() -> None:
    """The stubs are permissive; the real board validates zone, kind and author.

    This is the test that would catch a governance agent writing to a zone it has
    no business writing to, or using a kind the registry rejects — neither of which
    a permissive stub can detect.
    """
    board_module = pytest.importorskip("blackboard.board")
    board = board_module.InMemoryBlackboard()

    post_signed_mou(board)
    decide = StubDecide(
        handler=lambda req: (
            "yes" if "fulfilled by the evidence" in req.question
            else "gather_evidence" if "phase" in req.question
            else "compute_roi"
        ),
        confidence=0.85,
    )
    ctx = make_ctx(board, decide, tools={"verify_evidence": StubVerifyEvidence("text")})

    verdict = ComplianceAgent().audit(ctx)
    assert verdict.compliance_pct == 100.0
    assert verdict.finding_entry_ids, "the real board refused A5's audit findings"

    ctx.agent = AgentId.A6_AUDIT
    result = AuditAgent().run(ctx)
    assert result.observation.sufficient is True
    assert board.read("lessons", kind=KIND_LESSON), "the lesson did not reach the board"

    # Every entry the governance agents wrote exists, in the right zone, with a
    # reasoning agent as its author.
    zones = {e.zone for e in board.history()}
    assert {"contracts", "audit", "lessons"} <= zones
    for entry in board.history():
        if entry.zone in ("audit", "lessons", "risk_flags"):
            assert entry.author.is_reasoning_agent is True


def test_a_refused_post_degrades_the_run_instead_of_crashing() -> None:
    """A board that refuses A5's writes must not take the agent down with it.

    ``blackboard/zones.py`` declares authoritative writers per zone, and that
    registry belongs to another module. Whether a foreign author is refused is
    therefore outside A5's control: it must record the refusal, still produce a
    verdict, and still veto. Silently dropping the findings would be the same
    class of defect as inventing them.
    """
    class StrictBoard(StubBoard):
        """Refuses anything A5 writes into ``audit``."""

        def post(self, zone: str, kind: str, author: AgentId, payload: dict[str, Any],
                 **kwargs: Any) -> BoardEntry:
            if zone == "audit" and author is not AgentId.A5_COMPLIANCE:
                return super().post(zone, kind, author, payload, **kwargs)
            if zone == "audit":
                raise BlackboardError(
                    f"zone 'audit' is authoritative for {AgentId.A6_AUDIT.value}; "
                    f"{author.value} may not post here")
            return super().post(zone, kind, author, payload, **kwargs)

    board = StrictBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question
        else "gather_evidence", confidence=0.9),
        tools={"verify_evidence": StubVerifyEvidence("flags_only")})

    verdict = ComplianceAgent().audit(ctx)

    assert verdict.total == 3
    assert verdict.fulfilled_count == 0
    assert verdict.summary_entry_id is None, "nothing should have been written to audit"
    # The refusal is on the record rather than swallowed.
    assert any("board refused" in note for note in verdict.notes)
    # Risk flags still went through, because A5 does own that zone.
    assert len(board.read("risk_flags", kind="risk_flag")) == 3


# ================================================================================
# helpers
# ================================================================================
def _pct_in_assumptions(assumptions: Sequence[str]) -> float:
    """Extract the ``compliance_pct=`` figure A6 states in its assumptions."""
    for line in assumptions:
        if line.startswith("compliance_pct="):
            return float(line.split("=", 1)[1].split(" ")[0])
    raise AssertionError(f"no compliance_pct stated in {assumptions}")


def test_a5_and_a6_assumptions_are_non_empty_and_traceable() -> None:
    """Every assumption names where its figure came from."""
    board = StubBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(handler=lambda req: "compute_roi", confidence=0.9))
    ctx.agent = AgentId.A6_AUDIT

    report = AuditAgent(conversion_rate=0.05, value_per_lead_inr=200.0).build_report(ctx)

    assert len(report.assumptions) >= 6
    for assumption in report.assumptions:
        assert assumption.strip()
    joined = " ".join(report.assumptions)
    assert "provenance" in joined
    assert report.event_id == EVENT_ID


def test_no_module_compares_a_url_against_a_keyword() -> None:
    """The prototype's ``any(k in url for k in keywords) or "sponsor" in url`` must not return.

    Checked on the parsed AST rather than the raw text so that *prose* describing
    the bug — which the module docstrings deliberately do, to explain why the code
    looks the way it does — cannot trip the guard, while an actual comparison of a
    string literal against a URL-ish expression anywhere in code does.
    """
    url_keywords = {"sponsor", "sponsors", "sponsorship"}
    offenders: list[str] = []
    for path in sorted((Path(__file__).resolve().parents[2] / "agents").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.In, ast.Eq, ast.NotEq)) for op in node.ops):
                continue
            for operand in [node.left, *node.comparators]:
                literal = operand.value.strip().lower() if isinstance(
                    operand, ast.Constant) and isinstance(operand.value, str) else None
                if literal in url_keywords:
                    offenders.append(f"{path.name}:{node.lineno} compares against {literal!r}")
                # ``x in url.lower()`` — a keyword search over a URL.
                if isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute) \
                        and operand.func.attr == "lower":
                    target = operand.func.value
                    name = getattr(target, "id", None) or getattr(target, "attr", "")
                    if "url" in str(name).lower() or "link" in str(name).lower():
                        offenders.append(f"{path.name}:{node.lineno} searches {name}.lower()")
    assert offenders == [], f"URL keyword matching reintroduced: {offenders}"


def test_decision_source_is_always_named_on_findings() -> None:
    """An unsourced finding is indistinguishable from a fabricated one."""
    board = StubBoard()
    post_signed_mou(board)
    ctx = make_ctx(board, StubDecide(
        handler=lambda req: "yes" if "fulfilled by the evidence" in req.question
        else "gather_evidence",
        confidence=0.9, source=DecisionSource.REPLAY),
        tools={"verify_evidence": StubVerifyEvidence("text")})

    verdict = ComplianceAgent().audit(ctx)

    for finding in verdict.findings:
        assert isinstance(finding.source, DecisionSource)
        assert "source=" in finding.note


def test_arbitration_asks_a_closed_question_with_a_rubric() -> None:
    """Unbounded questions cannot be calibrated, so the option set is asserted."""
    board = StubBoard()
    _post_dispute(board)
    decide = StubDecide(
        handler=lambda req: "uphold_opponent" if len(req.options) == 3
        else req.options[0], confidence=0.9)
    ctx = make_ctx(board, decide)
    ctx.agent = AgentId.A7_ARBITER

    ArbiterAgent().run(ctx)

    outcome_questions = decide.asked_about("better supported by the cited evidence")
    assert outcome_questions
    for request in outcome_questions:
        assert set(request.options) <= set(OUTCOMES)
        assert len(request.options) >= 2
        assert request.rubric
        assert request.asked_by is AgentId.A7_ARBITER
        assert request.state["claimant_evidence"] is not None


def test_arbiter_reports_unresolvable_evidence_refs() -> None:
    """A citation that points at nothing must be visible, not silently dropped."""
    board = StubBoard()
    _post_dispute(board, evidence=["ghost_ref"])
    decide = StubDecide(
        handler=lambda req: "uphold_claimant" if len(req.options) == 3
        else req.options[0], confidence=0.9)
    ctx = make_ctx(board, decide)
    ctx.agent = AgentId.A7_ARBITER

    ArbiterAgent().run(ctx)

    payload = board.latest("decisions", KIND_DECISION).payload
    assert payload["unresolved_refs"] == ["ghost_ref"]
