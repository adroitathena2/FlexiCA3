"""A6 Audit — honest ROI, plus the Reflexion lesson the next run reads.

What this agent is for
----------------------
A6 turns what actually happened into two things: an :class:`ROIReport` whose
every derived figure is traceable to a stated assumption, and a :class:`Lesson`
that changes behaviour on the *next* run. Shinn et al. (2023, NeurIPS) call the
second part Reflexion — no weights change, a verbal lesson is retrieved as
context. That only works if the lesson is retrievable, so this module writes it
with ``refs`` pointing at the board entries that motivated it and a ``rule`` that
names the zone and kind to read. A lesson nobody can locate is a diary entry.

Three defects this file exists to make impossible
--------------------------------------------------
1. **Undocumented assumptions.** ``ROIReport.assumptions`` is validated as
   non-empty by ``core.schemas``, deliberately. The prototype nevertheless used an
   invented 3% conversion and ₹150/lead, and its shipped artifact claimed 13% and
   ₹4,500 while the code said 3% and ₹150. Every figure A6 emits therefore
   carries both its basis *and* its provenance, and the numbers quoted in the
   prose are the numbers used in the arithmetic.
2. **Recomputing compliance.** The prototype printed ``compliance_score: 100.0``
   and ``compliance_pct: 0.0`` in one run because two modules measured
   compliance separately. A6 never measures compliance: it reuses A5's
   :class:`AuditFinding` entries from the ``audit`` zone. When A5 also published a
   ``compliance_summary``, A6 cross-checks its percentage against the findings
   and raises a ``COMPLIANCE_SUMMARY_MISMATCH`` risk flag if they differ, rather
   than quietly picking one.
3. **The silent divide.** The prototype wrote ``denom = spend if spend > 0 else
   1.0``, so zero spend silently produced an ROI of 22500x. :func:`_safe_roi_multiple`
   returns 0.0 with an explicit "undefined" note instead. Reporting *undefined*
   is a useful answer; a fake multiple is worse than none, because it gets quoted.

Every conditional decision goes through ``ctx.decide``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import ValidationError

from core.config import Settings, get_settings
from core.errors import BlackboardError, DecisionFailed, DecisionUnavailable, SchemaError
from core.ids import new_id, utcnow
from core.protocols import ActResult, AgentContext, BoardEntry, Observation, Plan, Reflection
from core.schemas import (
    AgentId,
    AuditFinding,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    Lesson,
    MoU,
    QuestionType,
    RiskFlag,
    ROIReport,
    Severity,
)

from .base import ReActAgent

__all__ = [
    "AuditAgent",
    "A6Audit",
    "RoiBasis",
    "ZONE_AUDIT",
    "ZONE_CONTRACTS",
    "ZONE_LESSONS",
    "ZONE_EVENT",
    "ZONE_RISK_FLAGS",
    "KIND_MOU",
    "KIND_AUDIT_FINDING",
    "KIND_COMPLIANCE_SUMMARY",
    "KIND_ROI_REPORT",
    "KIND_LESSON",
    "KIND_RISK_FLAG",
]

_M = TypeVar("_M")


# --------------------------------------------------------------------------- zones
def _resolve_zone(attr: str, fallback: str) -> str:
    """Prefer ``blackboard.zones``' own constant when that module is present.

    ``blackboard/`` is written in parallel with this file, so importing it
    unconditionally would leave A6 unimportable until it lands. Three spellings
    are tried because the constant may be named ``AUDIT`` or ``ZONE_AUDIT``; once
    ``blackboard.zones`` exists this is a plain lookup and the two definitions
    cannot drift apart.
    """
    try:  # pragma: no cover - depends on a concurrent module landing
        from blackboard import zones as _zones  # type: ignore[import-not-found]
    except (ImportError, BlackboardError, SchemaError):  # pragma: no cover
        return fallback
    for name in (attr, f"ZONE_{attr}", attr.lower()):
        value = getattr(_zones, name, None)
        if isinstance(value, str) and value:
            return value
    return fallback


ZONE_AUDIT = _resolve_zone("AUDIT", "audit")
ZONE_CONTRACTS = _resolve_zone("CONTRACTS", "contracts")
ZONE_LESSONS = _resolve_zone("LESSONS", "lessons")
ZONE_EVENT = _resolve_zone("EVENT", "event")
ZONE_RISK_FLAGS = _resolve_zone("RISK_FLAGS", "risk_flags")

KIND_MOU = "mou"
KIND_AUDIT_FINDING = "audit_finding"
KIND_COMPLIANCE_SUMMARY = "compliance_summary"
KIND_ROI_REPORT = "roi_report"
KIND_LESSON = "lesson"
KIND_EVENT = "event_profile"
KIND_RISK_FLAG = "risk_flag"


# ------------------------------------------------------------------ board helpers
def _read(ctx: AgentContext, zone: str, *, kind: str | None = None,
          limit: int | None = None) -> list[BoardEntry]:
    """``board.read`` with a positional-only fallback; a zone fault gives ``[]``."""
    try:
        try:
            return list(ctx.board.read(zone, kind=kind, limit=limit))
        except TypeError:
            entries = list(ctx.board.read(zone))
            if kind is not None:
                entries = [e for e in entries if e.kind == kind]
            if limit is not None:
                entries = entries[-limit:]
            return entries
    except BlackboardError:
        return []


def _as_model(entry: BoardEntry, model_cls: type[_M]) -> _M | None:
    """Reconstruct a typed artefact from a board payload.

    Strict validation first, then one *projected* retry that keeps only the
    fields the model declares, because another agent may have posted extra
    diagnostic keys beside the artefact. Two failures mean the entry simply is not
    this artefact, and the caller records a gap rather than inventing a value.
    """
    payload = getattr(entry, "payload", None)
    if not isinstance(payload, dict):
        return None
    try:
        return model_cls.model_validate(payload)
    except ValidationError:
        pass
    known = set(model_cls.model_fields)
    projected = {k: v for k, v in payload.items() if k in known}
    if not projected or projected == payload:
        return None
    try:
        return model_cls.model_validate(projected)
    except ValidationError:
        return None


def _latest_model(ctx: AgentContext, zone: str, model_cls: type[_M],
                  *, kind: str | None = None) -> _M | None:
    """Newest typed entry, preferring the board's ``latest_model`` helper."""
    getter = getattr(ctx.board, "latest_model", None)
    if callable(getter):
        try:
            found = getter(zone, model_cls, kind=kind)
        except TypeError:
            found = getter(zone, model_cls)
        except (BlackboardError, SchemaError):
            found = None
        if isinstance(found, model_cls):
            return found
    for entry in reversed(_read(ctx, zone, kind=kind, limit=25)):
        model = _as_model(entry, model_cls)
        if model is not None:
            return model
    return None


# ============================================================================ inputs
@dataclass(slots=True)
class RoiBasis:
    """Every input to the ROI arithmetic, with where it came from.

    Separated from the report so a reviewer can inspect provenance *before* the
    numbers, and so the same basis can produce a second report without re-reading
    the board.
    """

    mous: list[MoU]
    mou_ids: list[str]
    #: Board entry ids of those MoUs. Distinct from ``mou_ids``: a citation in
    #: ``BoardEntry.refs`` must name an entry, or the reference dangles and the
    #: lesson that cites it cannot be followed.
    mou_entry_ids: list[str]
    brands: list[str]
    footfall: int
    footfall_source: str
    findings: list[AuditFinding]
    finding_ids: list[str]
    conversion_rate: float
    value_per_lead_inr: float
    summary_pct: float | None = None
    summary_entry_id: str | None = None


# ============================================================================ agent
class AuditAgent(ReActAgent):
    """Compute an auditable ROI from signed MoUs, then write a Reflexion lesson."""

    id = AgentId.A6_AUDIT
    role = "Audit: ROI accounting with documented assumptions, and Reflexion"

    _SCRATCH = "a6_audit"

    #: The prototype's 15% spend ratio, now stated rather than buried in code.
    SPEND_RATIO = 0.15

    #: Model-decided audit bases. The first entry of each tuple is the
    #: historical behaviour, so a stub backend returning the first option (or
    #: an off-menu answer falling back to it) reproduces the exact numbers the
    #: existing tests assert.
    SPEND_BASIS_OPTIONS: tuple[str, ...] = (
        "apply_15pct_spend_ratio",
        "report_spend_undefined",
    )
    LEAD_BASIS_OPTIONS: tuple[str, ...] = (
        "apply_configured_conversion",
        "report_leads_undefined",
    )
    COMPLIANCE_READING_OPTIONS: tuple[str, ...] = (
        "report_recomputed_pct",
        "report_summary_mismatch",
        "report_undefined",
    )
    LESSON_FOCUS_OPTIONS: tuple[str, ...] = (
        "evidence_gap",
        "pricing_basis",
        "conversion_gap",
        "no_issue",
    )

    def __init__(self, *, step_budget: int = 2, deadline_s: float = 60.0,
                 settings: Settings | None = None,
                 value_per_lead_inr: float = 0.0,
                 conversion_rate: float = 0.0) -> None:
        """
        Parameters
        ----------
        value_per_lead_inr:
            What one captured lead is worth, in rupees. Defaults to ``0.0``, which
            gives ``pipeline_value = 0`` plus an assumption stating the value was
            not supplied. The prototype hardcoded ₹150 while its artifact claimed
            ₹4,500; zero makes the gap visible instead of filling it with a guess.
        conversion_rate:
            Share of footfall that becomes a lead, defaulting to ``0.0`` for the
            same reason. The prototype's 3% was invented; its artifact claimed 13%.
        """
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)
        self.settings = settings or get_settings()
        self.value_per_lead_inr = float(value_per_lead_inr)
        self.conversion_rate = float(conversion_rate)

    # ---------------------------------------------------------------- scratch
    def _state(self, ctx: AgentContext) -> dict[str, Any]:
        scratch = ctx.scratch.setdefault(self._SCRATCH, {})
        scratch.setdefault("phase", "load")
        scratch.setdefault("basis", None)
        scratch.setdefault("report", None)
        scratch.setdefault("notes", [])
        scratch.setdefault("lesson_entry_id", None)
        return scratch

    def _ask_basis(self, ctx: AgentContext, state: dict[str, Any], basis: RoiBasis,
                   *, decision_point: str, question: str,
                   options: list[str], instructions: str) -> str:
        """One basis decision with a recorded fallback to the first option.

        Backend failures and off-menu answers default to ``options[0]`` (the
        historical behaviour) with a note, so the audit never blocks on the
        model and direct ``build_report`` callers without a prior ``_plan``
        still see the historical numbers via the same default.
        """
        default = options[0]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=question,
            question_type=QuestionType.CHOICE,
            state={
                "signed_mous": len(basis.mous),
                "mou_ids": list(basis.mou_ids),
                "audit_findings": len(basis.findings),
                "footfall": basis.footfall,
                "conversion_rate": basis.conversion_rate,
                "value_per_lead_inr": basis.value_per_lead_inr,
                "summary_pct": basis.summary_pct,
            },
            options=list(options),
            instructions=instructions,
            asked_by=self.id,
            decision_point=decision_point,
        )
        try:
            choice = str(ctx.decide(request).choice or "")
        except (DecisionUnavailable, DecisionFailed) as exc:
            state["notes"].append(
                f"decision backend unavailable at {decision_point!r} ({exc}); "
                f"defaulted to {default!r}")
            return default
        except Exception as exc:  # noqa: BLE001 - a backend may raise anything
            state["notes"].append(
                f"decision backend failed at {decision_point!r} "
                f"({type(exc).__name__}: {exc}); defaulted to {default!r}")
            return default
        if choice not in options:
            state["notes"].append(
                f"decision returned {choice!r}, which is not one of "
                f"{options}; defaulted to {default!r}")
            return default
        return choice

    # ------------------------------------------------------------------ plan
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Gather inputs, then ask whether an ROI may be computed at all."""
        state = self._state(ctx)
        basis = self._load_basis(ctx, state)
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=("Should this audit compute an ROI report now, or stop because "
                      "the board has nothing signed to measure?"),
            question_type=QuestionType.CHOICE,
            state={
                "signed_mous": len(basis.mous),
                "mou_ids": list(basis.mou_ids),
                "audit_findings": len(basis.findings),
                "footfall": basis.footfall,
                "last_observation": obs.summary if obs is not None else "",
            },
            options=["compute_roi", "stop_no_inputs"],
            rubric=[
                "compute_roi: at least one signed MoU exists to account for",
                "stop_no_inputs: no signed MoU, so any ROI would be fabricated",
            ],
            instructions=("Never compute an ROI with no signed MoU. An empty report "
                          "stating why is the correct output in that case."),
            asked_by=self.id,
            decision_point="a6.audit.scope",
        )
        try:
            choice = str(ctx.decide(request).choice or "")
        except (DecisionUnavailable, DecisionFailed) as exc:
            choice = ""
            state["notes"].append(f"decision backend unavailable ({exc}); "
                                  f"scope decided by the stated rule")
        if choice not in ("compute_roi", "stop_no_inputs"):
            # A low-confidence or off-menu answer is not permission to invent a
            # figure; fall back to the rule, which is the conservative branch.
            choice = "compute_roi" if basis.mous else "stop_no_inputs"
            state["notes"].append(f"decision returned {choice!r}, which is not one of "
                                  f"the offered options; scope decided by the rule")
        if not basis.mous:
            # Hard floor. Nothing is signed, so there is no revenue to account
            # for, and a model preference cannot create one. An empty report that
            # says why is the honest output; a multiple over zero inputs is not.
            if choice != "stop_no_inputs":
                state["notes"].append(
                    f"decision chose {choice!r} but zone `{ZONE_CONTRACTS}` holds no "
                    f"signed MoU; scope overridden to stop_no_inputs")
            choice = "stop_no_inputs"
        # These bases are never overridden: they decide *how* the numbers are
        # worded, not *whether* an ROI may be computed. They run even when the
        # scope is stop_no_inputs, so an empty audit still reasons.
        state["spend_basis"] = self._ask_basis(
            ctx, state, basis,
            decision_point="a6.roi.spend_basis",
            question=("How should this audit account for organiser spend?"),
            options=list(self.SPEND_BASIS_OPTIONS),
            instructions=("Apply the stated 15% spend ratio, or report spend as "
                          "undefined rather than applying a ratio."),
        )
        state["lead_basis"] = self._ask_basis(
            ctx, state, basis,
            decision_point="a6.roi.lead_basis",
            question=("How should this audit estimate leads from footfall?"),
            options=list(self.LEAD_BASIS_OPTIONS),
            instructions=("Apply the configured conversion rate to footfall, or "
                          "report leads as undefined rather than converting."),
        )
        state["compliance_reading"] = self._ask_basis(
            ctx, state, basis,
            decision_point="a6.compliance.reading",
            question=("How should this audit read A5's compliance evidence?"),
            options=list(self.COMPLIANCE_READING_OPTIONS),
            instructions=("Report the recomputed percentage from AuditFinding "
                          "entries, flag a summary mismatch, or report undefined."),
        )
        state["lesson_focus"] = self._ask_basis(
            ctx, state, basis,
            decision_point="a6.lesson.focus",
            question=("What should the Reflexion lesson for this audit emphasise?"),
            options=list(self.LESSON_FOCUS_OPTIONS),
            instructions=("Focus the lesson on the gap that most threatens the "
                          "next run: missing evidence, pricing basis, conversion "
                          "basis, or no open issue."),
        )
        state["phase"] = "compute" if choice == "compute_roi" else "stop"
        return Plan(
            goal=f"audit scope={state['phase']}",
            steps=[f"scope:{choice}", f"signed_mous:{len(basis.mous)}"],
            rationale=(f"{len(basis.mous)} signed MoU(s) {basis.mou_ids}, "
                       f"{len(basis.findings)} A5 finding(s); decision chose {choice}"),
            confidence=1.0,
            source=DecisionSource.RULES,
            stop=state["phase"] == "stop",
        )

    # ------------------------------------------------------------------ act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        state = self._state(ctx)
        if str(state.get("phase")) == "stop":
            report = self._empty_report(ctx, plan.rationale)
            state["report"] = report
            # Publishing happens in _reflect, once, for both paths. Publishing here
            # as well would write the same report to the board twice.
            return ActResult(
                ok=True,
                output=report.model_dump(mode="json"),
                observations=["no signed MoU on the board; wrote an empty report "
                              "stating why instead of inventing a multiple"],
            )
        report = self._report_from(ctx, state["basis"])
        state["report"] = report
        return ActResult(
            ok=True,
            output=report.model_dump(mode="json"),
            observations=[f"ROI {report.report_id}: sponsored "
                          f"{report.total_sponsored_inr}, spend {report.spend_inr}, "
                          f"roi_multiple {report.roi_multiple}"],
            errors=[] if report.spend_inr > 0 else [
                "spend_inr is zero; ROI multiple is undefined and reported as 0.0"],
            degraded=not state["basis"].findings,
        )

    # --------------------------------------------------------------- observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        state = self._state(ctx)
        report = state.get("report")
        if not isinstance(report, ROIReport):
            return Observation(
                summary="no ROI report was produced",
                facts={"report_id": None},
                sufficient=False,
                gaps=["no ROI report was produced"],
            )
        gaps: list[str] = []
        if report.spend_inr <= 0.0:
            gaps.append("spend is zero, so roi_multiple is undefined and must be "
                        "read together with its assumption note")
        if not state["basis"].findings:
            gaps.append("A5 published no AuditFinding, so compliance is unmeasured "
                        "rather than zero")
        return Observation(
            summary=(f"ROI {report.report_id}: {report.total_sponsored_inr} sponsored, "
                     f"{report.estimated_leads} leads, spend {report.spend_inr}, "
                     f"roi_multiple {report.roi_multiple}"),
            facts={
                "report_id": report.report_id,
                "roi_multiple": report.roi_multiple,
                "spend_inr": report.spend_inr,
                "assumption_count": len(report.assumptions),
                "compliance_pct": _compliance_pct(state["basis"].findings),
                "lesson_entry_id": state.get("lesson_entry_id"),
            },
            sufficient=True,
            gaps=gaps,
        )

    # --------------------------------------------------------------- reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """Author the Reflexion lesson, post it, and return the same content.

        Posting is what makes the lesson retrievable next run; returning it
        satisfies the base class contract. Both point at one object, so the trace
        and the board cannot disagree about what was learned.
        """
        state = self._state(ctx)
        report = state.get("report")
        if not isinstance(report, ROIReport):
            return None
        return self._publish(ctx, report)

    # ================================================================ public API
    def build_report(self, ctx: AgentContext) -> ROIReport:
        """Compute the ROI report from the board without writing anything.

        The read-only half of :meth:`_act`, exposed so tests, the API layer and a
        dry run can see the arithmetic without mutating the blackboard. It shares
        :meth:`_report_from` with the agent path, so a dry run and a real run
        cannot produce different numbers.
        """
        state = self._state(ctx)
        basis = self._load_basis(ctx, state)
        return self._report_from(ctx, basis)

    def build_lesson(self, ctx: AgentContext, report: ROIReport,
                     sources: dict[str, list[str]] | None = None) -> Lesson:
        """Author the Reflexion lesson this audit implies.

        The rule is written for the *next* run's reader: it names the zone, the
        kind, and the concrete number that motivated it, so A1 or A2 can act on it
        rather than merely read it.
        """
        state = self._state(ctx)
        basis = state.get("basis")
        findings = basis.findings if isinstance(basis, RoiBasis) else []
        verified = sum(1 for f in findings if f.fulfilled)
        total = len(findings)
        pct = _compliance_pct(findings)
        if sources is None:
            sources = {
                "mous": list(basis.mou_entry_ids) if isinstance(basis, RoiBasis) else [],
                "findings": list(basis.finding_ids) if isinstance(basis, RoiBasis) else [],
                "footfall": [],
            }
        grounded = _lesson_refs(sources, str(state.get("report_entry_id") or ""), None)

        if total == 0:
            trigger = (f"zone `{ZONE_AUDIT}` held no A5 AuditFinding when this audit "
                       f"ran, so there was no compliance evidence to reflect on")
            rule = (f"Do not state a compliance percentage until A5 has posted "
                    f"{KIND_AUDIT_FINDING} entries to zone `{ZONE_AUDIT}`; report "
                    f"compliance as undefined (0.0), never as a default or 100%.")
            confidence = 0.9
        else:
            trigger = (f"A5 verified {verified} of {total} promised deliverables "
                       f"({pct:.2f}%) across {len(grounded)} board entry/entries "
                       f"{sorted(grounded)}")
            rule = (f"Treat every deliverable as unverified until an "
                    f"{KIND_AUDIT_FINDING} entry in zone `{ZONE_AUDIT}` says "
                    f"fulfilled=True. The audit at {report.report_id} verified "
                    f"{verified}/{total}; a promise with no inspectable evidence URL "
                    f"is not a promise kept.")
            confidence = 0.95 if verified < total else 0.8

        focus = state.get("lesson_focus") if isinstance(state, dict) else None
        focus = str(focus or self.LESSON_FOCUS_OPTIONS[0])
        if focus not in self.LESSON_FOCUS_OPTIONS:
            focus = self.LESSON_FOCUS_OPTIONS[0]
        if focus == "pricing_basis":
            trigger += (" Lesson focus per a6.lesson.focus=pricing_basis: "
                        "pricing basis (spend ratio and value per lead) needs review.")
            rule += (f" When pricing next, re-read {KIND_AUDIT_FINDING} entries in "
                     f"zone `{ZONE_AUDIT}` with focus pricing_basis "
                     f"(a6.lesson.focus=pricing_basis).")
        elif focus == "conversion_gap":
            trigger += (" Lesson focus per a6.lesson.focus=conversion_gap: "
                        "conversion from footfall to leads needs review.")
            rule += (f" When estimating next, re-read {KIND_AUDIT_FINDING} entries in "
                     f"zone `{ZONE_AUDIT}` with focus conversion_gap "
                     f"(a6.lesson.focus=conversion_gap).")
        elif focus == "no_issue":
            trigger += (" Lesson focus per a6.lesson.focus=no_issue: "
                        "no open gap beyond keeping provenance.")
            rule += (f" Keep reading {KIND_AUDIT_FINDING} entries in zone "
                     f"`{ZONE_AUDIT}` the same way (a6.lesson.focus=no_issue).")
        # evidence_gap is the historical focus: trigger/rule above are unchanged.

        corrections: list[str] = []
        if report.spend_inr <= 0.0:
            corrections.append(
                f"spend_inr was {report.spend_inr:.2f} so roi_multiple is undefined; "
                f"the prototype substituted 1.0 for the denominator and reported a "
                f"22500x ROI from zero spend")
        if report.estimated_leads == 0:
            corrections.append(
                "estimated_leads is 0 because no conversion rate was configured; the "
                "prototype published 650 leads from an undocumented 3%")
        if total == 0:
            corrections.append(
                "the prototype printed compliance_score 100.0 and compliance_pct 0.0 "
                "in one run; both figures now derive from one AuditFinding list")
        if not corrections:
            corrections.append(
                "no silent division and no undocumented assumption were needed this "
                "run; keep the provenance strings so a reader can re-derive the "
                "figures by hand")

        return Lesson(
            lesson_id=new_id("les"),
            event_id=ctx.event_id,
            author=self.id,
            trigger=trigger,
            correction="; ".join(corrections),
            rule=rule,
            confidence=confidence,
            created_at=utcnow(),
        )

    def _publish(self, ctx: AgentContext, report: ROIReport) -> Reflection | None:
        """Write the ROI report and its lesson to the board."""
        state = self._state(ctx)
        basis = state.get("basis")
        sources = {
            "mous": list(basis.mou_entry_ids) if isinstance(basis, RoiBasis) else [],
            "findings": list(basis.finding_ids) if isinstance(basis, RoiBasis) else [],
            "footfall": [],
        }
        report_entry_id = self._post(
            ctx, ZONE_AUDIT, KIND_ROI_REPORT, report.model_dump(mode="json"),
            refs=sources["mous"], source=DecisionSource.RULES,
        )
        state["report_entry_id"] = report_entry_id

        lesson = self.build_lesson(ctx, report, sources)
        lesson_entry_id = self._post(
            ctx, ZONE_LESSONS, KIND_LESSON, lesson.model_dump(mode="json"),
            refs=_lesson_refs(sources, str(report_entry_id or ""), lesson),
            confidence=lesson.confidence, source=DecisionSource.RULES,
        )
        state["lesson_entry_id"] = lesson_entry_id
        return Reflection(
            lesson_trigger=lesson.trigger,
            correction=lesson.correction,
            rule=lesson.rule,
            confidence=lesson.confidence,
        )

    # ============================================================== internals
    def _post(self, ctx: AgentContext, zone: str, kind: str, payload: dict[str, Any],
              *, refs: list[str] | None = None, confidence: float = 1.0,
              source: DecisionSource = DecisionSource.RULES) -> str | None:
        """:meth:`ReActAgent.post` with a recorded refusal instead of an exception.

        The board validates the zone/kind/author triple on every write and that
        registry belongs to another module. If it refuses, the run must still
        produce its verdict *and record that the write did not happen*.
        """
        try:
            return self.post(ctx, zone, kind, payload, refs=refs,
                             confidence=confidence, source=source)
        except (BlackboardError, SchemaError) as exc:
            self._state(ctx)["notes"].append(
                f"board refused {zone}/{kind} from {self.id.value}: "
                f"{type(exc).__name__}: {exc}")
            return None

    def _load_basis(self, ctx: AgentContext, state: dict[str, Any]) -> RoiBasis:
        """Read every input once per run and record its provenance."""
        mous, mou_ids, brands, mou_entry_ids = self._load_mous(ctx)
        footfall, footfall_source = self._load_footfall(ctx)
        findings, finding_ids = self._load_findings(ctx)
        summary_entry = _latest_entry(ctx, ZONE_AUDIT, KIND_COMPLIANCE_SUMMARY)
        summary_pct = None
        if summary_entry is not None and isinstance(summary_entry.payload, dict):
            raw = summary_entry.payload.get("compliance_pct")
            if isinstance(raw, (int, float)):
                summary_pct = float(raw)
        basis = RoiBasis(
            mous=mous, mou_ids=mou_ids, brands=brands, mou_entry_ids=mou_entry_ids,
            footfall=footfall, footfall_source=footfall_source,
            findings=findings, finding_ids=finding_ids,
            conversion_rate=self.conversion_rate,
            value_per_lead_inr=self.value_per_lead_inr,
            summary_pct=summary_pct,
            summary_entry_id=summary_entry.entry_id if summary_entry else None,
        )
        state["basis"] = basis
        self._cross_check(ctx, basis)
        return basis

    def _cross_check(self, ctx: AgentContext, basis: RoiBasis) -> None:
        """Compare A5's published percentage with the findings, and flag a gap.

        Two modules measuring compliance separately is how the prototype shipped
        ``compliance_score: 100.0`` beside ``compliance_pct: 0.0``. A6 uses the
        findings, but if A5's own summary disagrees with them the run has a real
        integrity problem, so it is recorded rather than resolved silently.
        """
        if basis.summary_pct is None or not basis.findings:
            return
        recomputed = _compliance_pct(basis.findings)
        if abs(recomputed - basis.summary_pct) < 1e-6:
            return
        flag = RiskFlag(
            flag_id=new_id("rsk"),
            event_id=ctx.event_id,
            brand=basis.brands[0] if basis.brands else "unknown",
            severity=Severity.MEDIUM,
            code="COMPLIANCE_SUMMARY_MISMATCH",
            message=(f"A5's compliance_summary reports {basis.summary_pct}% but its own "
                     f"AuditFinding entries recompute to {recomputed}%; A6 reports the "
                     f"recomputed value"),
            evidence=[basis.summary_entry_id or "", *basis.finding_ids],
            raised_by=self.id,
            raised_at=utcnow(),
        )
        self.post(ctx, ZONE_RISK_FLAGS, KIND_RISK_FLAG, flag.model_dump(mode="json"),
                  refs=[r for r in flag.evidence if r], source=DecisionSource.RULES)

    def _load_mous(self, ctx: AgentContext) -> tuple[list[MoU], list[str], list[str], list[str]]:
        """Signed MoUs only, with the board entry ids they came from.

        Unsigned drafts contribute nothing to revenue. Both ids are returned
        because they are used for different things: ``mou_id`` names the contract
        in prose, ``entry_id`` is what a ``refs`` citation must resolve to.
        """
        entries = _read(ctx, ZONE_CONTRACTS, kind=KIND_MOU)
        typed: list[tuple[str, MoU]] = []
        for entry in entries:
            model = _as_model(entry, MoU)
            if model is not None:
                typed.append((entry.entry_id, model))
        newest = _latest_model(ctx, ZONE_CONTRACTS, MoU, kind=KIND_MOU)
        if newest is not None and newest.mou_id not in {m.mou_id for _, m in typed}:
            typed.append(("", newest))
        signed: dict[str, tuple[str, MoU]] = {
            m.mou_id: (entry_id, m) for entry_id, m in typed if m.status == "signed"
        }
        return ([m for _, m in signed.values()],
                sorted(signed),
                sorted({m.brand for _, m in signed.values()}),
                sorted({e for e, _ in signed.values() if e}))

    def _load_footfall(self, ctx: AgentContext) -> tuple[int, str]:
        """Footfall from the event profile; 0 with an empty source rather than a guess."""
        profile = _latest_model(ctx, ZONE_EVENT, EventProfile, kind=KIND_EVENT) \
            or _latest_model(ctx, ZONE_EVENT, EventProfile)
        if profile is not None:
            return int(profile.footfall), f"EventProfile {profile.event_id} on the board"
        for entry in _read(ctx, ZONE_EVENT):
            raw = entry.payload.get("footfall") if isinstance(entry.payload, dict) else None
            if isinstance(raw, int) and raw >= 0:
                return raw, f"entry {entry.entry_id} ({entry.kind})"
        return 0, ""

    def _load_findings(self, ctx: AgentContext) -> tuple[list[AuditFinding], list[str]]:
        """A5's findings, reused verbatim. Compliance is never recomputed here."""
        entries = _read(ctx, ZONE_AUDIT, kind=KIND_AUDIT_FINDING)
        if not entries:
            # Shape-based fallback for a board where another agent used a
            # different kind name for the same artefact.
            entries = [e for e in _read(ctx, ZONE_AUDIT)
                       if isinstance(e.payload, dict)
                       and "promise" in e.payload and "fulfilled" in e.payload]
        findings: list[AuditFinding] = []
        ids: list[str] = []
        for entry in entries:
            model = _as_model(entry, AuditFinding)
            if model is not None:
                findings.append(model)
                ids.append(entry.entry_id)
        return findings, ids

    def _report_from(self, ctx: AgentContext, basis: RoiBasis) -> ROIReport:
        """Arithmetic with every input quoted in ``assumptions`` beside it."""
        scratch = ctx.scratch.get(self._SCRATCH) if isinstance(
            ctx.scratch.get(self._SCRATCH), dict) else {}
        spend_basis = str(scratch.get("spend_basis")
                          or self.SPEND_BASIS_OPTIONS[0])
        if spend_basis not in self.SPEND_BASIS_OPTIONS:
            spend_basis = self.SPEND_BASIS_OPTIONS[0]
        lead_basis = str(scratch.get("lead_basis")
                        or self.LEAD_BASIS_OPTIONS[0])
        if lead_basis not in self.LEAD_BASIS_OPTIONS:
            lead_basis = self.LEAD_BASIS_OPTIONS[0]
        compliance_reading = str(scratch.get("compliance_reading")
                                 or self.COMPLIANCE_READING_OPTIONS[0])
        if compliance_reading not in self.COMPLIANCE_READING_OPTIONS:
            compliance_reading = self.COMPLIANCE_READING_OPTIONS[0]
        lesson_focus = str(scratch.get("lesson_focus")
                           or self.LESSON_FOCUS_OPTIONS[0])
        if lesson_focus not in self.LESSON_FOCUS_OPTIONS:
            lesson_focus = self.LESSON_FOCUS_OPTIONS[0]

        total_sponsored = round(sum(m.amount_inr for m in basis.mous), 2)
        if lead_basis == "report_leads_undefined":
            leads = 0
            pipeline_value = 0.0
        else:
            leads = int(round(basis.footfall * basis.conversion_rate))
            pipeline_value = round(leads * basis.value_per_lead_inr, 2)
        if spend_basis == "report_spend_undefined":
            spend = 0.0
        else:
            spend = round(total_sponsored * self.SPEND_RATIO, 2) if total_sponsored > 0 else 0.0
        roi_multiple, roi_note = _safe_roi_multiple(pipeline_value + total_sponsored, spend)
        compliance = _compliance_pct(basis.findings)
        findings = list(basis.findings)

        if lead_basis == "report_leads_undefined":
            conversion_assumption = (
                f"lead_basis=report_leads_undefined per a6.roi.lead_basis: "
                f"estimated_leads reported as 0 (undefined) rather than applying "
                f"conversion_rate={basis.conversion_rate:.4f} to "
                f"footfall={basis.footfall}; provenance: decided basis, NOT measured "
                f"from any board entry.")
            value_assumption = (
                f"value_per_lead_inr={basis.value_per_lead_inr:.2f}; provenance: "
                f"constructor argument on AuditAgent, NOT measured. pipeline_value_inr "
                f"reported as 0.0 (undefined leads) per a6.roi.lead_basis="
                f"report_leads_undefined.")
        else:
            conversion_assumption = (
                f"conversion_rate={basis.conversion_rate:.4f} "
                f"({basis.conversion_rate * 100:.2f}% of footfall becomes a lead) x "
                f"footfall={basis.footfall} => estimated_leads={leads}; provenance: "
                f"constructor argument on AuditAgent, NOT measured from any board entry. "
                f"The previous prototype used an undocumented 3% while its shipped artifact "
                f"claimed 13%.")
            value_assumption = (
                f"value_per_lead_inr={basis.value_per_lead_inr:.2f}; provenance: "
                f"constructor argument on AuditAgent, NOT measured. pipeline_value_inr="
                f"{leads} x {basis.value_per_lead_inr:.2f} = {pipeline_value:.2f}. The "
                f"prototype hardcoded Rs150/lead while its artifact claimed Rs4,500.")
        if spend_basis == "report_spend_undefined":
            spend_assumption = (
                f"spend_basis=report_spend_undefined per a6.roi.spend_basis: "
                f"spend_inr reported as 0.0 (undefined) rather than applying "
                f"spend_ratio={self.SPEND_RATIO}; provenance: decided basis.")
        else:
            spend_assumption = (
                f"spend_ratio={self.SPEND_RATIO} applied to total_sponsored_inr gives "
                f"spend_inr={spend:.2f}; provenance: AuditAgent.SPEND_RATIO, the prototype's "
                f"own ratio, now stated instead of hidden.")
        if findings:
            _base_compliance = (
                f"compliance_pct={compliance:.4f} is fulfilled findings / total findings "
                f"({sum(1 for f in findings if f.fulfilled)}/{len(findings)}) over A5's "
                f"AuditFinding entries {basis.finding_ids or 'none'} in zone `{ZONE_AUDIT}`; "
                f"provenance: reused verbatim from A5, never recomputed.")
            if compliance_reading == "report_summary_mismatch":
                compliance_assumption = (
                    _base_compliance +
                    f" Reading per a6.compliance.reading=report_summary_mismatch: "
                    f"cross-checked against compliance_summary in zone `{ZONE_AUDIT}`; "
                    f"a mismatch raises COMPLIANCE_SUMMARY_MISMATCH in zone "
                    f"`{ZONE_RISK_FLAGS}`.")
            elif compliance_reading == "report_undefined":
                compliance_assumption = (
                    f"compliance_pct={compliance:.4f} with reading "
                    f"a6.compliance.reading=report_undefined: treated as undefined for "
                    f"lesson purposes, recomputed from "
                    f"({sum(1 for f in findings if f.fulfilled)}/{len(findings)}) over A5's "
                    f"AuditFinding entries {basis.finding_ids or 'none'} in zone "
                    f"`{ZONE_AUDIT}`; provenance: reused verbatim from A5.")
            else:
                compliance_assumption = _base_compliance
        else:
            _base_undefined = (
                f"compliance_pct is undefined: zone `{ZONE_AUDIT}` holds no A5 "
                f"AuditFinding, so this report makes no compliance claim. The prototype "
                f"printed compliance_score 100.0 and compliance_pct 0.0 in one run.")
            if compliance_reading == self.COMPLIANCE_READING_OPTIONS[0]:
                compliance_assumption = _base_undefined
            else:
                compliance_assumption = (
                    _base_undefined +
                    f" Reading per a6.compliance.reading={compliance_reading}.")

        assumptions = [
            f"total_sponsored_inr={total_sponsored:.2f} is sum(amount_inr) over "
            f"{len(basis.mous)} signed MoU(s) {basis.mou_ids} "
            f"(brands: {basis.brands or 'none'}); provenance: zone `{ZONE_CONTRACTS}`, "
            f"kind `{KIND_MOU}`. Unsigned drafts are excluded by design.",
            conversion_assumption,
            value_assumption,
            spend_assumption,
            (f"roi_multiple={roi_multiple} computed as (pipeline_value_inr + "
             f"total_sponsored_inr - spend_inr) / spend_inr; {roi_note}"),
            compliance_assumption,
            (f"footfall={basis.footfall}; provenance: "
             + (basis.footfall_source
                if basis.footfall_source
                else "no event profile found on the board, so 0 was used rather "
                     "than an invented figure")
             + "."),
        ]
        if basis.summary_pct is not None and findings:
            assumptions.append(
                f"A5's own compliance_summary reported {basis.summary_pct}%; A6 reports "
                f"{compliance:.4f}% recomputed from the same findings. A difference would "
                f"have raised COMPLIANCE_SUMMARY_MISMATCH in zone `{ZONE_RISK_FLAGS}`."
            )
        assumptions.append(
            f"decided bases: spend_basis={spend_basis}, lead_basis={lead_basis}, "
            f"compliance_reading={compliance_reading}, lesson_focus={lesson_focus}; "
            f"provenance: a6 plan decisions.")
        assumptions.append(f"computed_at={utcnow().isoformat()} for run {ctx.run_id}; "
                           f"no figure above is derived from an unstated constant.")

        return ROIReport(
            report_id=new_id("roi"),
            event_id=ctx.event_id,
            total_sponsored_inr=total_sponsored,
            footfall=basis.footfall,
            estimated_leads=leads,
            spend_inr=spend,
            pipeline_value_inr=pipeline_value,
            roi_multiple=roi_multiple,
            assumptions=assumptions,
            findings=findings,
        )

    def _empty_report(self, ctx: AgentContext, reason: str) -> ROIReport:
        """A valid, empty report. ``ROIReport`` refuses empty ``assumptions``."""
        scratch = ctx.scratch.get(self._SCRATCH) if isinstance(
            ctx.scratch.get(self._SCRATCH), dict) else {}
        spend_basis = str(scratch.get("spend_basis")
                          or self.SPEND_BASIS_OPTIONS[0])
        if spend_basis not in self.SPEND_BASIS_OPTIONS:
            spend_basis = self.SPEND_BASIS_OPTIONS[0]
        lead_basis = str(scratch.get("lead_basis")
                        or self.LEAD_BASIS_OPTIONS[0])
        if lead_basis not in self.LEAD_BASIS_OPTIONS:
            lead_basis = self.LEAD_BASIS_OPTIONS[0]
        compliance_reading = str(scratch.get("compliance_reading")
                                 or self.COMPLIANCE_READING_OPTIONS[0])
        if compliance_reading not in self.COMPLIANCE_READING_OPTIONS:
            compliance_reading = self.COMPLIANCE_READING_OPTIONS[0]
        lesson_focus = str(scratch.get("lesson_focus")
                           or self.LESSON_FOCUS_OPTIONS[0])
        if lesson_focus not in self.LESSON_FOCUS_OPTIONS:
            lesson_focus = self.LESSON_FOCUS_OPTIONS[0]
        return ROIReport(
            report_id=new_id("roi"),
            event_id=ctx.event_id,
            total_sponsored_inr=0.0,
            footfall=0,
            estimated_leads=0,
            spend_inr=0.0,
            pipeline_value_inr=0.0,
            roi_multiple=0.0,
            assumptions=[
                f"no signed MoU was present in zone `{ZONE_CONTRACTS}` when this audit "
                f"ran, so every figure is 0 and no rate was applied",
                "roi_multiple is undefined: with zero spend and zero revenue any "
                "multiple would be fabrication, so it is reported as 0.0 with this note",
                f"scope decision: {reason}",
                f"decided bases: spend_basis={spend_basis}, lead_basis={lead_basis}, "
                f"compliance_reading={compliance_reading}, lesson_focus={lesson_focus}; "
                f"provenance: a6 plan decisions still recorded on the empty path.",
                f"computed_at={utcnow().isoformat()} for run {ctx.run_id}",
            ],
            findings=[],
        )


#: Alias so ``agents/__init__.py`` can export either name.
A6Audit = AuditAgent


# ======================================================================= helpers
def _latest_entry(ctx: AgentContext, zone: str, kind: str) -> BoardEntry | None:
    """Newest entry of ``kind``, tolerating a board without keyword filters."""
    entries = _read(ctx, zone, kind=kind)
    if entries:
        return entries[-1]
    latest = getattr(ctx.board, "latest", None)
    if callable(latest):
        try:
            found = latest(zone, kind)
        except TypeError:
            found = latest(zone)
        except BlackboardError:
            return None
        return found if isinstance(found, BoardEntry) else None
    return None


def _safe_roi_multiple(revenue: float, spend: float) -> tuple[float, str]:
    """ROI multiple with the silent divide removed.

    The prototype wrote ``denom = spend if spend > 0 else 1.0``; zero spend then
    produced a huge, confident, meaningless number. Here zero spend returns 0.0
    with an explicit note, because "undefined" is a reportable result and a fake
    multiple is not — it will be quoted.
    """
    if spend <= 0.0:
        return 0.0, (
            f"UNDEFINED because spend_inr={spend:.2f} makes the denominator zero. The "
            f"prototype substituted 1.0 here and reported a 22500x ROI from zero spend."
        )
    return round((revenue - spend) / spend, 4), (
        f"defined because spend_inr={spend:.2f} > 0"
    )


def _compliance_pct(findings: Sequence[AuditFinding]) -> float:
    """Fulfilled / total over A5's findings — the same maths A5's verdict uses.

    Zero findings means *undefined*, reported as 0.0.
    """
    if not findings:
        return 0.0
    fulfilled = sum(1 for f in findings if getattr(f, "fulfilled", False))
    return round(100.0 * fulfilled / len(findings), 4)


def _lesson_refs(sources: dict[str, list[str]], report_entry_id: str,
                 lesson: Lesson | None) -> list[str]:
    """Board entries the lesson is grounded in, for ``BoardEntry.refs``.

    These are what make the lesson retrievable: A1/A2 next run can read the
    lesson, follow its refs to the MoUs and findings behind it, and see why the
    rule exists rather than trusting it. Only *entry ids* belong here — a domain
    id such as a ``mou_id`` would dangle, and a dangling citation is worse than
    none because it looks verified.
    """
    refs: list[str] = []
    for key in ("mous", "findings", "footfall"):
        for value in (sources or {}).get(key, []) or []:
            if isinstance(value, str) and value and value not in refs:
                refs.append(value)
    if report_entry_id and report_entry_id not in refs:
        refs.append(report_entry_id)
    if not refs and lesson is not None:
        refs.append(lesson.lesson_id)
    return refs
