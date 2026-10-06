"""A4 Contract — draft the MoU from the terms a sponsor actually accepted.

Why this agent is shaped the way it is
--------------------------------------
The previous prototype produced contracts for sponsors who had never agreed::

    def n_contract(self, s):
        if s.proposals:
            mou = draft_mou(s.event, s.proposals[0])   # <-- the FIRST offer
        ...

``proposals[0]`` is whichever offer was priced first, which has nothing to do
with which sponsor said yes. The result was a memorandum of understanding,
rendered to PDF, asserting a payment schedule and logo rights that no human being
had agreed to. A4 therefore:

* finds the accepted sponsor by *reading the board*: a ``Thread`` whose intent is
  ``Intent.YES`` (or whose status is ``closed_won``), naming the ``offer_id`` it
  accepted. The offer it then reads is **that** offer — the newest revision of
  that brand's terms — and never "the first one posted";
* refuses to draft at all when nothing is accepted, returning an observation with
  ``sufficient=False`` and the reason, so the graph can go back and sell rather
  than contract a stranger;
* writes real clause text. The shape of the boilerplate is reused from the
  prototype's ``old/backend/agents/contract.py`` (payment schedule, logo usage
  rights, cancellation notice, deliverable list, signature blocks) because that
  language was the one part of it worth keeping, but the clauses are generated
  from the accepted offer's own amount, deliverables, tier and version — no
  clause asserts a number that is not in the offer;
* renders through ``render_pdf`` when the registry has it, records the honest
  path when the render degrades, and marks the observation degraded rather than
  reporting a document that was never produced;
* gates release. An MoU is ``draft`` until a recorded ``HumanDecision`` says
  otherwise, and only a ``GateOutcome.APPROVE`` moves it to ``approved``.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.base import ReActAgent
from core.errors import (
    AgentError,
    BlackboardError,
    DecisionFailed,
    DecisionUnavailable,
    HumanGateRequired,
    SchemaError,
    ToolFailed,
    ToolTimeout,
    ToolUnavailable,
)
from core.ids import new_id, utcnow
from core.protocols import (
    ActResult,
    AgentContext,
    BoardEntry,
    Observation,
    Plan,
    Reflection,
    Tool,
    ToolResult,
)
from core.schemas import (
    AgentId,
    Approval,
    Decision,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    GateKind,
    GateOutcome,
    HumanDecision,
    HumanGate,
    Intent,
    MoU,
    Offer,
    QuestionType,
    Thread,
)

__all__ = ["ContractAgent", "RENDER_TOOL_NAMES", "PAYMENT_TERMS", "NOTICE_DAYS",
           "EVIDENCE_DAYS"]


# ============================================================================ zones
ZONE_EVENT = "event"
ZONE_OFFERS = "offers"
ZONE_THREADS = "threads"
ZONE_CONTRACTS = "contracts"
ZONE_APPROVALS = "approvals"
KIND_EVENT_PROFILE = "event_profile"
KIND_OFFER = "offer"
KIND_THREAD = "thread"
KIND_MOU = "mou"
KIND_APPROVAL = "approval"
KIND_HUMAN_DECISION = "human_decision"

#: Kind used when recording a pending gate. The shipped zone registry gives
#: ``approvals`` only the kinds ``approval`` and ``human_decision``, so this post
#: is expected to be refused; the refusal is recorded and, as in A3, grants
#: nothing. Release is judged solely from a recorded ``HumanDecision``.
GATE_ENTRY_KIND = "human_gate"

RENDER_TOOL_NAMES: tuple[str, ...] = (
    "render_pdf",
    "make_mou_pdf",
    "render_document",
)

#: Commercial terms A4 states. These are A4's house terms, published as
#: constants so a reader can see exactly what the agent promises on the
#: organiser's behalf — and so that changing them is a visible diff rather than
#: an edit buried in a template.
PAYMENT_ADVANCE_PCT: float = 50.0
PAYMENT_ADVANCE_DAYS: int = 7
PAYMENT_BALANCE_DAYS: int = 7
NOTICE_DAYS: int = 14
EVIDENCE_DAYS: int = 7

PAYMENT_TERMS: str = (
    f"{PAYMENT_ADVANCE_PCT:.0f}% of the sponsorship fee is payable within "
    f"{PAYMENT_ADVANCE_DAYS} days of signature of this MoU; the remaining "
    f"{100.0 - PAYMENT_ADVANCE_PCT:.0f}% is payable within "
    f"{PAYMENT_BALANCE_DAYS} days of the event date."
)

#: Model-decided contract options. The first entry of each tuple is the
#: long-standing house term, so a stub backend returning the first option
#: (or an off-menu answer falling back to it) reproduces the exact text the
#: existing tests assert.
PAYMENT_OPTIONS: tuple[str, ...] = (
    "advance_50_balance_7d",
    "advance_100_upfront",
    "advance_30_balance_30d",
)

EXCLUSIVITY_OPTIONS: tuple[str, ...] = (
    "non_exclusive",
    "category_exclusive",
)

EVIDENCE_WINDOW_OPTIONS: tuple[str, ...] = (
    "evidence_7_days",
    "evidence_14_days",
    "evidence_30_days",
)

MOU_RISK_OPTIONS: tuple[str, ...] = ("yes", "no")


def _payment_text(choice: str) -> str:
    """Payment-schedule clause body for a decided ``a4.payment_terms`` option."""
    if choice == "advance_100_upfront":
        return (
            "100% of the sponsorship fee is payable within "
            f"{PAYMENT_ADVANCE_DAYS} days of signature of this MoU."
        )
    if choice == "advance_30_balance_30d":
        return (
            "30% of the sponsorship fee is payable within "
            f"{PAYMENT_ADVANCE_DAYS} days of signature of this MoU; the remaining "
            "70% is payable within 30 days of the event date."
        )
    return PAYMENT_TERMS


def _exclusivity_text(choice: str) -> str:
    """Exclusivity clause body for a decided ``a4.exclusivity`` option."""
    if choice == "category_exclusive":
        return (
            "Category exclusivity is granted under this MoU. The Organiser "
            "shall not enter into another sponsorship in the Sponsor's category "
            "at the same tier without the Sponsor's prior written consent."
        )
    return (
        "No exclusivity is granted under this MoU. The "
        "Organiser may enter into other sponsorships at the same tier "
        "unless a written exclusivity addendum is signed by both parties."
    )


def _evidence_days(choice: str) -> int:
    """Evidence window in days for a decided ``a4.evidence_window`` option."""
    if choice == "evidence_14_days":
        return 14
    if choice == "evidence_30_days":
        return 30
    return EVIDENCE_DAYS


# =============================================================== module utilities
def _as_text(value: Any) -> str | None:
    """Trimmed non-empty string, or ``None``."""
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _is_offer(payload: Mapping[str, Any]) -> bool:
    """Cheap shape test for an ``offers`` entry before validating it."""
    return "offer_id" in payload and "amount_inr" in payload


def _probe_tool(tool: Tool) -> tuple[bool, str]:
    """Honour the ``Tool.available()`` contract before every call."""
    probe = getattr(tool, "available", None)
    if not callable(probe):
        return True, ""
    try:
        available, reason = probe()
    except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except (TypeError, ValueError) as exc:
        return False, f"available() returned an unusable value: {exc}"
    return bool(available), (str(reason) if reason else "")


def invoke(tool: Tool, args: Mapping[str, Any]) -> ToolResult:
    """Call ``tool.run`` with only the keyword arguments its signature accepts."""
    runner = getattr(tool, "run", None)
    if not callable(runner):
        raise ToolUnavailable(f"tool {getattr(tool, 'name', tool)!r} has no run()")
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        kwargs = dict(args)
    else:
        parameters = signature.parameters.values()
        if any(p.kind is p.VAR_KEYWORD for p in parameters):
            kwargs = dict(args)
        else:
            accepted = {p.name for p in parameters
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            kwargs = {k: v for k, v in args.items() if k in accepted}
    result = runner(**kwargs)
    if isinstance(result, ToolResult):
        return result
    return ToolResult.success(result,
                              source=f"{getattr(tool, 'name', 'unknown_tool')}:unwrapped")


def read_entries(board: Any, zone: str, *, kind: str | None = None,
                 limit: int | None = None) -> list[BoardEntry]:
    """``board.read`` with an unusable board reported as "nothing there"."""
    try:
        return list(board.read(zone, kind=kind, limit=limit))
    except TypeError:
        try:
            return list(board.read(zone))
        except (BlackboardError, SchemaError) as exc:
            raise AgentError(f"board.read({zone!r}) failed: {exc}") from exc
    except (BlackboardError, SchemaError) as exc:
        raise AgentError(f"board.read({zone!r}) failed: {exc}") from exc


def load_all(board: Any, zone: str, model_cls: type[BaseModel], *,
             kind: str | None = None,
             notes: list[str] | None = None) -> list[BaseModel]:
    """Every artefact of ``model_cls`` in ``zone``, oldest first."""
    record = notes if notes is not None else []
    try:
        entries = read_entries(board, zone, kind=kind)
    except AgentError as exc:
        record.append(str(exc))
        return []
    out: list[BaseModel] = []
    for entry in entries:
        try:
            out.append(model_cls.model_validate(entry.payload))
        except ValidationError as exc:
            record.append(
                f"skipped {entry.entry_id} in {zone!r}: not a valid "
                f"{model_cls.__name__}: {exc.error_count()} field error(s)"
            )
    return out


def load_latest(board: Any, zone: str, model_cls: type[BaseModel], *,
                kind: str | None = None,
                notes: list[str] | None = None) -> BaseModel | None:
    """Most recent artefact of ``model_cls`` in ``zone``, or ``None``."""
    latest_model = getattr(board, "latest_model", None)
    if callable(latest_model):
        try:
            return latest_model(zone, model_cls, kind=kind)
        except TypeError:
            try:
                return latest_model(zone, model_cls)
            except (BlackboardError, SchemaError, ValidationError) as exc:
                record = notes if notes is not None else []
                record.append(f"latest_model({zone!r}) unusable: {exc}")
        except (BlackboardError, SchemaError, ValidationError) as exc:
            record = notes if notes is not None else []
            record.append(f"latest_model({zone!r}) unusable: {exc}")
    found = load_all(board, zone, model_cls, kind=kind, notes=notes)
    return found[-1] if found else None


class ContractAgent(ReActAgent):
    """Draft the MoU from the accepted offer, and gate its release."""

    id = AgentId.A4_CONTRACT
    role = "contract: MoU drafted from accepted terms only, released behind a human gate"

    #: An unanswered MoU gate propagates :class:`~core.errors.HumanGateRequired`
    #: by default so the graph can interrupt; set False to stop locally with the
    #: MoU left at ``pending_approval``. Neither path releases anything.
    propagate_gate: bool = True

    def __init__(self, *, step_budget: int = 3, deadline_s: float = 60.0) -> None:
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)

    # ================================================================== planning
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Find the accepted sponsor, or refuse."""
        notes: list[str] = []
        ctx.scratch["a4_notes"] = notes
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        if event is None:
            return self._terminal_plan(
                "await the event profile; a contract needs a named event")

        accepted = self._accepted(ctx, notes)
        if not accepted:
            return self._terminal_plan(
                "no sponsor has accepted: refusing to draft an MoU",
                gap=("no thread on the board records Intent.YES or "
                     "closed_won; A4 will not contract an unaccepted offer"))

        pending = [(thread, offer) for thread, offer in accepted
                   if not self._has_mou(ctx, offer.offer_id, notes)]
        if not pending:
            return self._terminal_plan(
                f"an MoU already exists for every accepted offer "
                f"({len(accepted)} accepted sponsor(s))")

        if len(pending) > 1:
            thread, offer = self._choose(ctx, event, pending)
            confidence, source = self._last_decision(ctx)
        else:
            thread, offer = pending[0]
            confidence, source = 1.0, DecisionSource.RULES
        # Even a single acceptance reasons: decide the payment schedule now so
        # _plan itself issues a model decision. The full clause set is decided
        # again in _act_draft; this hint is recorded for the trace.
        try:
            plan_payment = self._decide_payment_terms(ctx, event, offer, notes)
        except Exception:  # noqa: BLE001 - fallback already handled inside
            plan_payment = PAYMENT_OPTIONS[0]
        ctx.scratch["a4_plan_terms"] = {"payment_terms": plan_payment}
        self._store(ctx, {"mode": "draft", "brand": offer.brand,
                          "offer_id": offer.offer_id, "thread_id": thread.thread_id})
        return Plan(
            goal=(f"draft the {offer.tier} MoU for {offer.brand} from the terms "
                  f"they accepted (INR {offer.amount_inr:,.0f}, offer "
                  f"{offer.offer_id} v{offer.version})"),
            steps=["draft"],
            tool_calls=[{"tool": RENDER_TOOL_NAMES[0],
                         "args": {"mou_id": new_id("mou")}}],
            rationale=(f"thread {thread.thread_id} is the acceptance of record; "
                       f"its cited offer is {offer.offer_id}, and that is the "
                       f"offer whose terms are contracted"),
            confidence=confidence,
            source=source,
        )

    # ========================================================================= act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        mode = plan.steps[0] if plan.steps else "idle"
        if mode != "draft":
            return ActResult(ok=True, output={"mode": mode, "noop": True},
                             observations=[f"A4 stood down: {plan.goal}"])
        return self._act_draft(ctx)

    def _act_draft(self, ctx: AgentContext) -> ActResult:
        """Build the MoU, render it, then ask a human before releasing it."""
        notes: list[str] = list(ctx.scratch.get("a4_notes") or [])
        state = self._state(ctx)
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        found = self._offer_with_entry(ctx, str(state["offer_id"]), notes)
        if event is None or found is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=["the event or the accepted offer vanished "
                                     "from the board mid-step"])
        offer_entry, offer = found
        threads = self._threads_by_id(ctx, notes)
        # The acceptance is named in the run output even though the document does
        # not cite it: the MoU's authority is the offer, and the thread is the
        # evidence that this sponsor said yes to *that* offer.
        thread = threads.get(str(state["thread_id"]))

        mou_id = new_id("mou")
        terms_options = self._decide_contract_terms(ctx, event, offer, notes)
        ctx.scratch["a4_notes"] = notes
        ctx.scratch["a4_terms_options"] = dict(terms_options)
        terms = self.compose_terms(event, offer, terms_options)
        deliverables = list(offer.deliverables)
        document_path, render_note, degraded = self._render(
            ctx, mou_id, event, offer, terms, deliverables)

        mou = MoU(
            mou_id=mou_id,
            event_id=event.event_id,
            brand=offer.brand,
            amount_inr=offer.amount_inr,
            terms=terms,
            deliverables=deliverables,
            status="pending_approval",
            document_path=document_path,
            version=offer.version,
        )
        try:
            draft_entry = self.post(ctx, ZONE_CONTRACTS, KIND_MOU,
                                    mou.model_dump(mode="json"),
                                    refs=[offer_entry.entry_id], confidence=1.0,
                                    source=DecisionSource.RULES)
        except (BlackboardError, SchemaError) as exc:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"could not post the MoU draft: {exc}"])

        errors: list[str] = [n for n in (render_note,) if n]
        try:
            outcome, gate_id = self._require_approval(
                ctx, GateKind.MOU,
                question=(f"Release the {offer.tier} MoU for {offer.brand} at "
                          f"INR {offer.amount_inr:,.0f}?"),
                preview=self._preview(terms),
                action=f"release MoU {mou_id} for {offer.brand}")
        except HumanGateRequired as exc:
            if self.propagate_gate:
                raise
            return ActResult(
                ok=False,
                output={"mode": "draft", "mou_id": mou_id, "brand": offer.brand,
                        "status": "pending_approval",
                        "document_path": document_path},
                errors=[f"MoU release blocked: {exc}"],
                observations=[f"MoU {mou_id} is drafted and parked at "
                              f"pending_approval; nothing was released"],
                degraded=degraded)

        status = {
            GateOutcome.APPROVE: "approved",
            GateOutcome.REJECT: "rejected",
            GateOutcome.REVISE: "pending_approval",
        }[outcome]
        final = mou.model_copy(update={"status": status})
        try:
            final_entry = self.post(ctx, ZONE_CONTRACTS, KIND_MOU,
                                    final.model_dump(mode="json"),
                                    refs=[draft_entry], confidence=1.0,
                                    source=DecisionSource.RULES)
        except (BlackboardError, SchemaError) as exc:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=[f"the MoU was drafted but its {status} state "
                                     f"could not be recorded: {exc}"])

        self._store(ctx, {"mode": "released", "brand": offer.brand,
                          "mou_id": mou_id, "status": status})
        return ActResult(
            ok=True,
            output={"mode": "draft", "mou_id": mou_id, "brand": offer.brand,
                    "tier": offer.tier, "amount_inr": offer.amount_inr,
                    "offer_id": offer.offer_id, "offer_version": offer.version,
                    "thread_id": state.get("thread_id"),
                    "accepted_by_thread": thread.thread_id if thread else None,
                    "status": status, "gate_id": gate_id,
                    "gate_outcome": outcome.value,
                    "document_path": document_path,
                    "render_degraded": degraded,
                    "entry_ids": [draft_entry, final_entry],
                    "deliverables": len(deliverables),
                    "payment_terms": terms_options.get("payment_terms"),
                    "exclusivity": terms_options.get("exclusivity"),
                    "evidence_window": terms_options.get("evidence_window"),
                    "mou_risk": terms_options.get("mou_risk")},
            observations=[f"MoU {mou_id} for {offer.brand}: {status} "
                          f"(offer {offer.offer_id} v{offer.version}, gate "
                          f"{gate_id} answered {outcome.value})"
                          + (f"; document at {document_path}" if document_path
                             else "; no document was rendered")
                          + (f"; {render_note}" if render_note else "")],
            errors=errors,
            degraded=degraded,
        )

    # ==================================================================== observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        mode = plan.steps[0] if plan.steps else "idle"
        output = result.output if isinstance(result.output, Mapping) else {}
        if mode == "halt":
            gaps = list(result.errors)
            if plan.steps and len(plan.steps) > 1:
                gaps.append(str(plan.steps[1]))
            return Observation(summary=plan.goal,
                               facts={"halted": True, "gate_note":
                                      ctx.scratch.get("a4_gate_note")},
                               sufficient=not gaps, gaps=gaps)
        if result.ok:
            return Observation(
                summary=(f"MoU {output.get('mou_id')} for {output.get('brand')} is "
                         f"{output.get('status')} from offer "
                         f"{output.get('offer_id')} v{output.get('offer_version')}"
                         + ("" if output.get("document_path")
                            else " (no document rendered)")),
                facts=dict(output), sufficient=True)
        return Observation(
            summary=f"A4 could not complete the MoU for "
                    f"{output.get('brand') or 'the accepted sponsor'}",
            facts=dict(output) if output else {"mode": mode},
            sufficient=False,
            gaps=list(result.errors) or ["MoU step failed with no reason recorded"])

    # =================================================================== reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        if obs.sufficient:
            return None
        facts = obs.facts or {}
        if "refusing" in obs.summary:
            return Reflection(
                lesson_trigger=obs.gaps[0] if obs.gaps else "no acceptance on record",
                correction=("the contract follows the acceptance, never the "
                            "posting order; proposals[0] is not a counterparty"),
                rule=("draft an MoU only for an offer named by a Thread with "
                      "Intent.YES or status=closed_won; otherwise report the gap"),
                confidence=0.95,
            )
        if facts.get("render_degraded") or not facts.get("document_path"):
            return Reflection(
                lesson_trigger="the MoU exists as text but not as a document",
                correction=("a document path that was never written must not be "
                            "reported as if it were"),
                rule=("record document_path only from a render tool that returned "
                      "status=OK; otherwise leave it null, mark the observation "
                      "degraded, and say which path was attempted"),
                confidence=0.8,
            )
        return None

    # ==================================================================== clauses
    def compose_terms(self, event: EventProfile, offer: Offer,
                      terms_options: Mapping[str, Any] | None = None) -> str:
        """The MoU body, generated from the accepted offer's own fields.

        Every commercial number here is read from ``offer``: the amount, the tier,
        the version, the deliverables. The house terms (advance percentage,
        notice period, evidence window) are the module constants above. Nothing is
        inherited from an unrelated proposal and nothing is invented.

        ``terms_options`` carries the model-decided clause variants
        (``payment_terms``, ``exclusivity``, ``evidence_window``). When ``None``
        or when an entry is unknown, the long-standing house text is used, so a
        caller that never asks the model reproduces the exact historical MoU.
        """
        opts = dict(terms_options) if isinstance(terms_options, Mapping) else {}
        payment_choice = str(opts.get("payment_terms") or PAYMENT_OPTIONS[0])
        if payment_choice not in PAYMENT_OPTIONS:
            payment_choice = PAYMENT_OPTIONS[0]
        exclusivity_choice = str(opts.get("exclusivity") or EXCLUSIVITY_OPTIONS[0])
        if exclusivity_choice not in EXCLUSIVITY_OPTIONS:
            exclusivity_choice = EXCLUSIVITY_OPTIONS[0]
        evidence_choice = str(opts.get("evidence_window") or EVIDENCE_WINDOW_OPTIONS[0])
        if evidence_choice not in EVIDENCE_WINDOW_OPTIONS:
            evidence_choice = EVIDENCE_WINDOW_OPTIONS[0]
        payment_terms = _payment_text(payment_choice)
        exclusivity_terms = _exclusivity_text(exclusivity_choice)
        evidence_days = _evidence_days(evidence_choice)
        deliverables = "\n".join(f"   {i + 1}. {item}"
                                for i, item in enumerate(offer.deliverables))
        return (
            f"MEMORANDUM OF UNDERSTANDING\n"
            f"Reference: offer {offer.offer_id} version {offer.version} "
            f"({offer.tier} tier)\n"
            f"Organiser: {event.name}\n"
            f"Event: {event.name} at {event.location} on {event.date}\n"
            f"Expected audience: {event.audience} (approx. {event.footfall:,} "
            f"attendees)\n"
            f"Sponsor: {offer.brand}\n"
            f"\n"
            f"1. SPONSORSHIP FEE. The Sponsor shall pay the Organiser "
            f"INR {offer.amount_inr:,.0f} (Indian Rupees "
            f"{int(round(offer.amount_inr)):d}) for the {offer.tier} tier of "
            f"sponsorship described in clause 5.\n"
            f"\n"
            f"2. PAYMENT SCHEDULE. {payment_terms}\n"
            f"\n"
            f"3. LOGO USAGE RIGHTS. The Sponsor's logo may appear only on the "
            f"collateral listed in clause 5 and only for the duration of the "
            f"event. The Organiser will not alter the logo's proportions or "
            f"colours, will not imply endorsement beyond the agreed tier, and "
            f"will remove the logo from materials published after the event on "
            f"request. Any use of the Organiser's name or marks by the Sponsor "
            f"requires the Organiser's prior written consent.\n"
            f"\n"
            f"4. CANCELLATION AND NOTICE. Either party may terminate this MoU by "
            f"{NOTICE_DAYS} days' written notice. If the Organiser cancels, any "
            f"amount already paid is refunded less costs already incurred and "
            f"non-refundable third-party charges. If the Sponsor cancels, amounts "
            f"paid are non-refundable. If the event is postponed beyond 90 days "
            f"or cancelled, the parties will renegotiate in good faith and any "
            f"unspent advance will be returned.\n"
            f"\n"
            f"5. DELIVERABLES. The Organiser shall provide:\n"
            f"{deliverables}\n"
            f"   Photo or written evidence of each delivered item will be "
            f"shared within {evidence_days} days of the event.\n"
            f"\n"
            f"6. EXCLUSIVITY. {exclusivity_terms}\n"
            f"\n"
            f"7. STATUS. This MoU records the parties' agreement on the terms "
            f"above. It becomes binding on signature by both parties.\n"
            f"\n"
            f"SIGNATURES\n"
            f"   For the Organiser ({event.name}): ______________________  "
            f"Name: ________________  Date: ____________\n"
            f"   For the Sponsor ({offer.brand}): ______________________  "
            f"Name: ________________  Date: ____________\n"
        )

    @staticmethod
    def _preview(terms: str) -> str:
        """What a human is shown before release: the whole body, not a summary.

        A gate that shows a summary and releases the document is not a gate.
        """
        return terms

    # ==================================================================== render
    def _render(self, ctx: AgentContext, mou_id: str, event: EventProfile,
                offer: Offer, terms: str, deliverables: Sequence[str]
                ) -> tuple[str | None, str, bool]:
        """Render the document. Returns ``(path, note, degraded)``.

        ``document_path`` is only ever a string a tool actually returned. If the
        registry has no renderer, or the renderer fails, the path is ``None`` and
        the note says so — the MoU still exists as validated text on the board,
        which is a real artefact, and the trace says the PDF does not exist.
        """
        tool, name = self._pick_render_tool(ctx)
        if tool is None:
            return None, (f"no renderer in the registry (looked for "
                          f"{list(RENDER_TOOL_NAMES)}); the MoU exists as text "
                          f"on the board only"), True
        available, reason = _probe_tool(tool)
        if not available:
            return None, f"renderer {name!r} unavailable: {reason}", True
        out_name = f"{mou_id}.pdf"
        try:
            result = invoke(tool, {
                "title": f"Memorandum of Understanding — {offer.brand}",
                "text": terms,
                "content": terms,
                "lines": terms.splitlines(),
                "out_path": out_name,
                "out_name": out_name,
                "filename": out_name,
                "data": {"mou_id": mou_id, "event": event.name,
                         "brand": offer.brand, "tier": offer.tier,
                         "amount": offer.amount_inr, "terms": terms,
                         "deliverables": list(deliverables)},
                "brand": offer.brand, "amount": offer.amount_inr,
                "amount_inr": offer.amount_inr,
            })
        except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
            return None, (f"renderer {name!r} raised {type(exc).__name__}: {exc}; "
                          f"the MoU exists as text only"), True
        if not result.ok:
            return None, (f"renderer {name!r} did not produce a document "
                          f"(status={result.status.value}): "
                          f"{result.reason or 'no reason given'}"), True
        path = _as_text(result.data)
        if path is None and isinstance(result.data, Mapping):
            path = _as_text(result.data.get("path") or result.data.get("out_path"))
        if not path:
            return None, (f"renderer {name!r} reported success but returned no "
                          f"path; no document can be claimed"), True
        return path, (f"rendered by {name!r} [source={result.source or 'unknown'}]"
                      if result.source else ""), bool(result.degraded)

    # ============================================================ the human gate
    def _require_approval(self, ctx: AgentContext, kind: GateKind, question: str,
                          preview: str, action: str) -> tuple[GateOutcome, str]:
        """Raise the release gate and return the human's recorded answer."""
        gate = HumanGate(
            gate_id=new_id("gat"),
            kind=kind,
            event_id=ctx.event_id,
            run_id=ctx.run_id,
            question=question,
            payload_preview=preview[:600],
            options=[GateOutcome.APPROVE, GateOutcome.REJECT, GateOutcome.REVISE],
            raised_at=utcnow(),
        )
        gate_entry, refusal = self._record_gate(ctx, gate)
        ctx.scratch["a4_gate_note"] = (
            f"pending {kind.value} gate {gate.gate_id} for {action}"
            + (f"; the board refused to store the gate itself ({refusal})"
               if refusal else ""))
        decision = self._find_human_decision(ctx, gate.gate_id, kind)
        if decision is None:
            raise HumanGateRequired(
                f"{kind.value} gate {gate.gate_id} is unanswered and no approval "
                f"record exists for {action}; the MoU stays pending_approval"
            )
        ctx.scratch["a4_gate_note"] = (
            f"{kind.value} gate {gate.gate_id} answered {decision.outcome.value} "
            f"by {decision.decided_by}"
            + (f"; the board refused to store the gate itself ({refusal})"
               if refusal else ""))
        if decision.outcome is GateOutcome.APPROVE:
            self._record_approval(ctx, gate, decision, action)
        return decision.outcome, gate.gate_id

    def _record_gate(self, ctx: AgentContext, gate: HumanGate
                     ) -> tuple[str | None, str | None]:
        try:
            entry_id = self.post(ctx, ZONE_APPROVALS, GATE_ENTRY_KIND,
                                 gate.model_dump(mode="json"),
                                 source=DecisionSource.RULES)
            return entry_id, None
        except (BlackboardError, SchemaError) as exc:
            return None, str(exc)

    def _find_human_decision(self, ctx: AgentContext, gate_id: str,
                             kind: GateKind) -> HumanDecision | None:
        """Exact ``gate_id`` match first; propagated graph approval on resume.

        See ``OutreachAgent._find_human_decision``: the graph answers a different
        gate id in ``gates/`` while A4 waits on its own id in ``approvals/``.
        A propagated ``approve`` for the same kind, handed via
        ``ctx.scratch["graph_approvals"]``, authorises the release on resume and
        is labelled as propagated in the instruction.
        """
        try:
            entries = read_entries(ctx.board, ZONE_APPROVALS, kind=KIND_HUMAN_DECISION)
        except AgentError:
            entries = []
        for entry in reversed(entries):
            try:
                decision = HumanDecision.model_validate(entry.payload)
            except ValidationError:
                continue
            if decision.gate_id == gate_id and decision.kind is kind:
                return decision
        try:
            propagated = ctx.scratch.get("graph_approvals") or []
        except Exception:
            propagated = []
        for item in reversed(list(propagated)):
            try:
                p_kind = str(item.get("kind") or "")
                p_outcome = str(item.get("outcome") or "")
            except Exception:
                continue
            if p_kind.rsplit(".", 1)[-1] != kind.value:
                continue
            if p_outcome.rsplit(".", 1)[-1] != GateOutcome.APPROVE.value:
                continue
            try:
                outcome = GateOutcome(p_outcome.rsplit(".", 1)[-1])
            except ValueError:
                continue
            return HumanDecision(
                gate_id=gate_id,
                kind=kind,
                outcome=outcome,
                decided_by=str(item.get("decided_by") or "unknown"),
                instruction=("propagated from graph gate "
                             f"{item.get('gate_id')}: {item.get('instruction') or ''}".strip()),
            )
        return None

    def _record_approval(self, ctx: AgentContext, gate: HumanGate,
                         decision: HumanDecision, action: str) -> str | None:
        approval = Approval(
            approval_id=new_id("apr"),
            gate_id=gate.gate_id,
            kind=gate.kind,
            outcome=decision.outcome,
            decided_by=decision.decided_by,
            action_taken=action[:240],
            at=utcnow(),
        )
        try:
            return self.post(ctx, ZONE_APPROVALS, KIND_APPROVAL,
                             approval.model_dump(mode="json"),
                             source=DecisionSource.RULES)
        except (BlackboardError, SchemaError):
            return None

    # ================================================================== decisions
    def _choose(self, ctx: AgentContext, event: EventProfile,
                pending: list[tuple[Thread, Offer]]) -> tuple[Thread, Offer]:
        """Which accepted sponsor to contract first, when several have accepted.

        A decision rather than an ``offers[0]``: with several acceptances the
        order is a commercial judgement (who pays fastest, who is easiest to
        serve), not a property of the board's append order.
        """
        options = [offer.brand for _, offer in pending]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=("Several sponsors have accepted. Which agreement should "
                      "Paytriq put into an MoU first?"),
            question_type=QuestionType.CHOICE,
            state={
                "event": event.name,
                "accepted": [{"brand": offer.brand, "tier": offer.tier,
                              "amount_inr": offer.amount_inr,
                              "version": offer.version,
                              "deliverables": offer.deliverables,
                              "thread_id": thread.thread_id}
                             for thread, offer in pending],
            },
            options=options,
            instructions=("Pick the agreement that is clearest to document and "
                          "least likely to be disputed."),
            asked_by=self.id,
            decision_point="a4.choose_accepted",
        )
        decision = self._decide(ctx, request)
        ctx.scratch["a4_last_decision"] = {
            "confidence": round(decision.confidence, 4),
            "source": decision.source.value,
        }
        chosen = (decision.choice or "").strip()
        for thread, offer in pending:
            if offer.brand == chosen:
                return thread, offer
        return pending[0]

    def _last_decision(self, ctx: AgentContext) -> tuple[float, DecisionSource]:
        state = ctx.scratch.get("a4_last_decision")
        if not isinstance(state, Mapping):
            return 1.0, DecisionSource.RULES
        source_text = str(state.get("source") or DecisionSource.RULES.value)
        try:
            source = DecisionSource(source_text)
        except ValueError:
            source = DecisionSource.RULES
        return float(state.get("confidence") or 1.0), source

    def _decide_with_default(self, ctx: AgentContext, request: DecisionRequest,
                             default: str, notes: list[str]) -> str:
        """Ask ``ctx.decide`` with a recorded fallback to ``default``.

        Backend failures and off-menu answers fall back to the house term
        (the first option) so a single-acceptance draft still reasons but never
        blocks on the model.
        """
        try:
            decision = ctx.decide(request)
        except (DecisionUnavailable, DecisionFailed, ToolUnavailable,
                AgentError) as exc:
            notes.append(
                f"decision backend unavailable at "
                f"{request.decision_point!r} ({type(exc).__name__}: {exc}); "
                f"defaulted to {default!r}")
            return default
        except Exception as exc:  # noqa: BLE001 - a backend may raise anything
            notes.append(
                f"decision backend failed at {request.decision_point!r} "
                f"({type(exc).__name__}: {exc}); defaulted to {default!r}")
            return default
        if not isinstance(decision, Decision):
            notes.append(
                f"ctx.decide returned {type(decision).__name__} for "
                f"{request.decision_point!r}, expected Decision; "
                f"defaulted to {default!r}")
            return default
        choice = (decision.choice or "").strip()
        if choice in request.options:
            return choice
        # NOUL vocabulary tolerance: System One answers true/false, rules
        # backends answer yes/no. Map explicitly rather than rejecting.
        lowered = choice.lower()
        if lowered in ("yes", "true", "1") and "yes" in request.options:
            return "yes"
        if lowered in ("no", "false", "0") and "no" in request.options:
            return "no"
        notes.append(
            f"decision returned {choice!r}, which is not one of "
            f"{list(request.options)}; defaulted to {default!r}")
        return default

    def _decide_payment_terms(self, ctx: AgentContext, event: EventProfile,
                              offer: Offer, notes: list[str]) -> str:
        """Which payment schedule the MoU states — a model decision, not a constant."""
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"What payment schedule should the MoU for {offer.brand} "
                      f"state for INR {offer.amount_inr:,.0f}?"),
            question_type=QuestionType.CHOICE,
            state={
                "brand": offer.brand,
                "tier": offer.tier,
                "amount_inr": offer.amount_inr,
                "event": event.name,
                "current_terms": PAYMENT_TERMS,
            },
            options=list(PAYMENT_OPTIONS),
            instructions=(
                "Choose the payment schedule that is clearest to enforce. "
                "The first option is the standard 50% advance, 50% balance term."),
            asked_by=self.id,
            decision_point="a4.payment_terms",
        )
        return self._decide_with_default(ctx, request, PAYMENT_OPTIONS[0], notes)

    def _decide_exclusivity(self, ctx: AgentContext, event: EventProfile,
                            offer: Offer, notes: list[str]) -> str:
        """Whether the MoU grants exclusivity — a model decision, not a default."""
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"Should the MoU for {offer.brand} grant exclusivity?"),
            question_type=QuestionType.CHOICE,
            state={
                "brand": offer.brand,
                "tier": offer.tier,
                "amount_inr": offer.amount_inr,
                "event": event.name,
            },
            options=list(EXCLUSIVITY_OPTIONS),
            instructions=(
                "Choose non_exclusive unless the tier and fee clearly justify "
                "a category lock-out. Exclusivity limits future sponsorships."),
            asked_by=self.id,
            decision_point="a4.exclusivity",
        )
        return self._decide_with_default(ctx, request, EXCLUSIVITY_OPTIONS[0], notes)

    def _decide_evidence_window(self, ctx: AgentContext, event: EventProfile,
                                offer: Offer, notes: list[str]) -> str:
        """How quickly evidence is due after the event — a model decision."""
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"Within how many days of {event.name} should evidence of "
                      f"{offer.brand}'s deliverables be due?"),
            question_type=QuestionType.CHOICE,
            state={
                "brand": offer.brand,
                "deliverables": list(offer.deliverables),
                "event": event.name,
                "current_days": EVIDENCE_DAYS,
            },
            options=list(EVIDENCE_WINDOW_OPTIONS),
            instructions=(
                "Choose the evidence window that balances sponsor assurance "
                "with organiser effort. The first option is the 7-day house term."),
            asked_by=self.id,
            decision_point="a4.evidence_window",
        )
        return self._decide_with_default(ctx, request, EVIDENCE_WINDOW_OPTIONS[0],
                                         notes)

    def _decide_mou_risk(self, ctx: AgentContext, event: EventProfile,
                         offer: Offer, notes: list[str]) -> str:
        """Is this MoU safe to draft? Recorded, never blocking."""
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"Is the MoU for {offer.brand} at INR "
                      f"{offer.amount_inr:,.0f} safe to draft?"),
            question_type=QuestionType.NOUL,
            state={
                "brand": offer.brand,
                "tier": offer.tier,
                "amount_inr": offer.amount_inr,
                "version": offer.version,
                "event": event.name,
            },
            options=list(MOU_RISK_OPTIONS),
            instructions=(
                "Answer yes if the accepted offer is clear enough to document. "
                "This is recorded only and never blocks drafting."),
            asked_by=self.id,
            decision_point="a4.mou_risk",
        )
        return self._decide_with_default(ctx, request, MOU_RISK_OPTIONS[0], notes)

    def _decide_contract_terms(self, ctx: AgentContext, event: EventProfile,
                               offer: Offer, notes: list[str]) -> dict[str, str]:
        """All clause-variant decisions for one draft, with safe defaults.

        The payment schedule reuses the ``a4.payment_terms`` decision taken in
        ``_plan`` (stored as ``ctx.scratch["a4_plan_terms"]``) when it names a
        valid option for this offer. That is what makes the plan-time decision
        load-bearing instead of cosmetic: the draft branches on it by skipping
        a second, duplicate model call. Any other clause is still decided here,
        and a missing or stale hint falls back to deciding fresh.
        """
        hint: Any = None
        try:
            stored = ctx.scratch.get("a4_plan_terms")
            hint = stored.get("payment_terms") if isinstance(stored, Mapping) else None
        except Exception:
            hint = None
        if isinstance(hint, str) and hint in PAYMENT_OPTIONS:
            payment = hint
            notes.append(f"reused plan payment_terms hint {hint!r} from "
                         f"a4.payment_terms instead of deciding twice")
        else:
            if hint is not None:
                notes.append(f"plan payment_terms hint {hint!r} is not a valid "
                             f"option; deciding the schedule fresh")
            payment = self._decide_payment_terms(ctx, event, offer, notes)
        exclusivity = self._decide_exclusivity(ctx, event, offer, notes)
        evidence = self._decide_evidence_window(ctx, event, offer, notes)
        risk = self._decide_mou_risk(ctx, event, offer, notes)
        return {"payment_terms": payment, "exclusivity": exclusivity,
                "evidence_window": evidence, "mou_risk": risk}

    def _decide(self, ctx: AgentContext, request: DecisionRequest) -> Decision:
        """The only route to a model in this agent."""
        try:
            decision = ctx.decide(request)
        except (DecisionUnavailable, DecisionFailed, ToolUnavailable) as exc:
            raise AgentError(
                f"decision backend failed at {request.decision_point!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(decision, Decision):
            raise AgentError(
                f"ctx.decide returned {type(decision).__name__} for "
                f"{request.decision_point!r}, expected core.schemas.Decision"
            )
        return decision

    # ====================================================================== board
    def _threads_by_id(self, ctx: AgentContext, notes: list[str]) -> dict[str, Thread]:
        try:
            entries = read_entries(ctx.board, ZONE_THREADS, kind=KIND_THREAD)
        except AgentError as exc:
            notes.append(str(exc))
            return {}
        latest: dict[str, Thread] = {}
        for entry in entries:
            try:
                thread = Thread.model_validate(entry.payload)
            except ValidationError:
                continue
            latest[thread.thread_id] = thread
        return latest

    def _accepted(self, ctx: AgentContext, notes: list[str]
                  ) -> list[tuple[Thread, Offer]]:
        """``(accepting thread, accepted offer)`` pairs, newest acceptance first.

        The offer comes from ``Thread.offer_id`` — the exact revision the sponsor
        answered — and is resolved to the newest offer for that brand only when
        the thread predates it. Positional selection (``proposals[0]``) never
        appears: it is the bug, not a shortcut.
        """
        threads = self._threads_by_id(ctx, notes)
        offers = load_all(ctx.board, ZONE_OFFERS, Offer, kind=KIND_OFFER, notes=notes)
        if not threads or not offers:
            return []
        by_id = {offer.offer_id: offer for offer in offers}
        latest_by_brand: dict[str, Offer] = {}
        for offer in offers:
            latest_by_brand[offer.brand] = offer

        pairs: list[tuple[Thread, Offer]] = []
        for thread in reversed(list(threads.values())):
            if thread.intent is not Intent.YES and thread.status != "closed_won":
                continue
            offer = by_id.get(thread.offer_id or "")
            if offer is None:
                note = latest_by_brand.get(thread.brand)
                if note is None:
                    notes.append(
                        f"thread {thread.thread_id} records an acceptance for "
                        f"{thread.brand} but names no offer that is on the board")
                    continue
                offer = note
            pairs.append((thread, offer))
        return pairs

    def _has_mou(self, ctx: AgentContext, offer_id: str, notes: list[str]) -> bool:
        """Has an MoU already been written for the terms on offer?

        Keyed on brand *and* amount, because those are what a MoU asserts. A new
        offer id at the same price is the same deal and needs no second document;
        a revised amount does. ``offer_id`` is accepted for signature symmetry
        with the other helpers and deliberately not used to match, because a
        board entry's ``refs`` hold *entry* ids, not offer ids, and comparing the
        two would be a type confusion dressed up as a lookup.
        """
        try:
            entries = read_entries(ctx.board, ZONE_CONTRACTS, kind=KIND_MOU)
        except AgentError as exc:
            notes.append(str(exc))
            return False
        accepted = [Offer.model_validate(e.payload) for e in
                    read_entries(ctx.board, ZONE_OFFERS, kind=KIND_OFFER)
                    if _is_offer(e.payload)]
        target = next((o for o in accepted if o.offer_id == offer_id), None)
        if target is None:
            notes.append(f"offer {offer_id!r} is not in the offers zone; "
                         "cannot tell whether an MoU already exists")
            return False
        for entry in entries:
            try:
                mou = MoU.model_validate(entry.payload)
            except ValidationError:
                continue
            if mou.brand == target.brand and mou.amount_inr == target.amount_inr:
                return True
        return False

    def _offer_with_entry(self, ctx: AgentContext, offer_id: str, notes: list[str]
                          ) -> tuple[BoardEntry, Offer] | None:
        try:
            entries = read_entries(ctx.board, ZONE_OFFERS, kind=KIND_OFFER)
        except AgentError as exc:
            notes.append(str(exc))
            return None
        for entry in reversed(entries):
            try:
                offer = Offer.model_validate(entry.payload)
            except ValidationError:
                continue
            if offer.offer_id == offer_id:
                return entry, offer
        notes.append(f"offer {offer_id!r} is not in the offers zone")
        return None

    # ==================================================================== helpers
    @staticmethod
    def _pick_render_tool(ctx: AgentContext, *,
                          purpose: str = "MoU PDF rendering",
                          decision_point: str = "a4.select_render_tool"
                          ) -> tuple[Tool | None, str | None]:
        """Model-directed renderer choice with function schemas in state."""
        from agents.tool_selection import select_tool

        tool, name, _ = select_tool(
            ctx, purpose, list(RENDER_TOOL_NAMES), decision_point)
        return tool, name

    def _terminal_plan(self, reason: str, gap: str | None = None) -> Plan:
        steps = ["halt"]
        if gap:
            steps.append(gap)
        return Plan(goal=reason, steps=steps, rationale=reason, confidence=1.0,
                    source=DecisionSource.RULES, stop=True)

    @staticmethod
    def _state(ctx: AgentContext) -> dict[str, Any]:
        value = ctx.scratch.get("a4_state")
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _store(ctx: AgentContext, state: Mapping[str, Any]) -> None:
        ctx.scratch["a4_state"] = dict(state)
