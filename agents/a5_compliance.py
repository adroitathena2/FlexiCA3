"""A5 Compliance — the only agent in Paytriq that holds veto authority.

What this agent is for
----------------------
A4 produces an MoU; A5 decides whether the promises inside it can honestly be
reported as fulfilled. If they cannot, the pipeline must stop rather than ship a
sponsorship report nobody can defend. That is why ``Severity.BLOCKING`` exists
in ``core.schemas`` and why A5 is the agent that raises it.

The defect this file exists to make impossible
---------------------------------------------
The previous prototype verified deliverables with::

    found = any(check_url_for_logo(u, promise).get("found") for u in urls)

and ``check_url_for_logo`` ended with::

    found = any(k in url.lower() for k in keywords) or "sponsor" in url.lower()

so a promise counted as "fulfilled" whenever *the URL contained the word
"sponsor"*. No page was ever fetched. The result was a shipped artifact reading
``compliance_pct: 100.0`` while the ROI section of the same file admitted zero
verified deliverables. Two invariants here make that unrepresentable:

1. **A URL is not evidence.** :func:`_extract_content` accepts only *content*
   returned by a tool. A boolean flag, a link, or a bare ``found=True`` becomes
   ``EvidenceRecord.inspectable == False`` with an explicit, recorded reason.
   This is the check the prototype lacked.
2. **Absence of evidence is not fulfilment.** Even if the decision model answers
   "yes" with 0.99 confidence, :meth:`ComplianceAgent._verdict_for` forces
   ``fulfilled = False`` when nothing was inspectable. The model's judgement is
   bounded by a policy floor; the floor is what stops a hallucinated "yes" from
   becoming a compliance percentage.

Every conditional decision — which phase to run, and the verdict on each
deliverable — goes through ``ctx.decide``. There is no keyword matching on free
text anywhere in this file and no direct model client.

The veto
--------
Findings and flags are written to the board *before* the veto, so the reason
survives the exception. The ReAct loop then raises
:class:`~core.errors.HumanGateRequired` when any ``BLOCKING`` flag was raised.
That is a control-flow signal, not a crash: the graph catches it, calls
``interrupt()``, and a human decides. Returning quietly is exactly how the old
system shipped a 100% compliance score nobody had checked.

One note on ``ReActAgent.flag_risk``: it mints its own ``flag_id`` and discards
the caller's object, which would leave an in-memory ``RiskFlag`` whose id does
not exist on the board. Since an unciteable flag is the same class of defect as
the compliance bug above, A5 builds the ``RiskFlag`` once and posts that exact
payload through :meth:`ReActAgent.post`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import ValidationError

from core.config import Settings, get_settings
from core.errors import (
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
    ToolResult,
)
from core.schemas import (
    AgentId,
    AuditFinding,
    Decision,
    DecisionRequest,
    DecisionSource,
    Deliverable,
    MoU,
    QuestionType,
    RiskFlag,
    Severity,
    ToolStatus,
)

from .base import ReActAgent

__all__ = [
    "ComplianceAgent",
    "A5Compliance",
    "ComplianceVerdict",
    "EvidenceRecord",
    "SEVERITY_BY_CODE",
    "ZONE_CONTRACTS",
    "ZONE_RISK_FLAGS",
    "ZONE_AUDIT",
    "KIND_MOU",
    "KIND_AUDIT_FINDING",
    "KIND_COMPLIANCE_SUMMARY",
]

_M = TypeVar("_M")


# --------------------------------------------------------------------------- zones
def _resolve_zone(attr: str, fallback: str) -> str:
    """Prefer ``blackboard.zones``' own constant when that module is present.

    ``blackboard/`` is being written concurrently with this file, so a hard
    import would leave A5 unimportable until it lands. Three spellings are tried
    because the constant may be named ``AUDIT`` or ``ZONE_AUDIT``. Once
    ``blackboard.zones`` exists this becomes a plain lookup and the two
    definitions cannot drift apart.
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


ZONE_CONTRACTS = _resolve_zone("CONTRACTS", "contracts")
ZONE_RISK_FLAGS = _resolve_zone("RISK_FLAGS", "risk_flags")
ZONE_AUDIT = _resolve_zone("AUDIT", "audit")

#: ``kind`` strings A5 writes, as constants so A6 and the tests can name them.
KIND_MOU = "mou"
KIND_AUDIT_FINDING = "audit_finding"
KIND_COMPLIANCE_SUMMARY = "compliance_summary"
KIND_RISK_FLAG = "risk_flag"


# ------------------------------------------------------------------ board helpers
def _read(ctx: AgentContext, zone: str, *, kind: str | None = None,
          limit: int | None = None) -> list[BoardEntry]:
    """``board.read`` with a fallback for boards taking fewer keywords.

    The protocol promises ``read(zone, *, kind, limit)``; a hand-written or
    stubbed board may honour only ``read(zone)``. Rather than fail the agent
    because of an adapter detail, filter locally. A zone fault yields ``[]`` — a
    missing zone is a gap to report, not a crash.
    """
    board = ctx.board
    try:
        try:
            return list(board.read(zone, kind=kind, limit=limit))
        except TypeError:
            entries = list(board.read(zone))
            if kind is not None:
                entries = [e for e in entries if e.kind == kind]
            if limit is not None:
                entries = entries[-limit:]
            return entries
    except BlackboardError:
        return []


def _as_model(entry: BoardEntry, model_cls: type[_M]) -> _M | None:
    """Reconstruct a typed artefact from a board payload.

    Strict validation first. If that fails, one *projected* retry keeps only the
    fields the model declares, because another agent may have posted extra
    diagnostic keys alongside the artefact. Two failures mean the entry simply is
    not this artefact, and the caller records a gap rather than inventing a value.
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
                  *, kind: str | None = None) -> tuple[_M | None, list[str]]:
    """Newest typed entry in ``zone``, plus notes about which path was used.

    Prefers the board's typed helpers (``latest_model``/``get_model``) and falls
    back to ``read``/``latest`` plus local reconstruction. Both paths are
    exercised by the tests because ``blackboard/`` is written in parallel and A5
    must work with either.
    """
    notes: list[str] = []
    getter = getattr(ctx.board, "latest_model", None)
    if callable(getter):
        try:
            found = getter(zone, model_cls, kind=kind)
        except TypeError:
            found = getter(zone, model_cls)
        except (BlackboardError, SchemaError) as exc:
            notes.append(f"latest_model({zone}) failed: {type(exc).__name__}: {exc}")
            found = None
        if isinstance(found, model_cls):
            return found, notes
        if found is not None:
            notes.append(f"latest_model({zone}) returned {type(found).__name__}; "
                         f"falling back to read() + reconstruction")
    for entry in reversed(_read(ctx, zone, kind=kind, limit=25)):
        model = _as_model(entry, model_cls)
        if model is not None:
            notes.append(f"board helper latest_model unavailable for {zone}; "
                         f"{model_cls.__name__} reconstructed from read()")
            return model, notes
    return None, notes


def _history(ctx: AgentContext) -> list[BoardEntry]:
    """Full board history, tolerating a board that does not implement it."""
    getter = getattr(ctx.board, "history", None)
    if not callable(getter):
        return []
    try:
        return list(getter())
    except BlackboardError:
        return []


# ----------------------------------------------------------------------- evidence
@dataclass(slots=True)
class EvidenceRecord:
    """What A5 could actually *inspect* for one promise.

    ``inspectable`` is the whole point. ``True`` means a tool returned content a
    human could read. It is never inferred from a URL, a filename, or a ``found``
    boolean — that inference is the exact defect being fixed.
    """

    promise: str
    brand: str
    candidate_url: str | None = None
    tool: str = ""
    status: str = ToolStatus.UNAVAILABLE.value
    retrievable: bool = False
    inspectable: bool = False
    content: str = ""
    reason: str = ""
    transient: bool = False
    entry_ids: list[str] = field(default_factory=list)

    def as_fact(self) -> dict[str, Any]:
        """JSON-safe projection for observations and trace attributes."""
        return {
            "promise": self.promise,
            "brand": self.brand,
            "candidate_url": self.candidate_url,
            "tool": self.tool,
            "status": self.status,
            "retrievable": self.retrievable,
            "inspectable": self.inspectable,
            "transient": self.transient,
            "reason": self.reason,
            "entry_ids": list(self.entry_ids),
        }

    @classmethod
    def from_fact(cls, fact: dict[str, Any] | None) -> EvidenceRecord:
        """Rebuild from :meth:`as_fact`, ignoring unknown keys."""
        data = fact if isinstance(fact, dict) else {}
        known = {k: v for k, v in data.items() if k in cls.__slots__}
        known.setdefault("promise", "")
        known.setdefault("brand", "")
        return cls(**known)


#: Payload keys whose *string* values count as inspectable content.
_CONTENT_KEYS: tuple[str, ...] = (
    "content", "text", "body", "excerpt", "html", "snippet", "caption",
    "captions", "observations", "matches", "found_in", "evidence_text",
    "page_text", "description", "alt",
)

#: Keys that look like evidence but never are: flags, booleans, and links.
_FLAG_ONLY_KEYS: tuple[str, ...] = (
    "found", "ok", "matched", "match", "present", "verified", "url", "link",
    "source_url", "evidence_url", "status", "confidence",
)

_EXCERPT_LIMIT = 600


def _extract_content(data: Any) -> tuple[str, str]:
    """Pull readable content out of a tool payload.

    Returns ``(content, reason)``. ``content`` is empty whenever the payload
    holds nothing but flags and links, and ``reason`` then names the keys that
    were seen — so a reviewer can distinguish "the page said no" from "the tool
    never fetched anything". That distinction is the whole fix.
    """
    if data is None:
        return "", "tool returned no payload at all, so nothing was inspected"

    def walk(node: Any, depth: int = 0) -> list[str]:
        if depth > 3 or node is None or isinstance(node, (bool, int, float)):
            return []
        if isinstance(node, str):
            return [node] if node.strip() else []
        if isinstance(node, list):
            out: list[str] = []
            for item in node[:20]:
                out.extend(walk(item, depth + 1))
            return out
        if isinstance(node, dict):
            out = []
            for key in _CONTENT_KEYS:
                if key in node:
                    out.extend(walk(node[key], depth + 1))
            return out
        return []

    seen = sorted(str(k) for k in data) if isinstance(data, dict) else []
    content = " ".join(walk(data)).strip()
    if content:
        return content[:_EXCERPT_LIMIT], ""
    flags = [k for k in seen if k in _FLAG_ONLY_KEYS]
    names = ", ".join(flags or seen) or "no keys"
    return "", (f"tool returned only non-evidence keys ({names}); a URL, a boolean, "
                f"or a status flag is not proof that a promise was fulfilled")


def _pick_tool(ctx: AgentContext, *,
               purpose: str = "evidence retrieval for deliverable verification",
               decision_point: str = "a5.select_evidence_tool"
               ) -> tuple[Any | None, str]:
    """Model-directed evidence-tool choice with function schemas in state.

    ``tools/browser.py`` is owned by another module and may register a
    ``verify_evidence`` tool under any of several names. Exact names are tried
    first; only then a name-substring match, iterating in sorted order so two
    candidate tools can never produce different runs. The ordered candidates
    are put to the decision model (see ``agents/tool_selection.py``) with
    their function schemas; an unreachable backend or off-menu answer falls
    back to the first candidate. Finding no tool is a first-class outcome,
    not an error: A5 then reports "unverifiable", which is the honest answer
    and raises a BLOCKING flag.
    """
    from agents.tool_selection import select_tool

    tools = getattr(ctx, "tools", None) or {}
    ordered: list[str] = []
    for name in ("verify_evidence", "evidence", "browser", "fetch", "verify"):
        if name in tools and name not in ordered:
            ordered.append(name)
    for name in sorted(tools):
        if (any(hint in name.lower()
                for hint in ("evidence", "verify", "browser", "fetch"))
                and name not in ordered):
            ordered.append(name)
    if not ordered:
        return None, ""
    tool, name, _ = select_tool(ctx, purpose, ordered, decision_point)
    if tool is None or name is None:
        return None, ""
    return tool, name


# --------------------------------------------------------------- severity policy
#: Documented rather than inline, because the veto is the most consequential
#: thing A5 does and a reviewer must be able to audit the policy rather than
#: reverse-engineer it.
SEVERITY_BY_CODE: dict[str, Severity] = {
    # Signed MoU + nothing inspectable: the organisation cannot substantiate a
    # promise it has already signed. This is the "fake 100%" scenario.
    "EVIDENCE_UNVERIFIABLE": Severity.BLOCKING,
    # Signed MoU + evidence retrieved that does not show the promise was met.
    "PROMISE_NOT_FULFILLED": Severity.BLOCKING,
    # The MoU asserts status=fulfilled while carrying no evidence.
    "UNSUPPORTED_FULFILMENT_CLAIM": Severity.BLOCKING,
    # The evidence tool could not be reached at all. Unverifiable, environmental,
    # and it still stops a published compliance claim.
    "EVIDENCE_TOOL_UNAVAILABLE": Severity.BLOCKING,
    # Nothing to verify. Worth surfacing; not a veto on its own.
    "NO_DELIVERABLES_SPECIFIED": Severity.MEDIUM,
    # Compliance ran before signature: verified, but nothing is binding yet.
    "MOU_NOT_SIGNED": Severity.MEDIUM,
}


# ============================================================================ agent
class ComplianceAgent(ReActAgent):
    """Verify promised deliverables against real evidence; veto when they fail."""

    id = AgentId.A5_COMPLIANCE
    role = "Compliance: deliverable verification with veto authority"

    _SCRATCH = "a5_compliance"

    def __init__(self, *, step_budget: int = 2, deadline_s: float = 60.0,
                 settings: Settings | None = None) -> None:
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)
        self.settings = settings or get_settings()

    # ---------------------------------------------------------------- scratch
    def _state(self, ctx: AgentContext) -> dict[str, Any]:
        """Per-run memo carried in ``ctx.scratch``.

        The base loop hands ``_act`` only a :class:`Plan`, so the phase travels
        through the run-scoped scratch dict rather than instance attributes —
        instance attributes would leak state between events that share one agent
        object.
        """
        scratch = ctx.scratch.setdefault(self._SCRATCH, {})
        scratch.setdefault("phase", "gather_evidence")
        scratch.setdefault("mous", [])
        scratch.setdefault("targets", [])
        scratch.setdefault("notes", [])
        return scratch

    def _retryable(self, state: dict[str, Any]) -> list[str]:
        """Promises whose evidence fetch failed for an environmental reason."""
        out: list[str] = []
        for target in state.get("targets", []):
            promise = target.get("promise")
            record = EvidenceRecord.from_fact(target.get("record"))
            if promise and record.transient:
                out.append(f"{record.brand}/{promise}")
        return out

    # ------------------------------------------------------------------ plan
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Choose the next phase. The choice itself goes through ``decide``."""
        state = self._state(ctx)
        retryable = self._retryable(state)
        if not state["targets"]:
            options = ["gather_evidence", "finalise_verdict"]
            default = "gather_evidence"
        else:
            options = ["finalise_verdict", "retry_evidence"]
            default = "retry_evidence" if retryable else "finalise_verdict"
        request = DecisionRequest(
            request_id=new_id("dec"),
            question="Which phase should compliance take next, given the evidence so far?",
            question_type=QuestionType.CHOICE,
            state={
                "deliverables_in_scope": len(state["targets"]),
                "retryable_evidence_failures": len(retryable),
                "mous_in_scope": len(state["mous"]),
                "last_observation": obs.summary if obs is not None else "",
            },
            options=options,
            rubric=[
                "gather_evidence: no promise has been examined yet",
                "retry_evidence: evidence was unreachable and may be reachable now",
                "finalise_verdict: every promise has been attempted at least once",
            ],
            instructions=(
                "Prefer finalise_verdict once each promise has had one attempt; a "
                "promise with no candidate URL cannot be rescued by retrying."
            ),
            asked_by=self.id,
            decision_point="a5.compliance.phase",
        )
        choice, notes = self._ask(ctx, request, options, default=default)
        phase = choice if choice in options else default
        # Normalise legacy short names ("gather"/"retry"/"finalise") so a
        # decision backend or test returning the short form still dispatches.
        _aliases = {"gather": "gather_evidence", "retry": "retry_evidence",
                    "finalise": "finalise_verdict", "finalize": "finalise_verdict"}
        phase = _aliases.get(phase, phase)
        state["phase"] = phase
        state["notes"].extend(notes)
        _, tool_name = _pick_tool(ctx)
        return Plan(
            goal=f"compliance phase={phase}",
            steps=[f"phase:{phase}", f"deliverables:{len(state['targets'])}"],
            tool_calls=[] if phase == "finalise_verdict" else [{"tool": tool_name or "none"}],
            rationale=(f"{len(state['targets'])} deliverable(s) in scope, "
                       f"{len(retryable)} retryable evidence failure(s); "
                       f"decision chose {phase}"),
            confidence=1.0,
            source=DecisionSource.RULES,
            stop=False,
        )

    # ------------------------------------------------------------------ act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        state = self._state(ctx)
        phase = str(state.get("phase", "gather_evidence"))
        # Accept both the canonical long names and the legacy short names so a
        # stored phase from an older run still dispatches correctly.
        if phase in ("gather", "gather_evidence"):
            return self._gather(ctx, state)
        if phase in ("retry", "retry_evidence"):
            return self._retry(ctx, state)
        return self._finalise(ctx, state)

    def _gather(self, ctx: AgentContext, state: dict[str, Any]) -> ActResult:
        """Enumerate MoUs, then attempt evidence for every promised deliverable."""
        tool, tool_name = _pick_tool(ctx)
        errors: list[str] = []
        observations: list[str] = []

        mous, notes = self._load_mous(ctx)
        state["mous"] = [m.model_dump(mode="json") for m in mous]
        state["notes"].extend(notes)
        if not mous:
            observations.append(
                "no MoU on the board; nothing to verify, so compliance is undefined "
                "rather than 100%"
            )

        targets: list[dict[str, Any]] = []
        for mou in mous:
            promises = _promises_of(mou)
            if not promises:
                targets.append({
                    "mou_id": mou.mou_id, "brand": mou.brand, "mou_status": mou.status,
                    "promise": None, "declared_status": None, "candidates": [],
                    "record": None,
                })
                continue
            for promise, declared in promises:
                candidates = self._evidence_candidates(ctx, mou, promise)
                record = self._inspect(ctx, tool, tool_name, mou.brand, promise, candidates)
                targets.append({
                    "mou_id": mou.mou_id,
                    "brand": mou.brand,
                    "mou_status": mou.status,
                    "promise": promise,
                    "declared_status": declared,
                    "candidates": [c["url"] for c in candidates],
                    "record": record.as_fact(),
                })
                if record.transient:
                    errors.append(f"evidence fetch failed for {mou.brand}/{promise}: "
                                  f"{record.reason}")
        state["targets"] = targets
        if tool is None:
            observations.append("no evidence tool is registered for this run; every "
                                "promise is reported as unverifiable")
        return ActResult(
            ok=True,
            output={"targets": len(targets), "tool": tool_name},
            observations=observations,
            errors=errors,
            degraded=tool is None or bool(errors),
        )

    def _retry(self, ctx: AgentContext, state: dict[str, Any]) -> ActResult:
        """Re-attempt only fetches that failed for environmental reasons.

        A promise with no candidate URL is not retried: there is nothing to
        fetch, and retrying would spend budget to reach the same answer.
        """
        tool, tool_name = _pick_tool(ctx)
        errors: list[str] = []
        retried = 0
        for target in state["targets"]:
            promise = target.get("promise")
            if not promise:
                continue
            record = EvidenceRecord.from_fact(target.get("record"))
            if not record.transient:
                continue
            retried += 1
            candidates = [{"url": url, "entry_id": None, "note": "retry"}
                          for url in target.get("candidates", [])]
            fresh = self._inspect(ctx, tool, tool_name, record.brand, str(promise), candidates)
            target["record"] = fresh.as_fact()
            if fresh.transient:
                errors.append(f"retry still failing for {record.brand}/{promise}: "
                              f"{fresh.reason}")
        return ActResult(
            ok=True,
            output={"retried": retried, "tool": tool_name},
            observations=[f"re-attempted evidence for {retried} promise(s)"],
            errors=errors,
            degraded=bool(errors) or tool is None,
        )

    def _finalise(self, ctx: AgentContext, plan: Plan) -> ActResult:
        """Publish findings and flags, then veto if anything is blocking."""
        del plan  # the phase lives in ctx.scratch; the plan is carried for the trace
        verdict = self.audit(ctx)
        if verdict.blocking:
            # A BLOCKING flag must stop downstream progress. HumanGateRequired is
            # the control-flow signal the graph turns into an interrupt(); every
            # finding and flag is already on the board, so the halt is auditable
            # rather than merely announced.
            raise HumanGateRequired(verdict.veto_message())
        return ActResult(
            ok=True,
            output=verdict.summary(),
            observations=list(verdict.notes),
            errors=[f.code for f in verdict.flags],
            degraded=verdict.degraded,
        )

    # --------------------------------------------------------------- observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        state = self._state(ctx)
        phase = str(state.get("phase", "gather_evidence"))
        sufficient = True
        gaps: list[str] = []
        if phase in ("gather", "gather_evidence", "retry", "retry_evidence"):
            # Gathering alone never completes the pass: the verdict is published
            # by finalise_verdict. Reporting sufficient here would stop the ReAct
            # loop before the veto can fire (the old gather/gather_evidence name
            # mismatch hid this by dispatching straight to finalise).
            retryable = self._retryable(state)
            if retryable:
                sufficient = False
                gaps = [f"evidence not retrievable yet for: {', '.join(retryable)}"]
            else:
                sufficient = False
                gaps = [f"evidence gathered in phase={phase}; verdict not yet finalised"]
        targets = state.get("targets", [])
        promises = [t for t in targets if t.get("promise")]
        return Observation(
            summary=(f"compliance phase={phase}: {len(promises)} promise(s) across "
                     f"{len(state['mous'])} MoU(s)"),
            facts={
                "phase": phase,
                "mous": [m.get("mou_id") for m in state["mous"] if isinstance(m, dict)],
                "promises": len(promises),
                "tool": _pick_tool(ctx)[1] or "none",
            },
            sufficient=sufficient,
            gaps=gaps,
        )

    # --------------------------------------------------------------- reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """Verbal lesson about evidence quality, returned to the caller.

        Not posted to the board: the base contract says the caller persists
        reflections, and A5's durable artefact is the ``compliance_summary``
        entry that A6 reads.
        """
        targets = [t for t in self._state(ctx).get("targets", []) if t.get("promise")]
        if not targets:
            return Reflection(
                lesson_trigger="no MoU to verify",
                correction="A previous run reported a compliance figure with no promise "
                           "to check, which reads as 100% but means nothing.",
                rule=(f"Read zone `{ZONE_CONTRACTS}` kind `{KIND_MOU}` before claiming any "
                      f"compliance number; with zero verifiable promises report "
                      f"compliance as undefined (0.0), never as 100%."),
                confidence=1.0,
            )
        unverifiable = sum(
            1 for t in targets if not EvidenceRecord.from_fact(t.get("record")).inspectable
        )
        return Reflection(
            lesson_trigger=(f"{unverifiable} of {len(targets)} promises had no "
                            f"inspectable evidence"),
            correction=("substring matches in URLs and tool booleans are not evidence; "
                        "the prototype accepted them and reported 100% compliance"),
            rule=(f"Require tool-returned content per promise before writing a verdict; "
                  f"a promise without content is fulfilled=False and raises "
                  f"EVIDENCE_UNVERIFIABLE in zone `{ZONE_RISK_FLAGS}`."),
            confidence=1.0 if unverifiable else 0.6,
        )

    # ================================================================ public API
    def audit(self, ctx: AgentContext) -> ComplianceVerdict:
        """Run one full compliance pass and publish it. Never raises.

        Callers that want the verdict without the veto (tests, a dry run, the
        audit API) use this. :meth:`_finalise` wraps it and raises when the
        verdict holds a BLOCKING flag, so there is exactly one implementation of
        the verdict and the two paths cannot disagree.
        """
        state = self._state(ctx)
        if not state["targets"]:
            state["phase"] = "gather_evidence"
            self._gather(ctx, state)
        return self._verify_and_publish(ctx, state)

    # ============================================================== internals
    def _load_mous(self, ctx: AgentContext) -> tuple[list[MoU], list[str]]:
        """Latest MoU per brand, whatever its status.

        Status is not filtered here: compliance runs against a draft MoU too, and
        says so via the ``MOU_NOT_SIGNED`` flag. Filtering would have hidden that
        the pipeline reached verification before signature.
        """
        notes: list[str] = []
        newest, latest_notes = _latest_model(ctx, ZONE_CONTRACTS, MoU, kind=KIND_MOU)
        notes.extend(latest_notes)
        typed: list[MoU] = [m for m in (_as_model(e, MoU) for e in _read(ctx, ZONE_CONTRACTS, kind=KIND_MOU))
                            if m is not None]
        if newest is not None and newest.mou_id not in {m.mou_id for m in typed}:
            notes.append(f"latest_model reported MoU {newest.mou_id} absent from read(); "
                         f"including the typed helper result")
            typed.append(newest)
        if not typed and not notes:
            notes.append(f"zone `{ZONE_CONTRACTS}` holds no {KIND_MOU} entry")
        seen: dict[str, MoU] = {}
        for model in typed:
            seen[model.brand] = model  # a later version supersedes an earlier one
        return list(seen.values()), notes

    def _evidence_candidates(self, ctx: AgentContext, mou: MoU,
                             promise: str) -> list[dict[str, Any]]:
        """Collect candidate evidence *addresses* for one promise.

        Sources, in priority order: the MoU's own ``Deliverable`` payload, then
        board entries that declare evidence for the same promise or for the same
        brand. A candidate is an address to check — never evidence itself, which
        is why this method's output can only ever lead to
        ``inspectable=False`` until a tool returns content.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(url: Any, entry_id: str | None, note: str) -> None:
            if not isinstance(url, str) or not url.strip() or url.strip() in seen:
                return
            seen.add(url.strip())
            out.append({"url": url.strip(), "entry_id": entry_id, "note": note})

        for deliverable in _deliverable_payloads(mou):
            if _same_deliverable(str(deliverable.get("promise", "")), promise):
                add(deliverable.get("evidence_url"), deliverable.get("_entry_id"),
                    str(deliverable.get("evidence_note") or "declared in the MoU payload"))
        if out:
            return out

        for entry in _history(ctx):
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            promised = str(payload.get("promise") or "")
            brand = payload.get("brand")
            same_promise = bool(promised) and _same_deliverable(promised, promise)
            # ``MoU.deliverables`` is ``list[str]`` in the frozen core, so a
            # promise cannot carry its own evidence address; another agent (A3
            # collects them from the sponsor, A4 from the signed pack) publishes
            # them on their own entry instead. Accept both shapes.
            mapped = payload.get("deliverable_evidence")
            if isinstance(mapped, dict):
                for name, value in mapped.items():
                    if _same_deliverable(str(name), promise):
                        add(value, entry.entry_id,
                            f"declared as {name!r} in {entry.zone}/{entry.kind}")
            # An entry without a promise qualifies only if it names this brand and
            # declares evidence explicitly. Anything looser would let an
            # unrelated link (a deck, an MoU PDF) count as deliverable evidence.
            qualified = same_promise or (brand is not None and str(brand) == mou.brand)
            if not qualified or promised and not same_promise:
                continue
            for key in ("evidence_url", "evidence_urls"):
                value = payload.get(key)
                if isinstance(value, list):
                    for item in value:
                        add(item, entry.entry_id, f"declared in {entry.zone}/{entry.kind}")
                else:
                    add(value, entry.entry_id, f"declared in {entry.zone}/{entry.kind}")
            if same_promise:
                break
        return out

    def _inspect(self, ctx: AgentContext, tool: Any | None, tool_name: str,
                 brand: str, promise: str,
                 candidates: Sequence[dict[str, Any]]) -> EvidenceRecord:
        """Attempt evidence retrieval. Never fabricates a result."""
        if not candidates:
            return EvidenceRecord(
                promise=promise, brand=brand, status="no_candidate",
                reason="no evidence address was found on the blackboard for this promise",
            )
        if tool is None:
            return EvidenceRecord(
                promise=promise, brand=brand, candidate_url=str(candidates[0]["url"]),
                status=ToolStatus.UNAVAILABLE.value,
                reason="no evidence tool is registered in ctx.tools for this run",
                transient=True,
            )

        entry_ids = [str(c["entry_id"]) for c in candidates if c.get("entry_id")]
        best_failure: EvidenceRecord | None = None
        for candidate in candidates:
            url = str(candidate["url"])
            try:
                result: ToolResult = tool.run(url=url, promise=promise, brand=brand)
            except (ToolUnavailable, ToolTimeout, ToolFailed) as exc:
                failure = EvidenceRecord(
                    promise=promise, brand=brand, candidate_url=url, tool=tool_name,
                    status=ToolStatus.UNAVAILABLE.value,
                    reason=f"{type(exc).__name__}: {exc}", transient=True,
                    entry_ids=entry_ids,
                )
                best_failure = best_failure or failure
                continue
            except Exception as exc:  # noqa: BLE001 - a tool must never crash the agent
                # Recorded rather than swallowed: the promise becomes
                # unverifiable and that fact is written to the board.
                failure = EvidenceRecord(
                    promise=promise, brand=brand, candidate_url=url, tool=tool_name,
                    status=ToolStatus.FAILED.value,
                    reason=f"tool raised {type(exc).__name__}: {exc}", transient=True,
                    entry_ids=entry_ids,
                )
                best_failure = best_failure or failure
                continue

            status = getattr(result, "status", ToolStatus.FAILED)
            ok = bool(getattr(result, "ok", False))
            if not ok or status not in (ToolStatus.OK, ToolStatus.CACHED):
                status_value = status.value if isinstance(status, ToolStatus) else str(status)
                failure = EvidenceRecord(
                    promise=promise, brand=brand, candidate_url=url, tool=tool_name,
                    status=status_value, retrievable=False,
                    reason=(f"evidence tool returned ok={ok} status={status_value}: "
                            f"{getattr(result, 'reason', '') or 'no reason given'}"),
                    transient=status in (ToolStatus.UNAVAILABLE, ToolStatus.FAILED),
                    entry_ids=entry_ids,
                )
                best_failure = best_failure or failure
                continue

            status_value = status.value if isinstance(status, ToolStatus) else str(status)
            evidence_url = getattr(result, "evidence_url", None) or url
            content, why_empty = _extract_content(getattr(result, "data", None))
            if not content:
                # Fetched something, inspected nothing. This is the exact branch
                # the old prototype missed, where a URL containing "sponsor" was
                # scored as a fulfilled promise.
                failure = EvidenceRecord(
                    promise=promise, brand=brand, candidate_url=str(evidence_url),
                    tool=tool_name, status=status_value, retrievable=True,
                    inspectable=False, reason=f"fetched {evidence_url} but {why_empty}",
                    entry_ids=entry_ids,
                )
                best_failure = best_failure or failure
                continue
            return EvidenceRecord(
                promise=promise, brand=brand, candidate_url=str(evidence_url),
                tool=tool_name, status=status_value, retrievable=True, inspectable=True,
                content=content, reason="tool returned readable content",
                entry_ids=entry_ids,
            )

        if best_failure is None:  # pragma: no cover - guarded by the empty check above
            return EvidenceRecord(promise=promise, brand=brand,
                                  reason="evidence attempt produced no result")
        return best_failure

    def _ask(self, ctx: AgentContext, request: DecisionRequest,
             options: Sequence[str], *, default: str) -> tuple[str, list[str]]:
        """Call ``ctx.decide`` and tolerate an unreachable backend.

        A decision backend that is down must not stop compliance: the agent
        degrades to its documented default and the run is marked degraded.
        Returns ``(choice, notes)``.
        """
        try:
            decision: Decision = ctx.decide(request)
        except (DecisionUnavailable, DecisionFailed) as exc:
            return default, [f"decision backend unavailable ({exc}); defaulted to {default}"]
        choice = str(decision.choice or default)
        if options and choice not in options:
            return default, [f"decision returned {choice!r}, which is not one of "
                             f"{list(options)}; defaulted to {default}"]
        return choice, []

    def _verdict_for(self, ctx: AgentContext, brand: str, promise: str,
                     record: EvidenceRecord) -> tuple[AuditFinding, bool]:
        """One ``noul`` question per promise, bounded by the evidence floor.

        The model answers "is this promise fulfilled by the evidence?"; the code
        then enforces that no evidence means no fulfilment. Both facts land in
        ``AuditFinding.note`` so they cannot disagree silently, and the floor is
        recorded explicitly when it overrides the model.
        """
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(f"Is the promise \"{promise}\" from {brand} fulfilled by the "
                      f"evidence that was actually retrieved?"),
            question_type=QuestionType.NOUL,
            state={
                "brand": brand,
                "promise": promise,
                "evidence_url": record.candidate_url,
                "evidence_status": record.status,
                "evidence_inspectable": record.inspectable,
                "evidence_excerpt": record.content[:400],
                "evidence_note": record.reason,
                "tool": record.tool,
            },
            instructions=(
                "Answer no unless the retrieved content itself shows the promise was "
                "delivered. A URL, a page title, or a boolean flag is not evidence. "
                "When evidence_inspectable is false the answer is no."
            ),
            asked_by=self.id,
            decision_point="a5.compliance.deliverable_verdict",
        )
        degraded = False
        try:
            decision = ctx.decide(request)
        except (DecisionUnavailable, DecisionFailed):
            decision = None
            degraded = True
            answer, source, confidence, probabilities = "unavailable", DecisionSource.RULES, 0.0, {}
        else:
            answer = str(decision.choice or "unknown").strip().lower()
            source = decision.source
            confidence = decision.confidence
            probabilities = dict(decision.probabilities)

        # A NOUL question is answered with a boolean. A System One decision model
        # returns "true"/"false"; a rules backend may return "yes"/"no". Accept
        # both vocabularies explicitly rather than testing for one string, because
        # a model that correctly answers "this promise was fulfilled" as `true`
        # must not then be read as having answered "no".
        _affirmative = answer in ("true", "yes", "1")
        _negative = answer in ("false", "no", "0")

        if not record.inspectable:
            # The floor. No content was read, so nothing can be called fulfilled.
            fulfilled = False
        elif _affirmative:
            fulfilled = True
        elif _negative:
            fulfilled = False
        else:
            # An unparseable choice, but a calibrated distribution is still usable.
            # Look under both vocabularies: a System One model keys a noul answer
            # "true", a rules backend keys it "yes". Reading only one of them would
            # make a correct high-confidence fulfilment look like "no distribution
            # available" and silently default it to not-fulfilled.
            p_true = probabilities.get("true")
            p_yes = probabilities.get("yes")
            p = p_true if p_true is not None else p_yes
            fulfilled = float(p) >= 0.5 if p is not None else False

        note = [
            f"decision={answer}",
            f"source={source.value if isinstance(source, DecisionSource) else source}",
            f"confidence={confidence:.3f}",
            f"inspectable={record.inspectable}",
            f"evidence={record.reason}",
        ]
        if not record.inspectable and answer == "yes":
            note.append("POLICY FLOOR: the model answered yes but no inspectable "
                        "evidence exists, so the promise is recorded as NOT fulfilled")
        if probabilities:
            note.append("probabilities=" + json.dumps(
                {k: round(float(v), 4) for k, v in probabilities.items()}, sort_keys=True))
        if degraded:
            note.append("degraded: decision backend unavailable; treated as not fulfilled")

        finding = AuditFinding(
            promise=promise,
            fulfilled=fulfilled,
            evidence_url=record.candidate_url,
            confidence=0.0 if degraded else round(min(1.0, confidence), 6),
            source=source if isinstance(source, DecisionSource) else DecisionSource.RULES,
            note="; ".join(note),
        )
        return finding, degraded

    def _verify_and_publish(self, ctx: AgentContext,
                            state: dict[str, Any]) -> ComplianceVerdict:
        """Decide every promise, write findings + flags + summary, return verdict."""
        findings: list[AuditFinding] = []
        flags: list[RiskFlag] = []
        finding_entry_ids: list[str] = []
        degraded = False
        notes: list[str] = list(state.get("notes", []))
        targets = state.get("targets", [])
        mou_ids: list[str] = []

        for target in targets:
            brand = str(target.get("brand") or "unknown")
            mou_id = str(target.get("mou_id") or "")
            if mou_id and mou_id not in mou_ids:
                mou_ids.append(mou_id)
            promise = target.get("promise")
            record = EvidenceRecord.from_fact(target.get("record"))
            declared = target.get("declared_status")
            # An unreachable tool degrades the run even though the verdict is
            # still produced: "unverifiable" and "verified" are not the same claim.
            degraded = degraded or record.transient

            if not promise:
                flags.append(self._build_flag(
                    ctx, brand, "NO_DELIVERABLES_SPECIFIED",
                    f"MoU {mou_id} for {brand} lists no deliverables, so compliance "
                    f"cannot be established for it",
                    evidence=[mou_id],
                ))
                continue

            finding, was_degraded = self._verdict_for(ctx, brand, str(promise), record)
            degraded = degraded or was_degraded
            findings.append(finding)
            entry_id = self._post(
                ctx, ZONE_AUDIT, KIND_AUDIT_FINDING, finding.model_dump(mode="json"),
                refs=[r for r in record.entry_ids if r],
                confidence=finding.confidence, source=finding.source,
            )
            if entry_id is not None:
                finding_entry_ids.append(entry_id)

            signed = str(target.get("mou_status")) == "signed"
            if not finding.fulfilled:
                blocking_context = signed or str(target.get("mou_status")) == "approved"
                if declared == "fulfilled" and not record.inspectable:
                    code = "UNSUPPORTED_FULFILMENT_CLAIM"
                    message = (f"MoU {mou_id} for {brand} marks \"{promise}\" as "
                               f"fulfilled but no inspectable evidence supports it")
                elif record.transient:
                    code = "EVIDENCE_TOOL_UNAVAILABLE"
                    message = (f"could not check \"{promise}\" for {brand}: "
                               f"{record.reason}")
                elif not record.inspectable:
                    code = "EVIDENCE_UNVERIFIABLE"
                    message = (f"no inspectable evidence for \"{promise}\" ({brand}): "
                               f"{record.reason}")
                else:
                    code = "PROMISE_NOT_FULFILLED"
                    message = (f"evidence for \"{promise}\" ({brand}) was retrieved and "
                               f"does not show the promise was delivered")
                flag = self._build_flag(ctx, brand, code, message,
                                        evidence=[entry_id, *record.entry_ids])
                if not blocking_context:
                    # An unfulfilled promise in an unsigned MoU is a note, not a
                    # veto: nothing is binding yet.
                    flag = flag.model_copy(update={"severity": Severity.MEDIUM})
                flags.append(flag)

        # The exact RiskFlag payload is posted, so the flag objects handed back in
        # the verdict are the same artefacts a reviewer will find on the board.
        flag_entry_ids = [
            entry_id for entry_id in (
                self._post(
                    ctx, ZONE_RISK_FLAGS, KIND_RISK_FLAG, flag.model_dump(mode="json"),
                    refs=list(flag.evidence), source=DecisionSource.RULES,
                )
                for flag in flags
            ) if entry_id is not None
        ]

        verdict = ComplianceVerdict(
            event_id=ctx.event_id,
            findings=findings,
            flags=flags,
            finding_entry_ids=finding_entry_ids,
            flag_entry_ids=flag_entry_ids,
            notes=notes,
            degraded=degraded,
            mou_ids=mou_ids,
        )
        verdict.summary_entry_id = self._post(
            ctx, ZONE_AUDIT, KIND_COMPLIANCE_SUMMARY, verdict.summary(),
            refs=list(verdict.finding_entry_ids) + list(verdict.flag_entry_ids),
            confidence=verdict.confidence, source=DecisionSource.RULES,
        )
        verdict.notes = list(state.get("notes", notes))
        return verdict

    def _post(self, ctx: AgentContext, zone: str, kind: str, payload: dict[str, Any],
              *, refs: list[str] | None = None, confidence: float = 1.0,
              source: DecisionSource = DecisionSource.RULES) -> str | None:
        """:meth:`ReActAgent.post` with a recorded refusal instead of an exception.

        The board validates the zone/kind/author triple on every write, and that
        registry belongs to another module. If it refuses a post, the run must
        still produce a verdict *and record that the write did not happen* —
        dropping an artefact silently would be the same class of defect as
        inventing one.
        """
        try:
            return self.post(ctx, zone, kind, payload, refs=refs,
                             confidence=confidence, source=source)
        except (BlackboardError, SchemaError) as exc:
            self._state(ctx)["notes"].append(
                f"board refused {zone}/{kind} from {self.id.value}: "
                f"{type(exc).__name__}: {exc}")
            return None

    def _build_flag(self, ctx: AgentContext, brand: str, code: str, message: str,
                    *, evidence: Sequence[str]) -> RiskFlag:
        """Build a :class:`RiskFlag` carrying the policy severity for ``code``."""
        return RiskFlag(
            flag_id=new_id("rsk"),
            event_id=ctx.event_id,
            brand=brand,
            severity=SEVERITY_BY_CODE.get(code, Severity.MEDIUM),
            code=code,
            message=message,
            evidence=[e for e in evidence if e],
            raised_by=self.id,
            raised_at=utcnow(),
        )


# ======================================================================= verdict
@dataclass(slots=True)
class ComplianceVerdict:
    """One compliance pass: the findings, the flags, and one percentage.

    ``compliance_pct`` is computed **once**, here, from the same
    :class:`AuditFinding` list that goes on the board. A6 reads that list instead
    of recomputing anything, which is why the two agents cannot print different
    compliance numbers for the same run — the defect behind
    ``compliance_score: 100.0`` and ``compliance_pct: 0.0`` appearing together.
    """

    event_id: str
    findings: list[AuditFinding] = field(default_factory=list)
    flags: list[RiskFlag] = field(default_factory=list)
    finding_entry_ids: list[str] = field(default_factory=list)
    flag_entry_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    degraded: bool = False
    summary_entry_id: str | None = None
    mou_ids: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ maths
    @property
    def total(self) -> int:
        return len(self.findings)

    @property
    def fulfilled_count(self) -> int:
        return sum(1 for f in self.findings if f.fulfilled)

    @property
    def compliance_pct(self) -> float:
        """Fulfilled findings / total findings, as a percentage.

        Zero promises means *undefined*, reported as 0.0. Reporting 100.0 for an
        empty denominator is how the old artifact claimed perfect compliance
        while verifying nothing at all.
        """
        if not self.findings:
            return 0.0
        return round(100.0 * self.fulfilled_count / len(self.findings), 4)

    @property
    def confidence(self) -> float:
        if not self.findings:
            return 0.0
        return round(sum(f.confidence for f in self.findings) / len(self.findings), 6)

    @property
    def blocking(self) -> list[RiskFlag]:
        return [f for f in self.flags if f.severity is Severity.BLOCKING]

    @property
    def blocking_codes(self) -> list[str]:
        return sorted({f.code for f in self.blocking})

    # ----------------------------------------------------------------- output
    def summary(self) -> dict[str, Any]:
        """Payload written to the ``compliance_summary`` board entry."""
        return {
            "event_id": self.event_id,
            "agent": AgentId.A5_COMPLIANCE.value,
            "mou_ids": list(self.mou_ids),
            "verified": self.fulfilled_count,
            "total_promises": self.total,
            "compliance_pct": self.compliance_pct,
            "compliance_pct_basis": (
                "fulfilled findings / total findings from the same AuditFinding list "
                "posted to this zone; 0.0 means undefined (nothing was verifiable), "
                "never 100%"
            ),
            "blocking_codes": sorted({f.code for f in self.blocking}),
            "flag_codes": sorted({f.code for f in self.flags}),
            "finding_entry_ids": list(self.finding_entry_ids),
            "flag_entry_ids": list(self.flag_entry_ids),
            "degraded": self.degraded,
            "helper_notes": list(self.notes),
            "computed_at": utcnow().isoformat(),
        }

    def veto_message(self) -> str:
        """Message carried by :class:`HumanGateRequired` when A5 vetoes."""
        detail = " | ".join(f"{f.code} ({f.brand}): {f.message}" for f in self.blocking)
        return (f"A5 compliance veto: {len(self.blocking)} blocking flag(s), "
                f"{self.compliance_pct}% of {self.total} promises verified. {detail}")

    def pairs(self) -> list[tuple[str, bool]]:
        """``(promise, fulfilled)`` pairs, for cross-agent comparison in tests."""
        return [(f.promise, f.fulfilled) for f in self.findings]


#: Alias so ``agents/__init__.py`` can export either name.
A5Compliance = ComplianceAgent


# ======================================================================= helpers
def _promises_of(mou: MoU) -> list[tuple[str, str | None]]:
    """``(promise, declared_status)`` pairs from an MoU.

    ``MoU.deliverables`` is declared ``list[str]``, but an agent may have posted
    ``Deliverable`` objects or dicts into the same field, and those carry the
    claimed status A5 exists to challenge. Both are accepted.
    """
    out: list[tuple[str, str | None]] = []
    for item in mou.deliverables:
        if isinstance(item, Deliverable):
            out.append((item.promise, item.status))
        elif isinstance(item, dict):
            promise = str(item.get("promise") or "").strip()
            if promise:
                status = item.get("status")
                out.append((promise, str(status) if status is not None else None))
        else:
            promise = str(item).strip()
            if promise:
                out.append((promise, None))
    return out


def _deliverable_payloads(mou: MoU) -> list[dict[str, Any]]:
    """Deliverables as dicts so an embedded ``evidence_url`` can be read."""
    out: list[dict[str, Any]] = []
    for item in mou.deliverables:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, Deliverable):
            out.append({
                "promise": item.promise,
                "status": item.status,
                "evidence_url": item.evidence_url,
                "evidence_note": item.evidence_note,
            })
    return out


def _same_deliverable(left: str, right: str) -> bool:
    """Normalised comparison of two deliverable *names*.

    Not free-text intent matching: both sides are deliverable labels taken from
    typed artefacts, compared after case-folding and whitespace collapsing.
    """
    a = " ".join(str(left).split()).casefold()
    b = " ".join(str(right).split()).casefold()
    return bool(a) and bool(b) and (a == b or a in b or b in a)
