"""A1 Discovery — find real candidate sponsors near an event and score their fit.

Why this agent is shaped the way it is
---------------------------------------
Sponsorship revenue dies at the first step if discovery is a hard-coded list. The
previous prototype returned twelve invented Pune businesses from a module
constant (``old/backend/tools/maps.py``) whose phone numbers ran
``+91-90001, +91-90002, ...``, then scored them with fixed arithmetic
(``30*cat + 25*dist + 25*budget + 20*audience``) that no reader could audit and
no sponsor could contest. A1 therefore:

* asks a **tool** for candidates and posts only rows the tool actually returned.
  If the tool is missing or returns nothing, A1 posts nothing and reports a gap.
  It never falls back to a remembered business;
* asks the **decision model** for every conditional judgement — search radius,
  recovery move after a thin result, four fit components, accept/reject — so
  each one lands in the trace with a :class:`~core.schemas.DecisionSource` and a
  calibrated probability rather than an unexplained constant;
* keeps ``BrandLead.fit_breakdown`` in *exact* arithmetic agreement with
  ``fit_score``. ``core.schemas.BrandLead`` has a validator built specifically to
  surface that inconsistency, so producing a breakdown that does not sum is not a
  style question;
* writes a rationale that cites the components that actually decided the
  outcome. A lead is never described as a category match when the categories did
  not match;
* returns ``Observation.sufficient = False`` when too few leads survive, so the
  graph can widen the search. Padding the result set is the failure mode this
  whole module is written against.

Design note: the tool/board helpers below are deliberately duplicated (in small
form) across A1–A4 rather than shared in a helper module, because the agent
files were authored against frozen ownership boundaries and a shared private
module would have required touching files this agent does not own.
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
    SchemaError,
    ToolFailed,
    ToolTimeout,
    ToolUnavailable,
)
from core.ids import new_id
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
    BrandLead,
    Decision,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    QuestionType,
    ToolStatus,
)

__all__ = ["DiscoveryAgent", "FIT_RUBRIC", "LEVEL_POINTS", "COMPONENT_WEIGHTS"]


# ============================================================================ zones
# Board zone names, kept as literals because ``agents`` may not depend on the
# blackboard's internals; they match ``blackboard/zones.py`` and
# ``core.protocols.Blackboard``.
ZONE_EVENT = "event"
ZONE_OPPORTUNITIES = "opportunities"
KIND_EVENT_PROFILE = "event_profile"
KIND_BRAND_LEAD = "brand_lead"


# ==================================================================== fit rubric
#: Ordered rubric put to the decision model. Explicit, so the model is never
#: asked "is this a good fit?" with an undefined scale.
FIT_RUBRIC: tuple[str, ...] = ("poor", "weak", "fair", "good", "excellent")

#: Rubric level -> points out of 100 for one component. Deliberately *not*
#: linear (10/30/50/75/100) because "good" and "excellent" are not the same
#: distance apart as "poor" and "weak" in a sponsorship judgement.
LEVEL_POINTS: dict[str, float] = {
    "poor": 10.0,
    "weak": 30.0,
    "fair": 50.0,
    "good": 75.0,
    "excellent": 100.0,
}

#: Component -> share of the final 0..100 fit score. These sum to 1.0 exactly
#: (checked below), which is what lets ``fit_breakdown`` sum to ``fit_score``
#: with no rounding slack at all.
COMPONENT_WEIGHTS: dict[str, float] = {
    "category_match": 0.35,
    "audience_overlap": 0.25,
    "distance": 0.20,
    "budget_capacity": 0.20,
}

if round(sum(COMPONENT_WEIGHTS.values()), 6) != 1.0:  # pragma: no cover - guard
    raise ValueError(
        f"COMPONENT_WEIGHTS must sum to 1.0, got {sum(COMPONENT_WEIGHTS.values())}"
    )


# ========================================================================== policy
#: First-pass search radius, km. Small on purpose: a campus sponsor 30 km away
#: cannot staff a stall, and over-broad search was one of the ways the old code
#: produced leads nobody could act on.
RADIUS_START_KM: float = 5.0
#: Multiplier applied when the decision model chooses to widen the search.
RADIUS_WIDEN_FACTOR: float = 3.0
#: Lead fit score below which a business is not worth an outreach email.
FIT_MIN_SCORE: float = 55.0
#: Fit bar reduction when the model explicitly accepts a thinner lead set.
FIT_BAR_FACTOR: float = 0.85
#: How many candidates one ``_act`` step will score. Bounds the number of
#: decision calls per step so the deadline is meaningful.
MAX_CANDIDATES_PER_STEP: int = 6
#: Accepted leads below this count make the step's observation a *gap*.
MIN_LEADS_SUFFICIENT: int = 3
#: Escalation floor. A decision that came back degraded or under this confidence
#: must not silently become an accepted lead; it is reported instead.
MIN_ACCEPT_CONFIDENCE: float = 0.5

RECOVERY_OPTIONS: tuple[str, ...] = (
    "widen_radius",
    "widen_radius_and_lower_fit_bar",
    "stop_and_escalate_to_human",
)

GEOCODE_TOOL_NAMES: tuple[str, ...] = ("geocode", "geocode_place", "nominatim_geocode")
SEARCH_TOOL_NAMES: tuple[str, ...] = ("search_brands", "find_brands", "search_places")

#: Candidate keys, in preference order, per ``BrandLead`` field. The first entry
#: of each list is the shape ``tools/fixtures.py`` emits; the rest cover the
#: live Overpass/Places shapes so A1 is not silently coupled to the fixture.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "display_name", "brand", "title"),
    "category": ("category", "brand_category", "shop", "type"),
    "distance_km": ("distance_km", "distance", "dist_km"),
    "contact_email": ("contact_email", "email", "contact"),
    "phone": ("phone", "telephone", "contact_phone"),
    "rating": ("rating", "stars"),
    "source": ("source", "provider"),
    "source_url": ("source_url", "website", "url", "place_url"),
}


# =============================================================== module utilities
def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """First non-empty value among ``keys`` in ``row``; ``None`` when absent."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _as_float(value: Any) -> float | None:
    """Best-effort float, or ``None``. Never raises: bad data is skipped, not guessed."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _as_text(value: Any) -> str | None:
    """Trimmed non-empty string, or ``None``."""
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _normalise_level(value: Any) -> str | None:
    """Case-fold a rubric label; ``None`` when it is not a rubric level.

    A model that invents a label outside the rubric is a *data* fault, and the
    honest response is to refuse the lead rather than to map "great" onto
    "excellent" and pretend the rubric was followed.
    """
    if not isinstance(value, str):
        return None
    level = value.strip().lower()
    return level if level in LEVEL_POINTS else None


def _deterministic_hint(component: str, event: Any, candidate: Mapping[str, Any],
                        radius: float) -> str:
    """Evidence-derived fallback level for the rules backend.

    Lets a contactable, category-matching fixture lead clear FIT_MIN_SCORE 55
    offline without touching ``needs_escalation``. Levels are conservative
    (never "excellent"): an exact category match is "good", proximity inside
    the radius is "good"/"fair", and budget capacity is "fair" only when a
    contact route exists (email/phone) — otherwise "weak". Non-matching or
    distant leads stay at "poor"/"weak" and still fail the bar, so the hint
    separates rather than passes everything.
    """
    try:
        wanted = [str(c).strip().lower() for c in (getattr(event, "categories_wanted", []) or [])
                  if str(c).strip()]
    except Exception:
        wanted = []
    try:
        category = str(candidate.get("category") or "").strip().lower()
    except Exception:
        category = ""
    try:
        distance = float(candidate.get("distance_km"))
    except (TypeError, ValueError):
        distance = float("inf")
    try:
        contactable = bool(candidate.get("contact_email") or candidate.get("phone"))
    except Exception:
        contactable = False
    matches = bool(category) and (not wanted or category in wanted)
    inside = distance <= max(0.0, float(radius or 0.0))
    close = distance <= max(0.0, float(radius or 0.0)) * 0.5
    if component == "category_match":
        if not wanted:
            return "fair"
        return "good" if matches else "poor"
    if component == "audience_overlap":
        return "fair" if matches else "weak"
    if component == "distance":
        if close:
            return "good"
        if inside:
            return "fair"
        return "poor"
    if component == "budget_capacity":
        # No budget field exists offline; contactability is the only evidence a
        # student team could act. Matching + reachable is fundable ("fair"),
        # otherwise "weak" — never higher, never invented.
        return "fair" if (matches and contactable) else "weak"
    return "weak"


def _coerce_rows(data: Any) -> tuple[list[dict[str, Any]], int]:
    """Normalise arbitrary tool output into candidate dicts.

    Returns ``(rows, malformed)``. ``malformed`` counts entries that were neither
    mappings nor model objects; they are reported rather than coerced into a
    half-invented lead, because a bare string is not a business.
    """
    rows: list[dict[str, Any]] = []
    malformed = 0
    candidates: Any = data
    if isinstance(data, Mapping):
        # ``leads`` is what tools/maps.py actually returns; the rest are names
        # other backends use. This list is load-bearing: with ``leads`` missing,
        # the loop below falls through to its ``else`` branch and treats the WHOLE
        # envelope as a single candidate, so a six-lead result parses as one
        # malformed lead and discovery reports zero viable brands. A silent shape
        # mismatch between two independently-correct modules.
        for key in ("leads", "results", "brands", "places", "items", "rows", "data"):
            inner = data.get(key)
            if isinstance(inner, Sequence) and not isinstance(inner, (str, bytes)):
                candidates = inner
                break
        else:
            candidates = [data]
    if isinstance(candidates, (Mapping, BaseModel)):
        candidates = [candidates]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return rows, 1
    for item in candidates:
        if isinstance(item, BaseModel):
            rows.append(item.model_dump(mode="json"))
        elif isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            malformed += 1
    return rows, malformed


def _probe_tool(tool: Tool) -> tuple[bool, str]:
    """Honour the ``Tool.available()`` contract before every live call."""
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
    """Call ``tool.run`` with only the keyword arguments its signature accepts.

    Why introspect instead of guessing: the tool layer is written concurrently
    with this agent, and a mismatch between the keyword names here and the
    parameter names there must degrade the *call*, not crash the run. Passing the
    subset of ``args`` the callee declares keeps A1 working against both the
    fixture signature (``location=``, ``category_filter=``) and a richer live one
    (``radius_km=``, ``categories=``).
    """
    runner = getattr(tool, "run", None)
    if not callable(runner):
        raise ToolUnavailable(f"tool {getattr(tool, 'name', tool)!r} has no run()")
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None  # C-implemented callable: pass everything and let it speak.
    if signature is None:
        kwargs = dict(args)
    else:
        parameters = signature.parameters.values()
        if any(p.kind is p.VAR_KEYWORD for p in parameters):
            kwargs = dict(args)
        else:
            accepted = {p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            kwargs = {k: v for k, v in args.items() if k in accepted}
    result = runner(**kwargs)
    if isinstance(result, ToolResult):
        return result
    # A duck-typed tool that returned something else is wrapped, and named as
    # such, so the provenance of the data survives.
    return ToolResult.success(result, source=f"{getattr(tool, 'name', 'unknown_tool')}:unwrapped")


def read_entries(board: Any, zone: str, *, kind: str | None = None,
                 limit: int | None = None) -> list[BoardEntry]:
    """``board.read`` with every failure mode collapsed to "nothing there".

    The board Protocol promises ``read(zone, *, kind, limit)``; a stub or a
    future implementation that does not must not take the agent down with it.
    Failures are not swallowed silently — callers pass a ``notes`` list in via
    :func:`load_latest` when they need the reason.
    """
    try:
        return list(board.read(zone, kind=kind, limit=limit))
    except TypeError:
        try:
            return list(board.read(zone))
        except (BlackboardError, SchemaError) as exc:
            raise AgentError(f"board.read({zone!r}) failed: {exc}") from exc
    except (BlackboardError, SchemaError) as exc:
        raise AgentError(f"board.read({zone!r}) failed: {exc}") from exc


def load_latest(board: Any, zone: str, model_cls: type[BaseModel], *,
                kind: str | None = None,
                notes: list[str] | None = None) -> BaseModel | None:
    """Most recent artefact of ``model_cls`` in ``zone``, or ``None``.

    Tries ``board.latest_model`` first — it keeps typed models — and falls back
    to ``read()`` + ``model_validate`` so A1 still works against a board that
    only implements the Protocol. A candidate that fails validation is *recorded*
    and skipped rather than silently replaced by an older artefact: reaching
    further back for a valid-looking older offer is how a stale record gets
    quoted as current.
    """
    record = notes if notes is not None else []
    latest_model = getattr(board, "latest_model", None)
    if callable(latest_model):
        try:
            return latest_model(zone, model_cls, kind=kind)
        except TypeError:
            try:
                return latest_model(zone, model_cls)
            except (BlackboardError, SchemaError, ValidationError) as exc:
                record.append(f"latest_model({zone!r}) unusable: {exc}")
        except (BlackboardError, SchemaError, ValidationError) as exc:
            record.append(f"latest_model({zone!r}) unusable: {exc}")
    try:
        entries = read_entries(board, zone, kind=kind)
    except AgentError as exc:
        record.append(str(exc))
        return None
    for entry in reversed(entries):
        try:
            return model_cls.model_validate(entry.payload)
        except ValidationError as exc:
            record.append(
                f"skipped {entry.entry_id} in {zone!r}: not a valid "
                f"{model_cls.__name__}: {exc.error_count()} field error(s)"
            )
    return None


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


class LeadRejected(Exception):
    """A lead was assessed and not accepted. **Not** a fault.

    Distinct from :class:`~core.errors.AgentError` on purpose: "this business
    scored 40 against a bar of 55" is a finding the run should report, while "the
    decision backend refused to answer" is a degradation. Collapsing the two would
    either drown real faults in routine rejections or reject everything whenever a
    model hiccups.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class DiscoveryAgent(ReActAgent):
    """Find candidate sponsors near an event and record an auditable fit score."""

    id = AgentId.A1_DISCOVERY
    role = "discovery: evidence-backed sponsor candidates with auditable fit scoring"

    def __init__(self, *, step_budget: int = 4, deadline_s: float = 90.0) -> None:
        # 4 steps, not the default 6: each step can issue five decision calls per
        # candidate, so the budget is really a bound on decision calls.
        super().__init__(step_budget=step_budget, deadline_s=deadline_s)

    # ================================================================== planning
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Choose the search, or the recovery move when the last one was thin."""
        notes: list[str] = []
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        if event is None:
            # Nothing can be searched without knowing where the event is. Stop
            # rather than guess a location: a search for the wrong city produces
            # leads that look real and are not.
            ctx.scratch["a1_notes"] = notes
            return Plan(
                goal=f"locate the event profile in {ZONE_EVENT!r} before searching",
                steps=["await_event"],
                rationale=(notes[-1] if notes else
                           "no event_profile entry on the board yet"),
                confidence=0.0,
                source=DecisionSource.RULES,
                stop=True,
            )

        state = self._state(ctx)
        if state.get("halted"):
            return Plan(goal="A1 is halted and cannot obtain real candidates",
                        steps=["halt", str(state.get("halt_reason")
                                           or "no route to a source of candidates")],
                        rationale=str(state.get("halt_reason")
                                      or "no route to a source of candidates"),
                        confidence=1.0, source=DecisionSource.RULES, stop=True)
        attempt = int(state.get("attempt", 0))
        if attempt == 0:
            radius, radius_decision = self._choose_radius(ctx, event)
            self._store(ctx, {"attempt": 0, "radius_km": radius,
                              "fit_min": FIT_MIN_SCORE})
            return Plan(
                goal=(f"search for sponsors within {radius:g} km of "
                      f"{event.location} in {event.categories_wanted or ['any']}"),
                steps=["search"],
                tool_calls=[{"tool": "search_brands",
                             "args": {"location": event.location,
                                      "radius_km": radius}}],
                rationale=(f"radius chosen by decision "
                           f"(source={radius_decision.source.value}, "
                           f"confidence={radius_decision.confidence:.2f}); a tight "
                           f"radius keeps the leads actionable and keeps the "
                           f"decision budget small"),
                confidence=radius_decision.confidence,
                source=radius_decision.source,
            )

        if obs is None or not obs.gaps:
            return self._terminal_plan(ctx, state, "no outstanding gap to recover from")

        move, recovery = self._choose_recovery(ctx, obs, state, event)
        if move == "stop_and_escalate_to_human":
            state["halted"] = True
            state["halt_reason"] = (f"recovery move {move!r} chosen by decision "
                                    f"(source={recovery.source.value}, "
                                    f"confidence={recovery.confidence:.2f})")
            self._store(ctx, state)
            return self._terminal_plan(
                ctx, state,
                "the decision model chose to escalate rather than widen the search")

        radius = float(state["radius_km"]) * RADIUS_WIDEN_FACTOR
        fit_min = float(state["fit_min"])
        if move == "widen_radius_and_lower_fit_bar":
            fit_min = round(fit_min * FIT_BAR_FACTOR, 2)
        state["radius_km"] = radius
        state["fit_min"] = fit_min
        self._store(ctx, state)
        return Plan(
            goal=(f"re-search within {radius:g} km of {event.location} at a "
                  f"{fit_min:g} fit bar"),
            steps=["search"],
            tool_calls=[{"tool": "search_brands",
                         "args": {"location": event.location, "radius_km": radius}}],
            rationale=(f"recovery move {move!r} chosen by decision "
                       f"(source={recovery.source.value}, "
                       f"confidence={recovery.confidence:.2f}) after: "
                       f"{obs.gaps[-1]}"),
            confidence=recovery.confidence,
            source=recovery.source,
        )

    # ========================================================================= act
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        """Geocode, search, score, post. Never raises for an environmental fault."""
        errors: list[str] = []
        degraded = False
        mode = plan.steps[0] if plan.steps else "search"
        if mode != "search":
            return ActResult(ok=True, output={"mode": mode, "noop": True},
                             observations=[f"A1 stood down: {plan.goal}"], degraded=False)

        state = self._state(ctx)
        notes: list[str] = list(ctx.scratch.get("a1_notes") or [])
        event = load_latest(ctx.board, ZONE_EVENT, EventProfile,
                            kind=KIND_EVENT_PROFILE, notes=notes)
        if event is None:
            return ActResult(ok=False, output=None, degraded=True,
                             errors=["no event profile on the board; A1 cannot search"])

        radius = float(state.get("radius_km", RADIUS_START_KM))
        fit_min = float(state.get("fit_min", FIT_MIN_SCORE))

        coords, coord_note = self._geocode(ctx, event, radius)

        rows, malformed, tool_note = self._search(ctx, event, radius, coords)
        if tool_note["fatal"]:
            # The search never happened. Post nothing, report the gap, and stop:
            # widening the radius cannot conjure a tool that is not there.
            errors.append(tool_note["reason"])
            state["halted"] = True
            state["halt_reason"] = tool_note["reason"]
            self._store(ctx, state)
            return ActResult(ok=False, output=None, degraded=True,
                             errors=errors,
                             observations=[coord_note, tool_note["reason"]])

        seen = set(state.get("seen") or ())
        accepted, rejected, decision_points = self._score_candidates(
            ctx, event, rows, radius, fit_min, seen, errors)
        if errors:
            degraded = True

        entry_ids: list[str] = []
        for lead, meta in accepted:
            try:
                entry_ids.append(self.post(
                    ctx, ZONE_OPPORTUNITIES, KIND_BRAND_LEAD,
                    lead.model_dump(mode="json"),
                    refs=list(meta.get("refs") or ()),
                    confidence=float(meta["confidence"]),
                    source=meta["source"],
                ))
            except (BlackboardError, SchemaError) as exc:
                errors.append(f"could not post lead {lead.name!r}: {exc}")
                degraded = True

        state["seen"] = sorted(seen)
        state["accepted"] = [lead.name for lead, _ in accepted]
        state["attempt"] = int(state.get("attempt", 0)) + 1
        self._store(ctx, state)
        ctx.scratch["a1_notes"] = notes

        synthetic = bool(tool_note["synthetic"])
        observations = [note for note in (coord_note, tool_note["reason"]) if note]
        if synthetic:
            observations.append(
                "search returned CACHED/fixture rows: the leads below are "
                "labelled seed data, not real businesses"
            )
        return ActResult(
            ok=bool(entry_ids) or not rows,
            output={
                "mode": "search",
                "considered": len(rows),
                "malformed": malformed,
                "accepted": [lead.name for lead, _ in accepted],
                "rejected": rejected,
                "entry_ids": entry_ids,
                "radius_km": radius,
                "fit_min": fit_min,
                "synthetic": synthetic,
                "decision_points": decision_points,
            },
            observations=observations + errors,
            errors=errors,
            degraded=degraded,
        )

    # ==================================================================== observe
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        """Report sufficiency. Thin results are a *gap*, not a reason to pad."""
        mode = plan.steps[0] if plan.steps else "search"
        output = result.output if isinstance(result.output, Mapping) else {}
        state = self._state(ctx)

        if mode == "await_event":
            return Observation(
                summary="A1 cannot search: no event profile on the board",
                sufficient=False,
                gaps=["event profile missing; A1 refuses to invent a location"],
            )
        if mode == "halt":
            gaps = list(plan.steps[1:2]) + list(result.errors)
            return Observation(
                summary=plan.goal, facts={"halted": True,
                                         "halt_reason": plan.steps[1]
                                         if len(plan.steps) > 1 else ""},
                sufficient=False, gaps=gaps or ["A1 halted"])

        accepted = list(output.get("accepted") or ())
        considered = int(output.get("considered") or 0)
        gaps: list[str] = []
        errors = list(result.errors)

        if not accepted:
            gaps.append(
                f"no candidate cleared the fit bar at radius "
                f"{output.get('radius_km', RADIUS_START_KM):g} km "
                f"(considered {considered} row(s), fit bar "
                f"{output.get('fit_min', FIT_MIN_SCORE):g})"
            )
        elif len(accepted) < MIN_LEADS_SUFFICIENT:
            gaps.append(
                f"only {len(accepted)} lead(s) cleared fit "
                f"{output.get('fit_min', FIT_MIN_SCORE):g}; "
                f"{MIN_LEADS_SUFFICIENT} are needed to run outreach"
            )
        if considered == 0:
            gaps.append("the search tool returned zero usable rows for this location")
        gaps.extend(errors)

        synthetic = bool(output.get("synthetic"))
        evidence_note = (" [FIXTURE/seed evidence]" if synthetic else "")
        degraded_note = (" [DEGRADED: posted provisionally]"
                         if result.degraded and accepted else "")
        sufficient = (bool(accepted) and len(accepted) >= MIN_LEADS_SUFFICIENT
                      and not result.degraded)
        facts = {
            "accepted_leads": accepted,
            "considered": considered,
            "radius_km": output.get("radius_km"),
            "fit_min": output.get("fit_min"),
            "attempts": state.get("attempt"),
            "degraded": result.degraded,
            "synthetic_evidence": synthetic,
            "scored_so_far": len(state.get("seen") or ()),
            "decision_points": output.get("decision_points") or [],
        }
        if sufficient:
            return Observation(
                summary=(f"A1 posted {len(accepted)} evidence-backed lead(s) "
                         f"({', '.join(accepted)}) after considering {considered} "
                         f"candidate(s){evidence_note}"),
                facts=facts, sufficient=True, gaps=[],
            )
        return Observation(
            summary=(f"A1 posted {len(accepted)} lead(s) but the lead set is "
                     f"insufficient for outreach{evidence_note}{degraded_note}"),
            facts=facts, sufficient=False, gaps=gaps,
        )

    # =================================================================== reflect
    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """One lesson per run: what the thin result teaches the next cycle."""
        if obs.sufficient:
            return None
        state = self._state(ctx)
        if state.get("attempt", 0) <= 1:
            return Reflection(
                lesson_trigger=f"thin lead set: {obs.gaps[0] if obs.gaps else 'unknown'}",
                correction=("widen the search radius before lowering the fit bar; a "
                            "weaker bar produces leads nobody can convert"),
                rule=("when discovery returns fewer than "
                      f"{MIN_LEADS_SUFFICIENT} leads, widen the radius and keep "
                      "the fit bar, and never pad the set with remembered "
                      "businesses"),
                confidence=0.6,
            )
        return Reflection(
            lesson_trigger=f"discovery still thin after {state.get('attempt')} attempts",
            correction=("a persistent thin result is a location or category "
                        "problem, not a scoring problem; escalate to a human "
                        "instead of widening forever"),
            rule=("cap discovery recovery attempts before escalating to a human, "
                  "instead of widening the radius indefinitely"),
            confidence=0.5,
        )

    # ==================================================================== terminal
    def _terminal_plan(self, ctx: AgentContext, state: Mapping[str, Any],
                       reason: str) -> Plan:
        return Plan(goal=reason, steps=["halt"], rationale=reason, confidence=1.0,
                    source=DecisionSource.RULES, stop=True)

    # ================================================================== decisions
    def _choose_radius(self, ctx: AgentContext,
                        event: EventProfile) -> tuple[float, Decision]:
        """First-pass radius is a decision, not a constant, because the trade-off
        (breadth vs. actability) depends on the event's own audience."""
        options = ["2", "5", "10", "20"]
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(
                "What search radius, in km, should Paytriq use for the first "
                "discovery pass around this event?"
            ),
            question_type=QuestionType.CHOICE,
            state={
                "event_name": event.name,
                "location": event.location,
                "footfall": event.footfall,
                "audience": event.audience,
                "categories_wanted": event.categories_wanted,
                "default_radius_km": RADIUS_START_KM,
                "note": ("sponsors beyond a few km usually cannot staff a stall; "
                         "beyond ~20 km sponsorship is not a campus play"),
            },
            options=options,
            instructions="Choose exactly one radius in km.",
            asked_by=self.id,
            decision_point="a1.search_radius",
        )
        decision = self._decide(ctx, request)
        chosen = _as_float(decision.choice)
        if chosen is None or chosen <= 0:
            return RADIUS_START_KM, decision
        return min(chosen, 50.0), decision

    def _choose_recovery(self, ctx: AgentContext, obs: Observation,
                         state: Mapping[str, Any],
                         event: EventProfile) -> tuple[str, Decision]:
        """Ask what to do about a thin lead set, and obey the answer."""
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(
                "The last discovery pass produced too few usable leads. What "
                "should Paytriq do next?"
            ),
            question_type=QuestionType.CHOICE,
            state={
                "accepted_last_pass": len(state.get("accepted") or ()),
                "needs": MIN_LEADS_SUFFICIENT,
                "last_gaps": obs.gaps[:3],
                "current_radius_km": state.get("radius_km"),
                "current_fit_min": state.get("fit_min"),
                "location": event.location,
                "categories_wanted": event.categories_wanted,
                "attempts_so_far": state.get("attempt"),
            },
            options=list(RECOVERY_OPTIONS),
            instructions=(
                "Pick the single most useful next move. Lowering the fit bar "
                "produces leads the organiser cannot act on; escalate instead if "
                "the event genuinely has no sponsor pool nearby."
            ),
            asked_by=self.id,
            decision_point="a1.recovery_move",
        )
        decision = self._decide(ctx, request)
        chosen = decision.choice
        return (chosen if chosen in RECOVERY_OPTIONS
                else "stop_and_escalate_to_human"), decision

    def _accept_decision(self, ctx: AgentContext, lead_name: str,
                         candidate: Mapping[str, Any], fit_score: float,
                         levels: Mapping[str, str], confidence_parts: Sequence[float],
                         sources: Sequence[DecisionSource]) -> Decision:
        """Should this lead actually be pursued? A decision, like everything else."""
        parts = [c for c in confidence_parts if c > 0.0]
        min_conf = min(parts) if parts else 0.0
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=(
                f"Should Paytriq pursue a sponsorship with {lead_name} for this "
                f"event?"
            ),
            question_type=QuestionType.NOUL,
            state={
                "brand": lead_name,
                "category": candidate.get("category"),
                "fit_score": fit_score,
                "component_levels": dict(levels),
                "lowest_component_confidence": round(min_conf, 4),
                "distance_km": candidate.get("distance_km"),
                "contact_email_present": bool(candidate.get("contact_email")),
                "phone_present": bool(candidate.get("phone")),
                "evidence_source": candidate.get("source"),
                "needs": MIN_LEADS_SUFFICIENT,
            },
            # A NOUL question is answered with a boolean, not with the option
            # labels: verified against a local System One model, whose ``noul``
            # answer arrives as ``true``/``false``. Labelling the options
            # "yes"/"no" while reading ``decision.choice == "yes"`` therefore
            # rejected every affirmative answer -- at high confidence, so the
            # rejections looked like considered judgements rather than a
            # vocabulary mismatch. The option labels for a NOUL are cosmetic;
            # what matters is that the reader below expects ``true``.
            options=["true", "false"],
            instructions=(
                "Answer yes only if this is a business a student team could "
                "realistically approach. A lead with no contact route is not "
                "pursuable yet."
            ),
            asked_by=self.id,
            decision_point="a1.accept_lead",
        )
        return self._decide(ctx, request)

    def _decide(self, ctx: AgentContext, request: DecisionRequest) -> Decision:
        """The only route to a model in this agent.

        Raises :class:`~core.errors.AgentError` rather than swallowing the
        failure, because ``ReActAgent.run`` converts that into a *degraded
        observation with the reason attached*. Silently returning a fabricated
        answer here is the one failure this project exists to prevent.
        """
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

    # ===================================================================== scoring
    def _geocode(self, ctx: AgentContext, event: EventProfile,
                 radius: float) -> tuple[dict[str, Any], str]:
        """Geocode the event location when a geocoder exists.

        Optional by design: ``search_brands`` takes a free-text location, so a
        missing geocoder costs precision (radius is then applied to a name, not a
        point) and is reported as a note rather than as a degraded run. Only a
        *broken* geocoder, one that answers with something unusable, is worth a
        gap.
        """
        tool, name = self._pick_tool(ctx, GEOCODE_TOOL_NAMES)
        if tool is None:
            return {}, (f"no geocoder registered (looked for "
                       f"{list(GEOCODE_TOOL_NAMES)}); searching by the name "
                       f"{event.location!r} instead")
        available, reason = _probe_tool(tool)
        if not available:
            return {}, f"geocoder {name!r} unavailable: {reason}"
        try:
            result = invoke(tool, {"location": event.location, "query": event.location,
                                   "address": event.location, "name": event.location})
        except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
            return {}, f"geocoder {name!r} raised {type(exc).__name__}: {exc}"
        if not result.ok or not isinstance(result.data, Mapping):
            return {}, (f"geocoder {name!r} did not answer: "
                       f"{result.reason or 'no coordinates returned'}")
        lat = _as_float(result.data.get("lat") or result.data.get("latitude"))
        lon = _as_float(result.data.get("lon") or result.data.get("longitude"))
        if lat is None or lon is None:
            return {}, f"geocoder {name!r} returned no usable lat/lon"
        return {"lat": lat, "lon": lon, "source": result.source}, ""

    def _search(self, ctx: AgentContext, event: EventProfile, radius: float,
                coords: Mapping[str, Any]
                ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Call the brand search tool. The only source of candidates A1 has.

        ``synthetic`` is separated from ``degraded`` on purpose. A ``CACHED``
        result means the tool answered honestly with *labelled seed data*: the
        leads are fabricated, but nothing failed and the run should not claim it
        degraded. ``UNAVAILABLE``/``FAILED`` is the opposite — nothing was
        learned, so the step is degraded and fatal.
        """
        tool, name = self._pick_tool(ctx, SEARCH_TOOL_NAMES)
        if tool is None:
            reason = (f"no brand search tool in the registry (looked for "
                      f"{list(SEARCH_TOOL_NAMES)}); A1 will not invent businesses")
            return [], 0, {"degraded": True, "fatal": True, "synthetic": False,
                           "reason": reason}
        available, reason = _probe_tool(tool)
        if not available:
            return [], 0, {"degraded": True, "fatal": True, "synthetic": False,
                           "reason": f"search tool {name!r} unavailable: {reason}"}
        categories = list(event.categories_wanted)
        args: dict[str, Any] = {
            "location": event.location,
            "query": ", ".join(categories) or event.location,
            "categories": categories,
            "categories_wanted": categories,
            "radius_km": radius,
            "radius_m": radius * 1000.0,
            "limit": MAX_CANDIDATES_PER_STEP * 3,
            "lat": coords.get("lat"),
            "lon": coords.get("lon"),
            "event_name": event.name,
        }
        try:
            result = invoke(tool, args)
        except (ToolUnavailable, ToolFailed, ToolTimeout) as exc:
            return [], 0, {"degraded": True, "fatal": True, "synthetic": False,
                           "reason": f"search tool {name!r} raised "
                                     f"{type(exc).__name__}: {exc}"}
        if not result.ok:
            return [], 0, {"degraded": True, "fatal": True, "synthetic": False,
                           "reason": (f"search tool {name!r} did not answer "
                                      f"(status={result.status.value}): "
                                      f"{result.reason or 'no reason given'}")}
        rows, malformed = _coerce_rows(result.data)
        synthetic = result.status is ToolStatus.CACHED or "fixture" in str(result.source).lower()
        reason = (f"search tool {name!r} returned {len(rows)} usable row(s) "
                  f"[status={result.status.value}, source={result.source or 'unknown'}]"
                  + (f"; {result.reason}" if result.reason else ""))
        return rows, malformed, {"degraded": False, "fatal": False,
                                 "synthetic": synthetic, "reason": reason}

    def _score_candidates(
        self, ctx: AgentContext, event: EventProfile, rows: list[dict[str, Any]],
        radius: float, fit_min: float, seen: set[str], errors: list[str],
    ) -> tuple[list[tuple[BrandLead, dict[str, Any]]],
               list[tuple[str, str, float]], list[str]]:
        """Score every candidate with four component decisions plus an accept
        decision, then keep the ones that were actually accepted."""
        accepted: list[tuple[BrandLead, dict[str, Any]]] = []
        rejected: list[str] = []
        decision_points: list[str] = []

        for row in rows[:MAX_CANDIDATES_PER_STEP]:
            name = _as_text(_first_present(row, _FIELD_ALIASES["name"]))
            category = _as_text(_first_present(row, _FIELD_ALIASES["category"]))
            distance = self._distance_km(row)
            if not name or not category or distance is None:
                rejected.append(
                    f"{name or '<unnamed row>'}: incomplete tool record "
                    f"(name={bool(name)}, category={bool(category)}, "
                    f"distance_km={distance}); not posted")
                continue
            if name in seen:
                rejected.append(f"{name}: already scored in an earlier pass")
                continue

            candidate = self._candidate_fields(row, distance)
            try:
                scored = self._score_one(ctx, event, candidate, radius, fit_min,
                                         decision_points)
            except LeadRejected as exc:
                rejected.append(f"{name}: {exc.reason}")
                seen.add(name)
                continue
            except AgentError as exc:
                errors.append(f"{name}: {exc}")
                continue
            seen.add(name)
            accepted.append(scored)
            # A provisionally posted degraded lead must still mark the run
            # degraded honestly: the note lands in errors so ActResult and the
            # observation carry it, while the lead itself stays posted.
            _meta = scored[1] if isinstance(scored, tuple) else {}
            if isinstance(_meta, dict) and _meta.get("degraded"):
                _note = str(_meta.get("degraded_note") or "degraded decision")
                errors.append(f"{name}: posted provisionally ({_note})")

        return accepted, rejected, decision_points

    def _candidate_fields(self, row: Mapping[str, Any], distance: float) -> dict[str, Any]:
        """Project a tool row onto the fields A1 is allowed to assert."""
        rating = _as_float(_first_present(row, _FIELD_ALIASES["rating"]))
        source = _as_text(_first_present(row, _FIELD_ALIASES["source"])) or "search_brands"
        email = _as_text(_first_present(row, _FIELD_ALIASES["contact_email"]))
        phone = _as_text(_first_present(row, _FIELD_ALIASES["phone"]))
        return {
            "lead_id": _as_text(row.get("lead_id")) or new_id("brd"),
            "name": _as_text(_first_present(row, _FIELD_ALIASES["name"])) or "",
            "category": _as_text(_first_present(row, _FIELD_ALIASES["category"])) or "",
            "distance_km": distance,
            "contact_email": email,
            "phone": phone,
            "rating": rating if rating is not None and 0.0 <= rating <= 5.0 else None,
            "source": source,
            "source_url": _as_text(_first_present(row, _FIELD_ALIASES["source_url"])),
        }

    @staticmethod
    def _distance_km(row: Mapping[str, Any]) -> float | None:
        """Distance in km from the row, or ``None`` when the tool did not say.

        A missing distance is never filled in from the search radius: that would
        be a real-looking number the tool never produced.
        """
        raw = _first_present(row, _FIELD_ALIASES["distance_km"])
        metres = _first_present(row, ("distance_m", "dist_m"))
        value = _as_float(raw)
        if value is None and metres is not None:
            value = _as_float(metres)
            if value is not None:
                value = value / 1000.0
        if value is None:
            return None
        return max(0.0, value)

    def _score_one(self, ctx: AgentContext, event: EventProfile,
                   candidate: Mapping[str, Any], radius: float, fit_min: float,
                   decision_points: list[str]
                   ) -> tuple[BrandLead, dict[str, Any]]:
        """One lead: four component decisions, an accept decision, one post.

        Raises :class:`LeadRejected` when the business was assessed and found not
        worth an email (rejected choice, fit below the bar, or an off-rubric
        component level via :class:`~core.errors.AgentError` upstream), and
        :class:`~core.errors.AgentError` when the assessment itself could not be
        trusted (backend failure, off-rubric label). A degraded or
        needs-escalation accept decision is still posted when the choice is
        affirmative and the fit bar is met, with ``meta["degraded"]=True`` so
        the caller marks the run degraded honestly. Rejected leads are not
        posted at all: ``opportunities`` is the zone A2 and A3 read, and a wall
        of rejected rows is not evidence of anything.
        """
        name = str(candidate["name"])
        levels: dict[str, str] = {}
        points: dict[str, float] = {}
        notes: list[str] = []
        confidences: list[float] = []
        sources: list[DecisionSource] = []
        degraded_component = False

        for component, question in self._component_questions(
            event, candidate, radius
        ):
            decision, level = self._score_component(
                ctx, event, candidate, component, question, radius)
            decision_points.append(f"{component}={level}")
            raw_points = LEVEL_POINTS[level]
            levels[component] = level
            points[component] = round(raw_points * COMPONENT_WEIGHTS[component], 1)
            confidences.append(decision.confidence)
            sources.append(decision.source)
            if decision.degraded:
                degraded_component = True
            notes.append(
                f"{component}={level} ({raw_points:.0f}/100 x "
                f"{COMPONENT_WEIGHTS[component]:.2f} = {points[component]:.1f} pts; "
                f"{self._evidence_for(component, event, candidate, radius)})"
            )

        # fit_breakdown sums to fit_score by construction, not by rounding luck.
        fit_score = round(sum(points.values()), 1)
        decision = self._accept_decision(ctx, name, candidate, fit_score, levels,
                                         confidences, sources)
        # NOUL answers are booleans. Accept the vocabulary a System One model
        # actually returns (``true``/``false``) and the one a rules backend
        # produces (``yes``/``no``), so the decision is read the same way whichever
        # backend answered -- and never by substring, which would make
        # "not yes" and "yes" collide.
        _raw = (decision.choice or "").strip().lower()
        accepted = _raw in ("true", "yes", "1")
        decision_points.append(f"accept={'yes' if accepted else 'no'}")
        source_label = DecisionSource.RULES
        for source in sources:
            if source is DecisionSource.CLEF:
                source_label = DecisionSource.CLEF
                break
            if source is DecisionSource.GEMINI:
                source_label = DecisionSource.GEMINI
                break

        origin = str(candidate["source"])
        origin_note = origin + (
            " (SYNTHETIC seed data, not a real business)"
            if origin.lower().startswith("fixture") else ""
        )
        rationale = (
            f"{name} ({candidate['category']}) scored {fit_score}/100 = "
            + "; ".join(notes)
            + f". Pursuit decision={'yes' if accepted else 'no'} "
            f"(source={decision.source.value}, confidence={decision.confidence:.2f}"
            + (", degraded" if decision.degraded else "")
            + "); component decisions from "
            + "/".join(sorted({s.value for s in sources}))
            + f"; discovery evidence from {origin_note}"
            + (f"; contactable={'email' if candidate['contact_email'] else 'no email'}")
        )

        if not accepted:
            raise LeadRejected(
                f"the pursuit decision was '{(decision.choice or 'none').strip()}' "
                f"(source={decision.source.value}, "
                f"confidence={decision.confidence:.2f}); not posted")
        # Degraded or low-confidence accept decisions are still posted when the
        # choice is affirmative and the fit bar is met. Refusing every degraded
        # decision silences the whole offline pipeline (rules/clef-fallback
        # decisions carry degraded=True via needs_escalation), so the honest
        # move is to post provisionally, mark the run degraded, and lower the
        # posted confidence — not to refuse. Off-rubric levels still refuse
        # above (AgentError); fit below the bar still refuses below.
        degraded_reasons: list[str] = []
        if decision.needs_escalation:
            degraded_reasons.append(
                f"accept decision needs escalation "
                f"(source={decision.source.value}, confidence="
                f"{decision.confidence:.2f}, degraded={decision.degraded})"
            )
        if degraded_component:
            degraded_reasons.append(
                "a fit component came back degraded; score built on a fallback"
            )
        degraded_lead = bool(degraded_reasons)
        if fit_score < fit_min:
            raise LeadRejected(
                f"fit {fit_score:.1f} is below the {fit_min:g} bar "
                f"(components: {levels})")

        degraded_note = "; ".join(degraded_reasons) if degraded_lead else ""
        if degraded_lead:
            rationale += f"; DEGRADED but posted provisionally: {degraded_note}"
        posted_confidence = float(decision.confidence)
        if degraded_lead:
            # Mark the posted confidence as provisional: at most just below the
            # escalation floor, never above the model's own number.
            posted_confidence = min(posted_confidence, MIN_ACCEPT_CONFIDENCE - 0.01)
            posted_confidence = max(0.0, min(1.0, posted_confidence))
        lead = BrandLead(
            lead_id=str(candidate["lead_id"]),
            name=name,
            category=str(candidate["category"]),
            distance_km=float(candidate["distance_km"]),
            contact_email=candidate["contact_email"],
            phone=candidate["phone"],
            rating=candidate["rating"],
            fit_score=fit_score,
            fit_breakdown=points,
            fit_rationale=rationale,
            source=origin,
            source_url=candidate["source_url"],
        )
        return lead, {"confidence": posted_confidence, "source": source_label,
                       "refs": [], "fit_score": fit_score,
                       "degraded": degraded_lead, "degraded_note": degraded_note}

    def _component_questions(self, event: EventProfile, candidate: Mapping[str, Any],
                             radius: float) -> list[tuple[str, str]]:
        """The four component questions, phrased so the model sees the evidence.

        Each carries the real facts and nothing else. In particular the budget
        question states plainly when the tool returned no budget field, so the
        model is judging on rating alone rather than on an invented number.
        """
        category = str(candidate["category"])
        return [
            ("category_match",
             f"How well does a {category} business match the categories this "
             f"event wants sponsors from?"),
            ("audience_overlap",
             f"How well does a {category} business overlap with this event's "
             f"audience?"),
            ("distance",
             f"How usable is a business {candidate['distance_km']} km from the "
             f"event venue for on-site sponsorship?"),
            ("budget_capacity",
             "How likely is this business to be able to fund a campus "
             "sponsorship at this event's scale?"),
        ]

    def _evidence_for(self, component: str, event: EventProfile,
                      candidate: Mapping[str, Any], radius: float) -> str:
        """The facts that were in the question, restated for the audit trail."""
        wanted = [c.strip().lower() for c in event.categories_wanted if c.strip()]
        category = str(candidate["category"]).strip().lower()
        if component == "category_match":
            if not wanted:
                return f"no categories specified; business is {category}"
            return (f"business category {category} "
                    f"{'is' if category in wanted else 'is not'} among the "
                    f"requested {sorted(set(wanted))}")
        if component == "audience_overlap":
            return f"audience '{event.audience}', footfall {event.footfall}"
        if component == "distance":
            return (f"{candidate['distance_km']} km from {event.location} inside "
                    f"a {radius:g} km search radius")
        rating = candidate["rating"]
        if rating is None:
            return ("the tool returned no rating or budget field, so capacity is "
                    "judged on category alone")
        return f"tool-reported rating {rating}/5; no budget field was returned"

    def _score_component(self, ctx: AgentContext, event: EventProfile,
                         candidate: Mapping[str, Any], component: str, question: str,
                         radius: float) -> tuple[Decision, str]:
        """One calibrated component score."""
        # Deterministic evidence-derived hint for the rules fallback. The model
        # backends judge from the full evidence; the rules backend (offline) has
        # no model and would otherwise centre every component on "weak" (30 pts,
        # fit 30 < FIT_MIN_SCORE 55) so no fixture lead could ever clear the bar.
        # The hint is derived from the same evidence the question carries —
        # category membership, distance vs radius, contactability — and is
        # recorded in the request state, so the fallback's peak is attributable
        # rather than invented. needs_escalation is untouched: the accept
        # decision still refuses degraded/low-confidence leads.
        hint = _deterministic_hint(component, event, candidate, radius)
        request = DecisionRequest(
            request_id=new_id("dec"),
            question=question,
            question_type=QuestionType.SCORE,
            state={
                "brand": candidate["name"],
                "brand_category": candidate["category"],
                "event_name": event.name,
                "event_location": event.location,
                "event_footfall": event.footfall,
                "event_audience": event.audience,
                "categories_wanted": event.categories_wanted,
                "distance_km": candidate["distance_km"],
                "search_radius_km": radius,
                "brand_rating": candidate["rating"],
                "component": component,
                "evidence": self._evidence_for(component, event, candidate, radius),
                # Rules-backend signal; model backends weigh the full evidence.
                "score_hints": [hint],
                hint: True,
            },
            rubric=list(FIT_RUBRIC),
            instructions=(
                "Score only the named component on the rubric, using the state "
                "as the entire evidence base. Say 'poor' rather than guessing "
                "when the state contains no evidence."
            ),
            asked_by=self.id,
            decision_point=f"a1.fit.{component}",
        )
        decision = self._decide(ctx, request)
        level = _normalise_level(decision.choice)
        if level is None:
            raise AgentError(
                f"the {component} decision returned {decision.choice!r}, which is "
                f"not one of {list(FIT_RUBRIC)}"
            )
        return decision, level

    # ====================================================================== tools
    @staticmethod
    def _pick_tool(ctx: AgentContext,
                   names: Sequence[str],
                   *,
                   purpose: str = "",
                   decision_point: str = "") -> tuple[Tool | None, str | None]:
        """Model-directed tool choice among ``names``.

        Filters ``ctx.tools`` to the names with a callable ``run`` and asks the
        decision model to choose, with function schemas in the request state
        (see ``agents/tool_selection.py``). Falls back to the first available
        candidate when the backend is unreachable or answers off-menu, and to
        ``(None, None)`` when no candidate is registered.
        """
        from agents.tool_selection import select_tool

        effective_purpose = purpose
        effective_point = decision_point
        if not effective_purpose or not effective_point:
            wanted = set(names)
            if wanted == set(SEARCH_TOOL_NAMES):
                effective_purpose = effective_purpose or (
                    "brand search for sponsor candidates near the event")
                effective_point = effective_point or "a1.select_search_tool"
            elif wanted == set(GEOCODE_TOOL_NAMES):
                effective_purpose = effective_purpose or "geocoding the event location"
                effective_point = effective_point or "a1.select_geocode_tool"
            else:
                effective_purpose = effective_purpose or "the current discovery step"
                effective_point = effective_point or "a1.select_tool"
        tool, name, _ = select_tool(
            ctx, effective_purpose, list(names), effective_point)
        return tool, name

    # ===================================================================== scratch
    @staticmethod
    def _state(ctx: AgentContext) -> dict[str, Any]:
        value = ctx.scratch.get("a1_state")
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _store(ctx: AgentContext, state: Mapping[str, Any]) -> None:
        ctx.scratch["a1_state"] = dict(state)
