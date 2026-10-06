"""Request and response bodies for the Paytriq API.

The regression this module exists to prevent
--------------------------------------------
The previous version of this project shipped ``artifacts/input_sample.json``
whose **nine of ten** top-level keys were silently discarded. The request model
named its fields differently from the sample (``name`` vs ``event_name``,
``footfall`` vs ``footfall_expected``, ``date: str`` vs ``dates: [...]``,
``budget_inr`` vs ``budget_range_inr: [...]``) and Pydantic v2's default is
``extra="ignore"``. The documented ``curl`` command returned **HTTP 200** with an
event containing none of the submitted data, and nothing in the response said so.

Three things fix that, and all three are here:

1. ``CreateEventRequest.model_config`` sets ``extra="forbid"``. An unrecognised
   key is now a **422** that names it, so a mismatch between sample and schema is
   a loud failure at the first request instead of a silent data loss at the end.

2. The keys the sample actually uses are **accepted**, not rejected. Aliases and
   shape coercion below map ``event_name`` -> ``name``, ``dates: [...]`` ->
   ``date``, ``budget_range_inr: [lo, hi]`` -> ``budget_inr`` and so on. A shape
   that can be interpreted without inventing information is coerced; anything
   else is refused.

3. Every coercion is **reported**. ``CreateEventRequest.coercions`` lists what
   was reshaped and by what rule, and ``CreateEventResponse`` carries it to the
   caller. Silently choosing one number out of ``[20000, 100000]`` and returning
   no mention of the other would just be the old bug wearing a different hat.

Why ``extra="forbid"`` and not ``extra="allow"``
-----------------------------------------------
``allow`` would accept ``footfall_expected`` *and* ``footfall``, and then have to
pick. Picking silently is the original defect. Coercion with a written rule and a
report is the honest version of the same convenience.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.schemas import EventProfile

__all__ = [
    "CreateEventRequest",
    "CreateEventResponse",
    "PipelineStage",
    "StageRequest",
    "OutreachRequest",
    "ReplyRequest",
    "ContractRequest",
    "ComplianceRequest",
    "AuditRequest",
    "GateDecisionRequest",
    "GateActionResponse",
    "ResumeRequest",
    "HealthResponse",
    "UnavailableResponse",
    "ALIASES",
]


# ============================================================================ events
#: Legacy key -> canonical field. Every entry is a spelling the previous
#: prototype's ``input_sample.json`` used, so that sample now submits its data
#: instead of having it discarded.
ALIASES: dict[str, str] = {
    "event_name": "name",
    "event_title": "name",
    "title": "name",
    "footfall_expected": "footfall",
    "dates": "date",
    "budget": "budget_inr",
    "budget_range_inr": "budget_inr",
    "budget_range": "budget_inr",
    "contact": "contact_email",
}


def _as_int(value: Any, field: str) -> int:
    """Coerce a JSON number or a numeric string to ``int``, refusing anything else."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise ValueError(f"{field} must be a whole number, got {value}")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip().replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    raise ValueError(f"{field} must be an integer, got {type(value).__name__}")


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, not a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "").replace("₹", "").replace("INR", ""))
        except ValueError as exc:
            raise ValueError(f"{field} must be a number, got {value!r}") from exc
    raise ValueError(f"{field} must be a number, got {type(value).__name__}")


def _string_list(value: Any, field: str) -> list[str]:
    """Accept ``"a,b,c"``, ``["a","b","c"]`` or ``None``; always return a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                raise ValueError(f"{field}[{index}] must be a string, got {type(item).__name__}")
            out.append(str(item).strip())
        return [item for item in out if item]
    raise ValueError(f"{field} must be a string or a list of strings, "
                     f"got {type(value).__name__}")


class CreateEventRequest(BaseModel):
    """Body of ``POST /api/events``.

    ``extra="forbid"`` is the load-bearing line. Without it this class would
    reproduce the exact defect described in the module docstring: a caller sends
    ``footfall_expected``, pydantic drops it, and the response looks successful.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    #: Client-supplied id is optional; the server mints ``evt_<hex>`` when absent,
    #: so a client cannot collide with another event by guessing an id.
    event_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=200)
    footfall: int = Field(ge=0, le=10_000_000)
    #: Single date for a one-day event, ``"a to b"`` for a multi-day one.
    date: str = Field(min_length=4, max_length=120)
    audience: str = Field(min_length=1, max_length=500)
    budget_inr: float | None = Field(default=None, ge=0)
    categories_wanted: list[str] = Field(default_factory=list)
    deliverables_offered: list[str] = Field(default_factory=list)
    contact_email: str | None = None

    #: Populated **by the server** during coercion. Any client-supplied value is
    #: discarded in :meth:`_absorb` so a caller cannot forge a coercion report.
    coercions: list[str] = Field(default_factory=list, exclude=True)

    # ------------------------------------------------------------------ coercions
    @model_validator(mode="before")
    @classmethod
    def _absorb(cls, data: Any) -> Any:
        """Map aliases and reshape list/dict inputs; refuse everything else.

        Runs before field validation so that by the time pydantic checks types,
        ``footfall`` really is an ``int`` and ``date`` really is a ``str``.
        """
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        # The coercion log is server-owned. Discard whatever arrived under that
        # key before doing anything else.
        raw.pop("coercions", None)
        coercions: list[str] = []

        # ---- aliases -------------------------------------------------------
        for legacy, canonical in ALIASES.items():
            if legacy not in raw:
                continue
            legacy_value = raw.pop(legacy)
            if canonical in raw and raw[canonical] != legacy_value:
                raise ValueError(
                    f"both {legacy!r} and {canonical!r} were supplied with "
                    f"different values ({legacy_value!r} vs {raw[canonical]!r}); "
                    f"send one"
                )
            raw[canonical] = legacy_value
            coercions.append(f"{legacy} -> {canonical}")

        # ---- footfall ------------------------------------------------------
        if "footfall" in raw and raw["footfall"] is not None:
            try:
                raw["footfall"] = _as_int(raw["footfall"], "footfall")
            except ValueError as exc:
                raise ValueError(str(exc)) from exc

        # ---- date: str | list[str] | {"from":..,"to":..} ---------------------
        if "date" in raw and raw["date"] is not None:
            value = raw["date"]
            if isinstance(value, (list, tuple)):
                parts = [str(v).strip() for v in value if str(v).strip()]
                if not parts:
                    raise ValueError("dates[] was supplied but empty")
                raw["date"] = " to ".join(parts) if len(parts) == 2 else ", ".join(parts)
                coercions.append(f"date: {len(parts)} date(s) -> {raw['date']!r}")
            elif isinstance(value, dict):
                start = value.get("from") or value.get("start") or value.get("date")
                end = value.get("to") or value.get("end")
                if not start:
                    raise ValueError(f"date object needs 'from'/'start'/'date', got {sorted(value)}")
                raw["date"] = f"{start} to {end}" if end else str(start)
                coercions.append(f"date object -> {raw['date']!r}")
            elif not isinstance(value, str):
                raise ValueError(
                    f"date must be a string, a list of strings, or an object "
                    f"with 'from'/'to'; got {type(value).__name__}"
                )

        # ---- budget_inr: number | [lo, hi] | {"min":..,"max":..} ------------
        # A range has no single value, and inventing one silently is the bug
        # this module is about. The rule is fixed (upper bound == the amount
        # being sought, which is what A2 prices against) and reported.
        if "budget_inr" in raw and raw["budget_inr"] is not None:
            value = raw["budget_inr"]
            if isinstance(value, (list, tuple)):
                numbers = [_as_float(v, "budget_inr") for v in value]
                if not numbers:
                    raise ValueError("budget_range_inr[] was supplied but empty")
                raw["budget_inr"] = max(numbers)
                coercions.append(
                    f"budget range {numbers} -> budget_inr={max(numbers)} "
                    f"(upper bound: the amount being sought)"
                )
            elif isinstance(value, dict):
                lo = value.get("min")
                hi = value.get("max")
                if hi is None and lo is None:
                    raise ValueError(f"budget object needs 'min' and/or 'max', got {sorted(value)}")
                raw["budget_inr"] = _as_float(hi if hi is not None else lo, "budget_inr")
                coercions.append(
                    f"budget object {value} -> budget_inr={raw['budget_inr']} "
                    f"(upper bound: the amount being sought)"
                )
            else:
                raw["budget_inr"] = _as_float(value, "budget_inr")

        # ---- audience: str | {"age_range":..,"profile":..,"interests":[..]} --
        if isinstance(raw.get("audience"), dict):
            audience = raw["audience"]
            pieces: list[str] = []
            for key in ("age_range", "profile", "description"):
                value = audience.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            interests = audience.get("interests")
            if isinstance(interests, (list, tuple)):
                joined = ", ".join(str(i).strip() for i in interests if str(i).strip())
                if joined:
                    pieces.append(f"interests: {joined}")
            raw["audience"] = "; ".join(pieces) or "unspecified"
            coercions.append(
                "audience object -> string "
                f"({', '.join(k for k in ('age_range', 'profile', 'interests') if k in audience)})"
            )

        # ---- categories / deliverables: "a,b" | [...] -----------------------
        for key in ("categories_wanted", "deliverables_offered"):
            if isinstance(raw.get(key), str):
                raw[key] = _string_list(raw[key], key)
                coercions.append(f"{key}: comma-separated string -> list")

        # ---- contact: {"name":..,"email":..} | "a@b.c" ----------------------
        if isinstance(raw.get("contact_email"), dict):
            contact = raw["contact_email"]
            email = contact.get("email")
            if not email:
                raise ValueError(
                    f"contact object needs an 'email' field, got {sorted(contact)}"
                )
            raw["contact_email"] = email
            coercions.append("contact object -> contact_email")

        raw["coercions"] = coercions
        return raw

    @field_validator("contact_email")
    @classmethod
    def _email_shape(cls, value: str | None) -> str | None:
        """Same plausibility check as ``EventProfile``, at the edge."""
        if value is None or value == "":
            return None
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError(f"not a plausible email address: {value!r}")
        return value

    # ------------------------------------------------------------------- output
    def to_profile(self, *, event_id: str | None = None) -> EventProfile:
        """Build the frozen ``EventProfile`` this request describes.

        The server-generated id wins over any client-supplied one so two clients
        cannot claim the same ``event_id`` by guessing; a caller that needs a
        specific id should pass it explicitly to :func:`create_event`.
        """
        from core.ids import new_id

        return EventProfile(
            event_id=event_id or new_id("evt"),
            name=self.name,
            location=self.location,
            footfall=self.footfall,
            date=self.date,
            audience=self.audience,
            budget_inr=self.budget_inr,
            categories_wanted=list(self.categories_wanted),
            deliverables_offered=list(self.deliverables_offered),
            contact_email=self.contact_email,
        )


class CreateEventResponse(BaseModel):
    """What ``POST /api/events`` returns.

    ``coercions`` is not decoration: it is how the caller learns that a list of
    six deliverables arrived as a list of six, and that a budget *range* became a
    single figure.
    """

    model_config = ConfigDict(extra="forbid")

    event: EventProfile
    run_id: str
    board_entry_id: str
    #: Zone on the blackboard the profile was posted to.
    board_zone: str = "event"
    coercions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# =========================================================================== pipeline
class PipelineStage(BaseModel):
    """Static description of one pipeline stage, for the status panel."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    agent: str
    label: str
    #: ``None`` when the stage performs no irreversible side effect.
    gate_kind: str | None = None
    description: str = ""


class StageRequest(BaseModel):
    """Common body for the stage endpoints.

    ``extra="forbid"`` here too, for the same reason as on the event model: a
    typo in a stage parameter should be a 422, not a parameter that quietly does
    nothing.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    #: Overrides ``settings.agent_step_budget`` / ``agent_deadline_s`` for this
    #: call only, so a demo can make an agent visibly rush or dawdle.
    step_budget: int | None = Field(default=None, ge=1, le=64)
    deadline_s: float | None = Field(default=None, gt=0, le=3600)
    #: Human-readable label attached to the trace for this invocation.
    label: str | None = Field(default=None, max_length=120)
    #: LangGraph thread to resume. When supplied the stage continues that
    #: thread instead of starting a fresh one, so a gate retry resumes the
    #: parked run rather than re-running the pipeline from scratch.
    thread_id: str | None = Field(default=None, max_length=128)


class OutreachRequest(StageRequest):
    """Body of ``POST /api/outreach`` and ``/api/events/{id}/outreach``.

    There is no ``approve: bool`` field. The previous API had one, and the
    previous *graph* called ``send_day0(approve=True)`` unconditionally, so the
    flag was decorative: it read as a human decision and authorised nothing. The
    only thing that authorises a send here is an ``Approval`` row, and the only
    way to create one is ``POST /api/gates/{gate_id}``.
    """

    gate_id: str | None = Field(default=None, max_length=64)
    #: Which event's run to act on. Required by ``POST /api/outreach``; the
    #: event-scoped form (``/api/events/{id}/outreach``) supplies it from the path.
    event_id: str | None = Field(default=None, max_length=64)
    #: Restrict the wave to these brands. ``None`` means every ready thread.
    brands: list[str] | None = None


class ReplyRequest(StageRequest):
    """An inbound sponsor reply handed to the intent router.

    ``brand`` is optional: when absent the API falls back to the run's own
    state (newest thread/offer/lead brand, via ``graph.build._last_brand``)
    and only answers 422 when no brand exists anywhere. ``extra="forbid"``
    is kept on purpose -- an unknown key such as ``event_id`` in the body is
    still a loud 422, not a silently dropped field.
    """

    brand: str | None = Field(default=None, min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=8000)
    subject: str = Field(default="", max_length=400)


class ContractRequest(StageRequest):
    """Request to draft or release an MoU. Release requires a ``MOU`` approval.

    ``brand`` is optional for the same reason as on :class:`ReplyRequest`:
    the run usually already knows which sponsor it is contracting with, so a
    body of ``{}`` is accepted whenever the brand is inferable from state.
    ``extra="forbid"`` stays for the same reason: typos must 422, not vanish.
    """

    brand: str | None = Field(default=None, min_length=1, max_length=200)
    gate_id: str | None = Field(default=None, max_length=64)
    amount_inr: float | None = Field(default=None, ge=0)
    terms: str | None = Field(default=None, max_length=4000)


class ComplianceRequest(StageRequest):
    """Ask A5 to verify deliverable promises. Read-only; no gate needed."""

    brand: str | None = Field(default=None, max_length=200)


class AuditRequest(StageRequest):
    """Ask A6 for the ROI report. Read-only; no gate needed."""

    actual_spend: float | None = Field(default=None, ge=0)
    actual_leads: int | None = Field(default=None, ge=0)


# ============================================================================= gates
class GateDecisionRequest(BaseModel):
    """A human's answer to one gate.

    ``instruction`` is free text and is recorded verbatim on both the
    :class:`~core.schemas.HumanDecision` and the
    :class:`~core.schemas.Approval`. That is the whole reason the approve/reject
    split exists: a rejection without a reason is not a decision, it is a shrug.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    outcome: str = Field(min_length=1, max_length=16)
    #: Who answered. Recorded on the durable row; ``"auto"`` marks a
    #: non-human resolution and is the only value a script should use.
    decided_by: str = Field(min_length=1, max_length=120)
    instruction: str = Field(default="", max_length=4000)
    #: When false (the default), a REJECT leaves the pending gate on the books so
    #: the reviewer can revise and re-answer instead of re-running the pipeline.
    close_gate: bool = True

    @field_validator("outcome")
    @classmethod
    def _known_outcome(cls, value: str) -> str:
        from core.schemas import GateOutcome

        try:
            return GateOutcome(value.strip().lower()).value
        except ValueError as exc:
            valid = [o.value for o in GateOutcome]
            raise ValueError(f"outcome must be one of {valid}, got {value!r}") from exc


class GateActionResponse(BaseModel):
    """Result of answering a gate: what was decided, and the durable proof."""

    model_config = ConfigDict(extra="forbid")

    gate: dict[str, Any]
    decision: dict[str, Any]
    approval: dict[str, Any]
    run_id: str
    #: True when ``decided_by`` is not ``"auto"``.
    human: bool
    #: The effect this approval now authorises, and whether it has been exercised.
    authorises: str | None = None
    board_entry_ids: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ResumeRequest(BaseModel):
    """Resume a parked run on the same thread.

    ``gate_id`` names the API-ledger gate being answered (if any); the graph's
    own interrupt is resumed with the same ``outcome``/``decided_by`` so one
    human answer satisfies both ledgers instead of leaving two gates for one
    action.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    outcome: str = Field(min_length=1, max_length=16)
    decided_by: str = Field(min_length=1, max_length=120)
    instruction: str = Field(default="", max_length=4000)
    gate_id: str | None = Field(default=None, max_length=64)
    thread_id: str | None = Field(default=None, max_length=128)

    @field_validator("outcome")
    @classmethod
    def _known_outcome(cls, value: str) -> str:
        from core.schemas import GateOutcome

        try:
            return GateOutcome(value.strip().lower()).value
        except ValueError as exc:
            valid = [o.value for o in GateOutcome]
            raise ValueError(f"outcome must be one of {valid}, got {value!r}") from exc


# ============================================================================= health
class HealthResponse(BaseModel):
    """Deliberately permissive: the shape of ``health_payload`` changes as
    siblings land, and a strict response model would then make ``/health``
    return 500 during exactly the concurrent development this API is built for."""

    model_config = ConfigDict(extra="allow")


class UnavailableResponse(BaseModel):
    """The honest failure body. Returned with HTTP 503."""

    model_config = ConfigDict(extra="allow")

    error: str
    unavailable: bool = True
    reason: str = ""
    subsystem: str = ""
    where: str = ""
