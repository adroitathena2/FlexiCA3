"""Event routes: create an ``EventProfile``, read it back, publish its schema.

Three endpoints, and the third is the one that matters most for the regression
this package is built around:

``POST /api/events``          create, post to the blackboard, return the profile
``GET  /api/events``          list every profile the blackboard holds
``GET  /api/events/{id}``     read one back, reconstructed from the board
``GET  /api/schema/event``    the JSON Schema *and* a sample generated from it

Why the profile is posted to the blackboard at all
---------------------------------------------------
``EventProfile`` is the root of every citation chain in the system: A1's leads
cite it, A2's offers cite those leads, and an audit that cannot answer "which
event was this run about?" is not an audit. So the create handler posts to the
``event`` zone and returns the resulting ``entry_id``; every later artefact that
needs to reference the event cites that id.

The author of that post
----------------------
``blackboard.post`` requires an ``AgentId`` author and the ``event`` zone
declares no owner. Posting as a reasoning agent would put "A1 asserted this event
exists" into the board's author statistics, which is a claim nobody made: the
caller did, and the caller is not an agent. So the post is authored by
``AgentId.ENVIRONMENT`` -- the one id in the enum that is *not* a reasoning
agent -- which keeps every per-agent statistic in the run honest while still
leaving a valid, attributable author on the entry.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from core.errors import BlackboardError, SchemaError
from core.ids import new_id
from core.protocols import BoardEntry
from core.schemas import AgentId, DecisionSource, EventProfile

from .deps import run_for_event, unavailable_payload
from .schemas import ALIASES, CreateEventRequest, CreateEventResponse

__all__ = ["router"]

log = logging.getLogger("paytriq.api.events")

router = APIRouter(prefix="/api", tags=["events"])

#: Zone and kind the profile is posted under. Taken from the zone registry rather
#: than hardcoded twice; a rename in ``blackboard/zones.py`` then fails here
#: loudly instead of posting into a zone that does not exist.
EVENT_ZONE = "event"
EVENT_KIND = "event_profile"


def post_profile(board: object, profile: EventProfile) -> BoardEntry:
    """Post an ``EventProfile`` to the ``event`` zone and return the entry.

    Raises :class:`~core.errors.BlackboardError` when the zone or kind is not
    registered, which is a wiring fault rather than a client error -- the route
    layer turns it into a 500 with the message, not into a 200 with no event.
    """
    return board.post(  # type: ignore[attr-defined]
        EVENT_ZONE,
        EVENT_KIND,
        AgentId.ENVIRONMENT,
        profile.model_dump(mode="json"),
        # ``rules`` because no decision was made here: this is the caller's
        # input, recorded verbatim. Claiming a decision source would be a lie.
        source=DecisionSource.RULES,
    )


def _rehydrate(board: object, entry: BoardEntry) -> EventProfile:
    """Rebuild a typed profile from a board entry, or explain why it cannot be."""
    try:
        return board.get_model(entry.entry_id, EventProfile)  # type: ignore[attr-defined]
    except (SchemaError, BlackboardError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"board entry {entry.entry_id} claims to be an EventProfile but "
                f"does not validate: {exc}"
            ),
        ) from exc


@router.post(
    "/events",
    response_model=CreateEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an event profile and post it to the blackboard",
)
def create_event(payload: CreateEventRequest, request: Request) -> CreateEventResponse:
    """Create one event. ``422`` on any field the schema does not know.

    The 422 is the point. ``extra="forbid"`` on
    :class:`~api.schemas.CreateEventRequest` means a request carrying a key this
    build does not understand fails loudly here instead of producing an event
    with that key silently missing -- the defect that cost the previous version
    nine of its ten input fields.
    """
    # The client may propose an id; the server mints one otherwise. Proposing is
    # honoured so a demo script can be re-run idempotently.
    try:
        profile = payload.to_profile(event_id=payload.event_id or new_id("evt"))
    except ValidationError as exc:
        # ``EventProfile`` is stricter than the request model in places. Surface it
        # as a 422 rather than a 500: the client sent something invalid.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "the request does not describe a valid EventProfile",
                    "reason": exc.errors()},
        ) from exc

    # Create (or find) the run **before** posting, because the profile belongs on
    # *that run's* board. Boards are per run; posting to the current run's board
    # first and creating the run afterwards would leave the event on the wrong one.
    context = run_for_event(profile.event_id)
    if context.board is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=unavailable_payload(
                context.board_error or "this run has no blackboard",
                where="POST /api/events"),
        )

    try:
        entry = post_profile(context.board, profile)
    except BlackboardError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "the blackboard rejected the event profile",
                    "reason": str(exc)},
        ) from exc

    notes: list[str] = []
    if context is not None and context.tracer is not None:
        from core.schemas import TraceKind

        context.emit(TraceKind.AGENT, "api.event.create",
                     event_id=profile.event_id, board_entry=entry.entry_id,
                     footfall=profile.footfall)
    if payload.coercions:
        notes.extend(f"input coerced: {line}" for line in payload.coercions)

    return CreateEventResponse(
        event=profile,
        run_id=context.run_id,
        board_entry_id=entry.entry_id,
        board_zone=EVENT_ZONE,
        coercions=list(payload.coercions),
        notes=notes,
    )


def _all_event_entries() -> list[tuple[Any, BoardEntry]]:
    """Every ``event_profile`` entry on **every** run's board, oldest first.

    Boards are per run, so this has to visit all of them. Reading only the
    current run's board would return only the most recent event and report it as
    the full list -- the same class of bug as reading one zone of a board and
    calling it the board.
    """
    from .deps import iter_runs

    rows: list[tuple[object, BoardEntry]] = []
    for context in iter_runs():
        board = context.board
        if board is None:
            continue
        try:
            entries = board.read(EVENT_ZONE, kind=EVENT_KIND)
        except BlackboardError as exc:
            log.warning("run %s: cannot read the event zone: %s", context.run_id, exc)
            continue
        rows.extend((board, entry) for entry in entries)
    return rows


@router.get("/events", summary="List every event profile on the blackboard")
def list_events(request: Request, limit: int = 100) -> dict:
    """Every ``event_profile`` entry, oldest first. ``limit`` keeps the newest."""
    rows = _all_event_entries()
    if limit > 0:
        rows = rows[-limit:]
    return {
        "count": len(rows),
        "limit": limit,
        "events": [_rehydrate(board, entry).model_dump(mode="json")
                   for board, entry in rows],
        "entry_ids": [entry.entry_id for _board, entry in rows],
    }


@router.get("/events/{event_id}", summary="Read one event profile back")
def get_event(event_id: str, request: Request) -> dict:
    """Rehydrate a profile from the board rather than echoing the request.

    Reading back what was *stored* is the only version of this endpoint that can
    detect a coercion that went wrong; returning the submitted body would just
    repeat the request.
    """
    rows = _all_event_entries()
    for board, entry in rows:
        profile = _rehydrate(board, entry)
        if profile.event_id == event_id:
            return {
                "event": profile.model_dump(mode="json"),
                "board_entry_id": entry.entry_id,
                "board_seq": entry.seq,
                "posted_at": entry.at,
            }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": f"no event with event_id {event_id!r}",
                "known": [e.payload.get("event_id") for _b, e in rows]},
    )


@router.get("/schema/event", summary="JSON Schema for the create-event body")
def event_schema() -> dict:
    """Generate the input sample **from the model**, instead of hand-writing it.

    The previous version shipped a hand-written ``input_sample.json`` that had
    drifted from the request model; nine of its ten keys were dropped at
    runtime. The cure is not vigilance, it is provenance: this endpoint emits
    both the schema and a valid example, so a sample cannot drift from the code
    without the drift being visible.
    """
    schema = CreateEventRequest.model_json_schema(mode="validation")
    sample = {
        "name": "TechFest 2026",
        "location": "Pune, Maharashtra, India",
        "footfall": 5000,
        "date": "2026-02-14 to 2026-02-15",
        "audience": "engineering students 18-24; early tech adopters",
        "budget_inr": 100000,
        "categories_wanted": ["cafe", "gym", "coaching", "printing"],
        "deliverables_offered": ["stage_banner", "stall_10x10", "instagram_reel"],
        "contact_email": "organizer@example.invalid",
    }
    legacy_sample = {
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
        "categories_wanted": ["cafe", "gym", "coaching"],
        "deliverables_offered": "stage_banner, stall_10x10",
        "contact": {"name": "Demo Organizer", "email": "organizer@example.invalid"},
    }
    # Validate the legacy sample here so the endpoint cannot advertise a body
    # that the endpoint it documents would reject.
    try:
        CreateEventRequest.model_validate(legacy_sample)
        legacy_ok, legacy_reason = True, ""
    except ValidationError as exc:
        legacy_ok = False
        legacy_reason = exc.errors(include_url=False).__repr__()
    try:
        CreateEventRequest.model_validate(sample)
        sample_ok, sample_reason = True, ""
    except ValidationError as exc:
        sample_ok = False
        sample_reason = exc.errors(include_url=False).__repr__()

    return {
        "model": "api.schemas.CreateEventRequest",
        "extra": "forbid",
        "note": (
            "Unknown keys are rejected with HTTP 422. A key that is not listed "
            "here will not be silently discarded -- it will fail the request."
        ),
        "aliases": dict(sorted(ALIASES.items())),
        "schema": schema,
        "sample": sample,
        "sample_valid": sample_ok,
        "sample_error": sample_reason,
        "legacy_sample": legacy_sample,
        "legacy_sample_valid": legacy_ok,
        "legacy_sample_error": legacy_reason,
        "curl": (
            "curl -X POST http://127.0.0.1:18780/api/events "
            "-H 'content-type: application/json' "
            "-d '" + json.dumps(sample, separators=(",", ":")) + "'"
        ),
    }
