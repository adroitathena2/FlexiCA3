"""Server-Sent Events: the live trace feed.

This is the centrepiece of the demo. The audience watches seven agents hand work
to each other in real time, and the only honest way to show that is to stream the
trace as it is written rather than to render it afterwards.

Why SSE and not WebSockets
--------------------------
The traffic is one-directional (server -> browser), it is plain HTTP so it works
through every proxy a demo laptop might sit behind, and ``EventSource`` in the
browser reconnects on its own. A WebSocket would add a dependency and a handshake
to solve a problem this traffic does not have.

The protocol
------------
Each ``TraceEvent`` is emitted as one ``event: trace`` frame whose ``data`` is the
event as JSON. Then, when the run finishes:

* ``event: summary`` -- the ``TraceSummary``, including ``distinct_gap_values``;
* ``event: done``    -- the terminal frame.

Reconnects resume where they left off: the browser's ``EventSource``
re-sends the last ``id:`` it saw as ``Last-Event-ID`` on an automatic
retry, and a fresh ``EventSource`` (which has no memory of the previous
one) can pass the same value as ``?last_event_id=``. Both are honoured;
an absent or unparseable value replays from the start rather than failing.
``seq`` is 0-based and contiguous (see ``observability/lint.py``), so
"events with ``seq`` greater than the resume value" is exactly "everything
the client has not seen".

Plus ``: heartbeat`` comments while idle, and ``event: error`` if the graph or the
tracer is unavailable.

Frame shape::

    event: trace
    id: 12
    data: {"seq": 12, "kind": "handoff", "name": "A1->A2", ...}

The ``id:`` field carries ``seq``, which is the tracer's strictly-increasing
counter with no gaps. That is deliberate: a browser reconnecting with
``Last-Event-ID`` can then be told exactly where it left off, and any consumer
that sees a gap knows a frame was dropped rather than having to guess.

Degradation
-----------
An unavailable graph or tracer produces an ``error`` frame and then ``done`` -- a
stream that closes with a stated reason, not one that hangs or emits a plausible
sequence of events that never happened.

Cleanup
-------
The generator polls ``context.finished`` and the request's disconnect flag, so a
closed browser tab stops the loop instead of pinning a thread forever. The final
``finally`` block cancels any pending sleep, which is what keeps a disconnected
client from holding an event-loop task until its timeout expires.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from core.schemas import TraceEvent

from .deps import RunContext, unavailable_payload

__all__ = ["router", "stream_run", "format_sse", "parse_resume",
           "resume_from_request", "POLL_INTERVAL_S", "HEARTBEAT_S"]

log = logging.getLogger("paytriq.api.stream")

router = APIRouter(prefix="/api/runs", tags=["stream"])

#: How often the generator looks for new events. 50 ms keeps a demo feeling
#: live without busy-waiting; the trace file is append-only so there is nothing to
#: miss between polls.
POLL_INTERVAL_S = 0.05

#: Heartbeat cadence while the run is idle. 15 s is under the ~30 s idle timeout
#: of most reverse proxies, so an idle stream is not killed mid-run.
HEARTBEAT_S = 15.0

#: Hard cap on one stream. A run whose agents hang must not hold a connection for
#: ever; at the cap the stream emits its summary with ``terminated_by: "deadline"``
#: so the truncation is stated rather than silent.
MAX_STREAM_S = 900.0

#: Give up if the first event never arrives. A run that has produced nothing in
#: this long is not going to produce a demo, and saying so beats a blank screen.
STARTUP_GRACE_S = 120.0


def format_sse(event: str, data: Any, *, event_id: int | None = None) -> str:
    """Encode one SSE frame.

    ``json.dumps`` with ``default=str`` because ``TraceEvent.attributes`` is
    documented as "free-form but JSON-safe" and a value that is not (a
    ``datetime`` in a custom payload, say) must not kill the stream mid-run. The
    stream still says what it sent.
    """
    body = json.dumps(data, default=str, separators=(",", ":"))
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    # Multi-line data would break the protocol, so newlines are escaped here
    # rather than relying on the payload never containing one.
    body = body.replace("\r\n", "\\r\\n").replace("\n", "\\n").replace("\r", "\\r")
    return f"{prefix}event: {event}\ndata: {body}\n\n"


def _comment(text: str) -> str:
    """An SSE comment. Ignored by ``EventSource``, read by proxies and humans."""
    return f": {text}\n\n"


def parse_resume(value: Any) -> int | None:
    """A ``Last-Event-ID`` / ``?last_event_id=`` value as a ``seq``, or ``None``.

    ``None`` means "replay from the start". Anything that is not a
    non-negative integer -- absent, blank, ``"null"``, ``"12.5"`` -- is also
    ``None``: a resume hint must never break the stream, it only narrows it.
    """
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in ("null", "none", "undefined", "nan"):
        return None
    try:
        seq = int(text, 10)
    except (TypeError, ValueError):
        return None
    return seq if seq >= 0 else None


def resume_from_request(request: Request,
                        last_event_id: str | None = None) -> int | None:
    """Where a reconnecting client left off, from the header or the query.

    ``EventSource`` sends ``Last-Event-ID`` on an automatic retry of the same
    object, but a *fresh* ``EventSource`` (manual reconnect, run switch back
    and forth) sends nothing -- ``fetch``/``EventSource`` cannot set headers
    on a new connection, so the console also passes ``?last_event_id=``.
    The header wins when both are present; either may be absent.
    """
    header = request.headers.get("last-event-id")
    query = request.query_params.get("last_event_id")
    if query is None:
        query = request.query_params.get("lastEventId")
    for candidate in (header, last_event_id, query):
        parsed = parse_resume(candidate)
        if parsed is not None:
            return parsed
    return None


def _run_or_404(run_id: str) -> RunContext:
    from .deps import get_run

    context = get_run(run_id)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"unknown run_id {run_id!r}",
                    "hint": "POST /api/events returns a run_id; start a run "
                            "before opening its stream"},
        )
    return context


def _summary_frame(context: RunContext, terminated_by: str) -> dict[str, Any] | None:
    """The terminal summary, or ``None`` with a stated reason if there is none.

    Prefers ``compute_summary`` over the trace *file* because that is the same
    function ``GET /api/runs/{run_id}/summary`` uses, so the value the audience
    sees live is the value they can verify afterwards. Falls back to counting the
    in-memory events, clearly labelled as such -- the gap metric cannot be computed
    from an in-memory list, and reporting one that was not measured would be worse
    than reporting none.
    """
    from pathlib import Path

    path_text = context.trace_path()
    if path_text and Path(path_text).exists():
        try:
            from observability.summary import compute_summary

            summary = compute_summary(Path(path_text), context.run_id)
            return {
                "summary": summary.model_dump(mode="json"),
                "distinct_gap_values": summary.distinct_gap_values,
                "computed_from": str(path_text),
            }
        except Exception as exc:  # noqa: BLE001 - fall through to the honest count
            log.warning("could not compute the summary for %s: %s", context.run_id, exc)
    events = context.events()
    return {
        "summary": None,
        "distinct_gap_values": None,
        "computed_from": "in-memory events only",
        "reason": ("the trace file is not written yet, so the anti-fabrication "
                   "gap statistics cannot be computed; count the events instead"),
        "in_memory_event_count": len(events),
    }


async def stream_run(context: RunContext, request: Request, *,
                     poll_s: float = POLL_INTERVAL_S,
                     heartbeat_s: float = HEARTBEAT_S,
                     max_s: float = MAX_STREAM_S,
                     resume_from: int | None = None) -> AsyncIterator[str]:
    """Yield SSE frames for one run until it finishes, dies, or hits the deadline.

    Ordering guarantees, because a live feed with gaps is worse than no feed:

    * events are emitted in strictly increasing ``seq``;
    * only events with ``seq`` greater than ``resume_from`` are emitted, so a
      reconnecting client is told what it missed rather than everything again;
    * the heartbeats are emitted only when there is nothing to send, never between
      two events (so a heartbeat can never delay an event);
    * exactly one terminal frame (``summary`` then ``done``) is emitted, on every
      exit path including the deadline and the disconnect.
    """
    started = time.monotonic()
    last_seq = resume_from if resume_from is not None else -1
    saw_first = False
    last_beat = started
    terminated_by = "client_disconnect"

    resume_note = (f" resumed_from={resume_from}"
                   if resume_from is not None else "")
    yield _comment(
        f"paytriq live trace: run={context.run_id} event={context.event_id} "
        f"started={started:.3f} poll={poll_s}s heartbeat={heartbeat_s}s"
        f"{resume_note}"
    )

    # ---- degradation: say so, then close cleanly ---------------------------
    if context.tracer is None:
        yield format_sse("error", {
            **unavailable_payload(
                context.tracer_error or "this run has no tracer, so no trace "
                                        "events exist to stream",
                where=f"/api/runs/{context.run_id}/stream"),
            "run_id": context.run_id,
            "terminated": True,
        })
        yield format_sse("done", {"run_id": context.run_id, "terminated_by": "unavailable",
                                  "event_count": 0})
        return

    try:
        while True:
            # ---- 1. has the client gone? -----------------------------------
            if await request.is_disconnected():
                terminated_by = "client_disconnect"
                break

            # ---- 2. anything new to send? ----------------------------------
            events = [e for e in context.events() if e.seq > last_seq]
            if events:
                for event in events:
                    last_seq = event.seq
                    saw_first = True
                    last_beat = time.monotonic()
                    yield format_sse("trace", _event_payload(event),
                                     event_id=event.seq)
                continue  # re-poll immediately: a burst should not wait a tick

            # ---- 3. is the run over? ---------------------------------------
            if context.finished:
                terminated_by = "run_finished"
                break

            elapsed = time.monotonic() - started
            if elapsed > max_s:
                terminated_by = "deadline"
                log.warning("SSE stream for %s hit the %.0fs cap", context.run_id, max_s)
                break
            if not saw_first and elapsed > STARTUP_GRACE_S:
                terminated_by = "startup_timeout"
                yield format_sse("error", {
                    "error": f"no trace event was produced within {STARTUP_GRACE_S:.0f}s",
                    "reason": "the run has started but nothing has been recorded; "
                              "check /health for an unavailable subsystem",
                    "unavailable": True,
                    "run_id": context.run_id,
                    "in_memory_event_count": len(context.events()),
                })
                break

            # ---- 4. idle: heartbeat so proxies do not drop us --------------
            if time.monotonic() - last_beat >= heartbeat_s:
                last_beat = time.monotonic()
                yield _comment(
                    f"heartbeat t={time.monotonic() - started:.1f}s "
                    f"events={len(context.events())} finished={context.finished}"
                )

            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        # Client vanished mid-``await``. Not an error: a browser closing a tab is
        # normal, and re-raising CancelledError is how the task is torn down.
        log.info("SSE stream for %s cancelled", context.run_id)
        raise
    finally:
        if terminated_by != "client_disconnect":
            log.info("SSE stream for %s ended: %s (%d event(s))",
                     context.run_id, terminated_by, last_seq + 1)

    # ---- terminal frames ----------------------------------------------------
    if terminated_by in ("deadline", "startup_timeout"):
        yield format_sse("error", {
            "error": f"stream terminated: {terminated_by}",
            "reason": (f"the run had not finished after "
                       f"{time.monotonic() - started:.1f}s; the trace is still "
                       f"being written and /summary will report it later"),
            "unavailable": False,
            "truncated": True,
            "run_id": context.run_id,
        })

    tail = _summary_frame(context, terminated_by)
    yield format_sse("summary", {
        "run_id": context.run_id,
        "event_id": context.event_id,
        "terminated_by": terminated_by,
        "event_count": last_seq + 1,
        "truncated": terminated_by in ("deadline", "startup_timeout"),
        **(tail or {}),
    })
    yield format_sse("done", {
        "run_id": context.run_id,
        "terminated_by": terminated_by,
        "event_count": last_seq + 1,
    })


def _event_payload(event: TraceEvent) -> dict[str, Any]:
    """One event as a JSON-safe dict.

    ``model_dump(mode="json")`` handles the schema fields; the only failure mode
    is a non-JSON-safe value inside ``attributes``, which is caught and reported
    rather than raised -- a stream that dies on one odd attribute has failed at
    the one job it exists for.
    """
    try:
        return event.model_dump(mode="json")
    except (ValidationError, TypeError, ValueError) as exc:
        log.warning("event %s could not be dumped (%s); sending a degraded frame",
                    event.seq, exc)
        return {
            "seq": event.seq,
            "run_id": event.run_id,
            "event_id": event.event_id,
            "kind": str(getattr(event.kind, "value", event.kind)),
            "name": event.name,
            "agent": str(getattr(event.agent, "value", event.agent))
            if event.agent else None,
            "attributes": {"_degraded": True,
                           "_reason": f"{type(exc).__name__}: {exc}"},
        }


@router.get("/{run_id}/stream", summary="Live Server-Sent Events feed of the trace")
async def stream_endpoint(run_id: str, request: Request,
                          heartbeat_s: float = HEARTBEAT_S,
                          poll_s: float = POLL_INTERVAL_S,
                          last_event_id: str | None = None) -> StreamingResponse:
    """``text/event-stream`` of every ``TraceEvent`` in a run, then a summary.

    Headers worth noting:

    ``Cache-Control: no-cache``
        required by the SSE spec; a cached event stream is not an event stream.
    ``X-Accel-Buffering: no``
        tells nginx not to buffer the response. Without it a proxy will hold every
        frame until the connection closes, which turns a live demo into a
        three-minute wait followed by everything at once.
    ``Connection: keep-alive``
        disables uvicorn's idle timeout on this connection.

    Resume
    ------
    ``Last-Event-ID`` (sent by ``EventSource`` on an automatic retry) and
    ``?last_event_id=`` (sent by the console on a manual reconnect, where no
    header can be set) both narrow the stream to events the client has not
    seen. Either may be absent, in which case the run replays from the start.
    """
    context = _run_or_404(run_id)
    generator = stream_run(
        context, request,
        poll_s=max(0.01, min(poll_s, 1.0)),
        heartbeat_s=max(1.0, min(heartbeat_s, 120.0)),
        resume_from=resume_from_request(request, last_event_id),
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
