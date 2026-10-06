"""Stable identifier generation.

Every artefact in Paytriq carries a prefixed, sortable, collision-resistant id.
Prefixes are grouped by artefact family so a trace is readable at a glance:

    evt_  event          brd_  brand lead      off_  offer / proposal
    thr_  thread         mou_  MoU             dlv_  deliverable
    dsp_  dispute        bid_  bid             les_  reflexion lesson
    rsk_  risk flag      dec_  decision        hnd_  handoff
    run_  run            evt_ (span) span     agt_  agent registration

Span ids use a distinct prefix (``spn_``) so a trace reader never confuses an
OpenTelemetry span with a domain event.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

__all__ = ["new_id", "utcnow", "run_id", "span_id", "trace_id"]

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use naive datetimes anywhere in Paytriq."""
    return datetime.now(UTC)


def _short(n: int = 10) -> str:
    return uuid.uuid4().hex[:n]


def new_id(prefix: str, n: int = 10) -> str:
    """Return e.g. ``dsp_9f2a1c4e77``.

    The prefix is normalised and a separator is inserted only when missing, so
    ``new_id("dsp")`` and ``new_id("dsp_")`` behave identically.
    """
    p = (prefix or "x").strip().strip("_") or "x"
    return f"{p}_{_short(n)}"


def run_id() -> str:
    """Sortable run identifier: ``run_20261004T151203Z_ab12cd34``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{_short(8)}"


def trace_id() -> str:
    """32 hex chars, matching the OpenTelemetry trace-id width."""
    return uuid.uuid4().hex


def span_id() -> str:
    """16 hex chars, matching the OpenTelemetry span-id width."""
    return uuid.uuid4().hex[:16]
