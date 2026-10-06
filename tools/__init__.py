"""Paytriq tool layer: real backends, honestly labelled when there are none.

This package is deliberately boring about failure. Every tool implements
``core.protocols.Tool`` and returns a ``ToolResult`` whose ``status`` says what
actually happened:

* ``OK``          — a real backend produced this. For :class:`SendEmailTool`
                    specifically, OK means a transport accepted the message.
* ``CACHED``      — this is :mod:`tools.fixtures` seed data. Never a live result.
* ``UNAVAILABLE`` — the backend could not be reached, or is switched off.
* ``FAILED``      — the backend answered with something unusable.

The integrity rules this package exists to keep
-----------------------------------------------
1. **No fabrication, ever.** No invented businesses, distances, ratings, phone
   numbers, email addresses, logo counts, or fulfilment verdicts. If the data is
   not there, the tool says so.
2. **Fixtures are labelled beyond argument.** Seed records carry
   ``source="fixture"``, ``FIXTURE_``-prefixed names, and reach callers only
   through ``ToolStatus.CACHED`` + ``degraded=True`` results.
3. **Live calls are gated** behind ``settings.tools_live`` (default ``False``)
   via :func:`tools.base.live_allowed`, the only door to the network.
4. **Evidence is inspected, not pattern-matched.** A URL or filename can never
   satisfy a promise; :class:`~tools.browser.VerifyEvidenceTool` will answer
   "cannot-verify" instead.
5. **Every network call has a timeout**, and no ``except Exception`` appears
   anywhere in this package.

Modules
-------
``base``      timeout wrapper, redaction, live/fixture gating, HTTP helpers
``maps``      Overpass discovery, Nominatim geocoding, haversine distance
``browser``   page fetch, and the evidence verifier
``email``     Resend delivery or an honest local spool
``pdf``       real PDFs, or honestly named ``.txt``/``.html``
``vision``    Gemini vision, with no substitute for a model
``fixtures``  synthetic seed data (see the warning at the top of that file)
``registry``  ``build_registry`` and the demo status panel
"""
from __future__ import annotations

from .base import (
    EARTH_RADIUS_KM,
    REDACTED,
    TOOL_CLASSES,
    TOOL_SPECS,
    BaseTool,
    ToolSpec,
    annotate,
    describe_tools,
    fixture_result,
    guarded,
    haversine_km,
    http_client,
    http_result,
    live_allowed,
    redact,
    settings_secrets,
    shaped_email,
    split_emails,
    tool,
    valid_email,
    valid_phone,
)
from .browser import FetchPageTool, VerifyEvidenceTool, extract_text
from .email import SendEmailTool
from .maps import CATEGORY_TAGS, DistanceTool, GeocodeTool, SearchBrandsTool
from .pdf import RenderPdfTool
from .registry import TOOL_ORDER, build_registry, describe, summary
from .vision import AnalyseEvidenceTool

__all__ = [
    # base
    "BaseTool", "ToolSpec", "tool", "guarded", "redact", "fixture_result",
    "annotate", "live_allowed", "http_client", "http_result",
    "settings_secrets", "describe_tools", "valid_email", "shaped_email",
    "valid_phone",
    "split_emails", "haversine_km", "EARTH_RADIUS_KM", "REDACTED",
    "TOOL_CLASSES", "TOOL_SPECS",
    # tools
    "SearchBrandsTool", "GeocodeTool", "DistanceTool", "CATEGORY_TAGS",
    "FetchPageTool", "VerifyEvidenceTool", "extract_text",
    "SendEmailTool", "RenderPdfTool", "AnalyseEvidenceTool",
    # registry
    "build_registry", "describe", "summary", "TOOL_ORDER",
]
