"""Build the tool registry and the demo status panel.

``build_registry`` is the single wiring point named by
``core/protocols.py``: it instantiates every registered tool class with a
``Settings`` object and populates :data:`core.protocols.TOOL_REGISTRY`.

Two properties are maintained deliberately:

* **Instances are stateless.** Tools hold only their ``Settings``, a timeout,
  and the list of secrets to redact. Nothing accumulates between calls, so one
  instance can be shared across agents and across runs of a test.
* **The registry is rebuilt, not mutated.** Calling ``build_registry`` again
  replaces the contents. A test that flips ``tools_live`` gets a registry that
  actually reflects the new setting rather than a stale tool that quietly
  ignores it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.config import Settings, get_settings
from core.protocols import TOOL_REGISTRY, Tool

from .base import TOOL_CLASSES, TOOL_SPECS, BaseTool, describe_tools, live_allowed
from .browser import FetchPageTool, VerifyEvidenceTool
from .email import SendEmailTool
from .maps import DistanceTool, GeocodeTool, SearchBrandsTool
from .pdf import RenderPdfTool
from .vision import AnalyseEvidenceTool

__all__ = ["build_registry", "describe", "summary", "tool_classes", "TOOL_ORDER"]

#: Presentation order for the demo panel: discovery, then outreach, then the
#: gates that produce and verify artefacts. Anything not listed is appended
#: alphabetically, so adding a tool can never silently drop it from the panel.
TOOL_ORDER: tuple[str, ...] = (
    "search_brands",
    "geocode",
    "distance_km",
    "fetch_page",
    "verify_evidence",
    "send_email",
    "render_pdf",
    "analyse_evidence",
)

#: Explicit import-for-side-effect list. These imports are the reason this
#: module exists: the ``@tool`` decorator only runs when a tool module is
#: imported, so a registry built without them would be empty.
_MODULES = (SearchBrandsTool, GeocodeTool, DistanceTool, FetchPageTool,
            VerifyEvidenceTool, SendEmailTool, RenderPdfTool, AnalyseEvidenceTool)

#: Classes that must be present, in order, for the registry to be considered
#: complete. Kept separate from ``_MODULES`` so a typo in an import does not
#: quietly reduce coverage without failing.
_EXPECTED: tuple[str, ...] = TOOL_ORDER


def tool_classes() -> dict[str, type]:
    """Registered tool classes, keyed by tool name.

    Registration happens at import time via the ``@tool`` decorator, so this
    reflects exactly the modules imported by this package.
    """
    for cls in _MODULES:      # ensure the decorators have run
        del cls
    return dict(TOOL_CLASSES)


def build_registry(settings: Settings | None = None, *,
                   populate: bool = True) -> dict[str, Tool]:
    """Instantiate every registered tool and (by default) publish the registry.

    :param settings: the ``Settings`` to inject. Defaults to the process-wide
        instance. Tests pass their own, usually with ``tools_live`` toggled.
    :param populate: set ``False`` to build the mapping without touching the
        global registry — used by tests that must not affect other modules.
    :returns: name -> tool instance, in :data:`TOOL_ORDER` then alphabetical.
    :raises RuntimeError: if a registered tool class will not accept the
        ``settings`` keyword, or if a tool is missing. A silent omission would
        leave an agent calling a name that resolves to nothing.
    """
    cfg = settings if settings is not None else get_settings()
    instances: dict[str, Tool] = {}
    for name, cls in sorted(TOOL_CLASSES.items()):
        try:
            instances[name] = cls(settings=cfg)          # type: ignore[call-arg]
        except TypeError as exc:
            raise RuntimeError(
                f"tool {name!r} ({cls.__name__}) does not accept settings="
                f"{type(cfg).__name__}: {exc}"
            ) from None

    missing = [n for n in _EXPECTED if n not in instances]
    if missing:
        raise RuntimeError(
            f"tool registry is incomplete; missing {missing}. Import the tool "
            f"modules or update TOOL_ORDER.")

    ordered = {name: instances[name] for name in TOOL_ORDER if name in instances}
    ordered.update({n: t for n, t in sorted(instances.items()) if n not in ordered})

    if populate:
        TOOL_REGISTRY.clear()
        TOOL_REGISTRY.update(ordered)
    return ordered


def describe(tools: Mapping[str, Tool] | None = None,
             settings: Settings | None = None) -> list[dict[str, Any]]:
    """Rows for the demo status panel: availability and live/fixture mode.

    Pass ``tools`` to describe an existing mapping (the default is the global
    registry, so it reflects whatever ``build_registry`` last published).

    The ``mode`` column is the point of this function. A reader looking at a
    demo screenshot must be able to tell, without opening code, which tools are
    backed by Overpass and Resend and which are only serving labelled seed data.
    """
    mapping = tools if tools is not None else TOOL_REGISTRY
    rows = describe_tools(mapping)
    cfg = settings if settings is not None else get_settings()
    can_go_live, gate_reason = live_allowed(cfg)
    for row in rows:
        row["live_gate_open"] = can_go_live
        row["live_gate_reason"] = gate_reason
    return rows


def summary() -> str:
    """A fixed-width table of :func:`describe`, for the demo script.

    Written here rather than in the demo so the panel format is testable and
    identical everywhere it appears.
    """
    rows = describe()
    if not rows:
        return "(no tools registered — call tools.registry.build_registry() first)"
    widths = {
        "name": max(4, max(len(r["name"]) for r in rows)),
        "mode": max(4, max(len(str(r["mode"])) for r in rows)),
        "backend": max(7, max(len(str(r["backend"])) for r in rows)),
    }
    header = (f"{'tool'.ljust(widths['name'])}  {'mode'.ljust(widths['mode'])}  "
              f"{'backend'.ljust(widths['backend'])}  {'available':<9}  reason")
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name'].ljust(widths['name'])}  "
            f"{str(row['mode']).ljust(widths['mode'])}  "
            f"{str(row['backend']).ljust(widths['backend'])}  "
            f"{('yes' if row['available'] else 'no').ljust(9)}  "
            f"{row['reason']}"
        )
    return "\n".join(lines)


__all__ += ["BaseTool", "TOOL_SPECS"]
