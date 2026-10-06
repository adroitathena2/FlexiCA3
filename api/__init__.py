"""Paytriq API: the FastAPI layer over a 7-agent campus-sponsorship pipeline.

What lives here
---------------
``main``          app factory, CORS, lifespan, static mount
``schemas``       request/response models (strict: unknown fields are 422)
``deps``          lazy, fault-tolerant wiring for every sibling package
``routes_events`` create/read an ``EventProfile``; publish its JSON Schema
``routes_pipeline`` the seven stages, with gate enforcement and decision provenance
``routes_gates``  human approvals -- the control surface
``routes_trace``  trace, summary, blackboard, coordination artefacts, linter
``stream``        Server-Sent Events: the live trace feed

Imports are **lazy and fault-tolerant** on purpose. ``api`` is the only package
that depends on every other one, and those packages are written concurrently. A
module-scope ``import graph`` would make this package fail to start the moment a
sibling is mid-write; instead every dependency is constructed inside a function
behind :class:`api.deps._Lazy`, which caches either the instance or the reason it
could not be built. Handlers that need a missing subsystem return
``503 {"error": ..., "unavailable": true, "reason": ...}``.

:func:`~api.main.create_app` is the entry point. Run it with::

    uvicorn api.main:app --reload
    # or
    uvicorn api.main:create_app --factory
"""
from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["create_app", "__version__"]


def __getattr__(name: str) -> Any:
    """Expose ``create_app`` without importing ``fastapi`` at package import.

    ``api/__init__.py`` is imported by every module in this package, so anything
    it did eagerly would be paid for eight times over -- and a missing optional
    dependency in ``main`` would take down ``api.schemas`` with it. The function
    is resolved on first attribute access instead.
    """
    if name in ("create_app", "app"):
        from . import main

        return main.create_app if name == "create_app" else getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
