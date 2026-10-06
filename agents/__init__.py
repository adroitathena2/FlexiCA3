"""Paytriq reasoning agents.

Seven reasoning agents: A1 Discovery, A2 Pricing, A3 Outreach, A4 Contract,
A5 Compliance, A6 Audit, A7 Arbiter.

``AgentId.ENVIRONMENT`` is **not** an agent. It is the simulated sponsor
counterparty in :mod:`agents.environment`, part of the test environment rather
than the system under test — the same role the simulated opponents play in
SOTOPIA (Zhou et al., ICLR 2024) and NegotiationArena (Bianchi et al., ICML 2024).

Imports are **lazy** on purpose. The agents are developed as independent,
separately-owned modules, so this package must import cleanly even when only some
of them are present. ``graph.registry`` therefore discovers agents by scanning for
:class:`~agents.base.ReActAgent` subclasses rather than by importing names here.
"""
from __future__ import annotations

from .base import AgentResult, ReActAgent

__all__ = [
    "ReActAgent",
    "AgentResult",
    "AGENT_MODULES",
    "load_agent",
    "discover",
]

#: module name -> exported agent class name. Advisory only; discovery does not
#: rely on it, but it documents the intended layout and gives a helpful error.
AGENT_MODULES: dict[str, str] = {
    "a1_discovery": "DiscoveryAgent",
    "a2_pricing": "PricingAgent",
    "a3_outreach": "OutreachAgent",
    "a4_contract": "ContractAgent",
    "a5_compliance": "ComplianceAgent",
    "a6_audit": "AuditAgent",
    "a7_arbiter": "ArbiterAgent",
}


def load_agent(module: str, class_name: str | None = None):
    """Import one agent class lazily. Raises a clear error if it is absent."""
    import importlib

    name = class_name or AGENT_MODULES.get(module, "")
    if not name:
        raise KeyError(f"no class name known for agent module {module!r}")
    mod = importlib.import_module(f"{__name__}.{module}")
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise AttributeError(
            f"{module!r} does not export {name!r}; "
            f"available: {[n for n in vars(mod) if n.endswith('Agent')]}"
        ) from exc


def discover() -> dict:
    """Return every importable ``ReActAgent`` subclass, keyed by ``AgentId``.

    Scans the package so that a partially-built tree still yields a working set,
    and never raises for a missing sibling module.
    """
    import importlib
    import pkgutil

    from core.schemas import AgentId

    found: dict[AgentId, type[ReActAgent]] = {}
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("a") or info.name.endswith("_base"):
            continue
        if info.name == "environment":
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
        except ImportError:
            continue
        for attr in vars(mod).values():
            if (isinstance(attr, type) and issubclass(attr, ReActAgent)
                    and attr is not ReActAgent):
                try:
                    found[attr.id] = attr
                except AttributeError:
                    continue
    return found
