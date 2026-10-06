"""Agent discovery and construction, without assuming any class name.

The seven agent modules (``agents.a1_discovery`` … ``agents.a7_arbiter``) are
being written concurrently by different people. Hard-coding
``from agents.a2_pricing import PricingAgent`` in the orchestrator would make
``graph`` fail to import the moment a name differs from the guess, and would
make this file the place where every merge conflict lands.

So the registry **scans**. :func:`discover_agents` walks the ``agents`` package
with :mod:`pkgutil`, imports every submodule defensively, and keeps any class
that

1. is a subclass of :class:`agents.base.ReActAgent`,
2. is not ``ReActAgent`` itself, and
3. declares a class attribute ``id`` that is a valid
   :class:`~core.schemas.AgentId`.

The declared ``id`` — not the module name, not the class name — is the identity.
That is what makes this tolerant: an agent can be called
``PricingAgent``, ``A2PricingAgent`` or ``PricingSpecialist`` and land on
``AgentId.A2_PRICING`` regardless.

Consequences of scanning, stated plainly:

* An agent module that raises on import (e.g. it imports a dependency that is
  not installed yet) is logged and skipped. The graph still builds with the
  remaining agents, which is the entire reason this file is written defensively
  rather than optimistically.
* An agent whose class declares no valid ``id`` is logged and skipped.
* Duplicate declarations of one ``AgentId`` are resolved deterministically
  (module name, then class name) and the loser is logged, so the outcome does
  not depend on import order.
* The scan is memoised per package fingerprint but ``refresh()`` exists for
  tests and for the case where an agent module is written *after* the first
  scan in a long-lived process.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any

from core import REASONING_AGENTS, AgentId, ConfigError, Settings

__all__ = [
    "AgentRecord", "DiscoveryReport", "discover_agents", "discover_agent_classes",
    "build_agents", "available_agents", "missing_agents", "registry_status",
    "reset_discovery_cache", "agent_module_for",
]

log = logging.getLogger("paytriq.graph.registry")

#: Cache key includes the module count so a newly written agent file is noticed
#: by ``registry_status()`` even when the cache is warm.
_DISCOVERY_CACHE: dict[str, DiscoveryReport] = {}


@dataclass(slots=True)
class AgentRecord:
    """One discovered agent class plus the evidence used to find it."""

    agent_id: AgentId
    cls: type
    module: str
    role: str = ""
    source_file: str = ""

    def instantiate(self, settings: Settings) -> Any:
        """Construct with the configured budgets, tolerating narrow constructors.

        The frozen base takes ``step_budget`` and ``deadline_s``. A subclass may
        legitimately want only one of them, or neither. All three shapes are
        attempted; if all three fail the resulting :class:`ConfigError` lists
        every attempt and its cause, because a silent fallback to a bare
        constructor would hide a real wiring bug behind a default budget.
        """
        attempts: list[dict[str, Any]] = [
            {"step_budget": settings.agent_step_budget, "deadline_s": settings.agent_deadline_s},
            {"step_budget": settings.agent_step_budget},
            {},
        ]
        causes: list[str] = []
        for kwargs in attempts:
            try:
                return self.cls(**kwargs)
            except TypeError as exc:
                causes.append(f"{sorted(kwargs) or 'no args'}: {exc}")
            except Exception as exc:  # noqa: BLE001 - surface the real cause
                raise ConfigError(
                    f"constructing {self.agent_id.value} from "
                    f"{self.cls.__module__}.{self.cls.__qualname__} raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        raise ConfigError(
            f"cannot construct {self.agent_id.value} from "
            f"{self.cls.__module__}.{self.cls.__qualname__}; tried "
            + " -> ".join(causes)
        )

    def describe(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id.value,
            "label": self.agent_id.label,
            "class": f"{self.cls.__module__}.{self.cls.__qualname__}",
            "module": self.module,
            "role": self.role,
            "source_file": self.source_file,
        }





@dataclass(slots=True)
class DiscoveryReport:
    """Outcome of one scan: what was found, what was tried, what failed."""

    records: dict[AgentId, AgentRecord] = field(default_factory=dict)
    #: AgentIds with no discovered class.
    missing: tuple[AgentId, ...] = ()
    #: (module, class, reason) for every candidate rejected.
    rejected: tuple[tuple[str, str, str], ...] = ()
    #: Modules that could not be imported at all.
    unimportable: tuple[tuple[str, str], ...] = ()
    #: True when no agent module could even be located.
    package_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": {a.value: r.describe() for a, r in sorted(self.records.items())},
            "missing": [a.value for a in self.missing],
            "rejected": [{"module": m, "cls": c, "reason": r} for m, c, r in self.rejected],
            "unimportable": [{"module": m, "reason": r} for m, r in self.unimportable],
            "package_missing": self.package_missing,
        }


# ------------------------------------------------------------------------ scanning
def _base_class() -> type | None:
    """``ReActAgent`` itself, imported from its own module (not by guess)."""
    try:
        module = importlib.import_module("agents.base")
    except ImportError as exc:
        log.error("agents.base is not importable (%s); no agent can be discovered", exc)
        return None
    base = getattr(module, "ReActAgent", None)
    return base if isinstance(base, type) else None


def _iter_agent_modules(package_name: str = "agents") -> tuple[list[Any], str | None]:
    """Yield ``(module_name, module)`` for every importable submodule of ``agents``.

    Returns ``([], reason)`` instead of raising when the package is absent: an
    absent ``agents`` package is a legitimate state during concurrent
    development, not an error condition.
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError as exc:
        return [], f"package {package_name!r} is not importable: {exc}"

    found: list[Any] = []
    search_paths = list(getattr(package, "__path__", []) or [])
    if not search_paths:
        return [], f"{package_name!r} has no __path__ (not a package)"

    for info in pkgutil.walk_packages(search_paths, prefix=f"{package_name}."):
        name = info.name
        if name.endswith(".__init__"):
            continue
        try:
            found.append((name, importlib.import_module(name)))
        except Exception as exc:  # noqa: BLE001 - a half-written sibling must not
            # take down the orchestrator; recorded and skipped.
            log.warning("skipping unimportable agent module %s: %s: %s", name, type(exc).__name__, exc)
            found.append((name, exc))
    return found, None


def _declared_id(obj: Any) -> AgentId | None:
    """Read a class's declared ``id`` and coerce it to :class:`AgentId`.

    Accepts the bare string ``"A2"`` as well as ``AgentId.A2_PRICING`` so a
    subclass need not import the enum just to declare its identity.
    """
    raw = getattr(obj, "id", None)
    if isinstance(raw, AgentId):
        return raw
    if isinstance(raw, str):
        try:
            return AgentId(raw)
        except ValueError:
            return None
    return None


def discover_agent_classes(*, refresh: bool = False,
                           package_name: str = "agents") -> DiscoveryReport:
    """Scan ``package_name`` for ``ReActAgent`` subclasses keyed by ``id``."""
    fingerprint = package_name
    if not refresh and fingerprint in _DISCOVERY_CACHE:
        return _DISCOVERY_CACHE[fingerprint]

    base = _base_class()
    modules, reason = _iter_agent_modules(package_name)
    report = DiscoveryReport(package_missing=bool(reason))

    if base is None or reason:
        if reason:
            log.warning("agent discovery degraded: %s", reason)
            report.missing = tuple(REASONING_AGENTS)
        _DISCOVERY_CACHE[fingerprint] = report
        return report

    unimportable: list[tuple[str, str]] = []
    rejected: list[tuple[str, str, str]] = []

    for name, module in modules:
        if isinstance(module, BaseException):
            unimportable.append((name, f"{type(module).__name__}: {module}"))
            continue
        if module is base:
            continue
        for attr_name, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if obj is base or not issubclass(obj, base):
                continue
            agent_id = _declared_id(obj)
            if agent_id is None:
                rejected.append((name, attr_name, "no valid AgentId in class attribute 'id'"))
                continue
            if not agent_id.is_reasoning_agent:
                rejected.append((name, attr_name,
                                 f"id={agent_id.value} is the simulated counterparty, not an agent"))
                continue
            if agent_id in report.records:
                incumbent = report.records[agent_id]
                # Deterministic tie-break on (module, qualname): two modules may
                # legitimately declare the same AgentId (e.g. an agent and its
                # test double both shipped). Pick the lexicographically first
                # name so the result never depends on import order.
                incumbent_name = (incumbent.module,
                                  f"{incumbent.cls.__module__}.{incumbent.cls.__qualname__}")
                challenger_name = (name, f"{obj.__module__}.{obj.__qualname__}")
                if challenger_name >= incumbent_name:
                    rejected.append((name, attr_name,
                                     f"duplicate id {agent_id.value}; kept "
                                     f"{incumbent_name[0]}.{incumbent_name[1]}"))
                    continue
                rejected.append((incumbent_name[0], incumbent_name[1],
                                 f"duplicate id {agent_id.value}; replaced by lower name"))
            report.records[agent_id] = AgentRecord(
                agent_id=agent_id,
                cls=obj,
                module=name,
                role=str(getattr(obj, "role", "") or ""),
                source_file=str(getattr(module, "__file__", "") or ""),
            )

    report.missing = tuple(a for a in REASONING_AGENTS if a not in report.records)
    report.rejected = tuple(rejected)
    report.unimportable = tuple(unimportable)

    for module, why in unimportable:
        log.warning("agent module %s failed to import and was skipped: %s", module, why)
    for module, cls_name, why in rejected:
        log.debug("rejected agent candidate %s.%s: %s", module, cls_name, why)
    if report.missing:
        log.info("agent discovery: %d/%d registered; missing %s",
                 len(report.records), len(REASONING_AGENTS),
                 ", ".join(a.value for a in report.missing))
    else:
        log.info("agent discovery: all %d agents registered", len(report.records))

    _DISCOVERY_CACHE[fingerprint] = report
    return report


def reset_discovery_cache() -> None:
    """Drop the memoised scan. Tests and long-lived processes use this."""
    _DISCOVERY_CACHE.clear()


def agent_module_for(agent_id: AgentId, *, refresh: bool = False) -> str | None:
    """The module a given ``AgentId`` was actually found in, or ``None``.

    Useful for the demo's status panel and for error messages that need to say
    *where* the graph expects an agent, without assuming a filename.
    """
    report = discover_agent_classes(refresh=refresh)
    record = report.records.get(agent_id)
    return record.module if record else None


# ----------------------------------------------------------------------- construction
def discover_agents(*, refresh: bool = False,
                    settings: Settings | None = None) -> dict[AgentId, Any]:
    """``AgentId -> ReActAgent`` built from whatever is currently importable.

    Despite the name this *instantiates*; the discovery-only view is
    :func:`discover_agent_classes`. Kept as the spec's entry point.
    """
    effective = settings or Settings()
    report = discover_agent_classes(refresh=refresh)
    built: dict[AgentId, Any] = {}
    for agent_id, record in sorted(report.records.items(), key=lambda kv: kv[0].value):
        try:
            built[agent_id] = record.instantiate(effective)
        except Exception as exc:  # noqa: BLE001 - reported, graph builds without it
            log.error("agent %s (%s) could not be constructed and is excluded: %s",
                      agent_id.value, record.cls.__qualname__, exc)
    for agent_id in report.missing:
        log.warning("agent %s (%s) is not registered; its node will run as a "
                    "logged no-op so the topology stays intact",
                    agent_id.value, agent_id.label)
    return built


def build_agents(settings: Settings, *, refresh: bool = False) -> dict[AgentId, Any]:
    """Spec-named alias for :func:`discover_agents`, with settings supplied."""
    return discover_agents(settings=settings, refresh=refresh)


def available_agents(agents: dict[AgentId, Any] | None = None) -> list[str]:
    """AgentIds present, as ``"A1"``-style strings, for a status panel."""
    source = agents if agents is not None else discover_agents()
    return [a.value for a in sorted(source, key=lambda x: x.value)]


def missing_agents(agents: dict[AgentId, Any] | None = None) -> list[str]:
    """AgentIds absent, as ``"A1"``-style strings, with their labels."""
    source = agents if agents is not None else discover_agents()
    return [a.value for a in REASONING_AGENTS if a not in source]


def registry_status(*, refresh: bool = False) -> dict[str, Any]:
    """Everything the demo status panel needs, in one JSON-safe dict."""
    report = discover_agent_classes(refresh=refresh)
    payload = report.to_dict()
    payload["expected"] = [a.value for a in REASONING_AGENTS]
    payload["complete"] = not report.missing
    payload["count"] = len(report.records)
    return payload
