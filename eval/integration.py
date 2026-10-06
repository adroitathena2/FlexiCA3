"""The adapter between the evaluation harness and the project's **real** stack.

Why this module exists separately from :mod:`eval.conditions`
------------------------------------------------------------
:class:`~eval.conditions.RealAgentRuntime` is the *measured system's* boundary with
``eval``. The wiring it needs - a real :class:`~graph.state.GraphRuntime`, a real
:class:`~core.protocols.AgentContext`, a real decision callable, a real tool map - is
mechanical, verbose, and full of the kind of detail that is easy to get subtly
wrong. Keeping it here means the runtime in ``conditions.py`` reads as a statement
of *policy* ("a human gate parks the run; an escalation is recorded, not resolved")
rather than as a pile of plumbing, and every lazy ``import`` of a sibling package
lives in exactly one file.

The integrity rules this module exists to preserve
-------------------------------------------------
1. **Nothing is fabricated.** If the real agents cannot be built, the caller is told
   the *reason string* and must fall back explicitly; :func:`conditions.resolve_runtime`
   labels that fallback. No placeholder agent is ever constructed.
2. **Degradation is recorded, never disguised.** A decision the rules engine answered
   while ``clef`` was the intended head is counted as ``DecisionSource.RULES`` **and**
   ``degraded=True``. ``llm_calls`` only moves for a genuine model answer, so a sweep
   without clef cannot be read as one with it.
3. **A park is not a success.** ``HumanGateRequired`` is a *control-flow* signal, not
   an error. It becomes a parked run, never ``ok=True``.
4. **Scenario data is fixture data.** The harness plays the counterparty and
   publishes the scenario's event profile, leads, seeded offers, and the drawn sponsor
   reply, so all three conditions see byte-identical inputs. Everything the harness
   posts is stamped with the scenario's fixture source marker.

Import cost
-----------
Nothing here runs at ``import eval`` time. Every sibling package (``graph``,
``agents``, ``blackboard``, ``decision``, ``tools``) is imported *inside* a
function, so ``eval`` stays importable while any sibling is mid-author.
"""
from __future__ import annotations

import importlib
import inspect
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.protocols import ToolResult
from core.schemas import (
    AgentId,
    AuditFinding,
    Decision,
    DecisionRequest,
    DecisionSource,
    Dispute,
    GateKind,
    HumanGate,
    Intent,
    MoU,
    Offer,
    RiskFlag,
    RunMode,
    Severity,
    Thread,
)

from .conditions import (
    ContractTerms,
    EvalBoard,
    RunCounters,
    ToolInvoker,
    _build_decision,
    _rules_intent_decision,
)
from .scenarios import FIXTURE_SOURCE, Scenario

__all__ = [
    # probing
    "RealStackStatus", "probe_real_stack", "AGENT_PACKAGE_FACTORIES",
    "decision_provenance",
    # tools
    "CountingTool", "build_tool_map",
    # board
    "MirroredBoard", "ZONE_PROJECTION",
    # decisions
    "RealDecisionBridge",
    # run scope
    "RunScope", "open_run_scope", "publish_world",
    "SIMULATED_GATE_DECIDER", "simulated_gate_approvals",
    # sponsor counterparty (flag-gated demo path, default off for eval)
    "build_sponsor_environment", "sponsor_env_enabled",
    "environment_reply_for",
    # projection
    "DeltaView", "read_delta", "blocking_flags_from", "gate_kind_from",
    "contract_terms_from_mou", "latest_arbitration",
]


#: Env var enabling the live sponsor counterparty in demo paths. Default off:
#: eval sweeps must see byte-identical hardcoded replies, not model-driven ones.
SPONSOR_ENV_FLAG = "PAYTRIQ_SPONSOR_ENV"


# ===================================================================== probing
#: Names probed on the bare ``agents`` package, in order of preference. A fallback
#: only: ``graph.registry.build_agents`` is the factory the orchestrator itself uses,
#: and probing for names that no longer exist is how a working registry gets reported
#: as absent.
AGENT_PACKAGE_FACTORIES: tuple[str, ...] = (
    "build_registry", "build_agents", "make_registry", "default_registry", "get_registry",
)

#: Container types a registry factory is allowed to return. A ``str``/``bytes`` is
#: deliberately not among them: an agent registry is never text.
_REGISTRY_CONTAINERS: tuple[type, ...] = (dict, list, tuple, set, frozenset)


@dataclass(slots=True)
class RealStackStatus:
    """The outcome of asking "is the real agent stack there, and what is it?".

    ``reason`` is the load-bearing field. It is printed verbatim in the report's
    provenance block, so it names the factory that answered, how many agents were
    built and which are missing - because "the real stack" is only a meaningful claim
    if the reader can see which parts of it were actually present.
    """

    ok: bool
    reason: str
    #: ``"graph.registry.build_agents"``, ``"agents.build_registry"``, ...
    factory: str = ""
    agents: dict[AgentId, Any] = field(default_factory=dict)
    settings: Any = None
    #: ``AgentId`` values with no registered class, as strings.
    absent: tuple[str, ...] = ()


def probe_real_stack(*, prefer_real: bool = True, settings: Any = None) -> RealStackStatus:
    """Find the real agent factory and build the registry. Lazy by construction.

    Order matters and is the point of the fix: ``graph.registry.build_agents`` is the
    factory production wiring uses, so it is tried first. The bare ``agents``-package
    probe list is kept as a fallback so a tree without ``graph`` still produces a real
    measurement rather than a silent downgrade to the stub.

    Never raises. Every failure path returns ``ok=False`` with the *specific* reason -
    absent package, factory raised, factory built nothing - because that string is
    what the report shows, and a generic "unavailable" tells a reader nothing they
    could act on.
    """
    if not prefer_real:
        return RealStackStatus(ok=False, reason="real agents not requested (prefer_real=False)")

    try:
        from core.config import Settings as _Settings
    except Exception as exc:  # noqa: BLE001 - core is a hard dependency, but say why
        return RealStackStatus(
            ok=False, reason=f"core.config unavailable ({type(exc).__name__}: {exc})")
    resolved_settings = settings if settings is not None else _Settings()

    try:
        registry_module = importlib.import_module("graph.registry")
    except Exception as exc:  # noqa: BLE001 - a sibling may be mid-author
        return _probe_agents_package(
            resolved_settings,
            prefix=(f"graph.registry unavailable ({type(exc).__name__}: {exc}); "
                    "probed the agents package instead"),
        )

    factory = getattr(registry_module, "build_agents", None)
    if not callable(factory):
        return _probe_agents_package(
            resolved_settings,
            prefix=("graph.registry exposes no build_agents(); "
                    "probed the agents package instead"),
        )
    try:
        built = factory(resolved_settings)
    except Exception as exc:  # noqa: BLE001 - the factory is user code
        return _probe_agents_package(
            resolved_settings,
            prefix=(f"graph.registry.build_agents() raised {type(exc).__name__}: {exc}; "
                    "probed the agents package instead"),
        )
    return _status_from_agents(built, factory="graph.registry.build_agents",
                               settings=resolved_settings, prefix="")


def _probe_agents_package(settings: Any, *, prefix: str) -> RealStackStatus:
    """Second choice: a registry factory exported by the bare ``agents`` package."""
    try:
        module = importlib.import_module("agents")
    except Exception as exc:  # noqa: BLE001 - absence is a normal state here
        reason = f"agents package unavailable ({type(exc).__name__}: {exc})"
        return RealStackStatus(ok=False, reason=f"{prefix}; {reason}" if prefix else reason)

    offered = [name for name in AGENT_PACKAGE_FACTORIES
               if callable(getattr(module, name, None))]
    if not offered:
        reason = f"agents package exposes none of {list(AGENT_PACKAGE_FACTORIES)}"
        return RealStackStatus(ok=False, reason=f"{prefix}; {reason}" if prefix else reason)

    # Both calling conventions are attempted because the bare-package factories in the
    # wild take either no argument or the settings, and the first failure must not be
    # mistaken for the end of the search.
    for name in offered:
        factory = getattr(module, name)
        failures: list[str] = []
        for label, call in (("no arguments", lambda f=factory: f()),
                            ("settings", lambda f=factory: f(settings))):
            try:
                built = call()
            except TypeError as exc:
                failures.append(f"called with {label}: TypeError: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - user code
                failures.append(f"called with {label}: {type(exc).__name__}: {exc}")
                break
            status = _status_from_agents(built, factory=f"agents.{name}", settings=settings,
                                         prefix=prefix)
            if status.agents:
                return status
        reason = f"agents.{name}() built nothing ({'; '.join(failures) or 'no detail'})"
        return RealStackStatus(ok=False, reason=f"{prefix}; {reason}" if prefix else reason)

    reason = f"agents package exposes none of {list(AGENT_PACKAGE_FACTORIES)}"
    return RealStackStatus(ok=False, reason=f"{prefix}; {reason}" if prefix else reason)


def _status_from_agents(built: Any, *, factory: str, settings: Any,
                        prefix: str) -> RealStackStatus:
    """Normalise whatever a factory returned into a :class:`RealStackStatus`.

    A factory may legitimately return a mapping keyed by ``AgentId`` or an iterable of
    agents each declaring ``id``; both are accepted, because the harness must not be
    the reason a working registry is rejected. Anything else is reported, not coerced.
    """
    registry: dict[AgentId, Any] = {}
    if isinstance(built, Mapping):
        for key, value in built.items():
            try:
                registry[AgentId(key)] = value
            except (TypeError, ValueError):
                continue
    elif isinstance(built, _REGISTRY_CONTAINERS):
        for agent in built:
            agent_id = getattr(agent, "id", None)
            if isinstance(agent_id, AgentId):
                registry[agent_id] = agent
    else:
        reason = (f"{factory} returned {type(built).__name__}, which is neither a "
                  f"mapping of AgentId nor an iterable of agents")
        return RealStackStatus(ok=False,
                               reason=f"{prefix}; {reason}" if prefix else reason,
                               factory=factory)

    from core import REASONING_AGENTS

    if not registry:
        reason = f"{factory} built 0 agents; cannot run the real agentic stack"
        return RealStackStatus(ok=False,
                               reason=f"{prefix}; {reason}" if prefix else reason,
                               factory=factory)

    absent = tuple(a.value for a in REASONING_AGENTS if a not in registry)
    detail = f"{factory} -> {len(registry)} agents"
    if absent:
        detail += f"; absent: {list(absent)}"
    return RealStackStatus(ok=True, reason=detail, factory=factory, agents=registry,
                           settings=settings, absent=absent)


# ======================================================================= tools
class CountingTool:
    """A :class:`~core.protocols.Tool` that records its outcome on the run's counters.

    Why wrap at all: the real tools are already honest, but the ablation needs every
    call attributed to a specific ``(scenario, condition, seed)`` run and broken down
    by ``ToolStatus`` so a reader can tell which numbers came from a live backend and
    which from the labelled fixtures ``TOOLS_LIVE=false`` serves. Counting at the call
    site is the only way to guarantee the breakdown matches the run.

    ``run`` takes ``**kwargs`` and filters them against the *wrapped* tool's real
    signature. The agents introspect whatever tool they are handed
    (``agents.a1_discovery.invoke``) and pass only what that signature accepts, so a
    wrapper with a fixed signature would silently starve them of arguments.
    """

    __slots__ = ("name", "description", "_tool", "_counters", "_accepted")

    def __init__(self, name: str, tool: Any, counters: RunCounters) -> None:
        self.name = name
        self.description = str(getattr(tool, "description", "") or name)
        self._tool = tool
        self._counters = counters
        self._accepted = _accepted_keywords(getattr(tool, "run", None))

    def available(self) -> tuple[bool, str]:
        """Delegate to the real tool, reporting a fault rather than swallowing it."""
        probe = getattr(self._tool, "available", None)
        if not callable(probe):
            return (True, "wrapped tool exposes no available(); assumed usable")
        try:
            ok, reason = probe()
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return (False, f"{self.name}.available() raised {type(exc).__name__}: {exc}")
        return (bool(ok), str(reason) if reason else "")

    def run(self, **kwargs: Any) -> ToolResult:
        """Call the real tool, recording the ``ToolStatus`` it returned.

        A tool that raises is converted to ``ToolResult.failed`` and counted as failed:
        the protocol says tools must not raise for an environmental problem, and if one
        does the harness records that it did rather than pretending the call never
        happened.
        """
        call_kwargs = kwargs if self._accepted is None else {
            k: v for k, v in kwargs.items() if k in self._accepted
        }
        try:
            result = self._tool.run(**call_kwargs)
        except Exception as exc:  # noqa: BLE001 - a tool fault is data, not a crash
            result = ToolResult.failed(f"{type(exc).__name__}: {exc}", source=self.name)
        if not isinstance(result, ToolResult):
            result = ToolResult.success(result, source=f"{self.name}:unwrapped")
        self._counters.add_tool(result.status)
        return result

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CountingTool {self.name} -> {type(self._tool).__name__}>"


def _accepted_keywords(runner: Any) -> frozenset[str] | None:
    """Keyword names ``runner`` accepts, or ``None`` to pass everything through.

    ``None`` means "no filtering possible or needed": either the callable is
    C-implemented and has no introspectable signature, or it accepts ``**kwargs`` and
    wants everything.
    """
    if not callable(runner):
        return frozenset()
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return None
    params = signature.parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return frozenset(p.name for p in params
                     if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))


def build_tool_map(invoker: ToolInvoker, counters: RunCounters) -> tuple[dict[str, Any], str]:
    """``name -> Tool`` over the real registry, plus an honest one-line reason.

    Returns an **empty** map when no registry is reachable. That is deliberate: an
    agent handed an empty ``ctx.tools`` says so in its own observation ("A1 will not
    invent businesses") instead of being handed a fake that answers. An empty map is a
    measurement; a fabricated one is not.
    """
    ok, why = invoker.available()
    if not ok:
        return ({}, f"no tool registry: {why}")
    registry = invoker.registry or {}
    wrapped = {name: CountingTool(name, tool, counters) for name, tool in registry.items()}
    return (wrapped, f"{len(wrapped)} tools from {why}")


# ====================================================================== board
#: Real ``(zone, kind)`` -> the eval harness's ``(zone, kind)``.
#:
#: The conditions read the board through :class:`~eval.conditions.EvalBoard` using
#: eval-local zone names (``discovery``, ``pricing``, ``outreach``, ``contract``, ...).
#: The real blackboard uses the domain names from ``blackboard/zones.py``. Rather than
#: change the conditions - which would couple the stub and the real stack through a
#: translation table both would have to honour - the projection happens here, in one
#: place, and every mirrored payload carries ``_real_zone``/``_real_kind`` so nothing
#: about the original post is lost.
#:
#: Anything not listed is mirrored verbatim under its own names, which is the honest
#: default: an unmapped zone must be visible, not silently swallowed.
ZONE_PROJECTION: dict[tuple[str, str], tuple[str, str]] = {
    ("opportunities", "brand_lead"): ("discovery", "brand_lead"),
    ("offers", "offer"): ("pricing", "offer"),
    ("threads", "thread"): ("outreach", "thread"),
    ("contracts", "mou"): ("contract", "mou_draft"),
    ("bids", "bid"): ("pricing", "bid"),
    ("tasks", "bid"): ("pricing", "bid"),
    ("audit", "audit_finding"): ("roi", "audit_findings"),
    ("audit", "roi_report"): ("roi", "roi_report"),
    ("audit", "compliance_summary"): ("roi", "compliance_summary"),
    ("risk_flags", "risk_flag"): ("risk_flags", "risk_flag"),
    ("disputes", "dispute"): ("disputes", "dispute"),
    ("decisions", "arbitration_decision"): ("decisions", "arbitration"),
    ("decisions", "escalation"): ("decisions", "escalation"),
    ("approvals", "human_gate"): ("approvals", "human_gate"),
    ("approvals", "human_decision"): ("approvals", "human_decision"),
    ("approvals", "approval"): ("approvals", "approval"),
    ("handoffs", "handoff"): ("handoffs", "handoff"),
    ("lessons", "lesson"): ("lessons", "lesson"),
    ("event", "event_profile"): ("event", "event_profile"),
}


class MirroredBoard:
    """The real blackboard, with every accepted post projected into the eval board.

    Two boards, two jobs, one write path:

    * the **real** board is what the agents read and write. It validates zones, kinds
      and typed payloads, so an agent posting where it should not fails loudly instead
      of being quietly absorbed by an unvalidated harness board.
    * the **eval** board is what the conditions read. It is a *projection*, exactly as
      ``graph.build.collect_board_delta`` projects the same board into graph state, so
      the harness's control flow sees the agents' real output rather than a parallel
      fiction.

    Only posts the real board **accepted** are mirrored. A rejected post never reaches
    the conditions, so a condition can never route on an artefact the system refused
    to write.
    """

    __slots__ = ("real", "projection", "_seq")

    def __init__(self, real: Any, projection: EvalBoard) -> None:
        self.real = real
        self.projection = projection
        self._seq = 0

    def post(self, zone: str, kind: str, author: AgentId, payload: dict[str, Any], *,
             refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> Any:
        """Write to the real board, then mirror. The real board decides validity."""
        entry = self.real.post(zone, kind, author, dict(payload or {}), refs=refs,
                               confidence=confidence, source=source)
        self._seq += 1
        zone_name = str(getattr(entry, "zone", zone) or zone)
        kind_name = str(getattr(entry, "kind", kind) or kind)
        projected_zone, projected_kind = ZONE_PROJECTION.get(
            (zone_name, kind_name), (zone_name, kind_name))
        mirrored = dict(getattr(entry, "payload", {}) or {})
        mirrored["_real_zone"] = zone_name
        mirrored["_real_kind"] = kind_name
        self.projection.post(projected_zone, projected_kind, author, mirrored,
                             refs=list(refs or []), confidence=float(confidence),
                             source=source)
        return entry

    # -- Blackboard protocol passthroughs -------------------------------------
    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[Any]:
        return self.real.read(zone, kind=kind, limit=limit)

    def latest(self, zone: str, kind: str | None = None) -> Any:
        getter = getattr(self.real, "latest", None)
        if callable(getter):
            try:
                return getter(zone, kind)
            except TypeError:
                return getter(zone)
        rows = self.real.read(zone, kind=kind)
        return rows[-1] if rows else None

    def zones(self) -> list[str]:
        return list(self.real.zones())

    def history(self) -> list[Any]:
        return list(self.real.history())

    def post_model(self, zone: str, model: Any, **kwargs: Any) -> Any:
        """Delegate ``post_model`` to the real board, then mirror the entry it made.

        ``post_model`` is what derives ``author``/``confidence``/``source`` from the
        model, so the real board must do the writing. The mirror is applied to the entry
        that came back, which is why this cannot simply delegate to :meth:`post`: the
        entry id, sequence number and normalised payload are the real board's, and the
        conditions must see the same artefact.
        """
        poster = getattr(self.real, "post_model", None)
        if not callable(poster):
            dump = getattr(model, "model_dump", None)
            payload = dict(dump(mode="json")) if callable(dump) else dict(model)
            kind = str(kwargs.pop("kind", "") or "artifact")
            author = getattr(model, "author", None)
            if not isinstance(author, AgentId):
                author = AgentId.A1_DISCOVERY
            return self.post(zone, kind, author, payload, **kwargs)
        entry = poster(zone, model, **kwargs)
        self._seq += 1
        zone_name = str(getattr(entry, "zone", zone) or zone)
        kind_name = str(getattr(entry, "kind", "") or "artifact")
        projected_zone, projected_kind = ZONE_PROJECTION.get(
            (zone_name, kind_name), (zone_name, kind_name))
        mirrored = dict(getattr(entry, "payload", {}) or {})
        mirrored["_real_zone"] = zone_name
        mirrored["_real_kind"] = kind_name
        self.projection.post(projected_zone, projected_kind,
                             getattr(entry, "author", AgentId.A1_DISCOVERY), mirrored,
                             refs=list(getattr(entry, "refs", []) or []),
                             confidence=float(getattr(entry, "confidence", 1.0) or 1.0),
                             source=getattr(entry, "source", DecisionSource.RULES))
        return entry

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MirroredBoard real={type(self.real).__name__} entries={self._seq}>"


# =================================================================== decisions
class RealDecisionBridge:
    """``ctx.decide`` wired to :func:`decision.registry.build_registry`.

    This is the one place the real decision layer enters the harness, and two things
    it guarantees - both of which the ablation's whole argument depends on:

    * **A real backend answers when one can.** With ``clef`` running, the head of the
      chain answers and ``Decision.source`` is ``DecisionSource.CLEF``, which the
      report's ``DecisionSource`` column then shows.
    * **A fallback is never laundered.** When the chain degrades to ``rules`` the
      decision keeps the source the rules engine actually set and gains
      ``degraded=True`` whenever the configured head was something else.

    Note that ``decision.registry`` exposes ``build_registry``, not the
    ``build_backend``/``get_backend``/``resolve_backend`` names an earlier revision of
    this harness probed for. Probing for names that do not exist is exactly how a
    working decision layer gets reported as absent.
    """

    def __init__(self, counters: RunCounters, *, settings: Any, run_mode: RunMode,
                 seed: int = 0) -> None:
        self._counters = counters
        self._settings = settings
        self._run_mode = run_mode
        self._seed = seed
        self._registry: Any = None
        self._reason = "not probed"
        self._probed = False
        #: The backend the configuration *intended*. ``RULES`` in OFFLINE mode is
        #: intended, not degraded.
        self.intended_source: DecisionSource = (
            DecisionSource.RULES if run_mode is RunMode.OFFLINE else DecisionSource.CLEF)

    # ------------------------------------------------------------------ probing
    def probe(self) -> tuple[bool, str]:
        """Build the registry once and report what it can actually reach."""
        if self._probed:
            return (self._registry is not None, self._reason)
        self._probed = True
        try:
            module = importlib.import_module("decision.registry")
        except Exception as exc:  # noqa: BLE001 - absence is a normal condition
            self._reason = (f"decision.registry unavailable ({type(exc).__name__}: {exc}); "
                            "serving the eval rules classifier")
            return (False, self._reason)
        factory = getattr(module, "build_registry", None)
        if not callable(factory):
            self._reason = ("decision.registry exposes no build_registry(); "
                            "serving the eval rules classifier")
            return (False, self._reason)
        try:
            registry = factory(self._settings)
        except Exception as exc:  # noqa: BLE001 - user code
            self._reason = (f"decision.registry.build_registry() raised "
                            f"{type(exc).__name__}: {exc}; "
                            "serving the eval rules classifier")
            return (False, self._reason)
        try:
            ok, why = registry.available()
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self._reason = (f"registry.available() raised {type(exc).__name__}: {exc}; "
                            "serving the eval rules classifier")
            return (False, self._reason)
        if not ok:
            self._reason = (f"registry present but unusable: {why}; "
                            "serving the eval rules classifier")
            return (False, self._reason)
        self._registry = registry
        self._reason = f"decision.registry.build_registry -> {why}"
        return (True, self._reason)

    def provenance(self) -> str:
        """One line naming the decision layer that answers this run."""
        ok, why = self.probe()
        head = getattr(self._settings, "effective_backend", lambda: "unknown")()
        return f"configured {head}; {why}"

    # ------------------------------------------------------------------ calling
    def __call__(self, request: DecisionRequest) -> Decision:
        """Answer ``request`` with the real chain, or say honestly that it could not."""
        ok, _why = self.probe()
        if ok and self._registry is not None:
            try:
                decision = self._registry.decide(request)
                if not isinstance(decision, Decision):
                    raise TypeError(f"expected core.schemas.Decision, got "
                                    f"{type(decision).__name__}")
                self._counters.add_decision(decision)
                return decision
            except Exception as exc:  # noqa: BLE001 - degrade the call, not the run
                self._counters.notes.append(
                    f"decision.registry.decide() failed for {request.request_id} "
                    f"({type(exc).__name__}: {exc}); fell back to the eval rules "
                    "classifier and recorded the substitution as degraded")
        return self._rules_fallback(request)

    def _rules_fallback(self, request: DecisionRequest) -> Decision:
        """The offline rules answer, labelled ``RULES`` and marked degraded.

        Deliberately the *same* calibrated classifier the local stub uses rather than
        a first-option guess: a run that degrades its decision layer should still be a
        run in which the decision layer had a policy, so a difference between the three
        conditions stays attributable to control flow rather than to a fallback that
        invented a preference.
        """
        options = [str(o) for o in request.options if o]
        intent_options = {i.value for i in Intent}
        if options and set(options) <= intent_options:
            rules = _rules_intent_decision(request, str(request.state.get("reply", "")))
            choice = str(rules.choice or Intent.UNKNOWN.value)
            probs = dict(rules.probabilities)
        else:
            choice = options[0] if options else "unknown"
            probs = {choice: 1.0}
        decision = _build_decision(
            request, choice, probs, DecisionSource.RULES, model="eval.rules-fallback",
            degraded=self.intended_source is not DecisionSource.RULES)
        self._counters.add_decision(decision)
        return decision


#: Probed-chain cache. Keyed by the head of the chain plus the run mode, because those
#: are the two things that change what the chain contains. The value records whether it
#: was a probe or configuration-only read, so the provenance string can say which.
_PROBE_CACHE: dict[tuple[str, str], str] = {}


def decision_provenance(settings: Any, *, probe: bool = True) -> str:
    """Describe the decision chain reachable right now, for a provenance block.

    With ``probe=True`` the chain is actually asked what it can reach and the answer is
    cached for the process, so a sweep pays for the round trip once rather than once per
    run. With ``probe=False`` only the configuration is reported.

    The distinction is not cosmetic. ``DecisionRegistry.health(probe=False)`` fills in
    ``available=True`` with the reason ``"not probed"``, so reporting that verbatim would
    claim clef is reachable on a machine where it is not - which is the exact class of
    lie the provenance block exists to prevent.
    """
    key = (str(getattr(settings, "effective_backend", lambda: "unknown")()),
           str(getattr(settings, "run_mode", "unknown")))
    if probe and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    try:
        module = importlib.import_module("decision.registry")
    except Exception as exc:  # noqa: BLE001 - absence is a normal condition
        text = f"decision.registry unavailable ({type(exc).__name__}: {exc})"
        if probe:
            _PROBE_CACHE[key] = text
        return text
    factory = getattr(module, "build_registry", None)
    if not callable(factory):
        return "decision.registry exposes no build_registry()"
    try:
        health = factory(settings).health(probe=probe)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return (f"decision.registry.build_registry().health() raised "
                f"{type(exc).__name__}: {exc}")
    chain = health.get("chain") or []
    if probe:
        parts = [f"{e.get('name')}={'up' if e.get('available') else 'down'}"
                 for e in health.get("backends", []) if e.get("in_chain")]
        detail = ", ".join(parts) or "no backend in the chain"
        text = (f"decision.registry.build_registry -> effective "
                f"{health.get('effective_backend')}; chain={chain}; probed -> {detail}")
    else:
        text = (f"decision.registry.build_registry -> effective "
                f"{health.get('effective_backend')}; chain={chain}; "
                "availability not probed")
    if probe:
        _PROBE_CACHE[key] = text
    return text


# ====================================================================== run scope
@dataclass(slots=True)
class RunScope:
    """One ``(scenario, condition, seed)`` run's real-stack collaborators.

    Held per ``run_id`` by :class:`~eval.conditions.RealAgentRuntime` so every step of
    a run shares one board, one tool map and one decision bridge, while two runs share
    nothing. That isolation is the whole reason the scope is keyed on ``run_id``: a
    blackboard carried across seeds would make the second seed's measurement a function
    of the first seed's leftovers, which is the one thing a seed sweep must not be.
    """

    run_id: str
    scenario: str
    seed: int
    runtime: Any                       # graph.state.GraphRuntime
    board: MirroredBoard
    decide: RealDecisionBridge
    tools: dict[str, Any] = field(default_factory=dict)
    tool_reason: str = "no tool registry"
    #: Board ``seq`` high-water mark for the entries the current step is responsible for.
    mark: int = 0
    world_published: bool = False
    notes: list[str] = field(default_factory=list)

    def agent(self, agent_id: AgentId) -> Any:
        """The registered real agent, or ``None`` when that agent was never built."""
        return self.runtime.agent(agent_id)

    def mark_board(self) -> int:
        """Record where the board is now, so the next delta is this step's output."""
        self.mark = len(self.board.history())
        return self.mark

    def delta(self, since: int) -> list[Any]:
        """Entries written after ``since``: this step's own output, nothing else."""
        return [e for e in self.board.history()
                if int(getattr(e, "seq", 0) or 0) > since]


def open_run_scope(request: Any, status: RealStackStatus, *, counters: RunCounters,
                   run_mode: RunMode | None = None,
                   settings: Any = None,
                   simulated_gates: bool = False) -> RunScope:
    """Assemble the real collaborators for one run through :func:`graph.build.build_runtime`.

    ``build_runtime`` is used rather than a hand-rolled ``GraphRuntime`` because it is
    the orchestrator's own wiring: the context the agents receive is the context
    production runs give them, which is the only way "we measured the real stack" can
    mean anything.

    With ``simulated_gates=True`` the scope's ``runtime.extras["graph_approvals"]``
    is seeded with labelled simulated-human approvals (SEND/MOU/COUNTER/ESCALATION)
    so real A3/A4 agents proceed past their ``HumanGateRequired`` parks in
    unattended eval sweeps. Labelled, never a real human approval.
    """
    from graph.build import build_runtime as _build_runtime

    resolved_settings = settings if settings is not None else status.settings
    mode = run_mode or getattr(resolved_settings, "run_mode", RunMode.LIVE)

    real_board, board_reason = _new_real_board()
    board = MirroredBoard(real_board, EvalBoard(request.event_id))
    tool_map, tool_reason = build_tool_map(request.tools, counters)
    decide = RealDecisionBridge(counters, settings=resolved_settings, run_mode=mode,
                                seed=int(getattr(request, "seed", 0)))

    runtime = _build_runtime(
        resolved_settings,
        board=board,
        tracer=None,
        tools=tool_map,
        decide=decide,
        agents=dict(status.agents),
        mode=mode,
        # Non-interactive: the harness never answers a gate on a human's behalf.
        # ``GatePolicy`` is only consulted by ``graph.interrupts``, which this adapter
        # deliberately does not use - a gate parks the run instead. With
        # simulated_gates the agents see seeded graph_approvals instead of parking.
        interactive=False,
        # The run id doubles as the thread id, so a parked run has a stable identity if
        # it is ever handed to LangGraph.
        thread_id=str(request.run_id),
    )
    runtime.run_id = str(request.run_id)
    runtime.event_id = str(request.event_id)
    if simulated_gates:
        try:
            runtime.extras.setdefault("graph_approvals", []).extend(
                simulated_gate_approvals())
        except Exception:
            pass

    scope = RunScope(run_id=str(request.run_id), scenario=str(request.scenario.name),
                     seed=int(getattr(request, "seed", 0)), runtime=runtime, board=board,
                     decide=decide, tools=tool_map, tool_reason=tool_reason)
    scope.notes.append(f"board: {board_reason}")
    scope.notes.append(f"tools: {tool_reason}")
    scope.notes.append(f"decisions: {decide.provenance()}")
    if simulated_gates:
        scope.notes.append(
            "gates: simulated-human approvals seeded for SEND/MOU/COUNTER/ESCALATION; "
            "no real person approved anything")
    if status.absent:
        scope.notes.append(
            f"agents absent from the real registry: {list(status.absent)}; a step that "
            "needs one is recorded as an explicit failure, never as a zero")
    return scope


#: Label for simulated eval gate approvals. Must not be ``"auto"`` (counted only)
#: and must not be a person's name (no real human approved anything).
SIMULATED_GATE_DECIDER = "simulated-human"


def simulated_gate_approvals(
        *, decider: str = SIMULATED_GATE_DECIDER) -> list[dict[str, Any]]:
    """Seed approvals real A3/A4 agents accept on resume in unattended eval sweeps.

    Each entry matches what :func:`graph.build._propagate_approval` stores in
    ``rt.extras["graph_approvals"]``: same ``kind``, ``outcome=approve``, and a
    labelled ``decided_by``. The agents' ``_find_human_decision`` accepts a
    propagated entry by kind+approve regardless of gate id, so one entry per
    gate kind covers every gate the run will raise. Best-effort by design: a
    caller that cannot seed still parks honestly.
    """
    kinds = ("send", "mou", "counter", "escalation")
    return [{
        "gate_id": f"sim_{kind}",
        "kind": kind,
        "outcome": "approve",
        "decided_by": decider,
        "human": False,
        "instruction": ("simulated-human approval for an unattended eval sweep; "
                        "not a real person and never to be read as one"),
    } for kind in kinds]


def _new_real_board() -> tuple[Any, str]:
    """``(board, reason)`` - the real blackboard if it imports, else the graph fallback.

    ``graph.state.InMemoryBoardLike`` is the documented stand-in and has the same
    semantics (append-only, monotonic ``seq``); using it when ``blackboard`` is absent
    keeps the harness measuring control flow rather than failing on a sibling another
    author is still writing. The reason is returned rather than logged, because it goes
    into the run's notes and therefore into the report.
    """
    from graph.state import InMemoryBoardLike

    try:
        module = importlib.import_module("blackboard")
        factory = getattr(module, "InMemoryBlackboard", None)
        if callable(factory):
            return (factory(), "blackboard.InMemoryBlackboard")
    except Exception as exc:  # noqa: BLE001 - absence is a normal condition
        return (InMemoryBoardLike(),
                f"FALLBACK graph.state.InMemoryBoardLike (blackboard unavailable: "
                f"{type(exc).__name__}: {exc})")
    return (InMemoryBoardLike(),
            "FALLBACK graph.state.InMemoryBoardLike (blackboard exposes no "
            "InMemoryBlackboard)")


# =============================================================== world seeding
def sponsor_env_enabled(request: Any = None) -> bool:
    """True only when the live sponsor counterparty is explicitly requested.

    Default off for eval: sweeps must replay byte-identical hardcoded replies.
    Enabled when ``request.use_sponsor_env`` is truthy or ``PAYTRIQ_SPONSOR_ENV``
    is set to a truthy value. Demo/negotiation paths opt in; eval stays off.
    """
    if request is not None:
        try:
            if bool(getattr(request, "use_sponsor_env", False)):
                return True
        except Exception:
            pass
    return str(os.getenv(SPONSOR_ENV_FLAG, "")).strip().lower() in (
        "1", "true", "yes", "on")


def build_sponsor_environment(scenario: Any) -> Any:
    """Sponsor counterparty with hidden reservation values for one scenario.

    Each brand gets a private profile derived from the scenario's negotiation
    bounds (when present): the reservation value and budget ceiling stay inside
    the environment object, never posted to the board and never placed in an
    agent-visible decision state. Tests may inspect them via
    ``constraints_for``; agents may not.
    """
    from agents.environment import SponsorEnvironment, SponsorProfile

    negotiation = getattr(scenario, "negotiation", None)
    seed_data = getattr(scenario, "seed_data", None)
    brands = list(getattr(seed_data, "brands", ()) or ())
    reservation = float(getattr(negotiation, "sponsor_reservation_inr", 0.0) or 0.0)
    ask = float(getattr(negotiation, "opening_ask_inr", 0.0) or 0.0)
    max_rounds = int(getattr(negotiation, "max_rounds", 4) or 4)
    env = SponsorEnvironment()
    for lead in brands:
        name = str(getattr(lead, "name", ""))
        if not name:
            continue
        ceiling = reservation if reservation > 0 else ask if ask > 0 else 50000.0
        env.register(SponsorProfile(
            brand=name,
            budget_ceiling_inr=ceiling,
            must_have=[],
            walk_away_conditions=[],
            patience=max_rounds,
            reservation_value_inr=reservation if reservation > 0 else ceiling,
            tone="neutral",
        ))
    return env


def environment_reply_for(scope: RunScope, request: Any, offer: Offer,
                           thread: Thread) -> Any:
    """One live counterparty reply for ``offer``/``thread`` via ``SponsorEnvironment``.

    Flag-gated demo path: builds the environment from ``request.scenario``,
    drives ``receive`` with a minimal agent context wired to the run's real
    decision bridge, and returns the ``SponsorReply``. Only the reply ``body``
    is ever published to the board; ``rationale``/reservation values stay inside
    the environment object.
    """
    from core.protocols import AgentContext

    scenario = getattr(request, "scenario", None)
    env = build_sponsor_environment(scenario)
    ctx = AgentContext(
        run_id=str(getattr(request, "run_id", "") or scope.run_id),
        event_id=str(getattr(request, "event_id", "") or scope.runtime.event_id),
        agent=env.id,
        board=scope.board,
        decide=scope.decide,
        tracer=scope.runtime.tracer,
        tools=dict(scope.tools),
        mode=getattr(scope.runtime, "mode", RunMode.LIVE),
        step_budget=2,
        deadline_s=30.0,
        scratch={},
    )
    return env.receive(offer, thread, ctx)


def publish_world(scope: RunScope, request: Any) -> list[str]:
    """Put the scenario's world in front of the real agents.

    The harness plays the counterparty and publishes the fixture, because the agents
    have no other way to learn who they are negotiating with or what the sponsor said.
    Everything posted here is stamped ``source=FIXTURE_SOURCE`` and the returned notes
    name it, so the report can state plainly that the *inputs* were synthetic while the
    *control flow* was the system's own.

    **What is published, and what deliberately is not.** The event profile, the
    discovered leads and the drawn sponsor reply are the *environment* - they are what
    the agents would otherwise have to go and find, and the agents cannot invent them.
    The scenario's pre-priced ``SeedOffer``s are **not** published: pricing is A2's job,
    and seeding the offer would let a run skip the pricing step that the ablation is
    partly measuring. One consequence is stated in ``docs/EVALUATION.md``: a seeded
    offer's clause facts (S1's 365-day exclusivity, its attendee-data clause) never reach
    the real agents, so those policy conflicts are not reproduced on the real stack. That
    is a finding about the system, not a gap in the harness, and the benchmark reports it
    as a missed conflict.

    The event, leads and replies are published once per run; the reply is republished on
    every step, because the board is append-only and each round draws a different reply.
    """
    scenario: Scenario = request.scenario
    seed_data = scenario.seed_data
    notes: list[str] = []

    if not scope.world_published:
        scope.world_published = True
        scope.board.post("event", "event_profile", AgentId.A1_DISCOVERY,
                         scenario.event.model_dump(mode="json"), source=DecisionSource.RULES)
        for lead in seed_data.brands:
            scope.board.post("opportunities", "brand_lead", AgentId.A1_DISCOVERY,
                             lead.model_dump(mode="json"), source=DecisionSource.RULES)
        notes.append(
            f"published the scenario world from {FIXTURE_SOURCE}: "
            f"{len(seed_data.brands)} lead(s), 1 event profile; the scenario's pre-priced "
            "offers are deliberately NOT published so A2 does the pricing itself")

    round_index = int(getattr(request, "round_index", 0) or 0)
    reply_text = str(getattr(request, "reply_text", "") or "")
    brand = str(request.focus_brand)
    # Flag-gated live counterparty: when explicitly enabled, the reply body is
    # composed by SponsorEnvironment from the actual offer + hidden reservation
    # values instead of the hardcoded fixture string. Default off for eval, so
    # sweeps stay byte-identical; demo paths opt in via use_sponsor_env / env.
    if sponsor_env_enabled(request):
        try:
            _offers = [e for e in scope.board.read("offers", kind="offer")]
            _offer_obj = None
            for _e in reversed(_offers):
                try:
                    _cand = Offer(**{k: v for k, v in (getattr(_e, "payload", {}) or {}).items()
                                     if not str(k).startswith("_")})
                except Exception:
                    continue
                if _cand.brand == brand:
                    _offer_obj = _cand
                    break
            if _offer_obj is not None:
                _tmp_thread = Thread(
                    thread_id=f"thr_{_slug(scenario.name)}_{_slug(brand)}",
                    event_id=str(request.event_id),
                    brand=brand,
                    email=_brand_email(seed_data, brand),
                    status="negotiating",
                    day=round_index,
                    intent=Intent.UNKNOWN,
                    reply_text=reply_text,
                    delivered=False,
                )
                _live = environment_reply_for(scope, request, _offer_obj, _tmp_thread)
                reply_text = str(getattr(_live, "body", "") or reply_text)
                notes.append(f"live sponsor reply via SponsorEnvironment "
                             f"(intent={getattr(_live.intent, 'value', _live.intent)}, "
                             f"round={_live.round_index}); reservation values stayed hidden")
        except Exception as exc:  # noqa: BLE001 - hardcoded reply is the safe fallback
            notes.append(f"sponsor environment unavailable ({type(exc).__name__}: {exc}); "
                         f"using hardcoded fixture reply")
    thread = Thread(
        thread_id=f"thr_{_slug(scenario.name)}_{_slug(brand)}",
        event_id=str(request.event_id),
        brand=brand,
        email=_brand_email(seed_data, brand),
        status="replied" if reply_text else "negotiating",
        day=round_index,
        intent=Intent.UNKNOWN,
        reply_text=reply_text,
        delivered=False,
    )
    scope.board.post("threads", "thread", AgentId.A3_OUTREACH,
                     thread.model_dump(mode="json"), source=DecisionSource.RULES)
    notes.append(f"published round {round_index} sponsor reply ({len(reply_text)} char(s)) "
                 f"from {FIXTURE_SOURCE}")
    return notes


def _brand_email(seed_data: Any, brand: str) -> str | None:
    lead = next((b for b in seed_data.brands if b.name == brand), None)
    email = getattr(lead, "contact_email", None) if lead is not None else None
    return str(email) if email else None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_") or "item"


# ============================================================ board -> StepResult
#: Real ``(zone, kind)`` -> the model that artefact is supposed to be.
#:
#: Keyed on the **real** blackboard's names, because :meth:`RunScope.delta` reads the
#: real board's history - that is where the agents wrote. The bucket an artefact lands
#: in is then derived through :data:`ZONE_PROJECTION`, so a reader can follow the same
#: path the conditions take from ``offers/offer`` to ``pricing/offer``.
_MODEL_FOR: dict[tuple[str, str], Any] = {
    ("offers", "offer"): Offer,
    ("threads", "thread"): Thread,
    ("contracts", "mou"): MoU,
    ("disputes", "dispute"): Dispute,
    ("audit", "audit_finding"): AuditFinding,
    ("risk_flags", "risk_flag"): RiskFlag,
    ("approvals", "human_gate"): HumanGate,
}

#: Real ``(zone, kind)`` pairs the harness reads as raw payloads rather than typed
#: artefacts. Both are genuinely untyped by design (``blackboard.zones.UNTYPED_KINDS``
#: and ``graph.build._CLASSIFY_RULES``), so their payloads are carried through whole.
_RAW_FOR: frozenset[tuple[str, str]] = frozenset({
    ("decisions", "arbitration_decision"),
    ("decisions", "escalation"),
    ("audit", "roi_report"),
    ("audit", "compliance_summary"),
    ("handoffs", "handoff"),
    ("lessons", "lesson"),
    ("approvals", "human_decision"),
    ("approvals", "approval"),
    ("tasks", "announcement"),
    ("tasks", "award"),
    ("tasks", "expedite"),
    ("event", "event_profile"),
})


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the projection bookkeeping keys before schema validation.

    ``_real_zone``/``_real_kind`` are added by :class:`MirroredBoard`, and every domain
    model is ``extra="forbid"``. Leaving them in would make every artefact fail to
    validate, which would look exactly like "the real agents produced nothing".
    """
    return {k: v for k, v in payload.items() if not str(k).startswith("_")}


@dataclass(slots=True)
class DeltaView:
    """Domain objects recovered from one agent step's board delta."""

    offers: list[Offer] = field(default_factory=list)
    threads: list[Thread] = field(default_factory=list)
    mouses: list[MoU] = field(default_factory=list)
    disputes: list[Dispute] = field(default_factory=list)
    findings: list[AuditFinding] = field(default_factory=list)
    flags: list[RiskFlag] = field(default_factory=list)
    gates: list[HumanGate] = field(default_factory=list)
    arbitrations: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[dict[str, Any]] = field(default_factory=list)
    roi: dict[str, Any] | None = None
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    lessons: list[dict[str, Any]] = field(default_factory=list)
    #: Zone/kind pairs the harness neither typed nor recognised. Never dropped: an
    #: artefact the benchmark could not read is a fact about the run, and hiding it
    #: would make a missing metric indistinguishable from one never produced.
    unparsed: list[str] = field(default_factory=list)

    def buckets(self) -> dict[str, Any]:
        """The projected ``(zone, kind)`` buckets this view found, for diagnostics."""
        found: dict[str, Any] = {}
        for name, items in (("offer", self.offers), ("thread", self.threads),
                            ("mou_draft", self.mouses), ("dispute", self.disputes),
                            ("audit_findings", self.findings), ("risk_flag", self.flags),
                            ("human_gate", self.gates), ("arbitration", self.arbitrations),
                            ("escalation", self.escalations), ("roi_report", [self.roi]),
                            ("handoff", self.handoffs), ("lesson", self.lessons)):
            if items and items != [None]:
                found[name] = len(items)
        return found


def read_delta(entries: Sequence[Any]) -> DeltaView:
    """Recover typed domain objects from raw board entries.

    An entry whose payload will not validate is *named* with the validation error
    rather than skipped: a silently dropped artefact would make a run that produced
    nothing look identical to a run whose artefacts the harness could not read.
    """
    view = DeltaView()
    for entry in entries:
        payload = getattr(entry, "payload", None)
        payload = dict(payload) if isinstance(payload, dict) else {}
        key = (str(getattr(entry, "zone", "") or ""), str(getattr(entry, "kind", "") or ""))
        projected = ZONE_PROJECTION.get(key, key)
        if key in _RAW_FOR:
            _collect_raw(view, projected, payload)
            continue
        model = _MODEL_FOR.get(key)
        if model is None:
            _note_unparsed(view, f"{key[0]}/{key[1]}")
            continue
        try:
            obj = model(**_identity(payload))
        except Exception as exc:  # noqa: BLE001 - a malformed artefact is reported
            _note_unparsed(view, f"{key[0]}/{key[1]}: {type(exc).__name__}: {exc}")
            continue
        if key == ("offers", "offer"):
            view.offers.append(obj)
        elif key == ("threads", "thread"):
            view.threads.append(obj)
        elif key == ("contracts", "mou"):
            view.mouses.append(obj)
        elif key == ("disputes", "dispute"):
            view.disputes.append(obj)
        elif key == ("audit", "audit_finding"):
            view.findings.append(obj)
        elif key == ("risk_flags", "risk_flag"):
            view.flags.append(obj)
        elif key == ("approvals", "human_gate"):
            view.gates.append(obj)
    return view


def _collect_raw(view: DeltaView, projected: tuple[str, str], payload: dict[str, Any]) -> None:
    """File an intentionally-untyped payload into the bucket it belongs to."""
    kind = projected[1]
    if kind == "arbitration":
        view.arbitrations.append(payload)
    elif kind == "escalation":
        view.escalations.append(payload)
    elif kind == "roi_report":
        view.roi = payload
    elif kind == "handoff":
        view.handoffs.append(payload)
    elif kind == "lesson":
        view.lessons.append(payload)
    elif kind == "compliance_summary":
        # No bucket of its own: A5's summary is read through the findings and risk
        # flags it produces, and a summary with neither is not a measurement.
        return


def _note_unparsed(view: DeltaView, label: str) -> None:
    """Record an untyped artefact once per label rather than once per entry."""
    if label not in view.unparsed:
        view.unparsed.append(label)


def blocking_flags_from(view: DeltaView) -> list[str]:
    """Blocking ``RiskFlag`` codes, for ``RunCounters.blocking_flags``."""
    out: list[str] = []
    for flag in view.flags:
        severity = getattr(flag.severity, "value", flag.severity)
        if str(severity).rsplit(".", 1)[-1] == Severity.BLOCKING.value:
            out.append(str(flag.code or "BLOCKING_FLAG"))
    return out


def latest_arbitration(view: DeltaView) -> dict[str, Any] | None:
    """The most recent ``arbitration_decision`` payload, if A7 posted one."""
    return view.arbitrations[-1] if view.arbitrations else None


def gate_kind_from(view: DeltaView, message: str, fallback: GateKind) -> GateKind:
    """The kind of the gate that parked the run.

    Three sources, in descending order of authority: the gate the agent recorded on
    the board, the kind named in the exception message, and finally the caller's
    fallback.

    The message pattern deliberately requires the shape A3 and A4 actually write -
    ``"<kind> gate <id> is unanswered"`` - rather than a bare mention of a gate word. A
    looser pattern reads A5's veto message ("no MoU may be released") as an ``mou`` gate,
    mislabels a compliance escalation as a release request, and then has the harness
    count a gate the contract step never asked for. A5 posts no ``HumanGate`` at all, so
    the fallback covers a real case rather than existing as a formality.
    """
    if view.gates:
        try:
            return GateKind(str(view.gates[-1].kind).rsplit(".", 1)[-1])
        except ValueError:
            pass
    match = re.search(r"\b(send|counter|mou|escalation)\s+gate\b", (message or "").lower())
    if match:
        try:
            return GateKind(match.group(1))
        except ValueError:
            pass
    return fallback


def contract_terms_from_mou(mou: MoU) -> tuple[ContractTerms, list[str]]:
    """Build the harness's audit view of a real MoU, from the MoU alone.

    ``ContractTerms`` exists so the benchmark can check a produced contract against the
    policy caps **without trusting the agent that wrote it**. On the real stack the MoU
    is prose, so the clause facts are read out of that prose:

    * ``exclusivity_days`` - an explicit "no exclusivity" clause reads as ``0``. That is
      a real finding, not a missing field: the real A4 always writes that clause.
    * ``attendee_data_clause`` - true only when the body actually grants a transfer.
    * ``signer_authority_evidence`` / ``deliverables_evidence_url`` - whatever the real
      artefacts carry. They usually carry none, and that is reported as a policy
      violation rather than smoothed over, because it is a true statement about what
      the system's contract looks like.

    Nothing is imported from the scenario fixture here. Doing so would let the harness
    grade the real agents against numbers the agents never saw.
    """
    notes: list[str] = []
    body = str(mou.terms or "")
    deliverables = tuple(str(d) for d in (mou.deliverables or []))

    exclusivity = _exclusivity_days(body)
    notes.append(f"exclusivity_days={exclusivity}: "
                 + ("the MoU grants no exclusivity"
                    if exclusivity == 0 else
                    "read from the MoU body" if exclusivity is not None else
                    "the MoU body mentions exclusivity in a form this adapter "
                    "could not read; reported as unknown and defaulted to 0"))

    data_clause = bool(re.search(
        r"(share|transfer|provide|release|sell)[^.]{0,80}attendee[^.]{0,40}(data|list|information)",
        body, re.IGNORECASE))
    notes.append(f"attendee_data_clause={data_clause}: "
                 + ("the MoU grants an attendee-data transfer"
                    if data_clause else "the MoU grants no attendee-data transfer"))

    document_path = str(mou.document_path or "")
    evidence_url = document_path if document_path.startswith(("http://", "https://")) else None
    notes.append(f"deliverables_evidence_url={evidence_url!r}: the MoU's document_path is "
                 f"{mou.document_path!r} and nothing else on the board carries a "
                 "compliance evidence reference")

    authority = bool(re.search(r"(signatory|signer)\s+authority[^.]{0,60}"
                               r"(verified|evidence|record)", body, re.IGNORECASE))
    notes.append(f"signer_authority_evidence={authority}: "
                 + ("the MoU cites signer-authority evidence"
                    if authority else "nothing on the board evidences the signer's authority"))

    terms = ContractTerms(
        brand=str(mou.brand),
        amount_inr=float(mou.amount_inr),
        exclusivity_days=int(exclusivity or 0),
        attendee_data_clause=data_clause,
        signer_authority_evidence=authority,
        deliverables_evidence_url=evidence_url,
        deliverables=deliverables,
    )
    return (terms, notes)


def _exclusivity_days(body: str) -> int | None:
    """Days of exclusivity the MoU body grants, or ``None`` when that is unreadable.

    Scanned **clause by clause** rather than across the whole body. A single regex over
    the text cannot cross a sentence boundary, so "365 days of exclusivity are granted"
    and "6. EXCLUSIVITY. As set out in Schedule 2." both look like "no number found" -
    and one of them is a real 365-day grant while the other is a clause this adapter
    cannot read. Collapsing those into the same number would turn a clean contract and
    an unreadable one into the same measurement.
    """
    if re.search(r"\bno\s+exclusivity\b|\bexclusivity\s+is\s+not\s+granted\b", body,
                 re.IGNORECASE):
        return 0
    for clause in re.split(r"(?<=[.!?])\s+|\n", body):
        if not re.search(r"exclusiv", clause, re.IGNORECASE):
            continue
        # A heading ("6. EXCLUSIVITY.") carries no duration, so the scan continues
        # rather than stopping: the grant usually lives in the *next* clause. Only a
        # whole body that mentions exclusivity without ever stating one is unreadable.
        days = re.search(r"(\d+)\s*(?:-|\s)?\s*days?\b", clause, re.IGNORECASE)
        if days:
            return int(days.group(1))
    # Either the body never mentions exclusivity, or it mentions it without ever
    # stating a duration. Both are "no grant this adapter can read", and the caller's
    # note says which, so a reader can tell a clean contract from an opaque one.
    return None
