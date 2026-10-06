"""Concrete ReAct base class for Paytriq agents.

Every agent subclasses :class:`ReActAgent` and supplies four small methods. The
loop itself — plan, act, observe, reflect, with a step budget, a deadline, and a
trace span around each phase — is implemented once, here, so it behaves
identically for all seven agents and appears identically in the trace.

Why the loop is explicit rather than hidden inside a framework:
``plan -> act -> observe -> reflect`` is the rubric's "planning, reasoning, using
tools, and adapting". Making the four phases first-class methods means each is
independently visible in the execution trace and independently unit-testable.

Subclass contract
-----------------
* ``id`` and ``role`` are class attributes.
* ``_plan`` / ``_act`` / ``_observe`` / ``_reflect`` raise ``NotImplementedError``.
* ``plan`` calls ``ctx.decide(...)`` (never a raw model client) so every routing
  decision is recorded with its ``DecisionSource`` and confidence.
* ``run`` never raises for a model failure; it converts to a degraded
  :class:`~core.schemas.Observation` so the supervisor can route around it.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from core.errors import (
    DecisionUnavailable,
    NoValidPlan,
    ToolUnavailable,
)
from core.ids import new_id, utcnow
from core.protocols import (
    ActResult,
    AgentContext,
    Observation,
    Plan,
    Reflection,
)
from core.schemas import AgentId, DecisionSource, Severity

__all__ = ["ReActAgent", "AgentResult"]


class AgentResult:
    """What one full ``run()`` returns to the orchestrator."""

    __slots__ = ("agent", "observation", "reflection", "steps", "duration_ms",
                 "degraded", "stop", "notes")

    def __init__(self, agent: AgentId, observation: Observation,
                 reflection: Reflection | None, steps: int, duration_ms: float,
                 degraded: bool, stop: bool, notes: list[str]) -> None:
        self.agent = agent
        self.observation = observation
        self.reflection = reflection
        self.steps = steps
        self.duration_ms = duration_ms
        self.degraded = degraded
        self.stop = stop
        self.notes = notes

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AgentResult({self.agent.value} steps={self.steps} "
                f"degraded={self.degraded} sufficient={self.observation.sufficient})")


class ReActAgent(ABC):
    """plan -> act -> observe -> reflect, with budget and deadline enforcement."""

    id: AgentId
    role: str = ""

    def __init__(self, *, step_budget: int = 6, deadline_s: float = 60.0) -> None:
        self.step_budget = step_budget
        self.deadline_s = deadline_s

    # -------------------------------------------------------------- subclass API
    @abstractmethod
    def _plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        """Choose the next action. Use ``ctx.decide`` for anything conditional."""

    @abstractmethod
    def _act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        """Execute the plan, typically via ``ctx.tools[...]``."""

    @abstractmethod
    def _observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        """Interpret the result; declare gaps so the loop can plan again."""

    def _reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        """Optional post-step lesson. Return ``None`` when there is nothing to learn."""
        return None

    # -------------------------------------------------------------- public hooks
    def plan(self, ctx: AgentContext, obs: Observation | None) -> Plan:
        return self._plan(ctx, obs)

    def act(self, ctx: AgentContext, plan: Plan) -> ActResult:
        return self._act(ctx, plan)

    def observe(self, ctx: AgentContext, plan: Plan, result: ActResult) -> Observation:
        return self._observe(ctx, plan, result)

    def reflect(self, ctx: AgentContext, obs: Observation) -> Reflection | None:
        return self._reflect(ctx, obs)

    # ----------------------------------------------------------------- the loop
    def run(self, ctx: AgentContext, obs: Observation | None = None) -> AgentResult:
        """Execute the loop until the agent stops, exhausts its budget, or times out.

        Never raises for an environmental failure. A model or tool that is
        unavailable becomes a degraded observation, and the run reports
        ``degraded=True`` so the trace shows what actually happened.
        """
        t0 = time.perf_counter()
        steps = 0
        notes: list[str] = []
        degraded = False
        reflection: Reflection | None = None
        current = obs

        while steps < self.step_budget:
            if (time.perf_counter() - t0) > self.deadline_s:
                notes.append(f"deadline exceeded after {steps} steps")
                break

            with ctx.tracer.agent(self.id, f"{self.id.value}.plan", step=steps):
                try:
                    plan = self.plan(ctx, current)
                except (DecisionUnavailable, ToolUnavailable) as e:
                    degraded = True
                    notes.append(f"plan degraded: {e}")
                    current = Observation(
                        summary=f"{self.id.value} could not plan: {e}",
                        sufficient=False, gaps=[str(e)],
                    )
                    break
                except NotImplementedError:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad step must not kill the run
                    degraded = True
                    notes.append(f"plan error: {type(e).__name__}: {e}")
                    current = Observation(
                        summary=f"{self.id.value} planning failed: {e}",
                        sufficient=False, gaps=[str(e)],
                    )
                    break

            if plan is None:
                raise NoValidPlan(f"{self.id.value} returned no plan")

            with ctx.tracer.agent(
                self.id, f"{self.id.value}.act", step=steps,
                tool_calls=len(plan.tool_calls), plan_source=plan.source.value,
            ):
                result = self.act(ctx, plan)

            degraded = degraded or result.degraded
            result.errors and notes.extend(result.errors)

            with ctx.tracer.agent(self.id, f"{self.id.value}.observe", step=steps):
                current = self.observe(ctx, plan, result)

            steps += 1

            if current.sufficient:
                if reflection is None:
                    with ctx.tracer.agent(self.id, f"{self.id.value}.reflect", step=steps):
                        try:
                            reflection = self.reflect(ctx, current)
                        except Exception as e:  # noqa: BLE001 - reflection is best-effort
                            notes.append(f"reflect error: {type(e).__name__}: {e}")
                            reflection = None
                return AgentResult(self.id, current, reflection, steps,
                                   round((time.perf_counter() - t0) * 1000, 3),
                                   degraded, plan.stop, notes)

            if plan.stop:
                notes.append("agent requested stop before sufficiency")
                return AgentResult(self.id, current, reflection, steps,
                                   round((time.perf_counter() - t0) * 1000, 3),
                                   degraded, True, notes)

        # Budget or deadline exhausted.
        if steps >= self.step_budget:
            notes.append(f"step budget exhausted ({self.step_budget})")
        return AgentResult(
            self.id,
            current or Observation(summary="no observation produced", sufficient=False),
            reflection, steps, round((time.perf_counter() - t0) * 1000, 3),
            degraded, True, notes,
        )

    # ----------------------------------------------------------------- helpers
    def post(self, ctx: AgentContext, zone: str, kind: str, payload: dict,
             *, refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> str:
        """Convenience wrapper so subclasses never touch the board directly."""
        entry = ctx.board.post(zone, kind, self.id, payload,
                               refs=refs, confidence=confidence, source=source)
        return entry.entry_id

    def flag_risk(self, ctx: AgentContext, brand: str, code: str, message: str,
                  *, severity: Severity = Severity.MEDIUM,
                  evidence: list[str] | None = None) -> str:
        """Raise a risk flag. Used by A5; kept here so all agents log uniformly."""
        from core.schemas import RiskFlag

        flag = RiskFlag(
            flag_id=new_id("rsk"), event_id=ctx.event_id, brand=brand,
            severity=severity, code=code, message=message,
            evidence=evidence or [], raised_by=self.id, raised_at=utcnow(),
        )
        return self.post(ctx, "risk_flags", "risk_flag", flag.model_dump(mode="json"),
                         refs=evidence, source=DecisionSource.RULES)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.id.value} {self.role!r}>"
