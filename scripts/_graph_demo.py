"""Throwaway demo: build the graph with fake agents, print the diagram, run it.

Not part of the test suite. Run with::

    python scripts/_graph_demo.py
"""
from __future__ import annotations

import logging
import sys
from typing import Any

sys.path.insert(0, ".")

from agents.base import AgentResult
from core import (
    AgentId,
    Decision,
    DecisionRequest,
    DecisionSource,
    EventProfile,
    Intent,
    RunMode,
    Settings,
)
from core.protocols import Observation
from graph import build_graph, build_runtime, default_config, registry_status, replay
from graph.checkpointer import CheckpointSetup
from graph.interrupts import gate_summary
from graph.state import InMemoryBoardLike, NullTracer

logging.basicConfig(level=logging.INFO, format="    %(levelname)-7s %(name)s | %(message)s")

RULE = "-" * 78


def rule(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# --------------------------------------------------------------------- stub decide
class ScriptedDecide:
    """Answers by decision-point substring; records every question asked."""

    def __init__(self, script: dict[str, str], confidence: float = 0.78,
                 source: DecisionSource = DecisionSource.RULES) -> None:
        self.script = dict(script)
        self.confidence = confidence
        self.source = source
        self.calls: list[DecisionRequest] = []

    def __call__(self, request: DecisionRequest) -> Decision:
        self.calls.append(request)
        point = request.decision_point or request.question
        choice = next((v for k, v in self.script.items() if k in point), "proceed")
        options = [o for o in request.options if o] or [choice, "other"]
        if choice not in options:
            options = [*options, choice]
        target = min(max(self.confidence, 1.0 / len(options)), 1.0)
        rest = round((1.0 - target) / (len(options) - 1), 4) if len(options) > 1 else 0.0
        probabilities = {choice: round(target, 4)}
        probabilities.update({o: rest for o in options if o != choice})
        total = sum(probabilities.values()) or 1.0
        probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}
        return Decision(request_id=request.request_id, question=request.question,
                        choice=choice, probabilities=probabilities,
                        confidence=probabilities[choice], source=self.source,
                        model="demo.scripted", degraded=self.source is DecisionSource.RULES,
                        raw={"scripted": True, "decision_point": point})

    def asked(self) -> list[str]:
        return [c.decision_point for c in self.calls]


# --------------------------------------------------------------------- fake agents
def lead(i: int, name: str, fit: float) -> dict:
    return {"lead_id": f"brd_{i}", "name": name, "category": "cafe",
            "distance_km": 1.1 * i, "contact_email": f"{name.split()[0].lower()}@example.invalid",
            "phone": None, "rating": 4.1, "fit_score": fit, "fit_breakdown": {},
            "fit_rationale": "within walking distance of the venue", "source": "seeded",
            "author": AgentId.A1_DISCOVERY.value}


class FakeAgent:
    def __init__(self, agent_id: AgentId, posts: list[tuple[str, str, dict]] | None = None,
                 role: str = "demo") -> None:
        self.id, self.role, self.posts = agent_id, role, posts or []
        self.step_budget, self.deadline_s, self.calls = 4, 5.0, 0

    def run(self, ctx: Any) -> AgentResult:
        self.calls += 1
        for zone, kind, payload in self.posts:
            ctx.board.post(zone, kind, self.id, payload, source=DecisionSource.RULES)
        return AgentResult(self.id, Observation(summary=f"{self.id.value} demo step",
                                                sufficient=True),
                            None, 1, 2.0, False, False, [])

    def __repr__(self) -> str:
        return f"<FakeAgent {self.id.value} calls={self.calls}>"


def memory_cp() -> CheckpointSetup:
    from langgraph.checkpoint.memory import InMemorySaver

    return CheckpointSetup(saver=InMemorySaver(), kind="memory")


def build(script: dict[str, str], *, interactive: bool, max_replans: int,
          agents: dict[AgentId, Any] | None = None, thread_id: str = "t"):
    settings = Settings(decision_backend="rules", run_mode=RunMode.OFFLINE,
                        human_gates_interactive=interactive,
                        auto_approve_outcome="approve", max_replans=max_replans,
                        max_debate_rounds=2, max_auction_rounds=2)
    decide = ScriptedDecide(script)
    runtime = build_runtime(settings, board=InMemoryBoardLike(), tracer=NullTracer(),
                            decide=decide, agents=agents, thread_id=thread_id)
    graph = build_graph(settings, agents=agents, runtime=runtime,
                        checkpointer=memory_cp())
    return graph, runtime, decide


EVENT = EventProfile(event_id="evt_demo", name="TechFest Pune 2026",
                     location="Pune, Maharashtra", footfall=6000, date="2026-11-15",
                     audience="college students 18-24", budget_inr=250000,
                     categories_wanted=["cafe", "fintech", "electronics"])


def world(narrow: bool) -> dict[str, Any]:
    fit = 18.0 if narrow else [78.0, 71.0, 66.0, 52.0]
    names = ["Cafe One"] if narrow else ["Cafe One", "Book House", "Tech Mart", "Paytm Kiosk"]
    return {**{f"brand_{i}": lead(i, n, fit if narrow else fit[i - 1])
               for i, n in enumerate(names, start=1)}}


# ======================================================================= 1. diagram
rule("1. ARCHITECTURE DIAGRAM - generated by get_graph().draw_mermaid()")
settings = Settings(decision_backend="rules", human_gates_interactive=False)
reg = registry_status(refresh=True)
print(f"registry: {reg['count']}/7 agents discovered, complete={reg['complete']}")
for aid, meta in sorted(reg["found"].items()):
    print(f"  {aid}  {meta['class']}")
diagram_graph = build_graph(settings, agents={}, checkpointer=memory_cp())
print()
print(diagram_graph.mermaid())

# ================================================ 2. end-to-end, gates auto-resolved
rule("2. END-TO-END RUN - non-interactive gates (decided_by='auto')")
agents = {
    AgentId.A5_COMPLIANCE: FakeAgent(AgentId.A5_COMPLIANCE, [
        ("audit", "compliance_summary", {"compliance_score": 100.0,
                                         "flags": 0, "blocking": 0})]),
    AgentId.A6_AUDIT: FakeAgent(AgentId.A6_AUDIT, [
        ("audit", "roi_report", {"report_id": "roi_1", "event_id": "evt_demo",
                                 "total_sponsored_inr": 60000.0, "footfall": 6000,
                                 "estimated_leads": 240, "spend_inr": 9000.0,
                                 "pipeline_value_inr": 420000.0, "roi_multiple": 4.67,
                                 "assumptions": ["lead -> opportunity rate 4%",
                                                 "opportunity -> won 20%"],
                                 "findings": []})]),
    AgentId.A7_ARBITER: FakeAgent(AgentId.A7_ARBITER),
    AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, [
        ("leads", "brand_lead", lead(i, n, f))
        for i, (n, f) in enumerate([("Cafe One", 78.0), ("Book House", 71.0),
                                    ("Tech Mart", 66.0)], start=1)]),
    AgentId.A2_PRICING: FakeAgent(AgentId.A2_PRICING, [
        ("offers", "offer", {"offer_id": "off_1", "event_id": "evt_demo",
                             "brand": "Cafe One", "tier": "silver", "amount_inr": 60000.0,
                             "deliverables": ["booth", "logo"], "pitch": "",
                             "fit_score": 78.0, "version": 1, "revised": False,
                             "revision_note": ""})]),
    AgentId.A3_OUTREACH: FakeAgent(AgentId.A3_OUTREACH, [
        ("threads", "thread", {"thread_id": "thr_1", "event_id": "evt_demo",
                               "brand": "Cafe One", "email": "x@example.invalid",
                               "status": "sent", "day": 0,
                               "intent": Intent.YES.value,
                               "reply_text": "Yes - approved, let's proceed and sign.",
                               "offer_id": "off_1", "sent_at": None, "delivered": True})]),
    AgentId.A4_CONTRACT: FakeAgent(AgentId.A4_CONTRACT, [
        ("contracts", "mou", {"mou_id": "mou_1", "event_id": "evt_demo",
                              "brand": "Cafe One", "amount_inr": 60000.0,
                              "terms": "standard tier-2 terms", "deliverables": ["booth", "logo"],
                              "status": "draft", "document_path": None, "version": 1})]),
}
graph, runtime, decide = build({
    "route.after_discovery": "proceed",
    "route.after_proposal": "outreach",
    "route.after_auction": "execute",
    "route.after_reply": "yes",
    "route.after_compliance": "audit",
    "route.progress": "progress",
    # A2 prices and A3 talks; their feasibility and effort genuinely differ, so
    # the award is a trade-off rather than a restatement of agent ordering.
    "contract_net.bid.A2": "low",
    "contract_net.bid.A3": "high",
    "contract_net.expedite": "accept",
}, interactive=False, max_replans=2, agents=agents, thread_id="t_auto")

final = graph.run(event=EVENT, thread_id="t_auto")
print(f"status={final['status']}  phase={final['phase']}")
print(f"brands={len(final['brands'])} offers={len(final['offers'])} "
      f"threads={len(final['threads'])} mous={len(final['mous'])}")
print(f"handoffs={len(final['handoffs'])} bids={len(final['bids'])} "
      f"approvals={len(final['approvals'])} disputes={len(final['disputes'])}")
print("\nrouting decisions recorded (every one names a source and a confidence):")
for h in final["handoffs"]:
    print(f"  {h['from_agent']} -> {h['to_agent']:<3} "
          f"[{h['decision_source']:<5} {float(h['confidence']):.2f}] {h['reason'][:78]}")
print("\nhuman gates:")
print(" ", gate_summary(final["approvals"]))
print("\ncontract-net bids (each agent was asked a differently framed question):")
for b in sorted(final["bids"], key=lambda x: -float(x["expected_value"]) * float(x["feasibility"]) / max(1, int(x["effort"]))):
    utility = float(b["expected_value"]) * float(b["feasibility"]) / max(1, int(b["effort"]))
    print(f"  {b['agent']}  feasibility={b['feasibility']}  "
          f"expected_value={b['expected_value']}  effort={b['effort']}  "
          f"utility={utility:.3f}")
print("\ncontract-net award:")
for e in final["run_events"]:
    if "winner" in e:
        print(f"  task      : {e.get('title')}")
        print(f"  winner    : {e.get('winner')}  at utility {e.get('utility'):.3f}")
        print(f"  ranking   : {[(r['agent'], round(r['utility'], 3)) for r in e.get('ranking', [])]}")
        for note in e.get("notes", []):
            print(f"  note      : {note}")

print(f"\nROI: multiple={(final.get('roi_report') or {}).get('roi_multiple')}  "
      f"compliance_score={final.get('compliance_score')}")
print(f"\ndecisions asked: {len(decide.calls)}")
for point in dict.fromkeys(decide.asked()):
    print(f"  - {point}")

# ======================================= 3. interrupt / resume cycle (human gate)
rule("3. HUMAN-IN-THE-LOOP - real interrupt() parked, Command(resume=...) applied")
agents_hitl = {k: FakeAgent(v.id, v.posts) for k, v in agents.items()}
graph, runtime, decide = build({
    "route.after_discovery": "proceed",
    "route.after_proposal": "outreach",
    "route.after_auction": "execute",
    "route.after_reply": "yes",
    "route.after_compliance": "audit",
    "route.progress": "progress",
    "contract_net.expedite": "accept",
}, interactive=True, max_replans=2, agents=agents_hitl, thread_id="t_hitl")

parked = graph.run(event=EVENT, thread_id="t_hitl")
pending = graph.pending_interrupts(thread_id="t_hitl")
print(f"after first pass: status={parked['status']}  (run is PARKED)")
for i in pending:
    p = i.value
    print(f"  interrupt id={i.id}")
    print(f"    kind      : {p['kind']}")
    print(f"    raised_by : {p['raised_by']}")
    print(f"    question  : {p['question']}")
    print(f"    preview   : {p['payload_preview']}")
    print(f"    options   : {p['options']}")

resume_value = {"outcome": "approve", "decided_by": "demo.operator",
                "instruction": "ship it, the deck is ready"}
resumed = graph.resume(resume_value, thread_id="t_hitl")
print(f"\nafter resume: status={resumed['status']}")
d = resumed["human_decisions"][-1]
print(f"  HumanDecision gate_id={d['gate_id']} kind={d['kind']} "
      f"outcome={d['outcome']} decided_by={d['decided_by']!r} instruction={d['instruction']!r}")
print(f"  A3 was invoked {agents_hitl[AgentId.A3_OUTREACH].calls} time(s) across the "
      f"park/resume cycle (memoised, so the side effect happens once)")
print("  gate summary:", gate_summary(resumed["approvals"]))

# ================================================== 4. replan: stall counter fires
rule("4. REPLAN - stall counter fires, Task Ledger rewritten, radius widened")
narrow_agents = {
    AgentId.A1_DISCOVERY: FakeAgent(AgentId.A1_DISCOVERY, [
        ("leads", "brand_lead", lead(1, "Lonely Cafe", 12.0))]),
}
graph, runtime, decide = build({
    "route.after_discovery": "replan",     # never enough leads -> replan loop
    "route.progress": "stalled",           # nothing changes -> stall
    "route.after_reply": "no",
}, interactive=False, max_replans=1, agents=narrow_agents, thread_id="t_replan")
replanned = graph.run(event=EVENT, thread_id="t_replan")
print(f"status={replanned['status']}  stall_count={replanned['stall_count']}  "
      f"replan_count={replanned['replan_count']}")
print(f"discovery_radius_km: 10.0 -> {replanned['discovery_radius_km']}")
print(f"task_ledger revision={replanned['task_ledger']['revision']} "
      f"written_by={replanned['task_ledger']['written_by']}")
print("  new plan:")
for step in replanned["task_ledger"]["plan"]:
    print(f"    - {step}")
print("  handoffs mentioning a stall or a replan:")
for h in replanned["handoffs"]:
    if "stall" in h["reason"] or "replan" in h["reason"].lower():
        print(f"    [{h['decision_source']} {float(h['confidence']):.2f}] {h['reason']}")

# ============================================ 5. time travel: fork to another branch
rule("5. TIME TRAVEL - rewind to the reply, fork, take a different branch")
agents_replay = {k: FakeAgent(v.id, v.posts) for k, v in agents.items()}
agents_replay[AgentId.A3_OUTREACH] = FakeAgent(AgentId.A3_OUTREACH, [
    ("threads", "thread", {"thread_id": "thr_1", "event_id": "evt_demo",
                           "brand": "Cafe One", "email": "x@example.invalid",
                           "status": "sent", "day": 0, "intent": Intent.PUSHBACK.value,
                           "reply_text": "Too expensive. Reduce it to 40k or we pass.",
                           "offer_id": "off_1", "sent_at": None, "delivered": True})])
graph, runtime, decide = build({
    "route.after_discovery": "proceed", "route.after_proposal": "outreach",
    "route.after_auction": "execute", "route.after_reply": "pushback",
    "route.after_compliance": "audit", "route.progress": "progress",
    "contract_net.expedite": "accept",
}, interactive=False, max_replans=3, agents=agents_replay, thread_id="t_replay")
original = graph.run(event=EVENT, thread_id="t_replay")
print(f"original run : status={original['status']}  "
      f"negotiation_rounds={original['negotiation_rounds']}  "
      f"mous={len(original['mous'])}")
for h in original["handoffs"][-4:]:
    print(f"    {h['from_agent']} -> {h['to_agent']:<3} {h['reason'][:74]}")

history = replay.list_checkpoints(graph, "t_replay")
print(f"\ncheckpoints on thread t_replay: {len(history)} (newest first)")
for c in history[:6]:
    print(f"  step={c.step:<3} next={str(list(c.next_nodes)):.<26} id={c.checkpoint_id[:8]}")

target = next(c for c in history if c.next_nodes == ("reply",))
print(f"\nrewinding to checkpoint {target.checkpoint_id[:8]} (pending node: reply)")

branching = ScriptedDecide({
    "route.after_reply": "yes",            # <-- the one different answer
    "route.after_compliance": "audit", "route.progress": "progress",
}, confidence=0.83)
forked = replay.fork_and_run(graph, "t_replay", target.checkpoint_id,
                             target_thread="t_fork_yes", decide=branching)
print(f"forked run   : status={forked['status']}  "
      f"negotiation_rounds={forked['negotiation_rounds']}  mous={len(forked['mous'])}")
for h in forked["handoffs"][-4:]:
    print(f"    {h['from_agent']} -> {h['to_agent']:<3} {h['reason'][:74]}")

diff = replay.compare_branches(graph, "t_replay", "t_fork_yes")
print(f"\nbranches identical: {diff['identical']}")
for d in diff["differences"]:
    print(f"  {d['field']:<16} original={d['left']!r:<10} fork={d['right']!r}")
after = graph.compiled.get_state(default_config("t_replay")).values
print(f"\noriginal thread still says: {after['status']} "
      f"(forking did not mutate it)")
print("\nRULE: every difference above traces to one decision_point: "
      "'route.after_reply.intent'")

rule("DEMO COMPLETE")
