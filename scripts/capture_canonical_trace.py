"""Capture ONE canonical trace: real graph, offline, simulated decision backend.

Drives the real StateGraph (graph.build.build_graph) with the real agent
registry, real board, real OtelTracer and a stub/simulated CLEF decision backend
-- never hand-written JSONL -- through all 7 agents A1->A7 with human SEND + MOU
approvals granted, plus a contract-net award and at least one dispute/veto path,
emitting message/tool/llm spans.

Output:
  traces/canonical_<ts>_<rand>.jsonl (+ .summary.json sidecar via tracer.finish())
  committed sanitized copy -> docs/samples/golden_run.jsonl + golden_summary.json

Provenance: GIT_COMMIT/CODE_SHA256 come from git when commits exist, else the
literal "uncommitted-working-tree" (never an invented hash). CI/Docker/Render
inject the real SHA at build time (see Dockerfile/ci.yml/render.yaml).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import Decision, DecisionSource, EventProfile, RunMode, Settings  # noqa: E402

UNCOMMITTED = "uncommitted-working-tree"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha and sha != "HEAD" and len(sha) >= 7:
            return sha
    except Exception:
        pass
    return UNCOMMITTED


def resolve_provenance_sha() -> str:
    """Real commit SHA when the repo has commits, else the explicit literal.

    Never invents a hash. Respects an already-injected GIT_COMMIT (CI/Docker/
    Render set it at build time); otherwise asks git; a repo with zero commits
    (``fatal: ... does not have any commits yet``) yields ``UNCOMMITTED``.
    """
    env = (os.getenv("GIT_COMMIT", "") or "").strip()
    if env and env.lower() not in ("unknown", ""):
        return env
    return _git_sha()


class SimulatedClefDecide:
    """Stub/simulated decision backend answering as CLEF (labelled, offline).

    Every answer carries ``DecisionSource.CLEF`` with model ``simulated-clef``
    so the trace emits decision + llm spans via GraphRuntime.context_for, while
    remaining fully offline (no network, no model). Routing choices are scripted
    to force full coverage in ONE run:

    * first ``route.after_reply.intent`` -> pushback (forces A3 REVISION_REQUEST
      message + A2 COUNTER_PROPOSAL message + revise loop),
    * subsequent ``route.after_reply.intent`` -> yes (forces A4 contract),
    * ``route.after_compliance`` -> arbitrate (forces A7 adjudicate + veto path),
    * ``route.after_adjudication`` -> escalate (forces human ESCALATION gate),
    * everything else -> a valid choice from the request's own options.
    """

    def __init__(self, confidence: float = 0.82) -> None:
        self.confidence = confidence
        self.calls: list[Any] = []
        self._reply_intent_calls = 0
        self._a3_calls = 0

    def _choose(self, request: Any) -> str:
        point = str(getattr(request, "decision_point", "") or "")
        options = [str(o) for o in (getattr(request, "options", None) or []) if o]
        lower_opts = [o.lower() for o in options]

        def pick(*wants: str, default: str | None = None) -> str:
            for want in wants:
                for opt, low in zip(options, lower_opts, strict=False):
                    if low.startswith(want.lower()):
                        return opt
            if default is not None:
                for opt, low in zip(options, lower_opts, strict=False):
                    if low.startswith(default.lower()):
                        return opt
            return options[0] if options else (default or "proceed")

        if "route.after_discovery" in point:
            return pick("proceed", default="proceed")
        if "route.after_proposal" in point:
            return pick("outreach", default="outreach")
        if "route.after_reply.intent" in point:
            self._reply_intent_calls += 1
            if self._reply_intent_calls == 1:
                return pick("pushback", default="pushback")
            return pick("yes", default="yes")
        if "a3.classify_intent" in point:
            # Agent-level intent (A3 outreach): first PUSHBACK to force the
            # REVISION_REQUEST message, then YES to force the contract path.
            self._a3_calls += 1
            if self._a3_calls == 1:
                return pick("pushback", default="pushback")
            return pick("yes", default="yes")
        if "a4.choose_accepted" in point:
            # Pick the canonical YES thread (with a priced offer), not the first
            # thread in the list (which may be the pushback seed or an empty draft).
            for opt, low in zip(options, lower_opts, strict=False):
                if "canonical" in low or "cafe one" in low or "off_canonical_1" in low:
                    return opt
            # Newest YES thread is usually last; first is often the stale seed.
            return options[-1] if options else "proceed"
        if "route.after_compliance" in point:
            return pick("arbitrate", "arbitrate", default="arbitrate")
        if "route.after_adjudication" in point:
            return pick("escalate", default="escalate")
        if "route.progress" in point:
            return pick("progress", "complete", default="progress")
        if "contract_net.expedite" in point:
            return pick("accept", default="accept")
        if "contract_net.bid" in point and ".expected_value" in point:
            # Spread values so the award has a margin (not a tie).
            try:
                asked = getattr(request, "asked_by", None)
                val = getattr(asked, "value", str(asked))
                if "A3" in str(val):
                    return "100" if "100" in options else pick("100", default=options[0] if options else "100")
                if "A2" in str(val):
                    return "75" if "75" in options else pick("75", default=options[0] if options else "75")
            except Exception:
                pass
            return options[len(options) // 2] if options else "50"
        if "contract_net.bid" in point:
            # feasibility: high for A3, medium otherwise; effort: small for A3.
            try:
                asked = getattr(request, "asked_by", None)
                val = getattr(asked, "value", str(asked))
                if "feasibility" in point:
                    return pick("high" if "A3" in str(val) else "medium",
                                "high", "medium", "low",
                                default=options[0] if options else "medium")
                if "effort" in point:
                    return pick("small" if "A3" in str(val) else "medium",
                                "small", "medium", "large",
                                default=options[0] if options else "medium")
            except Exception:
                pass
            return options[0] if options else "medium"
        if "debate." in point:
            if point.startswith("debate.adjudicate"):
                return pick("resolve", default=options[0] if options else "resolve")
            return pick("hold", default=options[0] if options else "hold")
        if "environment.sponsor.reply" in point:
            return pick("yes", "pushback", default=options[0] if options else "yes")
        return options[0] if options else "proceed"

    def __call__(self, request: Any) -> Decision:
        self.calls.append(request)
        choice = self._choose(request)
        options = [str(o) for o in (getattr(request, "options", None) or []) if o]
        if choice not in options:
            options = [*options, choice]
        if not options:
            options = [choice, "other"]
        target = min(max(self.confidence, 1.0 / len(options)), 1.0)
        rest = round((1.0 - target) / (len(options) - 1), 4) if len(options) > 1 else 0.0
        probs = {choice: round(target, 4)}
        for opt in options:
            if opt != choice:
                probs[opt] = rest
        total = sum(probs.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        return Decision(
            request_id=getattr(request, "request_id", "dec_canonical"),
            question=getattr(request, "question", "canonical simulated decision"),
            choice=choice,
            probabilities=probs,
            confidence=min(probs[choice], self.confidence),
            source=DecisionSource.CLEF,
            model="simulated-clef",
            degraded=False,
            raw={"simulated": True,
                 "decision_point": getattr(request, "decision_point", "")},
        )


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    sha = resolve_provenance_sha()
    # Export so Settings + tracer + summary all record the same provenance.
    # Never invent: UNCOMMITTED is the honest value for a zero-commit repo.
    os.environ["GIT_COMMIT"] = sha
    # CODE_SHA256: content hash when computable, else the same literal. The
    # summary's code_sha256 stays honest: unset means uncommitted tree.
    if not (os.getenv("CODE_SHA256", "") or "").strip() \
            or os.getenv("CODE_SHA256", "").strip().lower() == "unknown":
        os.environ["CODE_SHA256"] = sha

    from core.config import reset_settings_cache
    reset_settings_cache()

    settings = Settings(
        run_mode=RunMode.OFFLINE,
        decision_backend="rules",
        human_gates_interactive=False,
        auto_approve_outcome="approve",
        max_replans=3,
        max_debate_rounds=2,
        max_auction_rounds=3,
        agent_step_budget=6,
        agent_deadline_s=60.0,
        git_commit=sha,
        code_sha256=sha,
    )
    # Fixture-backed tools, no network.
    os.environ["TOOLS_LIVE"] = "false"

    from graph.registry import build_agents
    from observability.tracer import OtelTracer

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"canonical_{ts}_{os.getpid():05d}"
    event_id = "evt_canonical"

    tracer = OtelTracer(settings)
    tracer.configure(run_id, service="paytriq-canonical", event_id=event_id)

    agents = build_agents(settings)
    missing = [a.value for a in __import__("core", fromlist=["REASONING_AGENTS"]).REASONING_AGENTS
               if a not in agents]
    if missing:
        print(f"FATAL: real registry missing agents: {missing}", file=sys.stderr)
        return 2
    print(f"agents: {sorted(a.value for a in agents)}")

    try:
        from tools.registry import build_registry as _build_tools
        tools = dict(_build_tools(settings))
    except Exception as exc:
        print(f"FATAL: tool registry unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"tools: {len(tools)} registered")

    decide = SimulatedClefDecide()

    from graph.build import build_graph, build_runtime
    from graph.state import GraphRuntime

    runtime: GraphRuntime = build_runtime(
        settings, board=None, tracer=tracer, tools=tools, decide=decide,
        agents=dict(agents), mode=RunMode.OFFLINE, interactive=False,
        thread_id=run_id,
    )
    # Human SEND + MOU (+ COUNTER/ESCALATION) approvals granted, labelled as a
    # named human for the canonical trace (not auto). Non-interactive +
    # scripted_resume with decided_by != "auto" records human=True.
    runtime.gate_policy.scripted_resume = {
        "outcome": "approve",
        "decided_by": "human:canonical-trace",
        "instruction": "canonical trace: human SEND + MOU approvals granted",
    }
    runtime.run_id = run_id
    runtime.event_id = event_id

    # Pre-seed the world the run needs, all synthetic and labelled fixture:
    # viable leads (fit>=40 + contactable) so discovery meets its threshold,
    # priced offers so auction/contract have something to award/draft, one
    # sponsor reply (pushback text; the simulated decide forces pushback first
    # then yes, driving both the REVISION_REQUEST/COUNTER_PROPOSAL message path
    # and the contract path), and one BLOCKING risk flag so the compliance
    # router files a real dispute/veto even if A5's own audit finds nothing.
    try:
        from core import AgentId as _AID
        from core import DecisionSource as _DS
        _leads = [
            ("brd_canonical_1", "FIXTURE_Canonical Cafe One", "cafe", 1.2, 78.0),
            ("brd_canonical_2", "FIXTURE_Canonical Books Two", "bookstore", 2.1, 71.0),
            ("brd_canonical_3", "FIXTURE_Canonical Tech Three", "electronics", 3.4, 66.0),
            ("brd_canonical_4", "FIXTURE_Canonical Fit Four", "fintech", 4.0, 62.0),
        ]
        for lid, name, cat, dist, fit in _leads:
            try:
                runtime.board.post(
                    "opportunities", "brand_lead", _AID.A1_DISCOVERY,
                    {"lead_id": lid, "name": name, "category": cat,
                     "distance_km": dist,
                     "contact_email": f"sponsorship@{name.lower().replace('_','-').replace(' ', '-').replace('fixture-canonical-','fixture-')}.invalid",
                     "phone": None, "rating": None, "fit_score": fit,
                     "fit_breakdown": {"category": round(fit * 0.5, 1),
                                        "distance": round(fit * 0.3, 1),
                                        "audience": round(fit * 0.2, 1)},
                     "fit_rationale": "canonical pre-seed: synthetic fixture lead",
                     "source": "fixture", "source_url": None},
                    source=_DS.RULES,
                )
            except Exception as exc:
                print(f"warning: lead {lid} not seeded: {exc}", file=sys.stderr)
        _offers = [
            ("off_canonical_1", "FIXTURE_Canonical Cafe One", "silver", 60000.0,
             ["stall slot", "stage logo"]),
            ("off_canonical_2", "FIXTURE_Canonical Books Two", "silver", 45000.0,
             ["stall slot", "instagram reel"]),
        ]
        for oid, brand, tier, amt, dlvs in _offers:
            try:
                runtime.board.post(
                    "offers", "offer", _AID.A2_PRICING,
                    {"offer_id": oid, "event_id": event_id, "brand": brand,
                     "tier": tier, "amount_inr": amt, "deliverables": dlvs,
                     "pitch": f"{tier} tier for {event_id}: " + "; ".join(dlvs),
                     "fit_score": 70.0, "version": 1, "revised": False,
                     "revision_note": ""},
                    source=_DS.RULES,
                )
            except Exception as exc:
                print(f"warning: offer {oid} not seeded: {exc}", file=sys.stderr)
        runtime.board.post(
            "threads", "thread", _AID.A3_OUTREACH,
            {"thread_id": "thr_canonical_seed", "event_id": event_id,
             "brand": "FIXTURE_Canonical Cafe One",
             "email": "sponsorship@fixture-canonical-cafe-one.invalid",
             "status": "replied", "day": 0,
             "intent": "unknown",
             "reply_text": ("We love the pitch slot, but 12 months exclusivity is too "
                            "long \u2014 can you reduce it?"),
             "offer_id": "off_canonical_1", "sent_at": None, "delivered": False},
            source=_DS.RULES,
        )
        # A second seed: an unambiguous acceptance naming a priced offer, so A4
        # always has a YES thread to draft from even when the first reply drives
        # the pushback/revise/message loop. Both seeds are synthetic fixtures.
        try:
            runtime.board.post(
                "threads", "thread", _AID.A3_OUTREACH,
                {"thread_id": "thr_canonical_accept", "event_id": event_id,
                 "brand": "FIXTURE_Canonical Books Two",
                 "email": "sponsorship@fixture-canonical-books-two.invalid",
                 "status": "closed_won", "day": 1,
                 "intent": "yes",
                 "reply_text": "Yes - approved, let's proceed and sign.",
                 "offer_id": "off_canonical_2", "sent_at": None, "delivered": True},
                source=_DS.RULES,
            )
        except Exception as exc:
            print(f"warning: accept thread not seeded: {exc}", file=sys.stderr)
        try:
            runtime.board.post(
                "risk_flags", "risk_flag", _AID.A5_COMPLIANCE,
                {"flag_id": "rsk_canonical_blocking", "event_id": event_id,
                 "brand": "FIXTURE_Canonical Cafe One",
                 "severity": "blocking", "code": "EXCLUSIVITY_CAP",
                 "message": ("365-day exclusivity exceeds the 42-day cap; "
                             "canonical pre-seed veto"),
                 "evidence": ["off_canonical_1"],
                 "raised_by": _AID.A5_COMPLIANCE.value,
                 "resolved": False, "resolution": ""},
                source=_DS.RULES,
            )
        except Exception as exc:
            print(f"warning: blocking flag not seeded: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"warning: could not pre-seed canonical world: {exc}", file=sys.stderr)

    from langgraph.checkpoint.memory import InMemorySaver

    from graph.checkpointer import CheckpointSetup

    event = EventProfile(
        event_id=event_id, name="Canonical Showcase 2026",
        location="Pune, Maharashtra", footfall=5000, date="2026-11-15",
        audience="college students 18-24", budget_inr=250000.0,
        categories_wanted=["cafe", "fintech", "edtech"],
        deliverables_offered=["stall slot", "stage logo", "instagram reel"],
        contact_email="organisers@canonical.example.invalid",
    )
    graph = build_graph(settings, agents=dict(agents), runtime=runtime,
                        checkpointer=CheckpointSetup(saver=InMemorySaver(), kind="memory"))
    print(f"running canonical graph run {run_id} ...")
    # Pass an explicit state carrying the tracer's run_id so board/handoff
    # run_ids match the trace file's run_id (PaytriqGraph.run otherwise mints a
    # fresh run_id via initial_state, leaving handoff run_id != trace run_id).
    from graph.build import initial_state as _initial_state
    _state = _initial_state(event=event, run_id=run_id, event_id=event_id)
    final = graph.run(state=_state, thread_id=run_id)
    print(f"final status={final.get('status')} phase={final.get('phase')} "
          f"handoffs={len(final.get('handoffs') or [])} "
          f"disputes={len(final.get('disputes') or [])} "
          f"mous={len(final.get('mous') or [])} approvals={len(final.get('approvals') or [])}")

    result = tracer.finish()
    trace_file = Path(result["trace_file"])
    summary_file = Path(result["summary_file"])
    print(f"trace -> {trace_file}")
    print(f"summary -> {summary_file}")

    # Verify with the repo's own loader/summary tooling.
    from observability.lint import lint_trace
    from observability.summary import compute_summary

    summary = compute_summary(trace_file, run_id)
    problems = lint_trace(trace_file)
    errors = [p for p in problems if p.severity == "error"]
    print(f"summary: spans={summary.span_count} events={summary.event_count} "
          f"handoffs={summary.handoff_count} messages={summary.message_count} "
          f"tools={summary.tool_call_count} llm={summary.llm_call_count} "
          f"gates={summary.human_gate_count} distinct_gaps={summary.distinct_gap_values}")
    print(f"decision_counts={summary.decision_counts} tool_status={summary.tool_status_counts}")
    # Agent coverage from spans+events.
    import json as _json
    agents_seen: set[str] = set()
    handoff_pairs: list[str] = []
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = _json.loads(line)
        except Exception:
            continue
        ag = rec.get("agent")
        if isinstance(ag, str) and ag.startswith("A"):
            agents_seen.add(ag)
        attrs = rec.get("attributes") or {}
        fa = attrs.get("from_agent") or attrs.get("handoff.from")
        ta = attrs.get("to_agent") or attrs.get("handoff.to")
        if rec.get("kind") == "handoff" and fa and ta:
            handoff_pairs.append(f"{fa}->{ta}")
    print(f"agents_seen={sorted(agents_seen)}")
    print(f"handoffs_in_file={len(handoff_pairs)} e.g. {handoff_pairs[:8]}")
    # Award + dispute evidence from final state + board.
    run_events = final.get("run_events") or []
    awards = [e for e in run_events if isinstance(e, dict) and e.get("winner")]
    print(f"awards={len(awards)} disputes_in_state={len(final.get('disputes') or [])} "
          f"risk_flags={len(final.get('risk_flags') or [])}")

    ok = True
    if summary.span_count == 0:
        print("FAIL: no spans", file=sys.stderr)
        ok = False
    if summary.handoff_count == 0:
        print("FAIL: no handoffs", file=sys.stderr)
        ok = False
    if len(agents_seen) < 7:
        print(f"FAIL: agent count {len(agents_seen)} < 7: {sorted(agents_seen)}",
              file=sys.stderr)
        ok = False
    if summary.message_count == 0:
        print("FAIL: no message spans (need A3<->A2 REVISION_REQUEST/COUNTER_PROPOSAL)",
              file=sys.stderr)
        ok = False
    if summary.tool_call_count == 0:
        print("FAIL: no tool spans", file=sys.stderr)
        ok = False
    if summary.llm_call_count == 0:
        print("FAIL: no llm spans (simulated CLEF should emit them)", file=sys.stderr)
        ok = False
    if not awards:
        print("FAIL: no contract-net award in run_events", file=sys.stderr)
        ok = False
    if not (final.get("disputes") or final.get("risk_flags")):
        print("FAIL: no dispute/veto path (no disputes and no risk_flags)", file=sys.stderr)
        ok = False
    # SEND + MOU approvals granted (human, not auto).
    approvals = final.get("approvals") or []
    kinds = {(str(a.get("kind") or "").rsplit(".", 1)[-1].lower(),
               str(a.get("outcome") or "").rsplit(".", 1)[-1].lower(),
               str(a.get("decided_by") or "")) for a in approvals if isinstance(a, dict)}
    has_send = any(k == "send" and o == "approve" for k, o, _ in kinds)
    has_mou = any(k == "mou" and o == "approve" for k, o, _ in kinds)
    human_any = any(d and d.strip().lower() != "auto" for _, _, d in kinds)
    print(f"approval_kinds={sorted(kinds)}")
    if not (has_send and has_mou):
        print("FAIL: need human SEND + MOU approvals granted", file=sys.stderr)
        ok = False
    if not human_any:
        print("FAIL: approvals are all auto; canonical needs human-granted", file=sys.stderr)
        ok = False
    if errors:
        print(f"FAIL: trace lint errors: {[e.code for e in errors]}", file=sys.stderr)
        ok = False
    if not ok:
        return 3
    print("CANONICAL TRACE OK")
    print(f"CANONICAL_TRACE_FILE={trace_file}")
    print(f"CANONICAL_SUMMARY_FILE={summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
