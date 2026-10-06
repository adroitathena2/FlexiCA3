# Running the evaluation

This file is the operating manual for Paytriq's evaluation harness: how to run it,
what every column means, and — most importantly — what the numbers are and are not
evidence of.

Nothing here is aspirational. If a figure could not be measured it renders as
`unavailable`; if a run could not complete it is listed as a failure with its reason.
Those two rules override every other consideration in this document.

---

## 1. What is being measured, and why it is hard

The project's central claim is that Paytriq is *genuinely agentic* rather than a
pipeline with extra steps. A claim cannot be graded, so `eval/` builds three
implementations of the same task that differ **only** in how control flow is chosen:

| Condition | Control flow |
| --- | --- |
| `fixed_pipeline` | hardcoded `A1 → A2 → A3 → A4 → A5 → A6` |
| `keyword_router` | the same agents; routing by substring match on the sponsor's reply |
| `agentic` | decision-layer routing, disputes, arbitration, replanning, gates |

All three receive the **same runtime and therefore the same agents**. A baseline built
from different, dumber agents would prove nothing about adaptivity.

---

## 2. How to run it

```powershell
# The real stack: the project's own agents via graph.registry.build_agents
python -m pytest -q tests/unit/test_eval_realstack.py   # fast, offline, ~10 s

# A full sweep. All five scenarios x three conditions x five seeds = 75 runs.
# Roughly five minutes: it is executing seven real ReAct agents, not a lookup table.
python -c @"
import logging, pathlib
logging.disable(logging.WARNING)
from eval.ablation import run_ablation
from eval.conditions import ALL_CONDITIONS
from eval.report import to_markdown, to_json
from eval.scenarios import ALL_SCENARIOS

report = run_ablation(ALL_SCENARIOS, ALL_CONDITIONS, (11, 22, 33, 44, 55))
pathlib.Path('realstack_ablation.md').write_text(to_markdown(report), encoding='utf-8')
pathlib.Path('realstack_ablation.json').write_text(to_json(report), encoding='utf-8')
print(to_markdown(report))
"@
```

### Simulated gates: letting an unattended sweep reach A4–A7

Without answered gates the real stack parks at SEND and no run reaches A4–A7
(see §6). `AgenticCondition(simulated_gates=True)` answers every gate with
`decided_by="simulated-human"` and seeds the same approvals into
`RealAgentRuntime` scopes, so sweeps proceed to contract, compliance veto,
arbitration and audit while the trace still records that no real person
approved anything:

```python
from core.schemas import RunMode
from eval.ablation import run_ablation
from eval.conditions import (
    AgenticCondition, FixedPipelineCondition, KeywordRouterCondition,
)
from eval.scenarios import ALL_SCENARIOS

conds = [
    FixedPipelineCondition(prefer_real_runtime=False),
    KeywordRouterCondition(prefer_real_runtime=False),
    AgenticCondition(prefer_real_runtime=False, run_mode=RunMode.OFFLINE,
                     simulated_gates=True),
]
report = run_ablation(ALL_SCENARIOS, conds, (11, 22, 33, 44, 55))
```

`auto` (counted only, never authorises) and `simulated-human` (authorises in
eval, labelled) are different values on purpose: an unattended sweep that
needs to measure A4–A7 uses the latter, and a reader can always tell which one
a run used from its notes and provenance.

### Which stack you actually got

The runtime is resolved once per sweep and **named in the provenance block**. There are
three possibilities and they are not interchangeable:

| Provenance line | What ran |
| --- | --- |
| `real-agents (graph.registry.build_agents -> N agents); decisions: …` | the real agents. This is the only row that is evidence about the production stack. |
| `FALLBACK local-rules-stub (deterministic, offline, DecisionSource.RULES, no LLM calls): …` | the pure-`core` stub. Honest, labelled, and **not** evidence about agent behaviour. |
| `local-rules-stub (requested explicitly)` | the stub, chosen deliberately by the caller. |

The word `FALLBACK` is load-bearing. It must never be removed to make a table look
better, and `tests/unit/test_eval.py` asserts its presence.

---

## 3. What each column means

`METRIC_FIELDS` in `eval/ablation.py` fixes the metric set; these are the two that
matter and the rest in context.

### `path_variation` — the headline

The number of **distinct agent paths** observed across seeds on byte-identical scenario
inputs.

* `fixed_pipeline` scores exactly **1**, and not by accident: a hardcoded sequence has
  no mechanism to produce a second path. That is the argument the ablation exists to
  test, and it is the one row that is true by construction.
* An adaptive system *can* exceed 1, because different inputs reach different agents.

`None` (rendered `unavailable`) means no run completed, so no variation exists. It is
deliberately not `0`, which would claim the system took exactly one path.

### The three columns that keep `path_variation` honest

**`path_variation` alone is not evidence of adaptivity.** A router that ping-pongs
between two agents also varies, and varies *more*. Every `path_variation` figure is
therefore reported next to:

| Column | Reads as | Why it is here |
| --- | --- | --- |
| `stall_rate` | runs that repeated an agent with no new board state | the oscillation detector. A router bouncing between two nodes scores maximum variation and maximum stalls. |
| `dispute_resolution_rate` | of the runs that surfaced a disagreement, the fraction resolved | does the variation lead anywhere, or does it just move? |
| `completion_rate` | runs that finished rather than failing | a system that varies by crashing has not adapted. |

A condition only has a case if it varies **and** resolves.

### The rest

| Column | Meaning |
| --- | --- |
| `steps_taken` | agent dispatches, including ones that parked at a gate. |
| `decision_calls` | every `Decision` produced, whatever asked for it. |
| `decisions_by_source` | the mix of `CLEF` / `GEMINI` / `RULES`. **`RULES`-only in a LIVE run is a degraded run**, and `degraded_decisions` counts it. |
| `llm_calls` | decisions that a *model* actually answered. A `RULES` answer never increments it, so a sweep run without clef cannot be read as one with it. |
| `tool_status_counts` | tool calls split by `ToolStatus`. `cached` means the real tool served its own labelled fixture (`TOOLS_LIVE=false`); `unavailable` means there was no backend. Both are visible rather than folded into a success count. |
| `human_gates_raised` | gates raised. Whether the gate was *answered* is a separate question — see the park semantics below. |
| `disputes_raised` / `_resolved` / `_escalated` | the conflict surface. `escalated` never means `resolved`. |
| `blocking_flags` | A5's `BLOCKING` risk flags — the veto, materialised as data. |
| `unverifiable_claims` | deliverables a system *claimed* were fulfilled where the scenario declares no evidence exists. The fabrication signal. |
| `negotiation` | `signed`, `rounds_to_deal`, `final_amount_inr`, `surplus_split`, `policy_violations`. `surplus_split` is `None` — never a fabricated share — when the final amount falls outside the bargaining range. |
| `path_variation_caveat` | non-empty whenever the row cannot be read as evidence about the production stack. Printed verbatim. |

---

## 4. Gate semantics on the real stack: a park is not a success

The real agents raise `HumanGateRequired` deliberately, so the graph can call
`interrupt()` and wait for a person. The harness has no human, so:

* the run **parks** — `ok=False`, `gate_pending=True`, the pending `GateKind` recorded,
  and the drafted MoU or thread left at `status="pending_approval"`;
* a park is **not** an error: `errors` is empty, because there is nothing to fix but a
  missing human;
* the harness's auto-resolved gate outcome (`decided_by="auto"`) is recorded **for the
  count only** and never authorises the side effect. A parked send is not recorded as a
  sent message.

This is why the real-stack sweep below shows `signed_rate = 0` for the two conditions
that gate, and why it is the *correct* number rather than a missing one.

A7's `ArbiterEscalation` is handled the same way: the escalation is recorded with its
dispute id and **not** resolved.

---

## 5. Stub vs. real: the caveat, stated plainly

The stub (`LocalRuleRuntime`) is a handful of arithmetic checks in pure `core`. It is
useful and it is honest about itself, but two of its numbers are structurally bounded:

* its decisions are `DecisionSource.RULES` — there are no LLM calls, ever;
* its `path_variation` ceiling is low by construction, because a table lookup has no
  temperature to sample.

So: **a stub number is a claim about control flow under a deterministic stub, not about
the production agent stack.** Wherever a stub run contributes, the report says so in the
provenance block and on every affected record.

The same applies to the *inputs*. The harness plays the counterparty: it publishes the
scenario's event profile, its discovered leads, and the drawn sponsor reply onto the
run's blackboard, stamped `source=fixture`. Those are the inputs, and they are synthetic
by construction (`SYNDETIC_BRAND_NOTE`). The *control flow* — who decided what, when,
and with what confidence — is the system's own.

One consequence is worth stating explicitly: the scenario's pre-priced `SeedOffer`s are
**not** published to the real stack, so that A2 does the pricing itself. A consequence of
*that* is that a seeded offer's clause facts — S1's 365-day exclusivity, its
attendee-data clause — never reach the real agents. The real A4's contract template
grants no exclusivity and no attendee-data transfer, so S1's policy conflict is not
reproduced on the real stack. The benchmark reports that as a **missed conflict**, which
is the correct reading: the system did not create the conflict the scenario was built
around.

---

## 6. The result with simulated gates (stub, offline)

75 runs: 5 scenarios × 3 conditions × 5 seeds (11, 22, 33, 44, 55), on the
local-rules stub (`local-rules-stub (requested explicitly)`) with the agentic
condition in simulated-gate mode:

```python
AgenticCondition(prefer_real_runtime=False, run_mode=RunMode.OFFLINE,
                 simulated_gates=True)
```

`simulated_gates=True` answers every SEND/MOU/COUNTER gate with
`decided_by="simulated-human"` (and seeds the same approvals into
`RealAgentRuntime` scopes) so unattended sweeps proceed past A3/A4 to A4–A7.
It is labelled, never a real human approval and never `auto`. Without it the
real stack parks at SEND (`A1 → A2 → A3 → A2 → A3 → …`, `path_variation = 1`,
`stall_rate = 1.00`, `disputes_raised = 0`) because no thread ever reaches
`accepted` — that parked finding is the reason this mode exists, not a result
it replaces.

```
| Condition         | Mean path_variation | Per-scenario            | stall_rate | signed_rate |
| fixed_pipeline    | 1.0                 | 1, 1, 1, 1, 1           | 0.00       | 1.00       |
| keyword_router    | 1.6                 | 2, 1, 1, 2, 2           | 0.48       | 0.48       |
| agentic           | 2.6                 | 4, 1, 4, 3, 1           | 0.48       | 0.16       |
```

Full health (from the same run): `fixed_pipeline` dispute 60% / resolved 0% /
replan 0% / gate 0%; `keyword_router` dispute 0% (resolution `unavailable`) /
replan 0% / gate 0%; `agentic` dispute 52% / resolved 61.5% / replan 32% /
gate 48%. Totals: 75 runs, 75 completed, 0 failed, 183 decision calls (all
`RULES`, 0 degraded, 0 LLM calls), 232 tool calls, 33 disputes raised, 16 human
gates raised.

**The agentic condition reaches A4–A7 and varies.** Its distinct paths include,
verbatim from the per-scenario table:

* S1 (4 paths): `A1->A2->A3->A6->A3->A4->A5->A7->A4->A2->A4` and
  `A1->A2->A3->A6->A3->A2->A3->A4->A5->A7->A4->A2->A4` (contract drafted,
  compliance vetoed, arbiter resolved, plan amended) alongside shorter
  no-contract paths;
* S4 (3 paths): `A1->A2->A3->A6->A4->A5->A4->A6` and
  `A1->A2->A3->A6->A3->A4->A5->A4->A6` (clean deal closes and audits);
* S5 (1 path): `A1->A2->A3->A6->A4->A5->A7->A4->A2->A4` (unverifiable
  deliverables audited honestly, dispute arbitrated, revision loop budgeted);
* S2 (1 path): `A1->A2->A3->A6->A7` (negative-EV lead challenged by A6 and
  arbitrated — deterministic across seeds, which is why this scenario scores 1).

Not every scenario varies (S2 and S5 score 1 by measurement, not by
construction), and that is reported rather than hidden. The headline 2.6 is the
mean of measured per-scenario variation, recomputable from the raw per-run
records in the JSON export.

### What these numbers prove, and what they do not

They prove **control-flow adaptivity under a deterministic stub with simulated
gates**: byte-identical scenario inputs with different seeds draw different
sponsor replies, the decision-layer router sends those replies to different
agents, disputes raised (52% of agentic runs) are resolved 61.5% of the time via
A7 arbitration, resolutions change the plan (32% replans: exclusivity cut,
data clause struck, lead retargeted), and gated releases proceed to audit
(48% gated, 16% signed cleanly). The `keyword_router` varies less (1.6) with
0 disputes and 48% stalls — its variation is early exits and oscillation, which
is exactly what the stall column is for.

They do **not** prove anything about the production agent stack with a live
model:

1. **The runtime is the stub.** `local-rules-stub` bounds achievable variation
   by construction (see `RunOutcome.path_variation_caveat`, printed verbatim in
   the report). Read this row as evidence about control flow, not about agent
   capability.
2. **The decisions are `RULES`-only.** 183 calls, 0 LLM calls, 0 degraded
   (OFFLINE: rules is intended, not degraded). A real backend replaces the
   calibrated rules classifier via `DecisionInvoker`; when it does, the
   `decisions_by_source` column says so.
3. **The gates are simulated.** `decided_by="simulated-human"` authorises the
   release in eval so A4–A7 are measured at all. It is not a human approval and
   must never be read as one; production gates park for a person.

Bringing `clef` up (temperature-sampled routing) and answering gates
interactively are the two real experiments this ablation is designed for; the
harness will report `DecisionSource.CLEF` and human `decided_by` when they
happen. More seeds will move the means but not the reading: variation with
resolution is adaptation, variation with stalls is oscillation.

### The oscillation caveat, restated

`path_variation` counts *distinct* paths. Nothing in that count distinguishes

```
A1 → A2 → A3 → A4 → A5 → A6      (productive: each step changed the next)
A1 → A2 → A3 → A2 → A3 → A2      (oscillation: nothing changed)
```

Both are one path. Both score 1. Any claim of adaptivity that rests on `path_variation`
alone is unfalsifiable, which is why `stall_rate`, `dispute_resolution_rate` and
`completion_rate` are reported beside it and must be read with it.

---

## 7. Reproducibility and limitations

* **Provenance is mandatory.** Every report opens with git commit, a content hash over
  the contributing packages, the run mode, the decision chain, and the runtime that
  produced each run. A number without provenance in this project is an unauditable
  number.
* **Unmeasured means `unavailable`,** never `0`, `N/A`, or a plausible substitute. A
  `stdev` over one seed is `unavailable`, because a standard deviation from one
  observation is noise presented as precision.
* **A run that cannot complete is a failure with a reason,** never a zero row.
* **Provenance probe, fixed.** `eval/ablation.py::_probe_decision_backends`
  probes `decision.registry` for `build_registry` first (with `build_backend`,
  `get_backend`, `resolve_backend` as fallbacks), which is the factory that
  module actually exposes — so the "Decision backends probed" line now names
  the live backend and its `order=`/`first_available=` chain instead of the
  old misleading `present but exposes none of [...]`. The "Runtimes used"
  lines (via `conditions.RealAgentRuntime.provenance()`) remain the
  authoritative record of what each run executed.
* **The real stack was mid-development during measurement.** `blackboard/zones.py` and
  `graph/build.py` were being edited concurrently with this run, and `agents.a7_arbiter`
  was briefly unimportable because the `tasks` zone declared kinds with no canonical
  model. If a step's agent is missing from the registry, the step is recorded as an
  explicit failure and the missing ids are named in the run's notes — never as a zero.
