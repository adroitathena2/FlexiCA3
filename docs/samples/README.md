# Golden run sample

Committed, citable evidence that a real run happened — a sanitized copy of one
execution trace. `traces/` itself is gitignored (raw runs accumulate, and some
lack their `.summary.json` sidecar), so this directory is the one trace
artefact that ships with the repo.

Regenerate with `python scripts/capture_canonical_trace.py` (real graph,
offline, simulated decision backend — never hand-written JSONL). The script
prints `CANONICAL_TRACE_FILE=` on success; the committed copy below is the
sanitized form of that run.

## Provenance

- Source: `traces/canonical_20261005T211014Z_53124.jsonl` (+ `.summary.json`),
  captured 2026-10-05T21:10Z (`run_mode=offline`, `environment=local`).
- 434 trace lines / 220 events, 214 spans, 10 handoffs, 1 message
  (`A3->A2 REVISION_REQUEST`), 15 tool calls, 57 LLM calls, 5 human gates,
  `distinct_gap_values=155` (many distinct inter-event gaps — the
  anti-fabrication signal; a hand-written trace has ~1).
- `git_commit` / `code_sha256` / `sha` are `uncommitted-working-tree` in the
  source: the repo still has zero commits (P0 owns git), so the capture records
  the explicit literal rather than inventing a hash. CI (`.github/workflows/
  ci.yml` → `GITHUB_SHA`), Docker (`Dockerfile` `ARG GIT_COMMIT/CODE_SHA256`)
  and Render (`render.yaml` `GIT_COMMIT/CODE_SHA256`, falling back to
  `RENDER_GIT_COMMIT`) inject the real SHA at build time; `core/config.py`,
  `observability/summary.py` and `graph/build.py` all honour those names.
- Sanitization (only changes vs. the source): `host_id` → `redacted`, and the
  absolute Windows `trace_file` path → bare filename. Event payloads, timings,
  confidences and decisions are byte-identical. `traces/` was **not** edited.

## Decision sources (MODEL vs FALLBACK)

Per-handoff `handoff.decision_source` in event handoffs (10):

| source | handoffs | meaning |
| --- | --- | --- |
| `clef` | 8 (A1→A2, A2→A3, A3→A2 ×2, A3→A4, A5→A7, A6→A7, A7→A6) | simulated CLEF backend (`simulated-clef`, offline stub, labelled, `degraded=false`) answered |
| `rules` | 2 (A2→A3 contract-net award, A7→A6 gate-outcome routing) | deterministic local decision, labelled |

Summary `decision_counts` is `clef=64, rules=2` over all decision spans (agent
+ router); `llm_call_count=57` comes from the simulated-CLEF answers via
`GraphRuntime.context_for` (decision + `llm` spans). `tool_status_counts` is
`cached=12, ok=1, unavailable=2` (fixture tools, `TOOLS_LIVE=false`).

So this is a **mixed, simulated-backend** run, not a pure-model run: read it as
evidence of control flow through real agents, not of decision-model quality.

## Human approvals (scripted for the canonical capture)

All 5 gates (`send` ×2, `counter`, `mou`, `escalation`) resolved with
`decided_by="human:canonical-trace"`, `human=true`, `outcome=approve`. That is
the canonical capture's scripted human approval so the unattended run proceeds
past SEND/MOU to A4–A7 with an award and a dispute — recorded as human (not
`auto`) and labelled `human:canonical-trace`, never a real operator. Eval
sweeps use the same shape with `decided_by="simulated-human"` (see
`docs/EVALUATION.md` §2/§6).

## Coverage (all 7 agents, award, dispute, MoU, message)

Agents seen in the file: A1, A2, A3, A4, A5, A6, A7 (7/7) with visible handoffs
(`A1->A2`, `A2->A3` incl. the `contract-net award went to A3` award handoff,
`A3->A2` pushback ×2, `A3->A4` acceptance, `A5->A7` veto, `A6->A7`,
`A7->A6` ×2). Final state: 1 contract-net award, 1 dispute
(`EXCLUSIVITY_CAP` blocking flag → filed dispute), 3 risk flags, 1 MoU drafted
and MOU-gate approved, 1 agent message (`REVISION_REQUEST` A3→A2 with span +
event). Loads with the repo's own tooling:

```python
from observability.summary import compute_summary
from observability.lint import assert_clean
s = compute_summary("docs/samples/golden_run.jsonl", s.run_id)
assert_clean("docs/samples/golden_run.jsonl")
```

`tests/golden/replay.jsonl` remains the (empty) replay stub; the canonical
committed copy lives here.
