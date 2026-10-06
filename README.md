# Paytriq

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Background reading: [`docs/CLEF.md`](docs/CLEF.md) explains what Cloudflare
> Clef (the decision model behind every agent routing choice here) is and why
> this system uses a decision model instead of a chat LLM.

Campus sponsorship dealmaking is still done by hand: a student organiser cold-mails
nearby businesses, quotes a flat fee copied from last year's fest, and negotiates over
email with no record of what was promised or why a number was chosen. The previous
prototype automated the shape of that workflow while keeping its failure modes — a
hardcoded list of invented businesses with sequential fake phone numbers, one flat
price for every event regardless of footfall, a reply router that tested for "yes"
before it tested for price objections, and a human approval gate that defaulted to
`approve=True`.

Paytriq replaces that pipeline with seven reasoning agents (A1 Discovery … A7 Arbiter)
coordinating through a typed append-only blackboard under a LangGraph `StateGraph`.
Every routing decision goes through a named decision backend (`clef → gemini → rules`)
and names its source; every outbound email and MoU release parks at a human gate
backed by a durable approval row; every figure in the final ROI report carries a
stated assumption. The execution trace — including `distinct_gap_values`, a timing
statistic that distinguishes a genuine capture from a hand-written one — is the
system's primary output, not an afterthought.

To try it, start the API (`uvicorn api.main:app`), open the dependency-free console in
`frontend/`, prefill the demo event, and drive the seven stages in order — approving
the SEND and MOU gates when they park. See `docs/SYNOPSIS.md` for the short version,
`docs/DESIGN.md` for the architecture, `docs/INPUT.md` / `docs/OUTPUT.md` for the
contracts, and `docs/EVALUATION.md` for what the numbers do and do not prove.

## Setup

Requires Python 3.11–3.13 (`>=3.11,<3.14` in `pyproject.toml`, the single
source of truth; 3.14+ is not supported yet because several transitive
dependencies still cap below it — `requirements.txt` and `Dockerfile` agree).

```bash
pip install -e .[dev]
# or: pip install -r requirements.txt  (runtime only; no pytest/ruff/mypy)

uvicorn api.main:app --reload --port 18780
```

Open the console: double-click `frontend/index.html` (works from `file://`), or visit
the backend root URL when served from the FastAPI static mount. The console defaults
its API base to the page origin over HTTP(S) and to `http://localhost:18780` from
`file://`. Interactive docs are at `/docs`; the machine-readable contract at
`/openapi.json`.

## Demo order

1. **Create event** — press *Prefill demo event*, then *Create event*
   (`POST /api/events`). The `run_id` is learned from the response and the trace
   stream opens by itself.
2. **Discover** (`A1`), then **Propose** (`A2`). Watch the handoff chain: control
   moves between agents, each hop carrying its decision source and confidence.
3. **Outreach** (`A3`) is expected to be refused with `403 {gated:true}` — that
   refusal is the feature. Approve the SEND gate card, then send again.
4. **Reply** with the *ambiguous* preset (*"Yesterday we thought the price was too
   high, but let's proceed."*), which carries both a pushback keyword and an
   acceptance keyword, to exercise the intent classifier rather than a substring
   router.
5. **Contract** (`A4`, MOU gate), **Compliance** (`A5`, veto authority), **Audit**
   (`A6`), then read the run summary (`distinct_gap_values` plus the raw gaps, so
   anyone can recompute it).

Full walkthrough with keys and fallback paths: `frontend/README.md`.

## Tests

```bash
python -m pytest -q tests/unit/test_graph.py
```

The full suite is offline by default; live-model and local-`clef` tests are gated
behind `--run-live` / `--run-clef` (see `tests/conftest.py`). The evaluation
harness and its operating manual live in `eval/` and `docs/EVALUATION.md`.

## Docs

- `docs/CLEF.md` — what Cloudflare Clef is and why Paytriq uses it. Read this first.
- `docs/SYNOPSIS.md` — the two-minute version of this file.
- `docs/DESIGN.md` — architecture diagram (generated from the compiled graph),
  agent roles, orchestration pattern, and tech justification.
- `docs/INPUT.md` — the `EventProfile` input contract with a working sample.
- `docs/OUTPUT.md` — the `MoU`, `ROIReport`, `Handoff`, and `TraceSummary`
  output contracts with a real captured sample.
- `docs/EVALUATION.md` — how to run the ablation, what each column means, and the
  headline negative finding, reported rather than hidden.
- `docs/samples/` — the committed golden run (`golden_run.jsonl`,
  `golden_summary.json` with `README.md` provenance): the one trace artefact
  that ships with the repo (`traces/` itself is gitignored).
- `frontend/README.md` — console operating manual.
