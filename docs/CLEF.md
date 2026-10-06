# Cloudflare Clef — what it is and why Paytriq is built on it

> New to Clef? Start here. This note explains the decision model behind every
> agent routing choice in Paytriq, and why a decision model — not a chat LLM —
> is the right tool for this system. For the system itself, see `SYNOPSIS.md`
> and `DESIGN.md`.

## What Clef is

Clef is Cloudflare's family of open-source **decision models** (Clef 27B and
Clef-flash 9B), **released 2026-10-01 — brand new and current state of the art**:
Clef currently leads the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index),
topping 7 of 10 decision benchmarks against Jev and other open decision models
(see benchmark table below). It is hosted on Workers AI. Unlike a chat LLM,
a decision model does not generate free-form text. You send it a `state` plus a
schema of typed `questions`, and it returns typed answers **with calibrated
probabilities** your code can act on immediately: route, block, escalate, or
proceed.

* Announcement and training story: [Introducing Clef: our open-source decision models](https://blog.cloudflare.com/clef-decision-models/)
* Product changelog: [Introducing Clef on Workers AI](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/)
* Hosted models: [`@cf/cloudflare/clef`](https://developers.cloudflare.com/workers-ai/models/clef/) and Clef-flash
* Open weights (Apache 2.0): [Cloudflare/clef on Hugging Face](https://huggingface.co/Cloudflare/clef) — local GGUF at `https://huggingface.co/ggml-org/Clef-Flash-GGUF`

Clef follows the **System One API** (same family as TypeSafe's Jev), so a Jev
integration switches over by changing endpoint and model. One request can ask up
to 64 questions in three types:

| Type | Meaning | Returns |
| --- | --- | --- |
| `noul` | Yes/no question | Probability the answer is yes |
| `choice` | Pick one of a closed set you define | Chosen option + probability per option + confidence |
| `score` | Rate against an ordered rubric | Probability-weighted score + probability per level |

```json
{
  "state": "Checkout has been failing for every customer for the last hour.",
  "questions": {
    "urgent": { "type": "noul", "instructions": "Is this support request urgent?" },
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Outages, errors, and configuration",
        "sales": "Plans and upgrades"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the customer impact?",
      "criteria": ["No impact", "Minor", "Major", "Critical"]
    }
  }
}
```

Example output for that input (probabilities, not prose) — one answer per
question type:

```json
{
  "answers": {
    "urgent": { "type": "noul", "noul": 0.97 },
    "team": {
      "type": "choice",
      "choice": "technical",
      "probabilities": { "billing": 0.02, "technical": 0.94, "sales": 0.04 },
      "confidence": 0.94
    },
    "severity": {
      "type": "score",
      "score": 3.8,
      "probabilities": { "No impact": 0.0, "Minor": 0.02, "Major": 0.18, "Critical": 0.8 },
      "confidence": 0.8
    }
  },
  "usage": { "prompt_tokens": 48, "completion_tokens": 0 }
}
```

The caller acts on this directly — treat as urgent (`noul` 0.97), route to
`technical` (`choice` 0.94), page whoever handles `Critical` impact (`score`
3.8) — no text parsing and no reasoning tokens to wait for.

Key properties that matter for agents: **strictly typed outputs** (no prose to
parse), **calibrated confidence** (a 0.85 means what it says, trained with
label-smoothed cross-entropy plus Brier loss), **64K context window**, a
**vision encoder** (unlike text-only Jev), and **millisecond latency** —
Clef-flash median 38.8 ms vs Jev 524.1 ms (~13x), Clef 209.3 ms (~2.5x), with
Clef leading 7 of 10 decision benchmarks (BFCL, BANKING77, CLINC150, Home
appliances, ToolRet, API-Bank, Amazon ESCI). Full numbers are in the launch
blog post linked above.

## How Paytriq uses it

Every conditional judgement in Paytriq goes through `ctx.decide()` — never a raw
model client or keyword match (`DESIGN.md`, agent roles table). The single entry
point is the decision registry (`decision/registry.py`):

```
clef → gemini → rules
```

* `clef` answers first when configured. The request is translated to one System
  One question by `decision/clef.py:build_body` (`state` + `noul`/`choice`/`score`)
  and posted to `POST /v1/systemone` — deliberately **not**
  `/v1/chat/completions`, which would return prose from the same weights
  (`decision/clef.py:17-46` documents this trap).
* The returned choice carries `source=clef`, calibrated `confidence`, and usage
  (`decision/clef.py:decide`). Every router records that source and confidence on
  its `Handoff` (`core/schemas.py`), and the frontend renders it as a model
  decision rather than a fallback.
* Below `CONFIDENCE_THRESHOLD` (default 0.62) the arbiter escalates instead of
  guessing; below `ESCALATION_THRESHOLD` (default 0.40) it escalates to a human
  rather than to a generative model (`core/config.py`, `.env.example`).
* If Clef is unreachable, the registry degrades **visibly** to Gemini then to
  deterministic `rules`, setting `degraded=True` and recording the substitution
  under `raw["_degradation"]` (`decision/registry.py:24-46`). A degraded run is
  still an honest run — see `EVALUATION.md §6` for what the ablation looks like.

Two transports speak the identical System One protocol (`decision/clef.py:48-57`):

* **Local** (`CLEF_BACKEND=ollama` — name frozen in config; the server is
  llama.cpp's `llama-server` on `:18781`): `llama-server.exe -m
  Clef-Flash-Q4_K_M.gguf --port 18781`. Plain HTTP, no auth, no cost, no rate
  limit — the recommended demo path. Do not use Ollama's `clef-flash` on
  Windows (see `ollama/ollama#18769`; the codebase rewrites `:11434` to `:18781`
  with a warning).
* **Cloud** (`CLEF_BACKEND=workers_ai`): `@cf/cloudflare/clef-flash` via
  `https://api.cloudflare.com/client/v4/accounts/.../ai/run`, bearer token,
  metered — for the deployed demo where no local GPU exists. `auto` probes
  local first.

## Why Clef fits this project

1. **Agents decide far more than they draft.** A1's radius/recovery/fit scoring,
   A2's tier/posture/concession, A3's reply-intent classification, A5's
   per-promise verdicts, A7's uphold/synthesis — each is a closed question over a
   fixed option set, exactly what `noul`/`choice`/`score` express. A chat LLM
   would bury each answer in prose that must be parsed back into a label;
   Clef returns the label with its probability directly.
2. **Confidence must be actionable.** Thresholds (0.62 escalate, 0.40 human) only
   work if probabilities are calibrated. Chat `logprobs` measure next-token
   uncertainty under a different objective and *look* like confidence without
   being it — the codebase therefore never touches them. Clef's Brier-trained
   calibration is what makes "escalate below threshold" a real policy instead of
   theatre.
3. **Routing sits on the hot path.** A single run makes dozens of decisions
   (radius, tier, intent, verdict, adjudication, progress). At ~39 ms median for
   Clef-flash, decision latency stays negligible next to tool calls and human
   gates; at chat-LLM latency the demo would stall between every handoff.
4. **Traces must be auditable.** Because every `Handoff` names its
   `decision_source` and `confidence`, `GET /api/runs/{run_id}/summary` can
   recompute `distinct_gap_values`, count `clef` vs `rules` decisions, and prove
   a trace is a genuine capture. Fabricating that provenance would require
   forging the decision layer itself.
5. **Demos must run anywhere.** Open weights plus the local llama-server path
   mean the full agentic loop runs offline on a laptop with no keys, no billing,
   and no network — while the Workers AI path means the same code deploys to
   Render with two env vars. The `clef → gemini → rules` chain keeps both
   honest: blank credentials produce a *less capable but truthful* run, never a
   fake one.
