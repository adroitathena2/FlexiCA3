# Paytriq frontend — live-trace console

A single-page, dependency-free UI for driving and inspecting a Paytriq run.
Vanilla JS + CSS. **No build step, no npm, no bundler, no API keys.** It loads
from `file://` or from a FastAPI static mount.

```
frontend/
  index.html   structure only — every panel is populated by app.js
  app.js       the whole application (IIFE, classic script, no modules)
  styles.css   dark theme, agent + event-kind colour system
  README.md    this file
```

---

## Run it

**1. Start the backend** (from the repo root):

```bash
uvicorn api.main:app --reload --port 18780
```

**2. Open the UI.** Either

* double-click `frontend/index.html` (works from `file://`), or
* visit whatever URL the backend mounts it on.

The console defaults its API base to the page's own origin when served over
HTTP(S) — which is the right answer for the static-mount case — and to
`http://localhost:18780` from `file://`. It is editable in the header and
persisted to `localStorage`, so you only set it once.

> From `file://` the page origin is `null`, so the backend must send
> `Access-Control-Allow-Origin: *` (CORS middleware). Under the static mount
> this is a non-issue: same origin.

**If the backend is down**, the console does not pretend. A persistent banner
appears at the top of the page naming the URL it tried and the command to start
the service, and stays pinned while you scroll. No panel silently shows "empty".

---

## What each control does

Every button on the page performs a real call or a real local action. There are
no placeholder controls.

| Control | Effect |
| --- | --- |
| **API base → Apply / reconnect** | Persists the base URL, re-checks health, re-opens the SSE stream |
| **↻ health** | `GET /health` |
| **Prefill demo event** | Fills the form with a realistic Pune tech-fest profile. Sends nothing |
| **Create event** | `POST /api/events` |
| **schema** | Shows the raw JSON Schema the backend served |
| **Run ▸ (per stage)** | The seven stage endpoints, in pipeline order |
| **Human gates → approve / reject / revise** | `POST /api/gates/{gate_id}` with `{outcome, instruction}` |
| **Sponsor reply presets** | Load one of five texts into the box; **Send reply** posts it |
| **shuffle** | Picks a random preset |
| **Pause / Resume** | Freezes the rendered feed; the stream keeps running and buffers |
| **load /trace** | One-shot `GET /api/runs/{run_id}/trace` — the fallback when the event stream cannot be used (proxy buffering, a CORS policy that blocks `EventSource`, demo-day wifi) |
| **Follow** | Auto-scroll the feed to the newest event |
| **Clear** | Empties the rendered rows and the de-dup memory. The stream stays connected |
| **kind chips** | Show/hide one event kind across the whole feed |
| **Connect / Stop** | Open or close the SSE connection |
| **derivation tree** | Toggles the collapsible citation view of the blackboard |
| **copy JSON** | Copies `/summary` verbatim to the clipboard (falls back to an on-page selectable block if the browser refuses) |
| **clear recorded runs** | Forgets the local run-comparison history |

### Keyboard

| Key | Action |
| --- | --- |
| `1`–`7` | Run discover, propose, outreach, reply, contract, compliance, audit |
| `n` | Run the next stage that has not been run yet |
| `p` | Pause / resume the feed |
| `g` | Refresh gates |
| `r` | Refresh health, board, summary and coordination |
| `c` | Copy the summary JSON |
| `f` | Focus mode — hide both side rails for the trace feed |

Shortcuts are suppressed while a text field has focus.

---

## The demo, in order

1. **Create event** (press *Prefill demo event* first, then *Create event*).
   The `run_id` is learned from the response and the trace stream opens by itself.
2. **Discover**, then watch the feed. The handoff chain across the top of the
   panel is the point of the demo: control moves *between* agents, and the rail
   shows a return leg (`A1 → A2 → A3 → A2 → A5 → A7`) that no script would produce.
3. **Outreach** is expected to fail with `403 {gated:true}`. That refusal is the
   feature: the gate card appears with *approve / reject / revise* and a free-text
   instruction. Approve it, then press **Send outreach** again.
4. **Reply** with the *ambiguous* preset — *"Yesterday we thought the price was
   too high, but let's proceed."* — which contains both a pushback keyword and an
   acceptance keyword. The intent card shows what the classifier returned, which
   agent it routed to, and which decision in the trace did the classifying.
5. **Contract**, **Compliance**, **Audit**, then read the **Run summary** panel.
   `distinct_gap_values` is the featured number, with the raw gaps beside it so
   anyone can recompute it.

---

## Endpoints used

Every one of these is optional at runtime: a 404 or an unrecognised body is
reported in the panel that needed it, and never crashes the page.

```
GET  /health                                             health strip, degraded banner
GET  /api/schema/event                                   event form fields
POST /api/events                                         create event
POST /api/events/{id}/discover|propose|contract|compliance|audit
POST /api/events/{id}/outreach                           403 {gated:true} until approved
POST /api/events/{id}/reply                              {text, brand?}
GET  /api/gates/{run_id}                                 outstanding gates
POST /api/gates/{gate_id}                                {outcome, instruction}
GET  /api/runs/{run_id}/stream                           SSE → live trace feed
GET  /api/runs/{run_id}/trace                            one-shot trace load (fallback)
GET  /api/runs/{run_id}/summary                          run summary + gap statistics
GET  /api/runs/{run_id}/board                            blackboard snapshot
GET  /api/runs/{run_id}/board/tree                       derivation view (text/plain)
GET  /api/runs/{run_id}/disputes|lessons|handoffs        coordination panel
```

`GET /api/events/{id}` and `GET /healthz` are in the contract but nothing in the
UI needs them, so they are not called.

---

## Honesty rules this UI enforces

The project's whole argument is that a trace can be checked rather than believed.
A console that fabricates would undercut that, so:

* **A field the backend did not send is never filled in.** It renders as
  `⟨not sent⟩` with the reason on hover. No zero, no dash, no plausible default.
* **Real, fixture-backed and unavailable are visually distinct.** A tool call
  carrying `source: fixture` is labelled *fixture-backed, not a live lookup* in
  the feed; `/health` reporting a tool as `unavailable` turns the banner amber
  and names the tool.
* **Every decision shows its `source`.** `clef`/`gemini` render as
  `MODEL · <source>`; `rules`/`replay` render as `FALLBACK · <source>`; `degraded`
  gets its own red `DEGRADED` chip. If the event carries no `source`, the card
  says `source not sent` rather than assuming a model.
* **Probability distributions are drawn only if the backend sent one.** If a
  decision event has no `probabilities`, the card says so and shows the
  confidence alone, labelled for what it is.
* **Bar lengths are proportional to the probability sent** (0–1 scale), not
  normalised to the largest option.
* **A failed refresh marks its panel stale** rather than leaving old data looking
  current. The note names the time of the last successful load.
* **No fetch error is swallowed.** Connectivity failures raise the persistent
  banner; HTTP errors are rendered where the data would have been. A 403 gated is
  treated as a real answer from a live backend, not as an outage. Uncaught JS
  errors and unhandled promise rejections are surfaced in the banner too.
* **Switching the API base clears every complaint about the old one**, and
  responses that arrive from a host you have already left are discarded.

---

## Reading the two headline panels

**Handoff chain.** One node per agent, one arrow per recorded handoff, with the
reason under each arrow. When the next hop starts somewhere other than where the
last one ended, the extra sender is inserted — so a return leg reads as a return
leg. Amber arrow and reason mean a deterministic fallback (`rules`/`replay`)
chose the routing.

**`distinct_gap_values`.** Real work has irregular timing, so consecutive start
instants differ on every run; a hand-typed trace advances on a round constant and
its gaps collapse to one distinct value. The panel shows the number, the count of
gaps it came from, the first gaps verbatim, and states its own limits: with fewer
than three operations the inference means nothing, and it detects synthetic
cadence rather than proving anything about a sophisticated fabricator.

---

## Notes and limits

* `localStorage` is used for the API base, the last `run_id`/`event_id`, and up to
  six recorded runs (so the "what changed because of a lesson" panel can diff two
  runs). Clearing site data resets it; nothing is sent anywhere.
* On reload, a known `run_id` is **not** auto-connected — press **Connect**. That
  is deliberate: silently replaying a finished run on load reads like a live one.
* `run_id` is learned from any response that carries one (the create call, a stage
  result, or a streamed event). You can also paste one by hand.
* The trace feed holds everything received in the page's lifetime and caps the
  decision inspector at the 40 most recent cards, saying so when it truncates.
* `prefers-reduced-motion` disables the arrival animation and smooth scrolling.