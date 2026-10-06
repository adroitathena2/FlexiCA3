# Paytriq — output contracts

Four typed artefacts leave the system (`core/schemas.py`). Field lists below are
excerpts — the code is authoritative. After the schemas, one real captured sample is
quoted verbatim with its provenance.

## `MoU` (`core/schemas.py:343-355`)

A4's memorandum of understanding: `mou_id`, `event_id`, `brand`, `amount_inr` (≥ 0),
`terms` (non-empty clause text generated from the *accepted* offer's amount,
deliverables, tier, and version — never `proposals[0]`), `deliverables: list[str]`,
`status` (`draft` → `pending_approval` → `approved` / `signed` / `rejected`;
`approved` only via an MOU-gate `APPROVE`), `document_path | null`, `version` (≥ 1),
`created_at`.

Provenance note: `document_path` is only ever a path a render tool actually
returned (`agents/a4_contract.py`, `tools/pdf.py`), so the chain
`run_id` → `mou_id` → rendered PDF is auditable from the board. Callers used to
pass `"<id>.pdf"` while the tool appended `.pdf` itself, emitting
`<id>.pdf.pdf` (now stripped in `tools/pdf.py:205-210`); the pre-existing files
on disk under `artifacts/documents/mou_*.pdf.pdf` exhibit the old naming and
are kept as-is.

## `ROIReport` (`core/schemas.py:369-392`)

A6's audit: `report_id`, `event_id`, `total_sponsored_inr`, `footfall`,
`estimated_leads`, `spend_inr`, `pipeline_value_inr`, `roi_multiple` (`0.0` with an
explicit "undefined" note when spend is zero — never a divide-by-zero multiple),
`assumptions: list[str]` (**validated non-empty**: an undocumented ROI is rejected by
the schema), `findings: list[AuditFinding]` (reused from A5, never recomputed),
`computed_at`.

## `Handoff` (`core/schemas.py:489-514`)

The core coordination record, written by every router: `handoff_id`, `event_id`,
`run_id`, `from_agent`, `to_agent` (must differ — self-handoffs are rejected),
`reason` (non-empty), `decision_source` (`clef` / `gemini` / `rules` / `replay`,
mandatory), `confidence` (0–1, mandatory), `summary`, `payload_refs` (board zone ids
the receiver should read), `at`.

## `TraceSummary` (`core/schemas.py:703-730`)

Derived statistics proving a trace is a genuine capture: `run_id`, `trace_file`,
`git_commit`, `code_sha256`, `event_count`, `span_count`, `root_span_count`,
`handoff_count`, `decision_counts`, `degraded_decision_count`, `wall_clock_ms`,
`consecutive_start_gaps_ms`, **`distinct_gap_values`** (many for a genuine capture,
exactly 1 for a hand-written trace with constant timing), `llm_call_count`,
`tool_call_count`, `tool_status_counts`, `human_gate_count`, `generated_at`.
Served live by `GET /api/runs/{run_id}/summary`.

## Real captured sample

The committed copy lives in `docs/samples/` (`golden_run.jsonl`,
`golden_summary.json`, provenance in `docs/samples/README.md`) — the one trace
artefact that ships with the repo, since `traces/` itself is gitignored.
Regenerate it with `python scripts/capture_canonical_trace.py` (real graph,
offline, simulated decision backend — never hand-written JSONL).

Provenance: `traces/canonical_20261005T211014Z_53124.summary.json` plus event
`seq 91` of `traces/canonical_20261005T211014Z_53124.jsonl`, run
`canonical_20261005T211014Z_53124`. This run drives all 7 agents A1→A7 with
human SEND + MOU approvals granted (`decided_by="human:canonical-trace"`), a
contract-net award, a compliance dispute/veto, an MoU, and message/tool/llm
spans — the captured evidence is the summary and one routing handoff:

```json
{
  "code_sha256": "uncommitted-working-tree",
  "decision_counts": {
    "clef": 64,
    "rules": 2
  },
  "degraded_decision_count": 0,
  "distinct_gap_values": 155,
  "event_count": 220,
  "git_commit": "uncommitted-working-tree",
  "handoff_count": 10,
  "human_gate_count": 5,
  "llm_call_count": 57,
  "message_count": 1,
  "root_span_count": 83,
  "run_id": "canonical_20261005T211014Z_53124",
  "sha": "uncommitted-working-tree",
  "span_count": 214,
  "tool_call_count": 15,
  "tool_status_counts": {
    "cached": 12,
    "ok": 1,
    "unavailable": 2
  },
  "trace_file": "canonical_20261005T211014Z_53124.jsonl",
  "wall_clock_ms": 179.428
}
```

```json
{
  "agent": "A1",
  "attributes": {
    "confidence": 0.82,
    "decision_source": "clef",
    "event_id": "evt_canonical",
    "from_agent": "A1",
    "handoff_id": "hnd_1c1fc8388c",
    "payload_refs": [
      "brd_canonical_1",
      "brd_canonical_2",
      "brd_canonical_3",
      "brd_canonical_4"
    ],
    "reason": "4 viable lead(s) at 10.0 km meets the threshold of 3",
    "run_id": "canonical_20261005T211014Z_53124",
    "summary": "task ledger revision 0 priced",
    "to_agent": "A2"
  },
  "duration_ms": null,
  "event_id": "evt_canonical",
  "kind": "handoff",
  "name": "A1->A2",
  "parent_span_id": null,
  "run_id": "canonical_20261005T211014Z_53124",
  "seq": 91,
  "span_id": "6edae5ddab34327f",
  "status": "ok",
  "trace_id": "146c68aa0521cf31f14a9a3ac87999f3",
  "ts": "2026-10-05T21:10:14.302065Z"
}
```

### MoU sample (shape-accurate reconstruction, not a verbatim trace line)

The trace file holds events/spans; the MoU artefact itself lives on the board
(zone `contracts`, queryable via `GET /api/runs/{run_id}/board`). The sample
below is reconstructed from the canonical run's board values — run
`canonical_20261005T211014Z_53124`, accepted offer `off_canonical_2`
(`FIXTURE_Canonical Books Two`, silver, INR 45000, deliverables `stall slot` +
`instagram reel`, version 1), event `evt_canonical` (Canonical Showcase 2026) —
with the clause shape from `ContractAgent.compose_terms`
(`agents/a4_contract.py:550-627`) and the canonical run's decided options
(`advance_50_balance_7d`, `non_exclusive`, `evidence_7_days`; trace seq
146–151). `mou_id`/`created_at` are per-run (`new_id("mou")`); the values shown
are illustrative. Status is `approved` via the MOU gate `gat_785ec95d00`
(`decided_by="human:canonical-trace"`, trace seq 159).

```json
{
  "mou_id": "mou_<hex, per run>",
  "event_id": "evt_canonical",
  "brand": "FIXTURE_Canonical Books Two",
  "amount_inr": 45000.0,
  "terms": "MEMORANDUM OF UNDERSTANDING\nReference: offer off_canonical_2 version 1 (silver tier)\nOrganiser: Canonical Showcase 2026\nEvent: Canonical Showcase 2026 at Pune, Maharashtra on 2026-11-15\nExpected audience: college students 18-24 (approx. 5,000 attendees)\nSponsor: FIXTURE_Canonical Books Two\n\n1. SPONSORSHIP FEE. The Sponsor shall pay the Organiser INR 45,000 (Indian Rupees 45000) for the silver tier of sponsorship described in clause 5.\n\n2. PAYMENT SCHEDULE. 50% of the sponsorship fee is payable within 7 days of signature of this MoU; the remaining 50% is payable within 7 days of the event date.\n\n3. LOGO USAGE RIGHTS. The Sponsor's logo may appear only on the collateral listed in clause 5 and only for the duration of the event. The Organiser will not alter the logo's proportions or colours, will not imply endorsement beyond the agreed tier, and will remove the logo from materials published after the event on request. Any use of the Organiser's name or marks by the Sponsor requires the Organiser's prior written consent.\n\n4. CANCELLATION AND NOTICE. Either party may terminate this MoU by 14 days' written notice. If the Organiser cancels, any amount already paid is refunded less costs already incurred and non-refundable third-party charges. If the Sponsor cancels, amounts paid are non-refundable. If the event is postponed beyond 90 days or cancelled, the parties will renegotiate in good faith and any unspent advance will be returned.\n\n5. DELIVERABLES. The Organiser shall provide:\n   1. stall slot\n   2. instagram reel\n   Photo or written evidence of each delivered item will be shared within 7 days of the event.\n\n6. EXCLUSIVITY. No exclusivity is granted under this MoU. The Organiser may enter into other sponsorships at the same tier unless a written exclusivity addendum is signed by both parties.\n\n7. STATUS. This MoU records the parties' agreement on the terms above. It becomes binding on signature by both parties.\n\nSIGNATURES\n   For the Organiser (Canonical Showcase 2026): ______________________  Name: ________________  Date: ____________\n   For the Sponsor (FIXTURE_Canonical Books Two): ______________________  Name: ________________  Date: ____________\n",
  "deliverables": [
    "stall slot",
    "instagram reel"
  ],
  "status": "approved",
  "document_path": "artifacts/documents/mou_<id>.pdf",
  "version": 1,
  "created_at": "2026-10-05T21:10:14.446472Z"
}
```

### ROIReport sample (shape-accurate reconstruction, not a verbatim trace line)

Same provenance as above: reconstructed from the canonical run's board values
(one MoU at INR 45000, `EventProfile` footfall 5000) with the arithmetic from
`AuditAgent._report_from` (`agents/a6_audit.py:757-904`) and the canonical run's
decided bases (`compute_roi`, `apply_15pct_spend_ratio`,
`apply_configured_conversion`, `report_recomputed_pct`; trace seq 196–202).
Constructor defaults apply (`conversion_rate=0.0`, `value_per_lead_inr=0.0`,
exposed rather than invented), so `estimated_leads=0`, `pipeline_value_inr=0.0`,
`spend_inr=45000 × 0.15=6750.0`, and `roi_multiple=(0 + 45000 − 6750) / 6750 =
5.6667` per `_safe_roi_multiple`. `report_id`/`computed_at` are per-run; the
findings reuse A5's two deliverable verdicts (`proceed` on `stall slot` and
`instagram reel`, trace seq 180–182).

```json
{
  "report_id": "roi_<hex, per run>",
  "event_id": "evt_canonical",
  "total_sponsored_inr": 45000.0,
  "footfall": 5000,
  "estimated_leads": 0,
  "spend_inr": 6750.0,
  "pipeline_value_inr": 0.0,
  "roi_multiple": 5.6667,
  "assumptions": [
    "total_sponsored_inr=45000.00 is sum(amount_inr) over 1 signed MoU(s) [mou_<hex>] (brands: ['FIXTURE_Canonical Books Two']); provenance: zone `contracts`, kind `mou`. Unsigned drafts are excluded by design.",
    "conversion_rate=0.0000 (0.00% of footfall becomes a lead) x footfall=5000 => estimated_leads=0; provenance: constructor argument on AuditAgent, NOT measured from any board entry. The previous prototype used an undocumented 3% while its shipped artifact claimed 13%.",
    "value_per_lead_inr=0.00; provenance: constructor argument on AuditAgent, NOT measured. pipeline_value_inr=0 x 0.00 = 0.00. The prototype hardcoded Rs150/lead while its artifact claimed Rs4,500.",
    "spend_ratio=0.15 applied to total_sponsored_inr gives spend_inr=6750.00; provenance: AuditAgent.SPEND_RATIO, the prototype's own ratio, now stated instead of hidden.",
    "roi_multiple=5.6667 computed as (pipeline_value_inr + total_sponsored_inr - spend_inr) / spend_inr; defined because spend_inr=6750.00 > 0",
    "compliance_pct=100.0000 is fulfilled findings / total findings (2/2) over A5's AuditFinding entries in zone `audit`; provenance: reused verbatim from A5, never recomputed.",
    "footfall=5000; provenance: EventProfile evt_canonical on the board.",
    "decided bases: spend_basis=apply_15pct_spend_ratio, lead_basis=apply_configured_conversion, compliance_reading=report_recomputed_pct, lesson_focus=evidence_gap; provenance: a6 plan decisions.",
    "computed_at=<per-run timestamp> for run canonical_20261005T211014Z_53124; no figure above is derived from an unstated constant."
  ],
  "findings": [
    {
      "promise": "stall slot",
      "fulfilled": true,
      "evidence_url": null,
      "confidence": 0.82,
      "source": "clef",
      "note": "A5 deliverable verdict: proceed"
    },
    {
      "promise": "instagram reel",
      "fulfilled": true,
      "evidence_url": null,
      "confidence": 0.82,
      "source": "clef",
      "note": "A5 deliverable verdict: proceed"
    }
  ],
  "computed_at": "2026-10-05T21:10:14.446472Z"
}
```

Provenance chain: `run_id` (`canonical_20261005T211014Z_53124`) → `mou_id` →
rendered PDF under `artifacts/documents/mou_*.pdf.pdf` (old double-extension
naming, kept as-is per the MoU section above) → outreach outbox copies under
`artifacts/outbox/msg_*.json` (`sent:false`, local-outbox only). Recompute any
of it live with `GET /api/runs/{run_id}/board` and
`GET /api/runs/{run_id}/summary`.

Notes on reading it: `distinct_gap_values: 155` over 213 gaps is the anti-fabrication
signal (irregular timing, not a constant cadence); `handoff_count: 10` with
`message_count: 1`, `tool_call_count: 15`, `llm_call_count: 57` and
`human_gate_count: 5` matches a full A1→A7 run with award, dispute, MoU and
human-granted SEND/MOU gates; `sha` is `uncommitted-working-tree` because the
repo has zero commits (P0 owns git) — CI/Docker/Render inject the real SHA at
build time. The handoff names its author (`A1 → A2`), its reason, its decision
source (`clef` from the simulated backend, confidence `0.82`), and the board
refs the receiver should read. Recompute with
`GET /api/runs/{run_id}/summary` and `GET /api/runs/{run_id}/trace` on any new run.
