# Paytriq — input contract

The system's real input is one object: an `EventProfile` (`core/schemas.py`). It is
created with `POST /api/events`, posted to the `event` zone of the blackboard, and
read by A1 from the board — never from request state. Unknown fields are rejected
with HTTP 422 (`extra="forbid"`); they are never silently discarded.

## `EventProfile` schema (excerpt, `core/schemas.py:233-263`)

| Field | Type | Constraints |
| --- | --- | --- |
| `event_id` | `str` | server-owned on create; echoed on read |
| `name` | `str` | 1–200 chars, required |
| `location` | `str` | 1–200 chars, required |
| `footfall` | `int` | 0–10,000,000, required — A2's pricing moves with this |
| `date` | `str` | 4–40 chars, as the organiser writes it, required |
| `audience` | `str` | 1–500 chars (who attends — this is what sponsors buy), required |
| `budget_inr` | `float \| null` | ≥ 0, optional |
| `categories_wanted` | `list[str]` | optional; accepts an array **or** `"a, b, c"` CSV (split server-side) |
| `deliverables_offered` | `list[str]` | optional; array or CSV, same as above |
| `contact_email` | `str \| null` | optional; must contain a plausible `@` shape |
| `created_at` | `datetime` | server timestamp |

The full machine-readable schema, with required flags and hints, is served live by
`GET /api/schema/event` (the frontend builds its form from it, falling back to a
labelled built-in list when the backend is unreachable).

## Sample event (works offline)

Taken from the console's `DEMO_EVENT` (`frontend/app.js:86-96`) with one deliberate
fix: the original `categories_wanted` (`beverages, fintech, edtech, telecom`) matches
**none** of the offline fixture categories, so an offline run would discover nothing.
Fixture filtering (`tools/fixtures.py`) is exact-match on category, and the seeded
categories are `cafe, gym, salon, printing, coaching, electronics, bookstore` — so
the sample below uses overlapping categories:

```json
{
  "name": "TechFest 2026",
  "location": "VIT Pune, Karad Road",
  "footfall": 5000,
  "date": "2026-02-14",
  "audience": "Second and third year engineering students, 62% male, hiring-season heavy",
  "categories_wanted": ["cafe", "gym", "coaching", "printing"],
  "deliverables_offered": ["main-stage logo backdrop", "4000 Instagram story mentions", "2 reels", "expo stall"],
  "budget_inr": 150000,
  "contact_email": "events@vit.edu.in"
}
```

`event_id` and `created_at` are server-owned and must not be sent; `categories_wanted`
/ `deliverables_offered` may also be sent as comma-separated strings.

> Category drift note (resolved): three event variants exist and all overlap
> the seeded fixture set, so offline runs discover leads either way. The
> **canonical event for evaluation is the TechFest 2026 sample above**
> (`categories_wanted: ["cafe", "gym", "coaching", "printing"]`). The others
> are alternates, not replacements: the live console default
> (`frontend/app.js:92`) reads `cafe, coaching, bookstore, printing`
> (`bookstore` replaces `gym`), and the capture script
> (`scripts/capture_canonical_trace.py:380-387`) uses `Canonical Showcase 2026`
> with `cafe, fintech, edtech`. Neither alternate changes the code contract;
> `app.js` and the capture script are intentionally left untouched.

## Calls

```bash
# What the backend accepts (schema + a validated sample + aliases):
curl http://127.0.0.1:18780/api/schema/event

# Create the event (returns the profile plus its run_id):
curl -X POST http://127.0.0.1:18780/api/events \
  -H "content-type: application/json" \
  -d '{"name":"TechFest 2026","location":"VIT Pune, Karad Road","footfall":5000,"date":"2026-02-14","audience":"Second and third year engineering students, 62% male, hiring-season heavy","categories_wanted":["cafe","gym","coaching","printing"],"deliverables_offered":["main-stage logo backdrop","4000 Instagram story mentions","2 reels","expo stall"],"budget_inr":150000,"contact_email":"events@vit.edu.in"}'

# Read it back (reconstructed from the board, not echoed from the request):
curl http://127.0.0.1:18780/api/events/{event_id}
```
