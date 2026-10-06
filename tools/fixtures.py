"""SYNTHETIC seed data. Nothing in this file describes a real business.

.. warning::

   **Every record below is invented.** The names, coordinates, distances and
   categories are fabricated for offline runs. They are *not* observations of
   Pune, and no record here may be presented as, quoted as, or emailed as a
   real prospective sponsor. Each name is prefixed ``FIXTURE_`` and each record
   carries ``"source": "fixture"`` precisely so that a reader who greps a trace,
   an artifact or a demo screenshot can tell within one glance that the data is
   seed data.

Why this file exists at all
---------------------------
The Paytriq pipeline has to run end-to-end with no network and no API keys,
because that is how it is demonstrated and tested. The previous prototype solved
this by hardcoding its "discovered businesses" inside the discovery module with
no marker at all, so a fixture and a live result were byte-identical to a
reader. Here the separation is enforced three ways:

* the records live in one obviously-named module with a module-level warning;
* they are only ever emitted through :func:`tools.base.fixture_result`, which
  forces ``status=ToolStatus.CACHED`` and ``degraded=True``;
* the ``FIXTURE_`` prefix travels into every derived artefact (brand name,
  email subject, outbox filename), so a leaked fixture is self-incriminating.

What is deliberately *absent*
-----------------------------
* **No phone numbers.** The old prototype used ``+91-90001``-style sequential
  fakes. A number-shaped string is a dialable artefact and invites someone to
  try it.
* **No email addresses.** Not even ``partnerships@{brand}.com`` — the old code
  invented a domain that does not exist and then reported a successful send to
  it.
* **No ratings.** A fixture rating would be an unverifiable number travelling
  downstream into a fit score.

Contact routes
--------------
Fixture leads carry a contact address in the **RFC 6761 reserved ``.invalid``
TLD** (``sponsorship@<fixture-name>.invalid``), which by specification can never
resolve to a real domain. This is a considered addition, not a slip:

* Without one, A1's accept-decision correctly refuses every fixture lead as
  uncontactable -- verified against the local decision model at 0.897 confidence
  for "false" -- so the whole offline pipeline runs and produces nothing. A demo
  in which the system refuses to do anything is indistinguishable from a broken
  one, and the refusal was *correct*, which is the worst case.
* The old prototype's mistake was not having a contact field; it was inventing a
  domain that **does** exist in shape (``partnerships@{brand}.com``) and then
  reporting a successful send to it. A ``.invalid`` address cannot be delivered
  to, cannot reach a real party, and cannot be confused with a live Overpass
  result: the address says ``.invalid`` and the lead's ``source`` still says
  ``fixture``.

So the contact route is present enough to exercise the negotiation and contract
path, and inert enough that nothing can be sent to it.
"""
from __future__ import annotations

from typing import Any

from core.ids import new_id
from core.schemas import BrandLead

from .base import EARTH_RADIUS_KM, haversine_km

__all__ = [
    "FIXTURE_SOURCE", "FIXTURE_NOTE", "FIXTURE_ORIGIN", "fixture_slug",
    "FIXTURE_BRAND_RECORDS", "FIXTURE_LOCATIONS", "FIXTURE_EVIDENCE_PAGES",
    "fixture_brands", "fixture_leads", "fixture_location", "fixture_page_text",
]

#: The value every record's ``source`` field carries. Compared against in tests.
FIXTURE_SOURCE = "fixture"

#: Appended to any reason string that refers to fixture-sourced data.
FIXTURE_NOTE = (
    "synthetic seed data from tools/fixtures.py; not a real business, not a real "
    "observation, not contactable"
)

#: Approximate centre of the seeded area (Shivajinagar, Pune). Used only to give
#: the seeded records plausible distances; not a claim about any venue.
FIXTURE_ORIGIN: tuple[float, float] = (18.5308, 73.8475)


def fixture_slug(name: str) -> str:
    """A DNS-safe slug for a fixture's reserved ``.invalid`` contact address.

    Deterministic, so the same fixture always yields the same address and a
    re-run produces a comparable trace.
    """
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in name.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "fixture"


def _brand(lead_name: str, category: str, lat: float, lon: float,
           blurb: str) -> dict[str, Any]:
    """One seeded record. ``distance_km`` is computed, never hand-typed."""
    return {
        "lead_id": new_id("brd"),
        "name": lead_name,
        "category": category,
        "lat": lat,
        "lon": lon,
        "distance_km": haversine_km(FIXTURE_ORIGIN[0], FIXTURE_ORIGIN[1], lat, lon),
        "phone": None,
        "contact_email": f"sponsorship@{fixture_slug(lead_name)}.invalid",
        "rating": None,
        "blurb": blurb,
        "source": FIXTURE_SOURCE,
        "source_url": None,
    }


#: Twelve records across six categories, all invented. Deliberately small: a
#: large fixture set would invite a demo to imply breadth it does not have.
FIXTURE_BRAND_RECORDS: tuple[dict[str, Any], ...] = (
    _brand("FIXTURE_Shivaji Cafe", "cafe", 18.5301, 73.8441,
           "Invented cafe near the seeded origin. No such business is asserted."),
    _brand("FIXTURE_Bridge Road Coffee", "cafe", 18.5364, 73.8452,
           "Invented cafe. Name is generic and may collide with a real firm."),
    _brand("FIXTURE_Deccan Fitness Studio", "gym", 18.5158, 73.8567,
           "Invented gym. No rating or phone is seeded."),
    _brand("FIXTURE_Kothrud Power Gym", "gym", 18.5074, 73.8077,
           "Invented gym west of the seeded origin."),
    _brand("FIXTURE_Laxmi Road Beauty Bar", "salon", 18.5271, 73.8533,
           "Invented salon."),
    _brand("FIXTURE_Laxmi Road Unisex Salon", "salon", 18.5249, 73.8551,
           "Invented salon."),
    _brand("FIXTURE_Shivajinagar Copy Point", "printing", 18.5312, 73.8432,
           "Invented print shop."),
    _brand("FIXTURE_University Road Printworks", "printing", 18.5349, 73.8421,
           "Invented print shop."),
    _brand("FIXTURE_Examveda Coaching Centre", "coaching", 18.5421, 73.8391,
           "Invented coaching centre."),
    _brand("FIXTURE_Law Garden Tuitors", "coaching", 18.5379, 73.8381,
           "Invented coaching centre."),
    _brand("FIXTURE_Fc Road Mobile Store", "electronics", 18.5441, 73.8428,
           "Invented electronics retailer."),
    _brand("FIXTURE_Bookworld Stationers", "bookstore", 18.5289, 73.8459,
           "Invented bookshop."),
)

#: Locations the geocoder can answer for without a network. Coordinates are
#: approximate public landmarks, included so a fixture-mode demo has an origin.
FIXTURE_LOCATIONS: dict[str, dict[str, Any]] = {
    "shivajinagar": {
        "query": "Shivajinagar, Pune, Maharashtra",
        "display_name": "Shivajinagar, Pune, Maharashtra, India (FIXTURE approx.)",
        "lat": FIXTURE_ORIGIN[0],
        "lon": FIXTURE_ORIGIN[1],
        "source": FIXTURE_SOURCE,
    },
    "kothrud": {
        "query": "Kothrud, Pune, Maharashtra",
        "display_name": "Kothrud, Pune, Maharashtra, India (FIXTURE approx.)",
        "lat": 18.5074,
        "lon": 73.8077,
        "source": FIXTURE_SOURCE,
    },
    "fc road": {
        "query": "Fergusson College Road, Pune, Maharashtra",
        "display_name": "Fergusson College Road, Pune, India (FIXTURE approx.)",
        "lat": 18.5441,
        "lon": 73.8428,
        "source": FIXTURE_SOURCE,
    },
    "baner": {
        "query": "Baner, Pune, Maharashtra",
        "display_name": "Baner, Pune, Maharashtra, India (FIXTURE approx.)",
        "lat": 18.5642,
        "lon": 73.7769,
        "source": FIXTURE_SOURCE,
    },
    "hinjewadi": {
        "query": "Hinjewadi, Pune, Maharashtra",
        "display_name": "Hinjewadi Phase 1, Pune, Maharashtra, India (FIXTURE approx.)",
        "lat": 18.5913,
        "lon": 73.7389,
        "source": FIXTURE_SOURCE,
    },
}

#: Page bodies for ``FetchPageTool``'s fixture mode. These are written so the
#: evidence verifier has something real to inspect *offline*: the text genuinely
#: mentions (or genuinely does not mention) the promised items, which is what
#: makes the compliance test meaningful without a network.
FIXTURE_EVIDENCE_PAGES: dict[str, str] = {
    "https://fixture.invalid/shivaji-cafe/sponsorship": (
        "FIXTURE PAGE — synthetic content, not a real website.\n"
        "Shivaji Cafe sponsorship confirmation 2026. We have approved the main-stage "
        "banner placement and the printed leaflet insert for the spring festival. "
        "The stall will be staffed from 09:00 to 18:00 on both event days.\n"
        "Signed off by the store manager, countersigned by the festival committee."
    ),
    "https://fixture.invalid/deccan-fitness/sponsorship": (
        "FIXTURE PAGE — synthetic content, not a real website.\n"
        "Deccan Fitness Studio has confirmed a cash sponsorship of INR 40,000 and will "
        "run a sponsor activation booth on day one. Logo placement on the backdrop is "
        "still under discussion and has not been agreed."
    ),
    "https://fixture.invalid/laxmi-beauty/sponsorship": (
        "FIXTURE PAGE — synthetic content, not a real website.\n"
        "Sponsor deck acknowledgement. Thank you for your interest. Our marketing "
        "team is reviewing budgets for the season and will revert next quarter. No "
        "commitment has been made on deliverables at this stage."
    ),
}


def fixture_brands(*, categories: list[str] | None = None,
                   limit: int | None = None) -> list[dict[str, Any]]:
    """Seeded brand records, optionally filtered by category.

    Filtering is exact on the normalised category so the demo behaves like the
    live path rather than silently returning everything.
    """
    wanted = {c.strip().lower() for c in (categories or []) if c and c.strip()}
    rows = [dict(r) for r in FIXTURE_BRAND_RECORDS]
    if wanted:
        rows = [r for r in rows if str(r["category"]).lower() in wanted]
    if limit is not None:
        rows = rows[:limit]
    return rows


def fixture_leads(*, categories: list[str] | None = None,
                  limit: int | None = None) -> list[BrandLead]:
    """Seeded records as schema-valid ``BrandLead`` objects.

    ``fit_score`` is deliberately left at ``0.0`` with an explanatory rationale:
    fit is A1's judgement, and a fixture that arrived with a score would let the
    rest of the pipeline treat a made-up number as a computed one.

    ``contact_email`` is a reserved ``.invalid`` address and ``phone`` stays
    ``None`` -- see "Contact routes" in the module warning. The address is
    deliberately non-dialable; it exists so the accept-decision has a contact
    route to weigh, not so anything can be delivered.
    """
    leads: list[BrandLead] = []
    for row in fixture_brands(categories=categories, limit=limit):
        slug = fixture_slug(str(row["name"]))
        leads.append(BrandLead(
            lead_id=str(row["lead_id"]),
            name=str(row["name"]),
            category=str(row["category"]),
            distance_km=float(row["distance_km"]),
            contact_email=f"sponsorship@{slug}.invalid",
            phone=None,
            rating=None,
            fit_score=0.0,
            fit_breakdown={},
            fit_rationale=f"unscored fixture lead. {FIXTURE_NOTE}.",
            source=FIXTURE_SOURCE,
            source_url=None,
        ))
    return leads


def fixture_location(query: str) -> dict[str, Any] | None:
    """Look up a seeded location by substring. ``None`` when nothing matches.

    Returning ``None`` (rather than the origin, or a fuzzy guess) is the point:
    an unrecognised place name has no honest answer offline.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return None
    for record in FIXTURE_LOCATIONS.values():
        haystack = f"{record['query']} {record['display_name']}".lower()
        if needle in haystack:
            return dict(record)
    return None


def fixture_page_text(url: str) -> str | None:
    """Seeded page body for ``url``, or ``None`` if nothing was seeded for it."""
    target = (url or "").strip().rstrip("/")
    if not target:
        return None
    for seeded_url, text in FIXTURE_EVIDENCE_PAGES.items():
        if target.rstrip("/") == seeded_url.rstrip("/"):
            return text
    for seeded_url, text in FIXTURE_EVIDENCE_PAGES.items():
        if target.startswith(seeded_url.rstrip("/")):
            return text
    return None


#: Exposed for the demo script so a reader can see the radius used.
FIXTURE_EARTH_RADIUS_KM = EARTH_RADIUS_KM
