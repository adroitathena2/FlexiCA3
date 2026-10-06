"""Real map data: Overpass business discovery, Nominatim geocoding, haversine.

Every lead this module produces came out of an OpenStreetMap response. If the
API is unreachable or answers with something unusable, the tool says so and
returns no leads — it does not fall back to a remembered list of "Pune brands".

That distinction is the whole design. The previous prototype's discovery step
returned twelve invented businesses with sequential phone numbers and no marker
of any kind, so the artefacts downstream (fit scores, outreach emails, ROI
figures) were built on fiction and read as measurements. Here:

* live mode talks to Overpass and parses ``tags`` into :class:`BrandLead`;
* fixture mode serves :mod:`tools.fixtures`, which stamps every record
  ``source="fixture"`` and arrives in a ``ToolStatus.CACHED``, ``degraded=True``
  result;
* no mode invents a business, a distance, a rating, or an email address.

Protocol notes that are easy to get wrong
-----------------------------------------
* Overpass **blocks** requests whose ``User-Agent`` looks like a default
  library one (returning HTTP 429), so ``settings.map_user_agent`` is always
  sent and an identifying contact is part of it.
* Nominatim's usage policy caps you at one request per second and requires a
  ``User-Agent``; it answers ``403`` otherwise.
* Neither API returns ratings. ``BrandLead.rating`` therefore stays ``None`` on
  the live path. A rating is left unset rather than approximated.
* OSM contact details are whatever the contributor typed in, so they are
  shape-checked before use. ``phone=no`` is common and is rejected.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.ids import new_id
from core.protocols import ToolResult
from core.schemas import BrandLead

from . import fixtures
from .base import (
    BaseTool,
    ToolSignal,
    annotate,
    fixture_result,
    guarded,
    haversine_km,
    http_client,
    http_result,
    tool,
    valid_email,
    valid_phone,
)

__all__ = [
    "SearchBrandsTool", "GeocodeTool", "DistanceTool",
    "CATEGORY_TAGS", "build_overpass_query",
]

#: Human category -> OSM tag filters. Only categories we can express as a real
#: tag filter go here; anything else falls back to a name search (see
#: :func:`build_overpass_query`), which still returns real businesses.
CATEGORY_TAGS: dict[str, tuple[str, ...]] = {
    "cafe": ('["amenity"="cafe"]', '["amenity"="coffee_shop"]'),
    "restaurant": ('["amenity"="restaurant"]',),
    "bakery": ('["shop"="bakery"]',),
    "gym": ('["leisure"="fitness_centre"]', '["leisure"="sports_centre"]'),
    "sports": ('["leisure"="sports_centre"]', '["leisure"="fitness_centre"]'),
    "salon": ('["shop"="hairdresser"]', '["amenity"="beauty_salon"]'),
    "hairdresser": ('["shop"="hairdresser"]',),
    "printing": ('["shop"="copyshop"]', '["shop"="printing"]'),
    "coaching": ('["amenity"="driving_school"]', '["amenity"="school"]',
                 '["amenity"="college"]'),
    "education": ('["amenity"="college"]', '["amenity"="school"]',
                  '["amenity"="university"]'),
    "electronics": ('["shop"="electronics"]', '["shop"="mobile_phone"]'),
    "mobile": ('["shop"="mobile_phone"]',),
    "bookstore": ('["shop"="books"]', '["shop"="newsagent"]'),
    "stationery": ('["shop"="stationery"]', '["shop"="books"]'),
    "supermarket": ('["shop"="supermarket"]', '["shop"="convenience"]'),
    "bank": ('["amenity"="bank"]',),
    "hotel": ('["tourism"="hotel"]', '["tourism"="guest_house"]'),
    "mall": ('["shop"="mall"]',),
    "optician": ('["shop"="optician"]',),
    "pharmacy": ('["amenity"="pharmacy"]',),
    "jeweller": ('["shop"="jewellery"]',),
    "textiles": ('["shop"="clothes"]', '["shop"="fabric"]'),
    "fitness": ('["leisure"="fitness_centre"]',),
    "coworking": ('["amenity"="community_centre"]', '["office"="coworking"]'),
    "office": ('["office"="company"]',),
    "real_estate": ('["shop"="estate_agent"]',),
    "event_venue": ('["amenity"="community_centre"]', '["amenity"="events_venue"]',
                    '["amenity"="banquet_hall"]'),
}

#: Tag keys that describe what a place *is*, used when the requested category
#: was a loose term and the caller wants the OSM truth.
_KIND_KEYS = ("amenity", "shop", "leisure", "tourism", "office", "healthcare",
              "craft", "club")

#: Overpass caps `around:` at 50 km; wider queries are silently useless.
_MAX_RADIUS_KM = 50.0
_MAX_LIMIT = 100


def _overpass_literal(value: str) -> str:
    """Escape a caller-supplied string for use inside an Overpass QL regex.

    The category names come from ``EventProfile.categories_wanted``, i.e. from a
    user, and Overpass QL is injected verbatim into a query string. Escaping
    regex metacharacters *and* the QL string delimiter keeps a category called
    ``foo"]["bar`` from turning into a query the caller did not ask for.
    """
    return re.sub(r'(["\\])', r"\\\1", re.escape(value or ""))


def build_overpass_query(lat: float, lon: float, radius_km: float,
                         categories: Sequence[str],
                         *, timeout_s: int = 25,
                         element_limit: int = 200) -> str:
    """Build the Overpass QL query for "named businesses of these kinds nearby".

    Each filter is emitted as its own statement inside a union, so a business
    tagged as both ``shop=hairdresser`` and ``amenity=beauty_salon`` is returned
    once per matching statement; duplicates are collapsed downstream.

    Categories we have no tag filter for become a case-insensitive **name**
    search. That is a deliberate widening rather than a guess: the result is
    still whatever OSM actually records under that word.
    """
    clauses: list[str] = []
    unnamed: list[str] = []
    for raw in categories:
        category = (raw or "").strip().lower()
        if not category:
            continue
        filters = CATEGORY_TAGS.get(category)
        if filters:
            clauses.extend(
                f'  nwr(around:{radius_km:.0f},{lat:.6f},{lon:.6f}){f};' for f in filters
            )
        else:
            unnamed.append(_overpass_literal(category))
    if unnamed:
        pattern = "|".join(unnamed)
        clauses.append(
            f'  nwr(around:{radius_km:.0f},{lat:.6f},{lon:.6f})'
            f'["name"~"^({pattern})$",i];'
        )
    if not clauses:
        # No usable category: return every *named* node/way in range so the
        # caller gets real records rather than nothing at all.
        clauses.append(
            f'  nwr(around:{radius_km:.0f},{lat:.6f},{lon:.6f})["name"];'
        )
    body = "\n".join(clauses)
    return (f"[out:json][timeout:{timeout_s}];\n(\n{body}\n);\n"
            f"out center {element_limit};")


@tool("search_brands",
      "Find real businesses near a coordinate within a radius, filtered by the "
      "event's wanted categories. Overpass API in live mode; labelled synthetic "
      "seed data in fixture mode. Never invents a business.",
      mode="live", backend="overpass")
class SearchBrandsTool(BaseTool):
    """Discover candidate sponsors around an event venue.

    Contract for callers:

    * ``ok=False`` means **no businesses were found**. ``reason`` says whether
      Overpass was unreachable, rejected the query, or simply had no matches.
    * ``data["leads"]`` is the only list of businesses you may use, and every
      element carries ``source`` — ``"overpass"`` or ``"fixture"``.
    * ``data["origin"]`` records the coordinate and radius used, so a distance
      in a downstream artefact can be re-derived from the trace.
    """

    def current_mode(self) -> str:
        return "live" if self.live else "fixture"

    def available(self) -> tuple[bool, str]:
        if not self.settings.tools_enabled:
            return False, "tools are disabled (settings.tools_enabled=False)"
        if self.live:
            return True, f"live Overpass at {self.settings.overpass_url}"
        return True, ("fixture mode (TOOLS_LIVE=false): serving labelled "
                      "synthetic seed data, no network call")

    # ------------------------------------------------------------------- public
    def run(self, lat: float | None = None, lon: float | None = None,
            radius_km: float = 3.0,
            categories: Sequence[str] | None = None,
            limit: int = 25,
            location: str = "") -> ToolResult:
        """Search for businesses within ``radius_km`` of ``(lat, lon)``.

        :param lat: venue latitude. ``None`` means "the seeded origin", which is
            only honoured in fixture mode; live mode refuses to guess.
        :param lon: venue longitude.
        :param radius_km: 0.05 - 50. Larger values are clamped, not rejected,
            because a caller asking for 500 km wanted *something*.
        :param categories: the event's ``categories_wanted``.
        :param location: free-text label recorded for provenance only. It is
            **not** geocoded here; use :class:`GeocodeTool` for that.
        """
        wanted = [c for c in (categories or []) if str(c).strip()]
        capped_limit = max(1, min(int(limit or 1), _MAX_LIMIT))
        radius = max(0.05, min(float(radius_km or 0.0), _MAX_RADIUS_KM))

        if lat is None or lon is None:
            if self.live:
                return self._unavailable(
                    "search_brands needs an explicit lat/lon in live mode; it "
                    "will not guess a venue coordinate (call geocode first)")
            origin = fixtures.FIXTURE_ORIGIN
            lat, lon = origin[0], origin[1]
        lat, lon = float(lat), float(lon)

        if not self.live:
            leads = fixtures.fixture_leads(categories=wanted, limit=capped_limit)
            payload = {
                "leads": leads,
                "count": len(leads),
                "origin": {"lat": lat, "lon": lon, "radius_km": radius,
                           "location_label": location or "fixture origin"},
                "categories_wanted": wanted,
                "backend": "fixtures",
                "note": fixtures.FIXTURE_NOTE,
            }
            return fixture_result(payload, "brand list")

        call = guarded(self._live_query, timeout=self.timeout,
                       tool_name=self.name, secrets=self.secrets)
        result = call(lat, lon, radius, wanted, capped_limit)
        if not result.ok:
            return result
        return annotate(result, source=f"{self.name}:overpass",
                        evidence_url=self.settings.overpass_url)

    # -------------------------------------------------------------------- live
    def _live_query(self, lat: float, lon: float, radius_km: float,
                    categories: Sequence[str], limit: int) -> dict[str, Any]:
        """Blocking Overpass round-trip. Errors are raised for :func:`guarded`.

        ``timeout`` is derived from ``settings.tool_timeout_s`` so the HTTP
        layer, not just the outer deadline, is bounded: Overpass regularly
        holds a connection open while it queues a heavy query.
        """
        http_timeout = max(5.0, min(self.timeout, 60.0))
        query = build_overpass_query(lat, lon, radius_km, categories,
                                    timeout_s=int(http_timeout),
                                    element_limit=max(limit * 4, 50))
        headers = {
            # Overpass 429s default library agents; this is required, not polite.
            "User-Agent": self.settings.map_user_agent,
            "Accept": "application/json",
        }
        with http_client(timeout=http_timeout, headers=headers,
                         settings=self.settings) as client:
            response = client.post(self.settings.overpass_url,
                                   data={"data": query},
                                   headers={"Content-Type": "application/x-www-form-urlencoded"})
            checked = http_result((response.status_code, response.content),
                                  evidence_url=self.settings.overpass_url,
                                  settings=self.settings, tool_name=self.name)
        if not checked.ok:
            # Carry the honest verdict (429 -> UNAVAILABLE, 400 -> FAILED)
            # instead of flattening it into a generic RuntimeError.
            raise ToolSignal(checked)

        try:
            payload = json.loads(checked.data[1])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Overpass returned a {len(checked.data[1])}-byte body that is "
                f"not JSON ({type(exc).__name__}); refusing to parse it as data"
            ) from None

        if not isinstance(payload, dict):
            raise TypeError(f"Overpass returned {type(payload).__name__}, expected an object")
        elements = payload.get("elements")
        if elements is None:
            raise ValueError(f"Overpass response has no 'elements' key: "
                             f"keys={sorted(payload)[:8]}")
        if not isinstance(elements, list):
            raise TypeError(f"Overpass 'elements' is {type(elements).__name__}, expected a list")

        remark = str(payload.get("remark") or "")
        leads: list[BrandLead] = []
        seen: set[tuple[str, str]] = set()
        skipped_no_name = 0
        skipped_no_coords = 0
        for element in elements:
            if not isinstance(element, dict):
                continue
            lead = self._to_lead(element, lat, lon)
            if lead is None:
                tags = element.get("tags")
                if isinstance(tags, dict) and not (
                        tags.get("name") or tags.get("brand") or tags.get("operator")):
                    skipped_no_name += 1
                else:
                    skipped_no_coords += 1
                continue
            key = (lead.name, lead.category)
            if key in seen:
                continue
            seen.add(key)
            leads.append(lead)

        leads.sort(key=lambda lead: lead.distance_km)
        trimmed = leads[:limit]
        return {
            "leads": trimmed,
            "count": len(trimmed),
            "origin": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "categories_wanted": list(categories),
            "backend": "overpass",
            "query": query,
            "remark": remark,
            "elements_returned": len(elements),
            "skipped_without_name": skipped_no_name,
            "skipped_without_coordinates": skipped_no_coords,
            "dropped_as_duplicates": max(0, len(leads) - len(trimmed)),
        }

    # ------------------------------------------------------------------ parsing
    def _to_lead(self, element: Mapping[str, Any], origin_lat: float,
                 origin_lon: float) -> BrandLead | None:
        """Turn one Overpass element into a ``BrandLead``, or ``None`` to skip it.

        Three refusals happen here, and each one prevents a specific kind of
        fabrication:

        * no usable name -> skipped. The previous prototype invented names.
        * no coordinates -> skipped. Without them there is no honest distance,
          and a made-up distance is worse than a missing lead.
        * unparseable contact details -> dropped, not repaired. OSM's
          ``phone=no`` must not become a contact number.
        """
        tags = element.get("tags")
        if not isinstance(tags, dict) or not tags:
            return None
        name = str(tags.get("name") or tags.get("brand") or tags.get("operator") or "").strip()
        if not name:
            return None

        lat, lon = _element_coordinates(element)
        if lat is None or lon is None:
            return None
        distance = max(0.0, haversine_km(origin_lat, origin_lon, lat, lon))

        raw_email = tags.get("contact:email") or tags.get("email") or ""
        contact_email = str(raw_email).strip() if valid_email(str(raw_email).strip()) else None
        raw_phone = tags.get("phone") or tags.get("contact:phone") or ""
        phone = str(raw_phone).strip() if valid_phone(str(raw_phone).strip()) else None

        category = _kind_of(tags) or "business"
        osm_type = str(element.get("type") or "")
        osm_id = str(element.get("id") or "")
        # A clickable OSM permalink is real evidence for where this record came
        # from; a bare "node/12345" is not a reference anything can open.
        source_url = (f"https://www.openstreetmap.org/{osm_type}/{osm_id}"
                      if osm_type and osm_id else None)
        # OSM records no ratings; BrandLead.rating stays None on purpose.
        return BrandLead(
            lead_id=new_id("brd"),
            name=name[:200],
            category=category[:120],
            distance_km=distance,
            contact_email=contact_email,
            phone=phone,
            rating=None,
            fit_score=0.0,
            fit_breakdown={},
            fit_rationale=(
                "Unscored: fit is A1's judgement. Distance is haversine from the "
                "search origin; no rating exists in OpenStreetMap."
            ),
            source="overpass",
            source_url=source_url,
        )


def _element_coordinates(element: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Latitude/longitude for a node, way or relation, or ``(None, None)``.

    Ways and relations carry a ``center`` object only when the query used
    ``out center``; without it there is no coordinate and therefore no distance.
    """
    lat = element.get("lat")
    lon = element.get("lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return float(lat), float(lon)
    center = element.get("center")
    if isinstance(center, dict):
        c_lat, c_lon = center.get("lat"), center.get("lon")
        if isinstance(c_lat, (int, float)) and isinstance(c_lon, (int, float)):
            return float(c_lat), float(c_lon)
    return None, None


def _kind_of(tags: Mapping[str, Any]) -> str:
    """What OSM says the place is, taken from its tagging.

    OSM stores the kind in one of a handful of keys; the value is the useful
    word ("cafe", "hairdresser"). An explicit ``category`` tag wins when present.
    """
    explicit = str(tags.get("category") or "").strip()
    if explicit:
        return explicit
    for key in _KIND_KEYS:
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ======================================================================= geocode
@tool("geocode",
      "Resolve a free-text location to latitude/longitude via the Nominatim "
      "OpenStreetMap geocoder, honouring the required User-Agent. Fixture mode "
      "answers only from labelled seed data and refuses anything else.",
      mode="live", backend="nominatim")
class GeocodeTool(BaseTool):
    """Turn ``"Shivajinagar, Pune"`` into a coordinate, or admit it cannot."""

    def current_mode(self) -> str:
        return "live" if self.live else "fixture"

    def available(self) -> tuple[bool, str]:
        if not self.settings.tools_enabled:
            return False, "tools are disabled (settings.tools_enabled=False)"
        if self.live:
            return True, f"live Nominatim at {self.settings.nominatim_url}"
        return True, ("fixture mode (TOOLS_LIVE=false): only the handful of "
                      "seeded locations in tools/fixtures.py can be resolved")

    def run(self, query: str, limit: int = 1) -> ToolResult:
        """Geocode ``query``. Returns ``data["results"]`` (possibly empty).

        An unresolvable place name is a ``failed`` result with a reason, never a
        zero coordinate pair — a silent ``(0.0, 0.0)`` would place the event in
        the Gulf of Guinea and quietly poison every distance downstream.
        """
        text = (query or "").strip()
        if not text:
            return ToolResult.failed(
                reason=f"{self.name}: empty query; nothing to geocode",
                source=f"{self.name}:input")

        if not self.live:
            record = fixtures.fixture_location(text)
            if record is None:
                return ToolResult.unavailable(
                    reason=(f"{self.name}: fixture mode has no seeded location "
                            f"matching {text!r}. Live geocoding is off "
                            f"(TOOLS_LIVE=false), so no coordinate can be "
                            f"returned honestly."),
                    source=f"{self.name}:fixture")
            payload = {
                "results": [{**record, "source": fixtures.FIXTURE_SOURCE}],
                "count": 1,
                "query": text,
                "backend": "fixtures",
                "note": fixtures.FIXTURE_NOTE,
            }
            return fixture_result(payload, "geocode result")

        call = guarded(self._live_geocode, timeout=self.timeout,
                       tool_name=self.name, secrets=self.secrets)
        result = call(text, max(1, min(int(limit or 1), 10)))
        if not result.ok:
            return result
        return annotate(result, source=f"{self.name}:nominatim",
                        evidence_url=self.settings.nominatim_url)

    def _live_geocode(self, query: str, limit: int) -> dict[str, Any]:
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": str(limit),
            # No addressdetails: this tool exists to get a pin, not a dossier.
            "addressdetails": "0",
        }
        url = self.settings.nominatim_url.rstrip("/") + "/search"
        headers = {
            # Nominatim's usage policy requires a real User-Agent, else 403.
            "User-Agent": self.settings.map_user_agent,
            "Accept": "application/json",
        }
        with http_client(timeout=max(5.0, self.timeout), headers=headers,
                         settings=self.settings) as client:
            response = client.get(url, params=params)
            checked = http_result((response.status_code, response.content),
                                  evidence_url=url, settings=self.settings,
                                  tool_name=self.name)
        if not checked.ok:
            raise ToolSignal(checked)
        try:
            payload = json.loads(checked.data[1])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Nominatim returned a non-JSON body ({type(exc).__name__})") from None
        if not isinstance(payload, list):
            raise TypeError(f"Nominatim returned {type(payload).__name__}, expected a list")

        results: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            lat, lon = item.get("lat"), item.get("lon")
            if not (isinstance(lat, str) and isinstance(lon, str)):
                continue
            try:
                float(lat), float(lon)
            except ValueError:
                continue
            results.append({
                "lat": float(lat),
                "lon": float(lon),
                "display_name": str(item.get("display_name") or ""),
                "osm_type": str(item.get("osm_type") or ""),
                "osm_id": str(item.get("osm_id") or ""),
                "source": "nominatim",
            })
        if not results:
            raise LookupError(
                f"Nominatim has no match for {query!r}; no coordinate returned")
        return {"results": results, "count": len(results), "query": query,
                "backend": "nominatim"}


# ======================================================================= distance
@tool("distance_km",
      "Great-circle distance in kilometres between two coordinates. Pure and "
      "deterministic: no network, no fixture, no I/O.",
      mode="pure", backend="haversine")
class DistanceTool(BaseTool):
    """Haversine distance. Always available, because it always works.

    Kept separate from :class:`SearchBrandsTool` so distances in the demo are
    reproducible: the same two coordinates always give the same number, with no
    backend that could be down, rate-limited, or silently changing.

    It is the one tool that stays available even when ``tools_enabled`` is
    ``False``, because it performs no I/O and has no side effect — there is
    nothing for a caller to protect against by refusing it.
    """

    def current_mode(self) -> str:
        return "pure"

    def available(self) -> tuple[bool, str]:
        return True, "pure function: no backend, no network, no credentials"

    def run(self, lat1: float | None = None, lon1: float | None = None,
            lat2: float | None = None, lon2: float | None = None) -> ToolResult:
        """Return ``data={"distance_km": float, ...}``.

        Invalid or missing coordinates are a ``failed`` result. Returning
        ``0.0`` for an unknown pair would make "next door" and "unknown" the
        same observation.
        """
        values: list[float] = []
        for label, raw in (("lat1", lat1), ("lon1", lon1), ("lat2", lat2), ("lon2", lon2)):
            try:
                value = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return ToolResult.failed(
                    reason=f"{self.name}: {label} is not a number: {raw!r}",
                    source=f"{self.name}:input")
            if not -90.0 <= value <= 90.0 and label.startswith("lat"):
                return ToolResult.failed(
                    reason=f"{self.name}: {label}={value} is outside [-90, 90]",
                    source=f"{self.name}:input")
            if not -180.0 <= value <= 180.0 and label.startswith("lon"):
                return ToolResult.failed(
                    reason=f"{self.name}: {label}={value} is outside [-180, 180]",
                    source=f"{self.name}:input")
            values.append(value)
        a_lat, a_lon, b_lat, b_lon = values
        distance = haversine_km(a_lat, a_lon, b_lat, b_lon)
        return ToolResult.success(
            {"distance_km": distance,
             "from": {"lat": a_lat, "lon": a_lon},
             "to": {"lat": b_lat, "lon": b_lon},
             "method": "haversine", "earth_radius_km": 6371.0088},
            source=f"{self.name}:pure",
        )
