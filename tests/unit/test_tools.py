"""Unit tests for the ``tools`` package. Fully offline — no network, ever.

Every test that exercises a network code path installs a fake ``http_client``
whose methods raise :class:`AssertionError` if anything tries to open a real
connection, so a regression that reintroduced a live call fails loudly instead
of silently reaching the internet.

The tests are organised around the module's integrity rules rather than around
its classes, because those rules are what actually matters:

* no bare ``except Exception`` anywhere in ``tools/`` (source-level check);
* no md5/filename "vision" counter survives in any form;
* fixtures are always ``CACHED`` + ``degraded``;
* live calls are gated behind ``settings.tools_live``;
* a URL can never satisfy a promise;
* a PDF path is only reported when a real PDF exists;
* local email spooling never reports ``ToolStatus.OK``.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:      # works with or without a repo conftest.py
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from core.config import Settings  # noqa: E402
from core.protocols import TOOL_REGISTRY  # noqa: E402
from core.schemas import ToolStatus  # noqa: E402
from tools import base as tools_base  # noqa: E402
from tools import fixtures  # noqa: E402
from tools.browser import (  # noqa: E402
    FetchPageTool,
    VerifyEvidenceTool,
    extract_text,
)
from tools.email import SendEmailTool  # noqa: E402
from tools.maps import (  # noqa: E402
    CATEGORY_TAGS,
    DistanceTool,
    GeocodeTool,
    SearchBrandsTool,
    build_overpass_query,
)
from tools.pdf import PDF_MAGIC, RenderPdfTool  # noqa: E402
from tools.registry import build_registry, describe  # noqa: E402
from tools.vision import AnalyseEvidenceTool  # noqa: E402

TOOLS_DIR = Path(tools_base.__file__).resolve().parent


# ============================================================================ fx
@pytest.fixture()
def cfg(tmp_path: Path) -> Settings:
    """Fixture-mode settings writing into a throwaway artifacts directory."""
    return Settings(tools_live=False, artifacts_dir=tmp_path / "artifacts",
                    tool_timeout_s=2.0)


@pytest.fixture()
def live_cfg(tmp_path: Path) -> Settings:
    """Live-mode settings. Tests using it must install a fake transport."""
    return Settings(tools_live=True, artifacts_dir=tmp_path / "artifacts",
                    tool_timeout_s=2.0)


class FakeResponse:
    """Minimal ``httpx.Response`` stand-in."""

    def __init__(self, status_code: int = 200, body: Any = b"",
                 headers: dict | None = None,
                 url: str = "https://fake.invalid/x") -> None:
        self.status_code = status_code
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = headers or {"content-type": "application/json"}
        self.url = url


class FakeClient:
    """Context-manager client whose ``get``/``post`` return queued responses.

    :param response: returned by every call, or an exception instance to raise.
    :param raise_on_open: when set, merely *entering* the context manager
        raises it — used to prove a tool never reached the transport at all.
    """

    def __init__(self, response: Any = None, *, calls: list | None = None,
                 raise_on_open: BaseException | None = None) -> None:
        self._response = response if response is not None else FakeResponse(200, b"{}")
        self._calls = calls if calls is not None else []
        self._raise_on_open = raise_on_open

    def __enter__(self) -> FakeClient:
        if self._raise_on_open is not None:
            raise self._raise_on_open
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self._calls.append(("GET", url, kwargs))
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self._calls.append(("POST", url, kwargs))
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


def install_fake(monkeypatch: pytest.MonkeyPatch, module: Any,
                 response: Any = None, *, calls: list | None = None,
                 client_calls: list | None = None,
                 raise_on_open: BaseException | None = None) -> FakeClient:
    """Point ``module.http_client`` at a :class:`FakeClient`.

    :param calls: records ``(method, url, kwargs)`` for every ``get``/``post``.
    :param client_calls: records ``(method, url, kwargs)`` for the client
        *factory* call, so a test can assert on headers (timeouts, User-Agent)
        that live on the client rather than the request.
    """
    fake = FakeClient(response, calls=calls, raise_on_open=raise_on_open)

    def factory(**kwargs: Any) -> FakeClient:
        if client_calls is not None:
            client_calls.append(("http_client", "", kwargs))
        return fake

    monkeypatch.setattr(module, "http_client", factory, raising=True)
    return fake


def explode(**kwargs: Any) -> Any:
    """A ``http_client`` replacement that fails the test if it is ever called."""
    raise AssertionError("a live HTTP call was attempted in an offline test")


# ============================================================ static integrity
def _python_sources(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.py"))


def _code_only(path: Path) -> str:
    """The file's *code*, with comments and string literals removed.

    The static checks below must not trip over this package's own docstrings,
    several of which quote the prototype's bad code verbatim in order to explain
    why it is wrong. Tokenising is the only honest way to scan code only.
    """
    import tokenize

    out: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(token.string)
    return " ".join(out)


def test_no_bare_except_exception_in_tools_package() -> None:
    """No bare ``except Exception`` / ``except:`` in the tools package.

    A bare handler is how a programming bug gets reported as an environmental
    failure, which is exactly the invisibility this module is meant to remove.
    The two documented, deliberate exceptions (if any) would need an explicit
    allowlist entry here; there are none.
    """
    offenders: list[str] = []
    for path in _python_sources(TOOLS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            kind = node.type
            bare = kind is None
            broad = isinstance(kind, ast.Name) and kind.id in {
                "Exception", "BaseException"}
            if bare or broad:
                offenders.append(f"{path.name}:{node.lineno} "
                                 f"({ast.unparse(node.type) if kind else 'bare'})")
    assert offenders == [], f"bare/broad except handlers found: {offenders}"


def test_no_hashing_fallback_for_vision() -> None:
    """The md5('filename') % 5 logo counter must not exist in any form."""
    banned = re.compile(r"\bmd5\b|\bsha1\b|hexdigest\(\)\s*%|%\s*\w+\s*\+\s*1\b"
                        r"|logo_count|footfall_estimate")
    for path in _python_sources(TOOLS_DIR):
        hits = [m.group(0) for m in banned.finditer(_code_only(path))]
        assert hits == [], f"{path.name} contains hash-derived output: {hits}"


def test_no_synthetic_email_from_brand_name() -> None:
    """No code path may derive an address from a brand name.

    The prototype built ``partnerships@{brand}.com`` and then reported a
    successful send to a domain that does not exist. The signature of that bug
    returning is an f-string that interpolates a brand-ish name into a template
    containing ``@``. Checked on the AST rather than the raw text so that this
    package's own explanatory docstrings (which quote the old code) do not
    trigger it.
    """
    offenders: list[str] = []
    for path in _python_sources(TOOLS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            # An address is *constructed* only when the f-string as a whole
            # mentions "@" and an interpolation is glued to an "@" before it or
            # to a bare domain suffix after it. Filenames like
            # ``f"{stem}.pdf"`` and diagnostics that merely say "@tld" in prose
            # do not count.
            literal_text = "".join(v.value for v in node.values
                                   if isinstance(v, ast.Constant)
                                   and isinstance(v.value, str))
            if "@" not in literal_text:
                continue
            for index, value in enumerate(node.values):
                if not isinstance(value, ast.FormattedValue):
                    continue
                before = "".join(v.value for v in node.values[:index]
                                 if isinstance(v, ast.Constant)
                                 and isinstance(v.value, str))
                after = ""
                for nxt in node.values[index + 1:]:
                    if isinstance(nxt, ast.Constant) and isinstance(nxt.value, str):
                        after = nxt.value
                        break
                glued_to_at = before.rstrip().endswith("@")
                domain_suffix = bool(re.fullmatch(
                    r"\.[A-Za-z0-9\-]{2,63}(\.[A-Za-z]{2,63})*", after))
                # A construction ending in the RFC 6761 reserved ``.invalid`` TLD is
                # deliberately exempt. Such an address can never resolve, so it
                # cannot become a dialable artefact or be mistaken for a real
                # recipient -- which is precisely what the original bug was. The
                # fixture layer needs one so the accept-decision has a contact
                # route to weigh; see "Contact routes" in tools/fixtures.py. The
                # exemption is narrow: a ``.com``/``.in``/``.org`` suffix glued to
                # an interpolated brand name still fails this check.
                reserved_invalid = after.strip() == ".invalid"
                if (glued_to_at or domain_suffix) and not reserved_invalid:
                    rendered = ast.unparse(node)
                    offenders.append(f"{path.name}:{node.lineno}: {rendered[:120]}")
    assert offenders == [], f"synthesised address construction: {offenders}"


def test_no_tool_module_falls_back_to_a_hardcoded_business_list() -> None:
    """``maps.py`` must not contain a literal list of business names.

    A quick check that the old hardcoded ``_CACHE`` of twelve invented brands did
    not get carried over: any list literal in ``maps.py`` with several long
    strings is suspicious.
    """
    tree = ast.parse((TOOLS_DIR / "maps.py").read_text(encoding="utf-8"))
    dunder_all_ids = {
        id(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "__all__"
        for node in [node]
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and id(node) not in dunder_all_ids:
            long_strings = [el for el in node.elts
                            if isinstance(el, ast.Constant)
                            and isinstance(el.value, str) and len(el.value) > 12]
            assert len(long_strings) <= 2, (
                f"maps.py line {node.lineno} has a list of "
                f"{len(long_strings)} long strings — that looks like a "
                f"hardcoded business list")


# ================================================================== base layer
def test_redact_removes_known_and_shaped_secrets() -> None:
    secret = "re_ABCDEF1234567890XYZ"
    text = f"calling https://api.resend.com with Authorization: Bearer {secret} " \
           f"and api_key=abcdefghijklmnop"
    out = tools_base.redact(text, [secret])
    assert secret not in out
    assert "abcdefghijklmnop" not in out
    assert tools_base.REDACTED in out


def test_redact_ignores_trivially_short_secrets() -> None:
    """A 3-char "secret" would corrupt unrelated text without protecting anything."""
    assert tools_base.redact("a running test", ["run"]) == "a running test"


def test_fixture_result_is_cached_and_degraded_and_names_the_fixture() -> None:
    result = tools_base.fixture_result({"x": 1}, "brand list")
    assert result.status is ToolStatus.CACHED
    assert result.degraded is True
    assert result.ok is True
    assert "FIXTURE brand list" in result.reason
    assert "synthetic" in result.reason


def test_guarded_converts_transport_error_to_unavailable() -> None:
    def boom() -> str:
        raise httpx.ConnectError("name resolution failed")

    result = tools_base.guarded(boom, timeout=1.0, tool_name="t")()
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert "ConnectError" in result.reason
    assert result.data is None


def test_guarded_converts_payload_error_to_failed() -> None:
    def boom() -> str:
        raise json.JSONDecodeError("bad", "{", 0)

    result = tools_base.guarded(boom, timeout=1.0, tool_name="t")()
    assert result.status is ToolStatus.FAILED
    assert "JSONDecodeError" in result.reason


@pytest.mark.slow
def test_guarded_enforces_a_timeout() -> None:
    import time as _time

    def slow() -> str:
        _time.sleep(2.0)
        return "never seen"

    result = tools_base.guarded(slow, timeout=0.2, tool_name="t")()
    assert result.ok is False
    assert result.status in {ToolStatus.FAILED, ToolStatus.UNAVAILABLE}
    assert "TimeoutError" in result.reason or "deadline" in result.reason


def test_guarded_redacts_secrets_in_failure_reasons() -> None:
    def boom() -> str:
        raise RuntimeError("auth failed for key re_SUPERSECRET123456")

    result = tools_base.guarded(boom, timeout=1.0, tool_name="t",
                                secrets=["re_SUPERSECRET123456"])()
    assert "re_SUPERSECRET123456" not in result.reason
    assert tools_base.REDACTED in result.reason


def test_valid_email_rejects_reserved_domains_and_tlds() -> None:
    assert tools_base.valid_email("sponsor@acme-brand.co")
    for bad in ("partner\nships@brand.com", "no-at-sign", "a@b",
                "a@example.com", "a@acme.invalid", "a@acme.test", "a@localhost"):
        assert tools_base.valid_email(bad) is False, bad


def test_valid_phone_rejects_osm_anti_patterns() -> None:
    assert tools_base.valid_phone("+91 98200 11223")
    assert not tools_base.valid_phone("no")
    assert not tools_base.valid_phone("none")
    assert not tools_base.valid_phone("")
    assert not tools_base.valid_phone("12")


# ================================================================== haversine
def test_haversine_zero_for_identical_points() -> None:
    assert tools_base.haversine_km(18.5308, 73.8475, 18.5308, 73.8475) == 0.0


def test_haversine_pune_to_mumbai_is_about_120_km() -> None:
    # Pune (18.53, 73.85) -> Mumbai (19.08, 72.88): ~118 km great-circle.
    km = tools_base.haversine_km(18.5308, 73.8475, 19.0760, 72.8777)
    assert 112.0 < km < 124.0


def test_haversine_one_degree_latitude_is_about_111_km() -> None:
    km = tools_base.haversine_km(0.0, 0.0, 1.0, 0.0)
    assert abs(km - 111.19) < 0.5


def test_haversine_is_symmetric() -> None:
    a = tools_base.haversine_km(18.5308, 73.8475, 18.5074, 73.8077)
    b = tools_base.haversine_km(18.5074, 73.8077, 18.5308, 73.8475)
    assert a == b


def test_distance_tool_run_matches_haversine_and_needs_no_network(
        cfg: Settings) -> None:
    tool = DistanceTool(settings=cfg)
    assert tool.available() == (True, tool.available()[1])
    result = tool.run(lat1=18.5308, lon1=73.8475, lat2=18.5074, lon2=73.8077)
    assert result.ok is True
    assert result.status is ToolStatus.OK
    assert result.degraded is False
    assert result.data["distance_km"] == tools_base.haversine_km(
        18.5308, 73.8475, 18.5074, 73.8077)
    assert 3.0 < result.data["distance_km"] < 6.0


def test_distance_tool_rejects_bad_coordinates_instead_of_returning_zero(
        cfg: Settings) -> None:
    tool = DistanceTool(settings=cfg)
    for kwargs in ({"lat1": None, "lon1": 1, "lat2": 1, "lon2": 1},
                   {"lat1": 999, "lon1": 1, "lat2": 1, "lon2": 1},
                   {"lat1": 1, "lon1": 1, "lat2": "abc", "lon2": 1}):
        result = tool.run(**kwargs)
        assert result.status is ToolStatus.FAILED
        assert result.data is None


# ======================================================================== maps
def test_search_brands_fixture_mode_is_cached_and_degraded(cfg: Settings) -> None:
    result = SearchBrandsTool(settings=cfg).run(categories=["cafe"])
    assert result.ok is True
    assert result.status is ToolStatus.CACHED
    assert result.degraded is True
    assert "FIXTURE" in result.reason
    leads = result.data["leads"]
    assert leads, "fixture mode should still return labelled seed data"
    assert all(lead.source == "fixture" for lead in leads)
    assert all(lead.name.startswith("FIXTURE_") for lead in leads)
    # A contact route is present so the offline pipeline can reach the
    # negotiation stage, but it is a reserved .invalid address: undeliverable by
    # specification. No phone is ever seeded.
    assert all(lead.contact_email.endswith(".invalid") for lead in leads)
    assert all(lead.phone is None for lead in leads)
    assert all(lead.rating is None for lead in leads)


def test_search_brands_live_mode_never_touches_the_network_when_gated(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    live_cfg.tools_live = False
    monkeypatch.setattr(tools_base, "http_client", explode, raising=True)
    result = SearchBrandsTool(settings=live_cfg).run(lat=18.5, lon=73.8)
    assert result.status is ToolStatus.CACHED


def test_search_brands_live_returns_unavailable_when_overpass_is_down(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    install_fake(monkeypatch, __import__("tools.maps", fromlist=["maps"]),
                 httpx.ConnectError("overpass-api.de: name resolution failed"))
    result = SearchBrandsTool(settings=live_cfg).run(lat=18.5, lon=73.8,
                                                     categories=["cafe"])
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.degraded is True
    assert "overpass" in result.reason.lower() or "ConnectError" in result.reason
    assert result.data is None, "a failed search must not hand back any leads"


def test_search_brands_live_refuses_non_json_body(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.maps as maps_mod

    install_fake(monkeypatch, maps_mod,
                 FakeResponse(200, b"<html>rate limited, please try later</html>"))
    result = SearchBrandsTool(settings=live_cfg).run(lat=18.5, lon=73.8,
                                                     categories=["cafe"])
    assert result.status is ToolStatus.FAILED
    assert "not JSON" in result.reason


def test_search_brands_live_reports_rate_limit_as_unavailable(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.maps as maps_mod

    install_fake(monkeypatch, maps_mod, FakeResponse(429, b"too many requests"))
    result = SearchBrandsTool(settings=live_cfg).run(lat=18.5, lon=73.8,
                                                     categories=["cafe"])
    assert result.status is ToolStatus.UNAVAILABLE
    assert "429" in result.reason


def test_search_brands_live_parses_overpass_elements(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.maps as maps_mod

    payload = {"elements": [
        {"type": "node", "id": 11, "lat": 18.5350, "lon": 73.8450,
         "tags": {"name": "Real Cafe", "amenity": "cafe",
                  "contact:email": "hello@real-cafe.co", "phone": "+91 98200 11223"}},
        {"type": "node", "id": 12, "lat": 18.5360, "lon": 73.8460,
         "tags": {"amenity": "cafe"}},                       # no name -> skipped
        {"type": "way", "id": 13, "center": {"lat": 18.5370, "lon": 73.8470},
         "tags": {"name": "Real Salon", "shop": "hairdresser", "phone": "no"}},
        {"type": "node", "id": 14, "tags": {"name": "No Coords Cafe",
                                            "amenity": "cafe"}},
    ]}
    calls: list = []
    client_calls: list = []
    install_fake(monkeypatch, maps_mod,
                 FakeResponse(200, json.dumps(payload).encode()),
                 calls=calls, client_calls=client_calls)

    result = SearchBrandsTool(settings=live_cfg).run(
        lat=18.5308, lon=73.8475, categories=["cafe", "salon"])
    assert result.ok is True, result.reason
    assert result.status is ToolStatus.OK
    assert result.degraded is False
    leads = result.data["leads"]
    assert [lead.name for lead in leads] == ["Real Cafe", "Real Salon"]
    assert leads[0].contact_email == "hello@real-cafe.co"
    assert leads[0].phone == "+91 98200 11223"
    assert leads[0].source == "overpass"
    assert leads[0].source_url == "https://www.openstreetmap.org/node/11"
    # OSM's phone=no must not become a contact number.
    assert leads[1].phone is None
    assert leads[1].rating is None
    assert result.data["skipped_without_name"] == 1
    assert result.data["skipped_without_coordinates"] == 1
    # A proper User-Agent went out: Overpass 429s default library agents.
    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == live_cfg.overpass_url
    assert client_calls[0][2]["headers"]["User-Agent"] == live_cfg.map_user_agent
    assert "data" in kwargs and "nwr(around" in kwargs["data"]["data"]


def test_search_brands_live_refuses_to_guess_a_missing_origin(
        live_cfg: Settings) -> None:
    result = SearchBrandsTool(settings=live_cfg).run(categories=["cafe"])
    assert result.status is ToolStatus.UNAVAILABLE
    assert "lat/lon" in result.reason


def test_overpass_query_escapes_caller_supplied_categories() -> None:
    """A category is user input; it must not be able to inject query syntax."""
    from tools.maps import _overpass_literal

    hostile = 'foo"]["name'
    literal = _overpass_literal(hostile)
    # Every regex metacharacter and every QL string delimiter is escaped.
    assert "[" not in literal.replace("\\[", "")
    assert "]" not in literal.replace("\\]", "")
    assert '"' not in literal.replace('\\"', "")

    query = build_overpass_query(18.5, 73.8, 2.0, [hostile, "cafe"])
    assert literal in query
    assert '["amenity"="cafe"]' in query
    # The name-search clause is still a well-formed filter.
    assert '["name"~"^(' in query
    assert query.count("nwr(") == 3
    # Quotes in the whole query are balanced, i.e. nothing escaped its literal.
    assert query.count('"') % 2 == 0


def test_overpass_query_without_usable_category_still_returns_named_records() -> None:
    query = build_overpass_query(18.5, 73.8, 2.0, [])
    assert '["name"]' in query


def test_category_tag_map_has_no_invented_categories() -> None:
    """Categories must map to real OSM tag filters, not to plausible guesses."""
    assert "cafe" in CATEGORY_TAGS
    for category, filters in CATEGORY_TAGS.items():
        for frag in filters:
            assert frag.startswith('["') and frag.endswith("]"), (category, frag)
            assert "=" in frag, (category, frag)


def test_geocode_fixture_mode_resolves_seeded_place(
        cfg: Settings) -> None:
    result = GeocodeTool(settings=cfg).run("Shivajinagar, Pune")
    assert result.status is ToolStatus.CACHED
    assert result.degraded is True
    assert result.data["results"][0]["source"] == "fixture"


def test_geocode_fixture_mode_refuses_unknown_place(cfg: Settings) -> None:
    """No seeded location means no answer — not a guessed coordinate."""
    result = GeocodeTool(settings=cfg).run("Atlantis, Nevada")
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None
    assert "Atlantis" in result.reason


def test_geocode_live_unavailable_when_nominatim_down(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.maps as maps_mod

    calls: list = []
    install_fake(monkeypatch, maps_mod,
                 httpx.ConnectTimeout("timed out"), calls=calls)
    result = GeocodeTool(settings=live_cfg).run("Kothrud, Pune")
    assert result.status is ToolStatus.UNAVAILABLE
    assert "ConnectTimeout" in result.reason


def test_geocode_live_returns_failure_when_no_match(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.maps as maps_mod

    install_fake(monkeypatch, maps_mod, FakeResponse(200, b"[]"))
    result = GeocodeTool(settings=live_cfg).run("nowhere at all")
    assert result.status is ToolStatus.FAILED
    assert "no match" in result.reason.lower()


# ===================================================================== fixtures
def test_every_fixture_record_is_labelled() -> None:
    assert fixtures.FIXTURE_SOURCE == "fixture"
    for record in fixtures.FIXTURE_BRAND_RECORDS:
        assert record["source"] == "fixture"
        assert record["name"].startswith("FIXTURE_")
        # The contact route is a reserved .invalid address, never a dialable one.
        # See "Contact routes" in tools/fixtures.py for why it exists.
        assert record["contact_email"].endswith(".invalid")
        assert record["contact_email"].startswith("sponsorship@")
        assert record["phone"] is None
        assert record["rating"] is None
        assert record["source_url"] is None
    for record in fixtures.FIXTURE_LOCATIONS.values():
        assert record["source"] == "fixture"
    assert all("FIXTURE PAGE" in text
               for text in fixtures.FIXTURE_EVIDENCE_PAGES.values())


def test_fixture_module_states_that_the_data_is_synthetic() -> None:
    source = (TOOLS_DIR / "fixtures.py").read_text(encoding="utf-8")
    assert "SYNTHETIC" in source
    assert "invented" in source.lower()


def test_fixture_leads_are_schema_valid_and_unscored() -> None:
    leads = fixtures.fixture_leads()
    assert leads
    for lead in leads:
        assert lead.source == "fixture"
        assert lead.fit_score == 0.0
        assert lead.fit_breakdown == {}
        assert "fixture" in lead.fit_rationale.lower()


def test_fixture_brand_filter_is_exact() -> None:
    only_cafes = fixtures.fixture_brands(categories=["cafe"])
    assert {r["category"] for r in only_cafes} == {"cafe"}
    assert fixtures.fixture_brands(limit=2) == [
        dict(fixtures.FIXTURE_BRAND_RECORDS[0]),
        dict(fixtures.FIXTURE_BRAND_RECORDS[1]),
    ]


# ====================================================================== browser
def test_extract_text_drops_scripts_styles_and_comments() -> None:
    html = ("<!-- <p>hidden</p> --><style>body{color:red}</style>"
            "<script>alert('x')</script><p>Visible &amp; readable</p>")
    text = extract_text(html)
    assert "Visible & readable" in text
    for forbidden in ("alert", "color:red", "hidden"):
        assert forbidden not in text


def test_fetch_page_fixture_mode_is_cached_and_degraded(cfg: Settings) -> None:
    result = FetchPageTool(settings=cfg).run(
        "https://fixture.invalid/shivaji-cafe/sponsorship")
    assert result.status is ToolStatus.CACHED
    assert result.degraded is True
    assert "FIXTURE" in result.reason
    assert "banner placement" in result.data["text"]


def test_fetch_page_fixture_mode_refuses_unseeded_url(cfg: Settings) -> None:
    result = FetchPageTool(settings=cfg).run("https://example.org/anything")
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None


def test_fetch_page_refuses_non_http_schemes(cfg: Settings) -> None:
    for bad in ("file:///etc/passwd", "ftp://example.org/x", "/etc/passwd",
                "javascript:alert(1)"):
        result = FetchPageTool(settings=cfg).run(bad)
        assert result.status is ToolStatus.FAILED, bad
        assert result.data is None


def test_fetch_page_live_strips_scripts_and_reports_final_url(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.browser as browser_mod

    calls: list = []
    client_calls: list = []
    install_fake(monkeypatch, browser_mod,
                 FakeResponse(200, b"<html><script>track()</script>"
                                    b"<p>Signed MoU for the fest</p></html>",
                              url="https://brand.invalid/final"),
                 calls=calls, client_calls=client_calls)
    result = FetchPageTool(settings=live_cfg).run("http://brand.invalid/start")
    assert result.ok is True
    assert result.data["final_url"] == "https://brand.invalid/final"
    assert "Signed MoU for the fest" in result.data["text"]
    assert "track()" not in result.data["text"]
    assert result.data["screenshot_path"] is None
    # Every request carries an identifying User-Agent and a finite timeout.
    assert client_calls[0][2]["headers"]["User-Agent"] == live_cfg.map_user_agent
    assert 0 < client_calls[0][2]["timeout"] <= 60


def test_fetch_page_live_reports_http_error_without_text(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.browser as browser_mod

    install_fake(monkeypatch, browser_mod, FakeResponse(404, b"not found"))
    result = FetchPageTool(settings=live_cfg).run("https://brand.invalid/gone")
    assert result.status is ToolStatus.FAILED
    assert result.data is None
    assert "404" in result.reason


def test_fetch_page_live_never_claims_an_uncaptured_screenshot(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    """Playwright is not installed here, so the screenshot must be a non-event."""
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_playwright_available", lambda: False)
    install_fake(monkeypatch, browser_mod, FakeResponse(200, b"<p>hi</p>"))
    result = FetchPageTool(settings=live_cfg).run("https://brand.invalid/x",
                                                  want_screenshot=True)
    assert result.data["screenshot_path"] is None
    assert "playwright" in result.data["screenshot_note"]
    assert result.degraded is True


# ============================================================ evidence verification
SPONSOR_URL = ("https://sponsor-logo-partnership.invalid/marketing/"
               "main-stage-banner-leaflet-insert-placement-fulfilment")
PROMISE = "main-stage banner placement and printed leaflet insert"


def test_url_substring_alone_never_fulfils_a_promise(cfg: Settings) -> None:
    """The prototype's bug, asserted against.

    Its check was ``found = any(word in url for word in promise_words)`` plus
    ``"sponsor" in url``. Here the URL contains *every* word of the promise and
    the literal string ``sponsor``, and the verdict is still "cannot verify".
    """
    tool = VerifyEvidenceTool(settings=cfg)
    assert tool.available()[0] is True

    result = tool.run(promise=PROMISE, evidence_url=SPONSOR_URL)

    assert result.ok is False, "no evidence was inspected, so nothing can be claimed"
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None, "no verdict may be produced from a URL alone"
    assert "cannot-verify" in result.reason
    # The URL really does contain every word of the promise, plus "sponsor".
    for word in ("banner", "leaflet", "insert", "placement", "sponsor"):
        assert word in SPONSOR_URL
    # ...and none of that was consulted. The reason never quotes the URL, and it
    # states plainly that the URL was not searched.
    assert SPONSOR_URL not in result.reason
    assert "not searched" in result.reason
    assert "fetch_page" in result.reason, "the reason must say what to do instead"


def test_url_substring_alone_never_satisfies_a_promise_even_with_an_image_ref(
        cfg: Settings) -> None:
    """Adding a path that also contains the words changes nothing."""
    tool = VerifyEvidenceTool(settings=cfg)
    result = tool.run(promise=PROMISE, evidence_url=SPONSOR_URL,
                      image_path="artifacts/evidence/main-stage-banner-leaflet-insert.jpg")
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None


def test_promise_text_alone_in_a_filename_or_url_is_not_evidence(
        cfg: Settings) -> None:
    """A file *named* after the promise proves nothing either."""
    tool = VerifyEvidenceTool(settings=cfg)
    result = tool.run(
        promise=PROMISE,
        evidence_url="https://cdn.invalid/",
        image_path="artifacts/evidence/main-stage-banner-leaflet-insert.png",
    )
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert "unanalysed image is not evidence" in result.reason


def test_unanalysed_image_without_text_is_cannot_verify(cfg: Settings) -> None:
    result = VerifyEvidenceTool(settings=cfg).run(
        promise="brand logo displayed on the backdrop",
        image_path="artifacts/evidence/backdrop.png")
    assert result.status is ToolStatus.UNAVAILABLE
    assert "cannot-verify" in result.reason


def test_fetched_text_matching_the_promise_is_fulfilled(cfg: Settings) -> None:
    result = VerifyEvidenceTool(settings=cfg).run(
        promise=PROMISE,
        page_text=fixtures.FIXTURE_EVIDENCE_PAGES[
            "https://fixture.invalid/shivaji-cafe/sponsorship"],
        evidence_url="https://fixture.invalid/shivaji-cafe/sponsorship")
    assert result.ok is True
    assert result.data["verdict"] == "fulfilled"
    assert result.data["fulfilled"] is True
    assert result.data["evidence_used"] == ["page_text"]
    assert result.data["coverage"] >= 0.6
    # Generic marketing vocabulary ("banner", "placement", "logo") is
    # deliberately excluded from the term list: a template page carries those
    # words whether or not anything was delivered.
    for filler in ("banner", "placement", "logo", "main", "stage"):
        assert filler not in result.data["matched_terms"]
    assert "leaflet" in result.data["matched_terms"]
    assert "insert" in result.data["matched_terms"]


def test_fetched_text_mentioning_the_sponsor_but_not_the_promise_is_not_fulfilled(
        cfg: Settings) -> None:
    """A page that says "sponsor" a lot is the old false positive, inverted."""
    body = ("Sponsorship acknowledgement. Thank you for your interest in our "
            "sponsorship programme. Our marketing team is reviewing budgets for "
            "the season and will revert next quarter. No commitment has been "
            "made on any deliverables at this stage. ") * 2
    result = VerifyEvidenceTool(settings=cfg).run(promise=PROMISE, page_text=body)
    assert result.ok is True
    assert result.data["verdict"] == "not-fulfilled"
    assert result.data["fulfilled"] is False
    assert "not supported" in result.data["reason"]
    assert "sponsor" not in result.data["matched_terms"]


def test_partial_text_match_is_not_fulfilled(cfg: Settings) -> None:
    body = ("We have agreed the stall staffing for both days and will staff from "
            "09:00 to 18:00. Everything else in the proposal remains under "
            "discussion and has not been finalised by either party. ") * 2
    result = VerifyEvidenceTool(settings=cfg).run(promise=PROMISE, page_text=body)
    assert result.data["verdict"] == "not-fulfilled"
    assert result.data["coverage"] < 0.6


def test_vision_findings_are_capped_and_labelled_model_generated(
        cfg: Settings) -> None:
    """Vision corroborates a text-based verdict but is capped and labelled."""
    vision = {
        "verification": "model-generated",
        "note": "Model-generated description; not independently verified.",
        "observations": [{"label": "logo", "text": "Shivaji Cafe leaflet insert",
                          "present": True, "confidence": 0.9}],
    }
    result = VerifyEvidenceTool(settings=cfg).run(
        promise=PROMISE,
        page_text=fixtures.FIXTURE_EVIDENCE_PAGES[
            "https://fixture.invalid/shivaji-cafe/sponsorship"],
        vision=vision,
        image_path="artifacts/evidence/x.png")
    assert result.ok is True
    assert result.data["verdict"] == "fulfilled"
    assert "vision:model-generated" in result.data["evidence_used"]
    assert "page_text" in result.data["evidence_used"]
    assert result.data["confidence"] <= 0.65, \
        "a model description must never outrank inspected page text"
    assert result.data["note"] == vision["note"]
    assert result.data["provenance"]["image_inspected"] is True


def test_vision_alone_never_produces_a_fulfilled_verdict(cfg: Settings) -> None:
    """Even a fully matching model description is not verification on its own."""
    vision = {"observations": [{"label": "banner",
                                "text": "printed leaflet insert main-stage",
                                "present": True, "confidence": 0.99}]}
    result = VerifyEvidenceTool(settings=cfg).run(
        promise=PROMISE, vision=vision, image_path="x.png")
    assert result.data["verdict"] == "cannot-verify"
    assert result.data["fulfilled"] is None
    assert result.degraded is True
    assert "model-generated" in result.data["reason"]


def test_vision_alone_cannot_upgrade_a_missing_promise_to_fulfilled(
        cfg: Settings) -> None:
    vision = {"observations": [{"label": "logo", "text": "leaflet insert",
                                "present": True, "confidence": 0.8}]}
    result = VerifyEvidenceTool(settings=cfg).run(
        promise=PROMISE, vision=vision, image_path="x.png")
    assert result.data["verdict"] == "cannot-verify"
    assert result.data["fulfilled"] is None
    assert result.degraded is True
    assert "cannot be verified" in result.data["reason"]


def test_verify_records_that_the_url_was_never_searched(cfg: Settings) -> None:
    result = VerifyEvidenceTool(settings=cfg).run(
        promise=PROMISE, page_text=fixtures.FIXTURE_EVIDENCE_PAGES[
            "https://fixture.invalid/shivaji-cafe/sponsorship"],
        evidence_url=SPONSOR_URL)
    assert result.data["checked_urls"] == []
    assert "never searched" in result.data["integrity_note"]


def test_verify_refuses_an_unmatchable_promise(cfg: Settings) -> None:
    result = VerifyEvidenceTool(settings=cfg).run(promise="the", page_text="x" * 400)
    assert result.status is ToolStatus.UNAVAILABLE
    assert "no content term" in result.reason


def test_verify_needs_substantive_evidence_not_a_stub(cfg: Settings) -> None:
    """A 3-character body must not be enough to call anything fulfilled."""
    result = VerifyEvidenceTool(settings=cfg).run(promise=PROMISE,
                                                  page_text="banner leaflet insert")
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert "200-char floor" in result.reason


def test_verify_does_not_match_substrings_of_longer_words(cfg: Settings) -> None:
    body = ("The organisers plan to stand in the entrance as ushers throughout "
            "the entire weekend, and to understand the crowd flow before "
            "announcing the next set of activities on stage. ") * 2
    result = VerifyEvidenceTool(settings=cfg).run(promise=PROMISE, page_text=body)
    assert result.data["verdict"] == "not-fulfilled"


# ======================================================================== email
def test_send_email_local_mode_does_not_report_ok(cfg: Settings) -> None:
    """The headline guarantee: spooled mail is never a successful send."""
    result = SendEmailTool(settings=cfg).run(
        to="sponsorship@acme-brand.co", subject="Sponsorship deck",
        body="Hello, here is the deck.", brand="Acme Brand", event_id="evt_1")
    assert result.status is not ToolStatus.OK
    assert result.ok is False
    assert result.degraded is True
    assert result.data["sent"] is False
    assert result.data["accepted_by_provider"] is False
    assert result.data["transport"] == "local-outbox"
    assert "NOT SENT" in result.reason
    assert Path(result.data["outbox_path"]).exists()


def test_send_email_local_mode_never_invents_an_address(cfg: Settings) -> None:
    """No recipient and a brand name must not become ``partnerships@brand``."""
    for kwargs in ({}, {"to": ""}, {"to": "@acme-brand.co"},
                   {"to": "acme-brand.co"}, {"to": "sponsorship@"}):
        result = SendEmailTool(settings=cfg).run(
            subject="Sponsorship deck", body="Hello.", brand="Acme Brand", **kwargs)
        assert result.status is ToolStatus.FAILED, kwargs
        assert result.data is None
        assert "inferred from a brand name" in result.reason or "refusing" in result.reason
    # Nothing was written and nothing was sent.
    assert not (Path(cfg.artifacts_dir) / "outbox").exists()


def test_send_email_refuses_reserved_and_derived_domains(cfg: Settings,
                                                          live_cfg: Settings) -> None:
    # Malformed shapes are refused everywhere; reserved TLDs are refused for
    # live sends (with a key, where a real send would be attempted) but spooled
    # as offline drafts (NOT sent).
    live_cfg.resend_api_key = "re_test_key_123456789"
    for bad in ("sponsor@example.com", "sponsor@acme.invalid", "sponsor@acme.test"):
        live = SendEmailTool(settings=live_cfg).run(
            to=bad, subject="Deck", body="Hello.")
        assert live.status is ToolStatus.FAILED, bad
        assert "routable" in live.reason
    for draft_addr in ("sponsor@acme.invalid", "sponsor@acme.test"):
        draft = SendEmailTool(settings=cfg).run(
            to=draft_addr, subject="Deck", body="Hello.")
        assert draft.status is ToolStatus.UNAVAILABLE, draft_addr
        assert draft.data["sent"] is False
        assert draft.data["recipient_routable"] is False


def test_send_email_refuses_one_bad_recipient_in_a_list(cfg: Settings,
                                                        live_cfg: Settings) -> None:
    live_cfg.resend_api_key = "re_test_key_123456789"
    live = SendEmailTool(settings=live_cfg).run(
        to="good@acme-brand.co, bad@acme.invalid", subject="Deck", body="Hello.")
    assert live.status is ToolStatus.FAILED
    assert live.data is None
    draft = SendEmailTool(settings=cfg).run(
        to="good@acme-brand.co, bad@acme.invalid", subject="Deck", body="Hello.")
    assert draft.status is ToolStatus.UNAVAILABLE
    assert draft.data["sent"] is False


def test_send_email_local_mode_is_idempotent(cfg: Settings) -> None:
    tool = SendEmailTool(settings=cfg)
    first = tool.run(to="sponsorship@acme-brand.co", subject="Deck", body="Hello.",
                     idempotency_key="evt_1:acme:day1")
    second = tool.run(to="sponsorship@acme-brand.co", subject="Deck", body="Hello.",
                      idempotency_key="evt_1:acme:day1")
    assert first.data["outbox_path"] == second.data["outbox_path"]
    assert second.status is ToolStatus.CACHED
    assert second.data["duplicate"] is True
    assert second.data["sent"] is False
    spooled = [p for p in Path(first.data["outbox_path"]).parent.glob("*.json")]
    assert len(spooled) == 1


def test_send_email_local_spool_file_records_that_it_was_not_sent(
        cfg: Settings) -> None:
    result = SendEmailTool(settings=cfg).run(
        to="sponsorship@acme-brand.co", subject="Deck", body="Hello.")
    record = json.loads(Path(result.data["outbox_path"]).read_text(encoding="utf-8"))
    assert record["sent"] is False
    assert record["transport"] == "local-outbox"
    assert record["body_sha256"]


def test_send_email_live_without_key_reports_unavailable(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.email as email_mod

    live_cfg.resend_api_key = None
    monkeypatch.setattr(email_mod, "http_client", explode, raising=True)
    result = SendEmailTool(settings=live_cfg).run(
        to="sponsorship@acme-brand.co", subject="Deck", body="Hello.")
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data["sent"] is False


def test_send_email_live_reports_resend_acceptance_as_accepted_not_delivered(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.email as email_mod

    live_cfg.resend_api_key = "re_test_key_123456789"
    live_cfg.email_from = "outreach@paytriq-demo.co"
    calls: list = []
    client_calls: list = []
    install_fake(monkeypatch, email_mod,
                 FakeResponse(200, {"id": "8f2c-4a91"}),
                 calls=calls, client_calls=client_calls)
    result = SendEmailTool(settings=live_cfg).run(
        to="sponsorship@acme-brand.co", subject="Deck", body="Hello.",
        idempotency_key="evt_1:acme:day1")
    assert result.status is ToolStatus.OK
    assert result.ok is True
    assert result.data["provider"] == "resend"
    assert "accepted" in result.reason.lower()
    assert "not delivery" in result.reason.lower()
    method, url, kwargs = calls[0]
    assert url == "https://api.resend.com/emails"
    headers = client_calls[0][2]["headers"]
    assert headers["Idempotency-Key"] == "evt_1:acme:day1"
    assert headers["Authorization"] == "Bearer re_test_key_123456789"
    assert "re_test_key_123456789" not in kwargs["content"].decode("utf-8")


def test_send_email_live_refuses_the_placeholder_sender(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    """The default EMAIL_FROM is a reserved domain, so a real send is refused."""
    import tools.email as email_mod

    live_cfg.resend_api_key = "re_test_key_123456789"
    assert live_cfg.email_from.endswith(".invalid")
    monkeypatch.setattr(email_mod, "http_client", explode, raising=True)
    result = SendEmailTool(settings=live_cfg).run(
        to="sponsorship@acme-brand.co", subject="Deck", body="Hello.")
    assert result.status is ToolStatus.FAILED
    assert result.data is None
    assert "would bounce" in result.reason


def test_send_email_live_failure_is_failed_not_sent(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings) -> None:
    import tools.email as email_mod

    live_cfg.resend_api_key = "re_test_key_123456789"
    live_cfg.email_from = "outreach@paytriq-demo.co"
    install_fake(monkeypatch, email_mod, FakeResponse(422, {"message": "bad from"}))
    result = SendEmailTool(settings=live_cfg).run(
        to="sponsorship@acme-brand.co", subject="Deck", body="Hello.")
    assert result.status is ToolStatus.FAILED
    assert result.data is None
    assert "422" in result.reason


def test_send_email_available_reflects_mode(cfg: Settings, live_cfg: Settings) -> None:
    assert SendEmailTool(settings=cfg).current_mode() == "local-outbox"
    assert SendEmailTool(settings=cfg).available()[0] is True
    ok, reason = SendEmailTool(settings=live_cfg).available()
    assert ok is False and "RESEND_API_KEY" in reason


# =========================================================================== pdf
def test_pdf_fallback_names_files_honestly(
        monkeypatch: pytest.MonkeyPatch, cfg: Settings) -> None:
    import tools.pdf as pdf_mod

    monkeypatch.setattr(pdf_mod, "_reportlab_available", lambda: False)
    result = RenderPdfTool(settings=cfg).run(
        kind="mou",
        data={"mou_id": "mou_abc123", "event_id": "evt_1", "brand": "FIXTURE_Shivaji Cafe",
              "amount_inr": 40000.0, "terms": "Two-day fest", "status": "draft"},
        out_name="mou_abc123")

    assert result.ok is True
    assert result.status is ToolStatus.OK
    assert result.degraded is True, "a missing PDF must be visible as degradation"
    assert result.data["pdf_path"] is None
    assert result.data["format"] == "txt"
    assert "reportlab is not installed" in result.reason

    documents = Path(cfg.artifacts_dir) / "documents"
    assert (documents / "mou_abc123.txt").exists()
    assert (documents / "mou_abc123.html").exists()
    assert not (documents / "mou_abc123.pdf").exists(), \
        "the fallback must never leave a text file named .pdf"
    assert list(documents.glob("*.pdf")) == []


def test_pdf_fallback_html_says_it_is_not_a_pdf(
        monkeypatch: pytest.MonkeyPatch, cfg: Settings) -> None:
    import tools.pdf as pdf_mod

    monkeypatch.setattr(pdf_mod, "_reportlab_available", lambda: False)
    result = RenderPdfTool(settings=cfg).run(kind="roi",
                                             data={"report_id": "rep_1",
                                                   "event_id": "evt_1",
                                                   "roi_multiple": 2.4,
                                                   "assumptions": ["footfall=5000"]})
    html = Path(result.data["html_path"]).read_text(encoding="utf-8")
    assert "NOT A PDF" in html


def test_pdf_rejects_unknown_kind_and_empty_data(cfg: Settings) -> None:
    tool = RenderPdfTool(settings=cfg)
    assert tool.run(kind="invoice", data={"a": 1}).status is ToolStatus.FAILED
    assert tool.run(kind="mou", data=None).status is ToolStatus.FAILED
    assert tool.run(kind="mou", data={}).status is ToolStatus.FAILED


def test_pdf_sanitises_the_output_filename(cfg: Settings) -> None:
    """A traversal attempt must not escape ``artifacts/documents``.

    Works whether or not reportlab is installed, so the sanitiser itself is
    always covered rather than only on the fallback path.
    """
    tool = RenderPdfTool(settings=cfg)
    result = tool.run(kind="mou", data={"mou_id": "m_1", "event_id": "e_1",
                                        "brand": "B", "amount_inr": 1.0,
                                        "terms": "t"},
                      out_name="../../etc/passwd")
    documents = Path(cfg.artifacts_dir) / "documents"
    written = Path(result.data.get("pdf_path") or result.data["text_path"])
    assert written.parent == documents
    assert ".." not in written.name
    assert written.exists()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("reportlab") is None,
    reason="reportlab is not installed; the PDF path cannot be exercised")
def test_pdf_writes_a_real_pdf_when_reportlab_is_available(
        cfg: Settings) -> None:
    result = RenderPdfTool(settings=cfg).run(
        kind="mou", data={"mou_id": "mou_abc", "event_id": "evt_1",
                          "brand": "FIXTURE_Shivaji Cafe", "amount_inr": 40000.0,
                          "terms": "Two-day fest", "status": "draft"},
        out_name="mou_real")
    assert result.data["pdf_path"] is not None
    assert result.degraded is False
    written = Path(result.data["pdf_path"])
    assert written.read_bytes().startswith(PDF_MAGIC)


def test_pdf_deletes_a_bogus_pdf_rather_than_shipping_it(
        monkeypatch: pytest.MonkeyPatch, cfg: Settings) -> None:
    """If the renderer produced non-PDF bytes, the .pdf must not survive."""
    import tools.pdf as pdf_mod

    monkeypatch.setattr(pdf_mod, "_reportlab_available", lambda: True)
    monkeypatch.setattr(pdf_mod, "_reportlab_errors", lambda: ())

    def fake_render(self: Any, out_path: Path, lines: Any, title: str) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("just text, not a pdf", encoding="utf-8")
        return out_path

    monkeypatch.setattr(pdf_mod.RenderPdfTool, "_render", fake_render, raising=True)
    result = RenderPdfTool(settings=cfg).run(
        kind="mou", data={"mou_id": "mou_bogus", "event_id": "e", "brand": "B",
                          "amount_inr": 1.0, "terms": "t"}, out_name="mou_bogus")
    assert result.data["pdf_path"] is None
    assert result.degraded is True
    documents = Path(cfg.artifacts_dir) / "documents"
    assert not (documents / "mou_bogus.pdf").exists()
    assert (documents / "mou_bogus.txt").exists()
    assert "%PDF" in result.reason


# ======================================================================= vision
def test_vision_without_image_reports_unavailable(cfg: Settings) -> None:
    result = AnalyseEvidenceTool(settings=cfg).run(None)
    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None
    assert "no image supplied" in result.reason


def test_vision_with_missing_file_reports_unavailable(cfg: Settings) -> None:
    result = AnalyseEvidenceTool(settings=cfg).run("no-such-image.png")
    assert result.status is ToolStatus.UNAVAILABLE
    assert "not found" in result.reason


def test_vision_with_no_key_never_invents_findings(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings, tmp_path: Path) -> None:
    live_cfg.gemini_api_key = None
    image = tmp_path / "logo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    result = AnalyseEvidenceTool(settings=live_cfg).run(str(image))

    assert result.ok is False
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None
    assert "GEMINI_API_KEY" in result.reason
    for banned in ("logo_count", "footfall_estimate", "counts"):
        assert banned not in json.dumps(result.data or {})


def test_vision_with_live_calls_off_never_invents_findings(
        monkeypatch: pytest.MonkeyPatch, live_cfg: Settings, tmp_path: Path) -> None:
    import tools.vision as vision_mod

    live_cfg.gemini_api_key = "AIzaSyTESTKEYTESTKEYTESTKEY"
    monkeypatch.setattr(vision_mod, "_genai_available", lambda: True)
    monkeypatch.setattr(vision_mod, "_genai_client", explode, raising=True)
    image = tmp_path / "logo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    live_cfg.tools_live = False
    result = AnalyseEvidenceTool(settings=live_cfg).run(str(image))
    assert result.status is ToolStatus.UNAVAILABLE
    assert result.data is None


def test_vision_rejects_a_non_image_file(cfg: Settings, tmp_path: Path) -> None:
    payload = tmp_path / "evidence.csv"
    payload.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    result = AnalyseEvidenceTool(settings=cfg).run(str(payload))
    assert result.status is ToolStatus.UNAVAILABLE
    assert "not an analysable image" in result.reason


def test_vision_rejects_an_empty_file(cfg: Settings, tmp_path: Path) -> None:
    empty = tmp_path / "blank.png"
    empty.write_bytes(b"")
    result = AnalyseEvidenceTool(settings=cfg).run(str(empty))
    assert result.status is ToolStatus.UNAVAILABLE
    assert "0 bytes" in result.reason


def test_vision_available_is_false_without_a_key(cfg: Settings, live_cfg: Settings) -> None:
    ok, reason = AnalyseEvidenceTool(settings=live_cfg).available()
    assert ok is False and "GEMINI_API_KEY" in reason
    live_cfg.gemini_api_key = "AIzaSyTESTKEYTESTKEY"
    live_cfg.tools_live = False
    ok, reason = AnalyseEvidenceTool(settings=live_cfg).available()
    assert ok is False and "TOOLS_LIVE" in reason


# ====================================================================== registry
def test_build_registry_populates_the_global_registry(cfg: Settings) -> None:
    registry = build_registry(cfg)
    assert set(registry) == {"search_brands", "geocode", "distance_km",
                             "fetch_page", "verify_evidence", "send_email",
                             "render_pdf", "analyse_evidence"}
    assert dict(TOOL_REGISTRY) == registry
    for name, instance in registry.items():
        assert instance.name == name
        assert instance.description
        assert instance.available()[0] in (True, False)


def test_build_registry_can_skip_the_global(cfg: Settings) -> None:
    before = dict(TOOL_REGISTRY)
    registry = build_registry(cfg, populate=False)
    assert registry
    assert dict(TOOL_REGISTRY) == before


def test_registry_instances_are_stateless(cfg: Settings) -> None:
    registry = build_registry(cfg)
    first = registry["distance_km"].run(lat1=0, lon1=0, lat2=1, lon2=0)
    second = registry["distance_km"].run(lat1=0, lon1=0, lat2=1, lon2=0)
    assert first.data["distance_km"] == second.data["distance_km"]


def test_describe_reports_mode_and_availability(cfg: Settings) -> None:
    rows = describe(build_registry(cfg), settings=cfg)
    by_name = {row["name"]: row for row in rows}
    assert by_name["search_brands"]["mode"] == "fixture"
    assert by_name["distance_km"]["mode"] == "pure"
    assert by_name["send_email"]["mode"] == "local-outbox"
    assert by_name["analyse_evidence"]["available"] is False
    assert by_name["search_brands"]["live_gate_open"] is False
    assert "tools_live" in by_name["search_brands"]["live_gate_reason"]

    live_settings = Settings(tools_live=True)
    live_rows = {row["name"]: row for row in
                 describe(build_registry(live_settings), settings=live_settings)}
    assert live_rows["search_brands"]["mode"] == "live"
    assert live_rows["search_brands"]["live_gate_open"] is True


def test_summary_renders_every_tool(cfg: Settings) -> None:
    build_registry(cfg)
    from tools.registry import summary

    text = summary()
    for name in ("search_brands", "verify_evidence", "send_email"):
        assert name in text
    assert "mode" in text and "available" in text


def test_tools_disabled_makes_every_tool_unavailable(cfg: Settings) -> None:
    disabled = Settings(tools_enabled=False)
    registry = build_registry(disabled)
    for name, instance in registry.items():
        if name == "distance_km":      # pure function, needs nothing
            continue
        ok, reason = instance.available()
        assert ok is False, name
        assert "disabled" in reason
