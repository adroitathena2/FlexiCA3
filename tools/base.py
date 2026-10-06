"""Shared plumbing for every Paytriq tool: timeouts, redaction, live/fixture gating.

Why this module exists
----------------------
The previous prototype's tool layer failed in a specific, diagnosable way: when a
real backend could not be reached, it returned *something shaped like an answer*.
A logo counter hashed a filename; a compliance check searched a promise's words
inside a URL; discovery returned a hardcoded list of invented businesses with
sequential phone numbers. Every one of those returned data that a reader could
not distinguish from a real observation. That is the failure this module is
built to make structurally impossible:

1. :func:`guarded` converts any expected failure into an honest
   :class:`~core.protocols.ToolResult` carrying the exception type and message.
2. :func:`fixture_result` is the *only* sanctioned way to emit offline seed data,
   and it stamps ``status=CACHED`` plus ``degraded=True`` with a reason naming
   the fixture.
3. :func:`live_allowed` is the single gate every live call passes through, so
   ``settings.tools_live`` cannot be bypassed by accident.
4. :func:`redact` scrubs credentials from anything that reaches a log or trace.

Two rules are enforced by construction rather than by convention:

* **No bare ``except Exception`` anywhere in this package.** ``guarded`` catches
  an explicit, documented tuple of exception types. An exception outside that
  tuple is a programming bug and is allowed to propagate loudly, because a bug
  disguised as a tool failure is exactly the kind of invisible degradation this
  package exists to prevent.
* **Fixtures can never masquerade as live results.** ``ToolResult`` built from
  seed data always carries ``status=ToolStatus.CACHED``.
"""
from __future__ import annotations

import functools
import json
import re
import socket
import ssl
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any, TypeVar

import httpx

from core.config import Settings, get_settings
from core.protocols import TOOL_REGISTRY, Tool, ToolResult
from core.schemas import ToolStatus

__all__ = [
    "TOOL_CLASSES", "TOOL_SPECS", "ToolSpec",
    "tool", "guarded", "redact", "redacted_repr", "fixture_result", "annotate",
    "live_allowed", "http_client", "http_result", "settings_secrets",
    "describe_tools", "BaseTool", "haversine_km", "EARTH_RADIUS_KM", "REDACTED",
    "valid_email", "shaped_email", "valid_phone", "split_emails", "ToolSignal",
    "ToolResult", "ToolStatus", "TOOL_REGISTRY",
]

T = TypeVar("T")

#: Placeholder written over anything credential-shaped.
REDACTED = "***REDACTED***"


# ================================================================== registration
class ToolSpec:
    """Static description of a tool, kept beside the class for the demo panel.

    ``mode`` is the field that matters for the integrity story: it is how
    ``describe()`` tells a reader which tools are backed by a real backend and
    which are only pretending well enough to run the pipeline.
    """

    __slots__ = ("name", "description", "cls", "mode", "backend")

    def __init__(self, name: str, description: str, cls: type, mode: str,
                 backend: str) -> None:
        self.name = name
        self.description = description
        self.cls = cls
        #: ``live`` | ``fixture`` | ``local`` | ``pure`` | ``deterministic``.
        self.mode = mode
        #: The real service this talks to when live (``"overpass"``, ``"resend"``...).
        self.backend = backend

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolSpec({self.name!r}, mode={self.mode!r}, backend={self.backend!r})"


#: name -> concrete tool class, populated by the :func:`tool` decorator at import.
TOOL_CLASSES: dict[str, type] = {}

#: name -> spec, populated by the :func:`tool` decorator at import.
TOOL_SPECS: dict[str, ToolSpec] = {}


def tool(name: str, description: str, *, mode: str = "live",
         backend: str = "") -> Callable[[type], type]:
    """Class decorator registering a Paytriq tool.

    Deliberately registers the **class**, not an instance: every tool takes a
    ``Settings`` object, and ``Settings`` may differ per run and per test. The
    instances that land in :data:`core.protocols.TOOL_REGISTRY` are created by
    :func:`tools.registry.build_registry`, which is also what
    ``core/protocols.py`` documents as the populator of that registry.

    :param mode: ``live`` / ``fixture`` / ``local`` / ``pure``. Used by
        ``describe()`` so a reader can tell at a glance which tools are real.
    :param backend: name of the real service used in live mode, for the panel.
    """

    def decorate(cls: type) -> type:
        if name in TOOL_SPECS:
            raise ValueError(f"tool {name!r} is already registered")
        cls.name = name                     # type: ignore[attr-defined]
        cls.description = description       # type: ignore[attr-defined]
        cls.mode = mode                     # type: ignore[attr-defined]
        cls.backend = backend or mode       # type: ignore[attr-defined]
        TOOL_CLASSES[name] = cls
        TOOL_SPECS[name] = ToolSpec(name, description, cls, mode, cls.backend)  # type: ignore[attr-defined]
        return cls

    return decorate


# ========================================================================= gating
def live_allowed(settings: Settings) -> tuple[bool, str]:
    """``(may_call_live, reason)`` — the single gate for every network call.

    ``settings.tools_live`` defaults to ``False``. That default is the whole
    point: with it off, tools serve labelled fixtures rather than quietly
    inventing data, and with it on, every call is a real one that shows up in
    the trace as such.
    """
    if not settings.tools_enabled:
        return False, "tools are disabled (settings.tools_enabled=False)"
    if not settings.tools_live:
        return False, "live tool calls are off (settings.tools_live=False)"
    return True, f"live calls enabled via {settings.map_user_agent!r}"


class BaseTool:
    """Shared base for every tool: a ``Settings``, a timeout, and the secrets.

    Stateless apart from configuration, so one instance per run is safe and
    ``build_registry`` can hand the same instance to several agents. The
    invariant every subclass must honour is that ``run()`` never raises for an
    environmental problem — it returns an honest ``ToolResult`` instead.
    """

    #: Filled in by the :func:`tool` decorator at import time.
    name: str = ""
    description: str = ""
    mode: str = "live"
    backend: str = ""

    def __init__(self, settings: Settings | None = None, *,
                 timeout: float | None = None) -> None:
        self.settings: Settings = settings if settings is not None else get_settings()
        self.timeout: float = float(
            self.settings.tool_timeout_s if timeout is None else timeout
        )
        self.secrets: tuple[str | None, ...] = settings_secrets(self.settings)

    # ------------------------------------------------------------------ gating
    @property
    def live(self) -> bool:
        """True only when a real backend call is permitted for this run."""
        return live_allowed(self.settings)[0]

    def live_block_reason(self) -> str:
        """Why a live call is not permitted, or ``""`` when it is."""
        ok, reason = live_allowed(self.settings)
        return "" if ok else reason

    def current_mode(self) -> str:
        """Effective mode for :func:`describe_tools`.

        Defaults to the class-level ``mode``; subclasses that can serve fixtures
        override it so the status panel shows ``fixture`` rather than ``live``
        when live calls are switched off.
        """
        return self.mode

    def _unavailable(self, reason: str) -> ToolResult:
        """Convenience wrapper keeping the ``source`` prefix uniform."""
        return ToolResult.unavailable(reason=reason, source=f"{self.name}:gated")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r} mode={self.current_mode()!r}>"


# ======================================================================= redaction
#: ``key=value`` / ``"key": "value"`` pairs where the value looks like a secret.
_CREDENTIAL_RE = re.compile(
    r"(?i)\b([a-z0-9_\-]*(?:api[_-]?key|apikey|token|secret|password|passwd|"
    r"authorization|bearer)[a-z0-9_\-]*)(\s*[=:]\s*)"
    r"([\"']?)([A-Za-z0-9_\-\.]{8,})\3"
)


def redact(text: Any, secrets: Sequence[str | None] = ()) -> str:
    """Remove credentials from ``text`` before it is logged or traced.

    Two passes, because a secret can leak in two ways:

    * verbatim — a configured API key appearing inside an error message or a
      request URL. Known secret values are replaced exactly.
    * by shape — an unknown credential such as a bearer token echoed back by a
      server. Anything matching ``*_token=...``/``api_key: ...`` is replaced
      regardless of its value.

    Secrets shorter than six characters are ignored: replacing them would
    corrupt unrelated text without protecting anything meaningful.
    """
    out = "" if text is None else str(text)
    for secret in secrets or ():
        if not secret:
            continue
        value = str(secret).strip()
        if len(value) >= 6:
            out = out.replace(value, REDACTED)
    return _CREDENTIAL_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)


def redacted_repr(obj: Any, secrets: Sequence[str | None] = ()) -> str:
    """:func:`redact` for an arbitrary object; falls back to ``repr``."""
    try:
        return redact(repr(obj), secrets)
    except (TypeError, ValueError):  # pragma: no cover - exotic __repr__
        return f"<unreprable {type(obj).__name__}>"


def settings_secrets(settings: Settings) -> tuple[str | None, ...]:
    """Every value in ``settings`` that must never appear in a log or trace."""
    return (
        settings.resend_api_key,
        settings.gemini_api_key,
        settings.clef_api_key,
    )


# ==================================================================== tool result
def annotate(result: ToolResult, *, degraded: bool | None = None,
             reason: str | None = None, source: str | None = None,
             evidence_url: str | None = None) -> ToolResult:
    """Add honesty metadata that the ``ToolResult`` factories do not take.

    The four factories cover the common cases, but a tool sometimes knows more
    than the factory does — e.g. "this succeeded, but only as a model-generated
    observation". Rather than hand-constructing a ``ToolResult`` at a dozen call
    sites (where a ``status`` could be forgotten and silently become ``OK``),
    callers mutate a factory result through this one helper.
    """
    if degraded is not None:
        result.degraded = degraded
    if reason is not None:
        result.reason = reason
    if source is not None:
        result.source = source
    if evidence_url is not None:
        result.evidence_url = evidence_url
    return result


def fixture_result(data: Any, what: str, *, source: str = "fixture") -> ToolResult:
    """Build the only kind of ``ToolResult`` a fixture is ever allowed to produce.

    ``ToolResult.cached`` gives us ``status=ToolStatus.CACHED`` and
    ``degraded=True``; we then replace the generic reason with one that names
    the fixture explicitly, because "served from cache" does not tell a reader
    that the data underneath is *synthetic*.

    :param what: human noun for what is being faked, e.g. ``"brand list"``. It is
        interpolated into the reason so the trace reads
        ``FIXTURE brand list (synthetic...)``.
    """
    result = ToolResult.cached(data, source=source)
    result.reason = (
        f"FIXTURE {what}: synthetic seed data from tools/fixtures.py, not a real "
        f"observation. Live backend not called (settings.tools_live=False)."
    )
    return result


# ========================================================================= timeouts
#: Errors that mean "the backend could not be reached". Reported as UNAVAILABLE.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,                 # covers Connect/Timeout/Status/Protocol errors
    urllib.error.URLError,           # urllib fallback path
    ConnectionError,                 # OSError subclass
    TimeoutError,                    # OSError subclass; also FutureTimeoutError
    socket.gaierror,                 # DNS failure
    ssl.SSLError,                    # TLS failure
    socket.timeout,
)

#: Errors that mean "something answered, but the answer was unusable".
_PAYLOAD_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError,
    UnicodeDecodeError,
    LookupError,                     # KeyError / IndexError on a malformed payload
    TypeError,
    ValueError,
    AttributeError,
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    OSError,                         # last resort for filesystem faults
    RuntimeError,
)

_EXPECTED_ERRORS: tuple[type[BaseException], ...] = _TRANSPORT_ERRORS + _PAYLOAD_ERRORS


def _run_with_deadline(fn: Callable[..., T], args: tuple, kwargs: dict,
                       timeout: float | None) -> T:
    """Call ``fn`` on a worker thread and abandon it after ``timeout`` seconds.

    ``httpx`` already enforces its own socket timeouts; this is the outer belt
    that also covers SDKs which do not (``google-genai``, ``reportlab``, a
    blocking Playwright launch). The abandoned thread may finish later and its
    result is discarded, which is the documented behaviour of a deadline: we
    stop waiting, we do not pretend to know the answer.

    A timeout of ``None`` or ``<= 0`` runs inline, which keeps pure functions
    free of thread overhead.
    """
    if timeout is None or timeout <= 0:
        return fn(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1,
                            thread_name_prefix="paytriq-tool") as pool:
        future: Future[T] = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except _FutureTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"deadline of {timeout:.1f}s exceeded"
            ) from None


class ToolSignal(Exception):
    """A live backend already decided *how* it failed; carry that decision.

    Without this, an Overpass 429 would be caught by :func:`guarded`'s generic
    handler and flattened to ``FAILED`` ("the call failed"), erasing the
    distinction between "the backend is down / rate-limiting us" and "the backend
    answered with nonsense". That distinction is exactly what a run needs to
    record, so a live helper that already knows the honest verdict raises this
    instead of a bare ``RuntimeError``.
    """

    def __init__(self, result: ToolResult) -> None:
        self.result = result
        super().__init__(result.reason or result.status.value)


def guarded(fn: Callable[..., T], *, timeout: float | None = None,
            tool_name: str = "tool",
            secrets: Sequence[str | None] = (),
            errors: Sequence[type[BaseException]] = ()) -> Callable[..., ToolResult]:
    """Wrap a live call so it can only return an honest ``ToolResult``.

    Usage::

        result = guarded(self._fetch, timeout=settings.tool_timeout_s,
                         tool_name="fetch_page")(url)
        if not result.ok:
            return result                     # honest UNAVAILABLE / FAILED

    The wrapped function returns **raw data**; the wrapper turns it into
    ``ToolResult.success`` and turns any expected exception into
    ``ToolResult.unavailable`` (backend unreachable) or ``ToolResult.failed``
    (backend answered unusably). The reason always includes the exception type
    and message, redacted, because "it didn't work" is not a diagnosable trace.

    :param errors: extra exception types a third-party SDK may raise, appended
        to the default tuple. Used for ``google-genai`` and ``playwright``, whose
        error classes cannot be imported at module scope without making them
        hard dependencies.
    :raises: anything outside the documented tuple. Those are bugs, not
        environmental problems, and must fail loudly.
    """
    expected: tuple[type[BaseException], ...] = _EXPECTED_ERRORS + tuple(errors)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            value = _run_with_deadline(fn, args, kwargs, timeout)
        except ToolSignal as signal:
            # The callee already formed an honest verdict; do not re-grade it.
            return signal.result
        except _TRANSPORT_ERRORS as exc:
            return ToolResult.unavailable(
                reason=redact(
                    f"{tool_name}: backend unreachable — "
                    f"{type(exc).__name__}: {exc}", secrets),
                source=f"{tool_name}:live",
            )
        except expected as exc:
            return ToolResult.failed(
                reason=redact(
                    f"{tool_name}: call failed — {type(exc).__name__}: {exc}",
                    secrets),
                source=f"{tool_name}:live",
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return ToolResult.success(value, source=f"{tool_name}:live",
                                  latency_ms=latency_ms)

    return wrapper


def http_client(*, timeout: float, headers: Mapping[str, str] | None = None,
                settings: Settings | None = None) -> httpx.Client:
    """Build the one ``httpx.Client`` shape every network tool in Paytriq uses.

    ``follow_redirects=True`` because brand sites redirect ``http`` -> ``https``
    and to canonical hosts constantly; the final URL is reported as the evidence
    URL rather than the requested one. No retries: a retry that eventually
    succeeds hides a flaky backend, and the trace should show the first failure.

    Tests monkeypatch this function (or ``httpx.Client`` beneath it) so the whole
    package can be exercised with no network at all.
    """
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers=dict(headers or {}),
    )


def http_result(payload: tuple[int, bytes], *, evidence_url: str, settings: Settings,
                tool_name: str) -> ToolResult:
    """Turn an HTTP response into an honest ``ToolResult``.

    Centralised because getting this wrong is how fake data enters a system: a
    200 carrying an HTML error page, or a 429 carrying a JSON body, must never be
    parsed as a result. The rules, applied to every HTTP-backed tool:

    * 2xx          -> parse the body. Parsing failures become ``failed``.
    * 408/425/429  -> ``unavailable`` ("busy"). Retrying now would be rude and
                      would turn a rate limit into a lie by retrying silently.
    * 5xx          -> ``unavailable`` ("backend down").
    * anything else-> ``failed`` ("backend rejected the request"), quoting the
                      status and a redacted body excerpt.

    :param payload: ``(status_code, body_bytes)``.
    :returns: ``ToolResult.success`` whose ``data`` is ``(status_code, body)``.
    """
    status = int(payload[0])
    body = payload[1]
    if 200 <= status < 300:
        return ToolResult.success((status, body), source=f"{tool_name}:live",
                                  evidence_url=evidence_url)
    excerpt = redact(body[:400].decode("utf-8", errors="replace"),
                     settings_secrets(settings))
    if status in (408, 425, 429):
        return ToolResult.unavailable(
            reason=f"{tool_name}: backend busy (HTTP {status}). {excerpt}",
            source=f"{tool_name}:live")
    if status >= 500:
        return ToolResult.unavailable(
            reason=f"{tool_name}: backend error (HTTP {status}). {excerpt}",
            source=f"{tool_name}:live")
    return ToolResult.failed(
        reason=f"{tool_name}: request rejected (HTTP {status}). {excerpt}",
        source=f"{tool_name}:live")


# ==================================================================== validators
_EMAIL_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[A-Za-z]{2,}$")

#: Hostnames that cannot receive mail. Anything here is a mistake, not an address.
_NON_ROUTABLE_DOMAINS = frozenset({
    "example.com", "example.org", "example.net", "example.invalid",
    "localhost", "local", "test", "invalid", "domain", "email",
})

#: Reserved top-level domains (RFC 2606 / RFC 6761). A hostname under one of
#: these can never receive mail either, so ``sponsor@acme.invalid`` is refused
#: for the same reason ``partnerships@acme.com`` was invented and then reported
#: as delivered by the previous prototype.
_NON_ROUTABLE_TLDS = frozenset({
    "invalid", "test", "example", "localhost", "localdomain",
})


def valid_email(address: str) -> bool:
    """Shape-and-routability check for one email address.

    Shape (``a@b.tld``) is necessary but not sufficient, so this also rejects
    reserved documentation domains and reserved TLDs. That is a direct response
    to the prototype's habit of inventing ``partnerships@{brand}.com`` and then
    reporting a successful send to a domain that does not exist: a plausible
    address is not a contactable one, and reporting one as deliverable is worse
    than reporting none.

    ``SendEmailTool`` refuses rather than corrects. It never rewrites an address
    into a valid-looking one.
    """
    candidate = (address or "").strip()
    if not _EMAIL_RE.match(candidate):
        return False
    domain = candidate.rsplit("@", 1)[1].lower()
    if ".." in domain or domain.endswith("."):
        return False
    if domain in _NON_ROUTABLE_DOMAINS:
        return False
    return domain.rsplit(".", 1)[-1] not in _NON_ROUTABLE_TLDS


def shaped_email(address: str) -> bool:
    """Shape-only check: ``a@b.tld`` with no routability judgement.

    Used to decide whether a draft can even be written. A malformed address
    (no ``@``, no TLD) is refused everywhere; a well-shaped but reserved-TLD
    address (``sponsor@acme.invalid``) is refused for live sends but may still
    be spooled as an offline draft, exactly like an unroutable sender. The
    draft is reported as NOT sent, so no deliverability is misrepresented.
    """
    candidate = (address or "").strip()
    if not _EMAIL_RE.match(candidate):
        return False
    domain = candidate.rsplit("@", 1)[1].lower()
    return not (".." in domain or domain.endswith("."))


def split_emails(raw: str) -> list[str]:
    """Split a comma/semicolon separated recipient list into trimmed parts."""
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]


def valid_phone(value: str) -> bool:
    """Shape check for a phone number taken from OSM tags.

    Guards against OSM's ``phone=no`` / ``phone=none`` anti-pattern, which would
    otherwise become a "contact number" of the literal string ``no``.
    """
    candidate = (value or "").strip()
    if not candidate or candidate.lower() in {"no", "none", "n/a", "unknown", "-"}:
        return False
    if not re.fullmatch(r"[+()\d][\d\s()+\-./]{5,19}", candidate):
        return False
    return sum(ch.isdigit() for ch in candidate) >= 6


# ======================================================================= describe
#: Mean Earth radius (IUGG), in kilometres.
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres. Pure, deterministic, no network.

    Lives in ``base`` rather than ``maps`` for one structural reason: both
    ``tools.fixtures`` and ``tools.maps`` need it, and keeping it here lets
    ``fixtures`` compute its own distances without importing ``maps`` (which
    imports ``fixtures``), which would be a cycle.

    The haversine formula is used rather than a planar approximation because the
    demo prints distances to two decimals next to real coordinates and a
    flat-earth shortcut is visibly wrong at city scale.

    :returns: kilometres, rounded to 3 decimal places (roughly 0.1 m).
    """
    import math

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2)
    return round(2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a)), 3)


def describe_tools(tools: Mapping[str, Tool]) -> list[dict[str, Any]]:
    """One row per tool for the demo status panel.

    Reports what each tool *is*, whether it can run right now, and whether its
    answer would come from a real backend or from labelled seed data. The
    ``mode`` column is the one that matters: it is what stops a fixture-backed
    demo from being read as a live one.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(tools):
        tool_obj = tools[name]
        spec = TOOL_SPECS.get(name)
        try:
            is_available, reason = tool_obj.available()
        except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
            is_available, reason = False, f"available() raised {type(exc).__name__}: {exc}"
        cfg = getattr(tool_obj, "settings", None)
        mode_fn = getattr(tool_obj, "current_mode", None)
        mode = mode_fn() if callable(mode_fn) else getattr(tool_obj, "mode", None)
        rows.append({
            "name": name,
            "description": getattr(tool_obj, "description",
                                   spec.description if spec else ""),
            "mode": mode or (spec.mode if spec else "unknown"),
            "backend": getattr(tool_obj, "backend", spec.backend if spec else ""),
            "available": bool(is_available),
            "reason": reason,
            "live_enabled": bool(getattr(cfg, "tools_live", False)),
            "tools_enabled": bool(getattr(cfg, "tools_enabled", False)),
        })
    return rows


def tool_registry_size() -> int:
    """Number of tools currently in ``core.protocols.TOOL_REGISTRY``."""
    return len(TOOL_REGISTRY)


def default_settings() -> Settings:
    """Convenience for callers that do not already hold a ``Settings``."""
    return get_settings()
