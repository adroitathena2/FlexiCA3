"""Application factory: middleware, lifespan, static mount, OpenAPI.

Two bugs from the previous version are fixed structurally here rather than by
convention.

1. **CORS.** The old app used ``allow_origins=["*"]`` together with
   ``allow_credentials=True``. That combination is rejected by browsers: the
   CORS specification forbids the wildcard origin on a credentialed request, and
   Chrome answers with an error rather than a response. It appeared to work only
   because ``fetch`` sent no credentials, so the browser's credentialed-CORS path
   was never exercised -- until someone turned on cookies. The rule, and the
   config below:

   * ``allow_origins=["*"]``  ->  ``allow_credentials`` **must** be ``False``.
   * ``allow_credentials=True`` ->  ``allow_origins`` **must** name origins
     explicitly (no wildcard).

   So the default is the first case -- open reads, no credentials -- which is
   correct for a public demo panel. The second is reached by setting
   ``PAYTRIQ_CORS_ORIGINS``, and the app **refuses to start** if it is configured
   with credentials and a wildcard, rather than silently shipping the invalid
   combination again.

2. **The static mount.** The old app mounted ``frontend/`` at ``/frontend`` inside
   a bare ``try/except Exception: pass``, so a failure to mount was invisible and
   the resulting 404 was indistinguishable from a missing file. Here the mount is
   attempted explicitly, the outcome is recorded, and ``/health`` reports it.

The lifespan
------------
On startup: build the blackboard and the tracer for a bootstrap run, so
``/health`` has live collaborators before the first request arrives.

On shutdown: ``tracer.finish()`` on **every** run, so a trace file and its summary
sidecar are always written -- including when the process is dying because
something raised. A run whose evidence is only flushed by a clean shutdown is a
run whose evidence is missing exactly when it is needed.

``/docs`` and ``/openapi.json`` are FastAPI's defaults and are left enabled on
purpose: a generated, always-current OpenAPI document is the cheapest available
evidence that this is a real backend with real typed routes, rather than a
hand-written list of URLs.
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import Settings

from . import deps, routes_events, routes_gates, routes_pipeline, routes_trace
from . import stream as stream_module


def _load_dotenv() -> None:
    """Load ``.env`` into ``os.environ`` without adding a dependency.

    Tries ``python-dotenv`` first; when it is not installed, parses the file
    by hand (``KEY=value``, ignoring blanks and ``#`` comments, stripping
    matching quotes). Existing environment values always win, and a missing
    file is not an error.
    """
    try:
        from dotenv import load_dotenv as _load  # type: ignore[import-not-found]

        _load()
        return
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - fall through to the manual parser
        pass
    try:
        candidate = Path.cwd() / ".env"
        if not candidate.is_file():
            return
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

__all__ = ["create_app", "cors_origins_from_env", "CONFIG_ERRORS", "TAGS"]

log = logging.getLogger("paytriq.api.main")

#: Human-facing descriptions for the OpenAPI tag list. Order is the order the
#: tags appear in the docs sidebar.
TAGS: list[dict[str, Any]] = [
    {"name": "events", "description":
     "Create an event profile and post it to the blackboard. Unknown fields are "
     "rejected with 422 rather than discarded."},
    {"name": "pipeline", "description":
     "The seven agents' stages. Outreach and MoU release are refused with 403 "
     "until an approval record exists; every response names the DecisionSource "
     "that produced its routing decision."},
    {"name": "gates", "description":
     "Human approval gates. Every answer writes a HumanDecision and an Approval, "
     "and an approval only authorises the single action kind it was raised for."},
    {"name": "trace", "description":
     "Inspection: trace events, the summary (including distinct_gap_values, the "
     "anti-fabrication metric), the blackboard, and the trace linter."},
    {"name": "stream", "description":
     "Server-Sent Events: every TraceEvent as it happens, then a summary."},
    {"name": "health", "description": "Liveness and per-subsystem availability."},
]

#: Startup failures worth surfacing loudly rather than as a 500 later.
CONFIG_ERRORS: tuple[type[BaseException], ...] = (RuntimeError, ValueError, OSError)


def cors_origins_from_env(raw: str | None = None) -> list[str]:
    """Parse ``PAYTRIQ_CORS_ORIGINS`` into a list of origins.

    Comma or space separated. ``*`` means "any origin, no credentials" and is the
    documented default for a public demo.
    """
    text = os.getenv("PAYTRIQ_CORS_ORIGINS", raw if raw is not None else "")
    if not text or not text.strip():
        return ["*"]
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    return parts or ["*"]


def _allow_credentials(origins: Sequence[str]) -> bool:
    """Whether to send ``Access-Control-Allow-Credentials``.

    Never ``True`` alongside a wildcard origin -- see the module docstring. The
    wildcard case means "this API is public and sends no cookies", which is what
    a demo panel wants; the credentialed case means "a real frontend is talking to
    this with a session", which requires the caller to name its origins.
    """
    return "*" not in origins


def _validate_cors(origins: Sequence[str], allow_credentials: bool) -> None:
    """Refuse the invalid wildcard-plus-credentials combination at startup.

    Raising here rather than shipping it is the point: the previous version's
    configuration was invalid, invalid configurations are easy to ship by
    accident, and a browser's failure mode (an opaque network error) does not
    name the cause.
    """
    if allow_credentials and "*" in origins:
        raise RuntimeError(
            "invalid CORS configuration: allow_credentials=True with a wildcard "
            "origin is forbidden by the CORS specification and browsers reject "
            "it. Either drop credentials (the default) or list explicit origins "
            "in PAYTRIQ_CORS_ORIGINS, e.g. "
            "PAYTRIQ_CORS_ORIGINS=https://myapp.example"
        )


def _frontend_dir(base_dir: Path) -> Path | None:
    """The frontend directory under ``base_dir``, if it exists.

    Honours ``repo_root`` exactly: if a caller says "the frontend lives here", the
    mount looks *here* and nowhere else. An earlier version fell back to the
    configured repository root when the override had no frontend, which made the
    mount's behaviour depend on the rest of the checkout -- a test pointing at an
    empty temp directory silently served the real repository's files.

    ``PAYTRIQ_SKIP_FRONTEND`` short-circuits the mount entirely.
    """
    if os.getenv("PAYTRIQ_SKIP_FRONTEND"):
        return None
    candidate = Path(base_dir) / "frontend"
    if candidate.is_dir() and (candidate / "index.html").exists():
        return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build collaborators on startup; flush every trace on shutdown.

    The shutdown half runs in a ``finally``, so it happens on an exception path
    too. ``tracer.finish()`` is what writes ``traces/<run_id>.jsonl`` and its
    ``.summary.json`` sidecar; a run that dies before it leaves no evidence, and a
    system whose evidence vanishes when it fails cannot be audited at the moment
    it most needs to be.
    """
    settings = deps.get_settings()
    deps.set_settings(settings)
    try:
        bootstrap = deps.create_run()
        app.state.bootstrap_run_id = bootstrap.run_id
        log.info("api ready: run=%s board=%s tracer=%s",
                 bootstrap.run_id,
                 "ok" if bootstrap.board is not None else bootstrap.board_error,
                 "ok" if bootstrap.tracer is not None else bootstrap.tracer_error)
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail the app
        # The API is still useful without a board or a tracer: /healthz, /health
        # and /openapi.json all work, and every feature endpoint answers 503 with
        # the reason. Refusing to start would leave nothing to diagnose with.
        log.error("could not build the bootstrap run: %s: %s", type(exc).__name__, exc)
        app.state.bootstrap_error = f"{type(exc).__name__}: {exc}"
    try:
        yield
    finally:
        finished: list[dict[str, Any]] = []
        for row in deps.list_runs():
            try:
                finished.append(deps.get_run(row["run_id"]).finish())  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 - shutdown must complete
                log.error("could not finish run %s: %s: %s",
                          row.get("run_id"), type(exc).__name__, exc)
        log.info("api shutting down; %d trace(s) finished: %s", len(finished),
                 [f.get("trace_file") for f in finished if f.get("trace_file")])


def create_app(settings: Settings | None = None, *,
               cors_origins: Sequence[str] | None = None,
               repo_root: Path | None = None) -> FastAPI:
    """Build the Paytriq application.

    Args:
        settings: overrides the process settings. Tests pass one pointing at a
            temp ``traces_dir`` so no test writes into the repository.
        cors_origins: explicit allowed origins. Defaults to
            ``PAYTRIQ_CORS_ORIGINS`` or ``["*"]``.
        repo_root: where to look for ``frontend/``. Defaults to the repository
            root recorded in ``core.config.Settings``.

    Raises:
        RuntimeError: for a CORS configuration browsers reject (see
            :func:`_validate_cors`). Failing at startup is deliberate.
    """
    if settings is not None:
        deps.set_settings(settings)
    cfg = deps.get_settings()

    origins = list(cors_origins) if cors_origins is not None else cors_origins_from_env()
    credentials = _allow_credentials(origins)
    _validate_cors(origins, credentials)

    # Named ``base_dir`` rather than ``root``: the service-index handler below is
    # called ``root()``, and a function would shadow the path.
    base_dir = Path(repo_root) if repo_root is not None else Path(cfg.repo_root)

    app = FastAPI(
        title="Paytriq API",
        version=cfg.app_version,
        summary="A 7-agent campus sponsorship pipeline: discovery to audited ROI.",
        description=(
            "Seven reasoning agents (A1 Discovery ... A7 Arbiter) coordinate through "
            "an append-only blackboard, take every routing decision through a named "
            "decision backend, and pause at a human gate before any irreversible "
            "action. Three properties this API is built to make checkable:\n\n"
            "* **No fabricated responses.** A missing subsystem returns HTTP 503 "
            "with `unavailable: true` and the real reason. A response never "
            "contains a plausible-looking substitute for a result.\n"
            "* **No unapproved side effects.** Outreach and MoU release return 403 "
            "unless an `Approval` row authorises them, and an approval only "
            "authorises the one action kind it was raised for.\n"
            "* **Distinguishable from a fabrication.** `GET /api/runs/{run_id}/summary` "
            "returns `distinct_gap_values`: many for a genuine capture, exactly 1 "
            "for a hand-written trace with constant timing."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = cfg
    app.state.cors_origins = origins
    app.state.cors_allow_credentials = credentials
    app.state.frontend_dir = None

    # ---------------------------------------------------------------- CORS
    # See the module docstring: wildcard origin implies credentials off.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
        max_age=600,
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next: Any) -> Any:
        """Stamp every response with the request id.

        One line, but it is the difference between "the demo broke" and "the
        demo broke on request 41, which is in trace run_20261004T...". The id is
        accepted from the client when supplied so a browser and a server log agree.
        """
        from core.ids import new_id

        incoming = request.headers.get("x-request-id")
        request_id = (incoming or new_id("req"))[:64]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.exception_handler(500)
    async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
        """Never leak a stack trace; always name the subsystem when there is one."""
        from core.errors import PaytriqError

        if isinstance(exc, deps.SubsystemUnavailable):
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                content=deps.unavailable_payload(exc))
        log.error("unhandled error on %s %s: %s: %s",
                  request.method, request.url.path, type(exc).__name__, exc)
        content: dict[str, Any] = {
            "error": f"{type(exc).__name__}: {exc}",
            "reason": "the API failed to handle this request; see the server log",
            "request_id": getattr(request.state, "request_id", ""),
        }
        if isinstance(exc, PaytriqError):
            content["unavailable"] = False
        return JSONResponse(status_code=500, content=content)

    # ------------------------------------------------------------- routers
    app.include_router(routes_events.router)
    app.include_router(routes_pipeline.router)
    app.include_router(routes_gates.router)
    app.include_router(routes_trace.router)
    app.include_router(stream_module.router)

    # ------------------------------------------------------------- health
    @app.get("/healthz", tags=["health"],
             summary="Liveness: is the process up?")
    def healthz() -> dict[str, Any]:
        """No dependencies. Deliberately touches nothing.

        Liveness and readiness are different questions. A liveness probe that
        touches the tracer would restart the process when a *dependency* is
        missing, which turns a degraded system into a dead one. This endpoint
        imports nothing and calls nothing; ``/health`` is where the dependency
        story lives.
        """
        import platform

        return {
            "status": "ok",
            "app": cfg.app_name,
            "version": cfg.app_version,
            "python": platform.python_version(),
        }

    @app.get("/health", tags=["health"],
             summary="Readiness: what is available, and what is not")
    def health(probe: bool = False) -> dict[str, Any]:
        """``health_payload()``: every subsystem, the decision chain, the tools.

        ``?probe=true`` asks each decision backend whether it answers *right now*.
        That is a socket to a local llama.cpp server and takes seconds to time
        out, so it is opt-in: the default reports what is configured and what
        answered last, which is enough to explain a degraded run.
        """
        return deps.health_payload(probe_backends=bool(probe))

    @app.get("/", tags=["health"], include_in_schema=True,
             summary="Service index")
    def root() -> dict[str, Any]:
        """Where everything is. Printed rather than served so it also works with
        no frontend present."""
        return {
            "service": cfg.app_name,
            "version": cfg.app_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "healthz": "/healthz",
            "health": "/health",
            "endpoints": {
                "events": ["POST /api/events", "GET /api/events",
                           "GET /api/events/{event_id}", "GET /api/schema/event"],
                "pipeline": [f"POST /api/events/{{event_id}}/{stage}"
                             for stage in routes_pipeline.STAGES]
                            + ["POST /api/outreach", "POST /api/reply"],
                "gates": ["GET /api/gates/{run_id}", "POST /api/gates/{gate_id}"],
                "trace": [f"GET /api/runs/{{run_id}}/{suffix}" for suffix in
                          ("trace", "summary", "board", "board/tree", "disputes",
                           "lessons", "handoffs", "bids")]
                         + ["POST /api/runs/{run_id}/lint"],
                "stream": ["GET /api/runs/{run_id}/stream"],
            },
            "frontend": ("mounted at /" if app.state.frontend_dir else
                         "not present in this checkout"),
        }

    # ------------------------------------------------------------- frontend
    # Mounted last so every route above wins the match. Starlette matches in
    # registration order, so a Mount at "/" only sees paths nothing else claimed.
    frontend = _frontend_dir(base_dir)
    if frontend is not None:
        try:
            app.mount("/", StaticFiles(directory=str(frontend), html=True),
                      name="frontend")
            app.state.frontend_dir = str(frontend)
            log.info("frontend mounted at / from %s", frontend)
        except (OSError, RuntimeError, ValueError) as exc:
            # Reported, not swallowed. A silently-missing frontend is how the
            # previous version shipped.
            log.error("frontend directory %s could not be mounted: %s: %s",
                      frontend, type(exc).__name__, exc)
            app.state.frontend_error = f"{type(exc).__name__}: {exc}"

    return app


#: Lazily-built module-level app, for ``uvicorn api.main:app``.
#:
#: PEP 562 ``__getattr__`` rather than ``app = create_app()`` at import time,
#: because ``create_app`` reads ``Settings``, whose constructor creates the trace
#: and artifact directories. Importing a module should not have that side effect;
#: the object is built on first access instead, so ``uvicorn api.main:app`` and
#: ``uvicorn api.main:create_app --factory`` both work and neither pays for the
#: other.
_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _port(default: int = 18780) -> int:
    """Port for ``python -m api.main`` / ``python api/main.py``.

    Honors ``$PORT`` (platform-assigned port) and falls back to 18780.
    """
    try:
        return int(os.getenv("PORT", str(default)))
    except ValueError:
        return default


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=_port(18780))
