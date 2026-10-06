"""Shared pytest configuration for Paytriq.

Two jobs.

**1. Offline by default, opt-in for real backends.** Live and clef tests are
skipped unless explicitly requested. This is done here rather than with ``-m`` in
``addopts`` because pytest ANDs multiple ``-m`` filters, so a default ``-m "not
live"`` in the ini file would make ``pytest -m clef`` silently evaluate to
*"not live AND clef"* and quietly run nothing. A ``--run-live`` flag has no such
trap.

**2. Import bootstrap.** Ensures the repo root is importable so ``import core``
works no matter where pytest is invoked from, and so a bare ``pytest`` (rather
than ``python -m pytest``) behaves identically.

Usage::

    pytest                       # offline; live + clef tests skipped
    pytest --run-clef            # also run tests against a local llama.cpp clef
    pytest --run-live            # also run tests that hit real network backends
    pytest -m "not slow"         # normal marker filtering still works
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("paytriq", "Paytriq backend opt-ins")
    group.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked 'live' that need real network backends",
    )
    group.addoption(
        "--run-clef",
        action="store_true",
        default=False,
        help="run tests marked 'clef' that need a local llama.cpp decision model",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: needs a real model or network backend (use --run-live)"
    )
    config.addinivalue_line(
        "markers",
        "clef: needs a local llama.cpp decision model (use --run-clef)",
    )
    config.addinivalue_line("markers", "slow: takes more than a couple of seconds")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in tests unless their flag was passed.

    The two flags are independent: ``--run-live`` opts into network-backed
    tests only, ``--run-clef`` into local-decision-model tests only. An earlier
    version short-circuited on ``run_live`` first, so ``pytest --run-live``
    silently also ran every ``clef`` test against a server nobody asked for.
    """
    run_live = config.getoption("--run-live")
    run_clef = config.getoption("--run-clef")

    skip_live = pytest.mark.skip(
        reason="needs a real backend; pass --run-live to include"
    )
    skip_clef = pytest.mark.skip(
        reason="needs a local decision model on $CLEF_BASE_URL; pass --run-clef to include"
    )
    for item in items:
        if "live" in item.keywords and not run_live:
            item.add_marker(skip_live)
        if "clef" in item.keywords and not run_clef:
            item.add_marker(skip_clef)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def settings_offline():
    """Settings forced offline, so a developer's real .env cannot leak into tests.

    Without this, a machine with ``GEMINI_API_KEY`` set would take a different
    code path than CI, and a test that passes locally would fail for a reviewer.
    """
    from core.config import Settings

    saved = {
        k: os.environ.get(k)
        for k in ("RUN_MODE", "GEMINI_API_KEY", "CLEF_API_KEY", "TOOLS_LIVE",
                  "RESEND_API_KEY", "DECISION_BACKEND", "HUMAN_GATES_INTERACTIVE")
    }
    os.environ.update({
        "RUN_MODE": "offline",
        "GEMINI_API_KEY": "",
        "CLEF_API_KEY": "",
        "TOOLS_LIVE": "false",
        "RESEND_API_KEY": "",
        "DECISION_BACKEND": "rules",
        "HUMAN_GATES_INTERACTIVE": "false",
    })
    from core.config import reset_settings_cache

    reset_settings_cache()
    try:
        yield Settings()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings_cache()
