"""Opt-in backend smoke tests: skipped unless ``--run-live`` / ``--run-clef``.

The default suite is fully hermetic (see ``tests/conftest.py``): every backend
is stubbed or driven through ``httpx.MockTransport``, so a developer without
keys or a local model gets the same result as CI. These two tests are the
exception, and they exist for one reason: to prove the opt-in flags actually
reach a real backend, and that ``--run-live`` does not drag the ``clef`` path
along with it (the gating bug fixed in ``conftest.py``).

* ``live`` spends no tokens: ``GeminiBackend.available()`` only builds the SDK
  client. Unreachability surfaces at call time as ``DecisionUnavailable``.
* ``clef`` makes exactly one trivial ``CHOICE`` call against the llama-server
  on ``$CLEF_BASE_URL``. Without a server it skips: an absent model is not a
  failure of the suite, it is the normal offline state.
"""
from __future__ import annotations

import os

import pytest

from core.config import Settings
from core.schemas import DecisionSource, QuestionType


@pytest.mark.live
def test_live_gemini_backend_is_configured(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``--run-live`` and a real key, the live path must resolve."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set; nothing live to check")
    # The deploy default, pinned so a regression to a nonexistent model id is
    # visible here rather than as a 404 in production.
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    settings = Settings()
    assert settings.gemini_model == "gemini-2.5-flash"

    from decision import GeminiBackend

    backend = GeminiBackend(settings)
    ok, reason = backend.available()
    assert ok, reason


@pytest.mark.clef
def test_clef_server_answers_a_trivial_choice() -> None:
    """With ``--run-clef`` and a local server, one choice round-trips."""
    from core.schemas import DecisionRequest
    from decision import ClefBackend

    settings = Settings()
    backend = ClefBackend(settings)
    ok, reason = backend.available()
    if not ok:
        pytest.skip(f"no clef server at {settings.clef_base_url}: {reason}")
    decision = backend.decide(DecisionRequest(
        request_id="req_optin_clef",
        question="Which team owns this issue?",
        question_type=QuestionType.CHOICE,
        options=["billing", "technical"],
        state={"text": "the invoice is wrong"},
    ))
    assert decision.source is DecisionSource.CLEF
    assert decision.choice in ("billing", "technical")
