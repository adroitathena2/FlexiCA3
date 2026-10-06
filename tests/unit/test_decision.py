"""Unit tests for the Paytriq decision layer.

Fully offline, by construction and by assertion:

* ``httpx.MockTransport`` replaces the HTTP transport, so ``/v1/systemone`` and
  the Cloudflare Workers AI route are exercised through the real request path --
  real JSON encoding, real status handling, real exception types -- without a
  socket;
* the Gemini client is injected, so no token is spent;
* an autouse fixture makes ``socket.socket.connect`` raise, so a test that
  accidentally reaches the network fails loudly instead of hanging.

The four things worth more than the rest of the coverage:

1. **No fabricated probabilities.** Every path that would have to invent a
   distribution is asserted to raise instead.
2. **The wrong-endpoint trap.** A chat-completions-shaped response must be
   rejected with a message naming ``/v1/systemone``.
3. **The intent regression.** "yesterday we thought the price was too high" must
   not become assent; the previous prototype's router tested ``"yes"`` first and
   routed that sentence into contract signing.
4. **The rules confidence ceiling.** A rule that claims 0.9 is lying, and every
   assertion here checks the ceiling.
"""
from __future__ import annotations

import json
import socket
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

# Allow `python -m pytest tests/unit/test_decision.py` and a bare `pytest` alike
# from the repository root, with or without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import decision  # noqa: E402
from core.config import Settings  # noqa: E402
from core.errors import DecisionFailed, DecisionUnavailable  # noqa: E402
from core.protocols import DecisionBackend  # noqa: E402
from core.schemas import (  # noqa: E402
    Decision,
    DecisionRequest,
    DecisionSource,
    Intent,
    QuestionType,
    RunMode,
)
from decision import (  # noqa: E402
    CONFIDENCE_CEILING,
    ClefBackend,
    DecisionRegistry,
    GeminiBackend,
    LlamaCppTransport,
    RulesBackend,
    WorkersAiTransport,
    build_registry,
    build_response_key,
    choice_from,
    normalise_probabilities,
    require_answers,
)

# ==============================================================================
# hermetic environment
# ==============================================================================
_ENV_KEYS = (
    "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL", "GEMINI_TIMEOUT_S",
    "CLEF_API_KEY", "CLEF_ACCOUNT_ID", "CLEF_BACKEND", "CLEF_BASE_URL",
    "CLEF_MODEL", "CLEF_TIMEOUT_S", "CLEF_HEALTH_TTL_S",
    "DECISION_BACKEND", "DECISION_MAX_RETRIES", "RUN_MODE",
    "CONFIDENCE_THRESHOLD", "ESCALATION_THRESHOLD",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every decision-related environment variable.

    ``core.config.Settings`` reads the process environment. A developer machine
    with ``GEMINI_API_KEY`` exported would otherwise make half of these tests
    exercise a different code path than the one on the grading machine.
    """
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound socket attempt an immediate, loud failure."""

    def deny(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "decision unit tests must never open a socket; use httpx.MockTransport"
        )

    monkeypatch.setattr(socket.socket, "connect", deny, raising=True)
    monkeypatch.setattr(socket, "create_connection", deny, raising=True)


# ==============================================================================
# fake System One server
# ==============================================================================
FAKE_MODELS = {"object": "list", "data": [{"id": "/models/clef-flash.Q4_K_M.gguf",
                                          "object": "model"}]}


def _answer_for(key: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a plausible System One answer for whatever was asked."""
    qtype = spec.get("type")
    if qtype == "noul":
        return {"type": "noul", "noul": 0.93}
    if qtype == "choice":
        criteria = list(spec.get("criteria") or {})
        head = criteria[0] if criteria else "a"
        probabilities = {head: 0.7}
        for option in criteria[1:]:
            probabilities[option] = round(0.3 / max(1, len(criteria) - 1), 6)
        return {"type": "choice", "choice": head, "confidence": probabilities[head],
                "probabilities": probabilities}
    levels = list(spec.get("criteria") or [])
    if not levels:
        return {"type": "score", "score": "major", "confidence": 0.6,
                "legend": {"0": "major"}, "probabilities": {"0": 0.6}}
    probabilities = {str(i): (0.66 if i == len(levels) - 1 else 0.17)
                     for i in range(len(levels))}
    return {
        "type": "score",
        "score": levels[-1],
        "confidence": 0.66,
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": probabilities,
    }


def fake_systemone(overrides: Mapping[str, Any] | None = None, *,
                   models: bool = True,
                   envelope: bool = False) -> Callable[[httpx.Request], httpx.Response]:
    """A fake ``/v1/systemone`` that answers whatever question it is asked.

    ``overrides`` maps a question key to the exact answer body to return, for
    the malformed-response cases. ``envelope=True`` wraps in the Cloudflare
    Workers AI ``{"result": ..., "success": true}`` envelope.
    """
    overrides = dict(overrides or {})
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            if not models:
                return httpx.Response(404, json={"error": "no models route"})
            return httpx.Response(200, json=FAKE_MODELS)
        body = json.loads(request.content)
        calls.append(body)
        key = next(iter(body["questions"]))
        spec = body["questions"][key]
        answer = overrides.get(key, _answer_for(key, spec))
        payload = {
            "answers": {key: answer},
            "usage": {"input_tokens": 412, "output_tokens": 0},
        }
        if envelope:
            return httpx.Response(200, json={"result": payload, "success": True,
                                             "errors": [], "messages": []})
        return httpx.Response(200, json=payload)

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def clef_backend(handler: Callable[[httpx.Request], httpx.Response], *,
                 settings: Settings | None = None,
                 max_retries: int = 0,
                 base_url: str = "http://localhost:8080",
                 ) -> ClefBackend:
    """A ``ClefBackend`` whose only transport is a mock."""
    s = settings or Settings(clef_backend="ollama", clef_base_url=base_url,
                             clef_model="clef-flash", decision_max_retries=max_retries)
    transport = LlamaCppTransport(base_url, model_label=s.clef_model,
                                  timeout_s=s.clef_timeout_s,
                                  max_retries=max_retries, backoff_s=0.0,
                                  transport=httpx.MockTransport(handler))
    return ClefBackend(s, transports={"ollama": transport}, backoff_s=0.0)


def choice_request(**kwargs: Any) -> DecisionRequest:
    payload: dict[str, Any] = {
        "request_id": "req_choice",
        "question": "Which team owns this issue?",
        "question_type": QuestionType.CHOICE,
        "options": ["billing", "technical"],
        "state": {"text": "the invoice is wrong"},
    }
    payload.update(kwargs)
    return DecisionRequest(**payload)


def score_request(**kwargs: Any) -> DecisionRequest:
    payload: dict[str, Any] = {
        "request_id": "req_score",
        "question": "How severe is this?",
        "question_type": QuestionType.SCORE,
        "rubric": ["none", "minor", "major"],
        "state": {"text": "a deliverable was missed"},
    }
    payload.update(kwargs)
    return DecisionRequest(**payload)


def noul_request(**kwargs: Any) -> DecisionRequest:
    payload: dict[str, Any] = {
        "request_id": "req_noul",
        "question": "Is this urgent?",
        "question_type": QuestionType.NOUL,
        "state": {"text": "we need an answer today"},
    }
    payload.update(kwargs)
    return DecisionRequest(**payload)


def intent_request(text: str, **kwargs: Any) -> DecisionRequest:
    payload: dict[str, Any] = {
        "request_id": "req_intent",
        "question": "What is the sponsor's intent?",
        "question_type": QuestionType.CHOICE,
        "options": [i.value for i in Intent],
        "state": {"reply_text": text},
    }
    payload.update(kwargs)
    return DecisionRequest(**payload)


class StubBackend:
    """A ``DecisionBackend`` that does exactly what a test tells it to.

    Used wherever the point of the test is the *chain*, not the HTTP path, so
    that no test has to invent a transport to make a backend misbehave.
    """

    def __init__(self, name: str, *, model: str = "stub",
                 available: bool = True,
                 decision: Decision | None = None,
                 error: BaseException | None = None) -> None:
        self.name = name
        self.model = model
        self._available = available
        self._decision = decision
        self._error = error
        self.calls = 0

    def available(self) -> tuple[bool, str]:
        return self._available, "configured" if self._available else "not configured"

    def decide(self, request: DecisionRequest) -> Decision:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._decision is not None:
            return self._decision
        return Decision(
            request_id=request.request_id, question=request.question,
            choice="billing", probabilities={"billing": 0.7, "technical": 0.3},
            confidence=0.7, source=DecisionSource(self.name), model=self.model,
        )

    def health(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model,
                "available": self._available, "calls": self.calls}


# ==============================================================================
# protocol conformance
# ==============================================================================
@pytest.mark.parametrize("backend_factory", [
    lambda: RulesBackend(Settings()),
    lambda: ClefBackend(Settings(clef_backend="ollama"),
                        transports={"ollama": LlamaCppTransport(
                            "http://localhost:8080", timeout_s=1,
                            transport=httpx.MockTransport(fake_systemone()))}),
    lambda: GeminiBackend(Settings(gemini_api_key="test-key")),
    lambda: DecisionRegistry(
        Settings(decision_backend="rules"),
        clef=StubBackend("clef"), gemini=StubBackend("gemini"),
        rules=StubBackend("rules")),
])
def test_backends_satisfy_the_decision_backend_protocol(backend_factory: Callable[[], Any]) -> None:
    """``core.protocols.DecisionBackend`` is the frozen contract; honour it."""
    backend = backend_factory()
    assert isinstance(backend, DecisionBackend)
    assert isinstance(backend.name, str) and backend.name
    assert isinstance(backend.model, str) and backend.model
    ok, reason = backend.available()
    assert isinstance(ok, bool) and isinstance(reason, str) and reason


# ==============================================================================
# probability normalisation
# ==============================================================================
class TestNormaliseProbabilities:
    def test_coerces_numeric_strings(self) -> None:
        out = normalise_probabilities({"a": "0.7", "b": "0.3"})
        assert out == {"a": 0.7, "b": 0.3}

    def test_clamps_out_of_range(self) -> None:
        out = normalise_probabilities({"a": 1.4, "b": -0.4})
        assert all(0.0 <= v <= 1.0 for v in out.values())
        assert abs(sum(out.values()) - 1.0) <= 1e-4

    def test_renormalises_unnormalised_input(self) -> None:
        out = normalise_probabilities({"a": 0.6, "b": 0.3})
        assert out["a"] > out["b"]
        assert abs(sum(out.values()) - 1.0) <= 1e-4

    def test_values_above_one_are_clamped_before_renormalising(self) -> None:
        """Clamping precedes normalisation, so 3.0 and 1.0 become 0.5 / 0.5."""
        out = normalise_probabilities({"a": 3.0, "b": 1.0})
        assert out == {"a": 0.5, "b": 0.5}

    def test_sum_is_within_one_e_minus_four(self) -> None:
        """Tighter than the ``Decision`` validator's 0.02 tolerance."""
        out = normalise_probabilities({f"o{i}": i + 1 for i in range(7)})
        assert abs(sum(out.values()) - 1.0) <= 1e-4

    def test_zero_sum_raises_rather_than_dividing_by_zero(self) -> None:
        with pytest.raises(DecisionFailed, match="zero distribution"):
            normalise_probabilities({"a": 0.0, "b": 0.0})

    @pytest.mark.parametrize("bad", [None, {"a": "high"}, {"a": True}, {"a": "NaN"},
                                     {"a": ["x"]}, {"a": {"n": 1}}])
    def test_unusable_values_raise(self, bad: Any) -> None:
        with pytest.raises(DecisionFailed):
            normalise_probabilities({"a": bad, "b": 0.5})

    def test_non_finite_is_refused(self) -> None:
        """Ollama's Windows clef bug (ollama/ollama#18769) shows up as NaN."""
        with pytest.raises(DecisionFailed, match="non-finite"):
            normalise_probabilities({"a": float("nan"), "b": 1.0})

    def test_empty_and_non_mapping_raise(self) -> None:
        with pytest.raises(DecisionFailed, match="empty"):
            normalise_probabilities({})
        with pytest.raises(DecisionFailed, match="mapping"):
            normalise_probabilities([0.5, 0.5])  # type: ignore[arg-type]


# ==============================================================================
# response parsing, per question type
# ==============================================================================
class TestChoiceFrom:
    # ------------------------------------------------------------------ noul
    def test_noul_uses_the_noul_probability(self) -> None:
        choice, probs, confidence = choice_from({"type": "noul", "noul": 0.93}, [])
        assert choice == "true"
        assert probs == {"true": 0.93, "false": 0.07}
        # confidence is absent for noul answers and must be derived, not invented.
        assert abs(confidence - 0.93) <= 1e-9

    def test_noul_below_half_flips_the_label(self) -> None:
        choice, probs, confidence = choice_from({"type": "noul", "noul": 0.2}, [])
        assert choice == "false"
        assert abs(confidence - 0.8) <= 1e-9
        assert probs["false"] == 0.8

    def test_noul_from_probabilities_only(self) -> None:
        choice, probs, _ = choice_from(
            {"type": "noul", "probabilities": {"true": 0.6, "false": 0.4}}, []
        )
        assert choice == "true" and probs["true"] == 0.6

    def test_noul_without_any_probability_raises(self) -> None:
        with pytest.raises(DecisionFailed, match="refusing to invent"):
            choice_from({"type": "noul"}, [])

    # ---------------------------------------------------------------- choice
    def test_choice_reads_choice_confidence_and_probabilities(self) -> None:
        payload = {"type": "choice", "choice": "billing", "confidence": 0.71,
                   "probabilities": {"billing": 0.71, "technical": 0.29}}
        choice, probs, confidence = choice_from(payload, ["billing", "technical"])
        assert choice == "billing"
        assert probs == {"billing": 0.71, "technical": 0.29}
        assert abs(confidence - 0.71) <= 1e-9

    def test_choice_missing_confidence_is_derived_from_the_distribution(self) -> None:
        payload = {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.64, "technical": 0.36}}
        _, _, confidence = choice_from(payload, ["billing", "technical"])
        assert abs(confidence - 0.64) <= 1e-9

    def test_choice_missing_type_is_inferred(self) -> None:
        payload = {"choice": "billing", "probabilities": {"billing": 0.6, "technical": 0.4}}
        choice, probs, _ = choice_from(payload, ["billing", "technical"])
        assert choice == "billing" and abs(sum(probs.values()) - 1.0) <= 1e-4

    def test_choice_missing_choice_falls_back_to_the_argmax(self) -> None:
        payload = {"type": "choice",
                   "probabilities": {"billing": 0.2, "technical": 0.8}}
        choice, _, confidence = choice_from(payload, ["billing", "technical"])
        assert choice == "technical" and abs(confidence - 0.8) <= 1e-9

    def test_choice_without_probabilities_is_refused(self) -> None:
        """The central no-fabrication rule: a label is not a distribution."""
        with pytest.raises(DecisionFailed, match="fabricated distribution"):
            choice_from({"type": "choice", "choice": "billing", "confidence": 0.9},
                        ["billing", "technical"])

    def test_choice_outside_the_offered_set_is_refused(self) -> None:
        payload = {"type": "choice", "choice": "legal",
                   "probabilities": {"legal": 0.9, "billing": 0.1}}
        with pytest.raises(DecisionFailed, match="not one of"):
            choice_from(payload, ["billing", "technical"])

    def test_choice_drops_probability_keys_the_request_never_offered(self) -> None:
        payload = {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.6, "technical": 0.3, "legal": 0.1}}
        choice, probs, _ = choice_from(payload, ["billing", "technical"])
        assert choice == "billing"
        assert set(probs) == {"billing", "technical"}
        assert abs(sum(probs.values()) - 1.0) <= 1e-4

    def test_choice_with_no_overlap_is_refused(self) -> None:
        payload = {"type": "choice", "choice": "legal",
                   "probabilities": {"legal": 1.0}}
        with pytest.raises(DecisionFailed, match="share no key"):
            choice_from(payload, ["billing", "technical"])

    def test_confidence_is_capped_by_the_chosen_probability(self) -> None:
        """``Decision`` forbids confidence > p(choice); renormalisation can
        legitimately move p(choice) under a reported confidence."""
        payload = {"type": "choice", "choice": "a", "confidence": 0.95,
                   "probabilities": {"a": 0.4, "b": 0.4, "c": 0.2}}
        choice, probs, confidence = choice_from(payload, ["a", "b", "c"])
        assert choice == "a"
        assert confidence <= probs["a"] + 1e-9

    # ----------------------------------------------------------------- score
    def test_score_maps_ordinal_keys_through_the_legend(self) -> None:
        payload = {"type": "score", "score": "major", "confidence": 0.66,
                   "legend": {"0": "none", "1": "minor", "2": "major"},
                   "probabilities": {"0": 0.08, "1": 0.26, "2": 0.66}}
        choice, probs, confidence = choice_from(payload, ["none", "minor", "major"])
        assert choice == "major"
        assert probs == {"none": 0.08, "minor": 0.26, "major": 0.66}
        assert abs(confidence - 0.66) <= 1e-9

    def test_score_accepts_label_keyed_probabilities(self) -> None:
        payload = {"type": "score", "score": "minor",
                   "probabilities": {"none": 0.2, "minor": 0.5, "major": 0.3}}
        choice, probs, _ = choice_from(payload, ["none", "minor", "major"])
        assert choice == "minor" and probs["minor"] == 0.5

    def test_score_without_probabilities_is_refused(self) -> None:
        with pytest.raises(DecisionFailed, match="carried no 'probabilities'"):
            choice_from({"type": "score", "score": "major",
                         "legend": {"0": "none", "1": "minor", "2": "major"}},
                        ["none", "minor", "major"])

    def test_score_level_outside_the_rubric_is_refused(self) -> None:
        payload = {"type": "score", "score": "catastrophic",
                   "legend": {"0": "catastrophic"},
                   "probabilities": {"0": 1.0}}
        with pytest.raises(DecisionFailed, match="not in the requested rubric"):
            choice_from(payload, ["none", "minor", "major"])

    def test_score_unmappable_key_is_refused(self) -> None:
        payload = {"type": "score", "score": "none",
                   "probabilities": {"7": 0.5, "9": 0.5}}
        with pytest.raises(DecisionFailed, match="no rubric level for key"):
            choice_from(payload, ["none", "minor", "major"])

    # ------------------------------------------------------------- degenerate
    @pytest.mark.parametrize("payload", [None, {}, [], "text", 7])
    def test_absent_answer_returns_none_rather_than_a_guess(self, payload: Any) -> None:
        assert choice_from(payload, ["a", "b"]) == (None, {}, 0.0)

    def test_unidentifiable_answer_raises(self) -> None:
        with pytest.raises(DecisionFailed, match="cannot tell what kind"):
            choice_from({"explanation": "because"}, ["a", "b"])


# ==============================================================================
# request construction
# ==============================================================================
class TestRequestConstruction:
    def test_response_key_is_stable_and_readable(self) -> None:
        first = build_response_key(choice_request())
        second = build_response_key(choice_request(request_id="req_other"))
        assert first == second, "the key must depend on the question, not the request id"
        assert first.startswith("q_which_team_owns_this_issue")
        assert build_response_key(choice_request(question="something else")) != first

    def test_response_key_is_json_object_safe(self) -> None:
        key = build_response_key(choice_request(question='weird "key" / \\ chars?'))
        assert json.loads(json.dumps({key: 1})) == {key: 1}

    def test_body_maps_state_options_and_rubric(self) -> None:
        from decision.clef import build_body

        key, body = build_body(choice_request())
        assert set(body) == {"state", "questions"}
        assert json.loads(body["state"]) == {"text": "the invoice is wrong"}
        question = body["questions"][key]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"billing", "technical"}
        assert question["instructions"] == "Which team owns this issue?"

        key, body = build_body(score_request())
        assert body["questions"][key]["criteria"] == ["none", "minor", "major"]

        key, body = build_body(noul_request())
        assert body["questions"][key]["type"] == "noul"
        assert "criteria" not in body["questions"][key]

    def test_richer_option_descriptions_are_used_when_supplied(self) -> None:
        from decision.clef import build_body

        request = choice_request(
            state={"criteria": {"billing": "invoices", "technical": "outages"}}
        )
        _, body = build_body(request)
        key = next(iter(body["questions"]))
        assert body["questions"][key]["criteria"] == {
            "billing": "invoices", "technical": "outages"
        }

    def test_custom_instructions_win_over_the_question(self) -> None:
        from decision.clef import build_body

        _, body = build_body(choice_request(instructions="  pick the owning team  "))
        key = next(iter(body["questions"]))
        assert body["questions"][key]["instructions"] == "pick the owning team"


# ==============================================================================
# wrong-endpoint guard
# ==============================================================================
class TestWrongEndpointGuard:
    def test_require_answers_rejects_a_chat_completion(self) -> None:
        payload = {"choices": [{"message": {"content": "The sponsor seems interested."}}],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 7}}
        with pytest.raises(DecisionFailed) as excinfo:
            require_answers(payload, "q_x", backend="clef", endpoint="/v1/systemone")
        message = str(excinfo.value)
        assert "/v1/systemone" in message
        assert "/v1/chat/completions" in message
        assert "chat completion" in message

    def test_require_answers_rejects_an_html_error_page(self) -> None:
        with pytest.raises(DecisionFailed, match="/v1/systemone"):
            require_answers({"detail": "Not Found"}, "q_x", backend="clef",
                            endpoint="/v1/systemone")

    def test_require_answers_names_the_question_key_it_asked(self) -> None:
        with pytest.raises(DecisionFailed, match="q_asked"):
            require_answers({"answers": {"q_other": {"type": "noul", "noul": 0.5}}},
                            "q_asked", backend="clef", endpoint="/v1/systemone")

    def test_clef_decide_raises_on_a_chat_completion(self) -> None:
        handler = fake_systemone()
        backend = clef_backend(lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "looks interested"}}]}))

        def dispatch(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return handler(request)
            return httpx.Response(200, json={"choices": [
                {"message": {"content": "looks interested"}}]})

        backend = clef_backend(dispatch)
        with pytest.raises(DecisionFailed) as excinfo:
            backend.decide(choice_request())
        assert "/v1/systemone" in str(excinfo.value)

    def test_clef_backend_does_not_read_logprobs(self) -> None:
        """A chat route offers logprobs; using them would fake a calibration."""
        payload = {"choices": [{"logprobs": {"content": [
            {"token": "billing", "logprob": -0.31}]}}]}
        with pytest.raises(DecisionFailed, match="/v1/systemone"):
            require_answers(payload, "q_x", backend="clef", endpoint="/v1/systemone")


# ==============================================================================
# clef
# ==============================================================================
class TestClefBackend:
    def test_decides_a_choice_question(self) -> None:
        backend = clef_backend(fake_systemone())
        decision_out = backend.decide(choice_request())
        assert decision_out.source is DecisionSource.CLEF
        assert decision_out.model == "clef-flash"
        assert decision_out.choice == "billing"
        assert decision_out.confidence == pytest.approx(0.7)
        assert abs(sum(decision_out.probabilities.values()) - 1.0) <= 1e-4
        assert decision_out.degraded is False

    def test_decides_a_score_question_through_the_legend(self) -> None:
        backend = clef_backend(fake_systemone())
        decision_out = backend.decide(score_request())
        assert decision_out.choice == "major"
        assert set(decision_out.probabilities) == {"none", "minor", "major"}
        assert decision_out.probabilities["major"] > decision_out.probabilities["none"]

    def test_decides_a_noul_question_without_a_confidence_field(self) -> None:
        backend = clef_backend(fake_systemone())
        decision_out = backend.decide(noul_request())
        assert decision_out.choice == "true"
        assert decision_out.probabilities == {"true": 0.93, "false": 0.07}
        assert decision_out.confidence == pytest.approx(0.93)

    def test_decision_invariants_hold_for_every_question_type(self) -> None:
        backend = clef_backend(fake_systemone())
        for request in (choice_request(), score_request(), noul_request()):
            out = backend.decide(request)
            assert out.choice in out.probabilities
            assert out.confidence <= out.probabilities[out.choice] + 1e-6
            assert abs(sum(out.probabilities.values()) - 1.0) <= 1e-4
            assert out.request_id == request.request_id
            assert out.question == request.question

    def test_raw_preserves_the_server_body_and_names_our_metadata(self) -> None:
        backend = clef_backend(fake_systemone())
        out = backend.decide(choice_request())
        assert out.raw is not None
        assert out.raw["usage"] == {"input_tokens": 412, "output_tokens": 0}
        assert out.raw["_transport"] == "clef/llama.cpp"
        assert out.raw["_endpoint"].endswith("/v1/systemone")
        assert out.raw["_question_key"].startswith("q_")

    def test_output_tokens_zero_is_recorded_not_assumed(self) -> None:
        """Clef is prefill-only; a non-zero output_tokens would mean text generation."""
        backend = clef_backend(fake_systemone())
        out = backend.decide(choice_request())
        assert out.raw is not None
        assert out.raw["usage"]["output_tokens"] == 0

    def test_a_malformed_answer_becomes_decision_failed(self) -> None:
        from decision.base import build_response_key

        key = build_response_key(choice_request())
        backend = clef_backend(fake_systemone(
            overrides={key: {"type": "choice", "choice": "billing"}}))
        with pytest.raises(DecisionFailed, match="fabricated distribution"):
            backend.decide(choice_request())

    # ---------------------------------------------------------------- probing
    def test_available_requires_a_real_probe(self) -> None:
        calls: list[str] = []
        inner = fake_systemone()

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return inner(request)

        backend = clef_backend(handler)
        ok, reason = backend.available()
        assert ok is True
        assert "/v1/systemone" in reason
        assert "/v1/models" in calls, "probe must actually hit the server"

    def test_available_is_false_when_nothing_listens(self) -> None:
        def refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        ok, reason = clef_backend(refused).available()
        assert ok is False
        assert "not reachable" in reason or "cannot reach" in reason

    def test_probe_result_is_cached_for_the_configured_ttl(self) -> None:
        inner = fake_systemone()
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return inner(request)

        settings = Settings(clef_backend="ollama", clef_base_url="http://localhost:8080",
                            clef_health_ttl_s=30.0)
        backend = clef_backend(handler, settings=settings)
        backend.available()
        first = len(calls)
        backend.available()
        assert len(calls) == first, "a cached probe must not re-hit the network"

    def test_probe_cache_expires(self) -> None:
        inner = fake_systemone()
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return inner(request)

        settings = Settings(clef_backend="ollama", clef_base_url="http://localhost:8080",
                            clef_health_ttl_s=0.0)
        backend = clef_backend(handler, settings=settings)
        backend.available()
        first = len(calls)
        backend.available()
        assert len(calls) > first

    def test_model_is_discovered_from_the_server(self) -> None:
        backend = clef_backend(fake_systemone())
        backend.available()
        assert backend.model == "/models/clef-flash.Q4_K_M.gguf"
        assert backend.health()["discovered_model"] is True
        # Decision.model always reports the configured name.
        assert backend.decide(choice_request()).model == "clef-flash"

    def test_model_falls_back_to_the_configured_label(self) -> None:
        backend = clef_backend(fake_systemone(models=False))
        backend.available()
        assert backend.model == "clef-flash"
        assert backend.health()["discovered_model"] is False

    def test_probe_rejects_a_server_that_only_speaks_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(404, json={"error": "nope"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        ok, reason = clef_backend(handler).available()
        assert ok is False
        assert "/v1/systemone" in reason

    # --------------------------------------------------------- error taxonomy
    def test_timeout_is_decision_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json=FAKE_MODELS)
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(DecisionUnavailable, match="timed out"):
            clef_backend(handler).decide(choice_request())

    def test_connection_refused_is_decision_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(DecisionUnavailable, match="cannot reach"):
            clef_backend(handler).decide(choice_request())

    def test_server_error_is_decision_unavailable_and_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(503, json={"error": "model loading"})

        with pytest.raises(DecisionUnavailable, match="503"):
            clef_backend(handler, max_retries=1).decide(choice_request())
        assert len(attempts) == 2, "a 5xx must consume the retry budget"

    def test_client_error_is_decision_failed_and_not_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(400, json={"error": "malformed questions"})

        with pytest.raises(DecisionFailed, match="400"):
            clef_backend(handler, max_retries=2).decide(choice_request())
        assert len(attempts) == 1, "retrying a 400 only wastes the run's wall clock"

    def test_a_transient_failure_is_retried_then_succeeds(self) -> None:
        state = {"n": 0}
        inner = fake_systemone()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != "/v1/models":
                state["n"] += 1
                if state["n"] == 1:
                    raise httpx.ConnectError("reset by peer", request=request)
            return inner(request)

        out = clef_backend(handler, max_retries=1).decide(choice_request())
        assert out.choice == "billing"
        assert out.source is DecisionSource.CLEF

    def test_non_json_body_is_decision_failed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json=FAKE_MODELS)
            return httpx.Response(200, text="<html>hello</html>")

        with pytest.raises(DecisionFailed, match="non-JSON"):
            clef_backend(handler).decide(choice_request())

    def test_every_call_carries_a_timeout(self) -> None:
        """An unbounded wait inside an agent step budget is a hang, not a wait."""
        settings = Settings(clef_timeout_s=7.5, decision_max_retries=1)
        transport = LlamaCppTransport("http://localhost:8080", timeout_s=settings.clef_timeout_s,
                                      max_retries=1)
        assert transport.timeout_s == 7.5
        assert transport.health()["timeout_s"] == 7.5

    # ------------------------------------------------------- base url rewriting
    def test_ollamas_default_port_is_moved_to_llama_cpp(self) -> None:
        """Ollama's clef-flash is broken on Windows; llama-server is on 18781."""
        transport = LlamaCppTransport("http://localhost:11434")
        assert transport.base_url == "http://localhost:18781"
        assert "ollama/ollama#18769" in transport.note

    def test_a_non_ollama_url_is_left_alone(self) -> None:
        transport = LlamaCppTransport("http://gpu-box.lan:9000/")
        assert transport.base_url == "http://gpu-box.lan:9000"
        assert transport.note == ""

    def test_endpoint_is_always_the_systemone_route(self) -> None:
        transport = LlamaCppTransport("http://localhost:8080")
        assert transport.endpoint == "http://localhost:8080/v1/systemone"


class TestWorkersAiTransport:
    def _backend(self, handler: Callable[[httpx.Request], httpx.Response],
                 **overrides: Any) -> ClefBackend:
        settings = Settings(clef_backend="workers_ai", clef_account_id="acct123",
                            clef_api_key="secret-token", **overrides)
        transport = WorkersAiTransport("acct123", "secret-token", model_label="clef-flash",
                                      timeout_s=5, max_retries=0, backoff_s=0.0,
                                      transport=httpx.MockTransport(handler))
        return ClefBackend(settings, transports={"workers_ai": transport}, backoff_s=0.0)

    def test_url_and_bearer_token(self) -> None:
        seen: dict[str, Any] = {}
        inner = fake_systemone(envelope=True)

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            return inner(request)

        backend = self._backend(handler)
        out = backend.decide(noul_request())
        assert seen["url"] == (
            "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/"
            "@cf/cloudflare/clef-flash"
        )
        assert seen["auth"] == "Bearer secret-token"
        assert out.source is DecisionSource.CLEF

    def test_cloudflare_envelope_is_unwrapped(self) -> None:
        backend = self._backend(fake_systemone(envelope=True))
        out = backend.decide(choice_request())
        assert out.choice == "billing"
        assert out.raw is not None and "answers" in out.raw

    def test_provider_errors_are_surfaced_not_swallowed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "result": None, "success": False,
                "errors": [{"code": 7003, "message": "no access to this model"}],
                "messages": []})

        with pytest.raises(DecisionFailed) as excinfo:
            self._backend(handler).decide(choice_request())
        message = str(excinfo.value)
        assert "@cf/cloudflare/clef-flash" in message
        assert "no access to this model" in message

    def test_missing_credentials_is_decision_unavailable(self) -> None:
        settings = Settings(clef_backend="workers_ai", clef_account_id="acct123",
                            clef_api_key=None)
        backend = ClefBackend(settings)
        ok, reason = backend.available()
        assert ok is False
        assert "clef_api_key" in reason and "CLEF_API_KEY" in reason

    def test_health_never_leaks_the_token(self) -> None:
        backend = self._backend(fake_systemone(envelope=True))
        blob = json.dumps(backend.health(), default=str)
        assert "secret-token" not in blob
        assert backend.health()["transport_health"]["authorised"] is True


# ==============================================================================
# gemini
# ==============================================================================
class FakeGeminiClient:
    """Scripted ``google-genai`` client. Each script entry is a response string
    or an exception to raise."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Models:
            calls = outer.calls

            def generate_content(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                item = outer.script.pop(0)
                if isinstance(item, BaseException):
                    raise item
                response = type("R", (), {})()
                response.text = item
                return response

        self.models = _Models()


class TestGeminiBackend:
    def _backend(self, script: list[Any], **overrides: Any) -> GeminiBackend:
        settings = Settings(gemini_api_key="test-key", **overrides)
        return GeminiBackend(settings,
                             client_factory=lambda key: FakeGeminiClient(script))

    def test_no_key_is_unavailable_with_a_clear_reason(self) -> None:
        ok, reason = GeminiBackend(Settings(gemini_api_key=None)).available()
        assert ok is False
        assert "GEMINI_API_KEY" in reason

    def test_choice_uses_the_distribution_the_model_supplied(self) -> None:
        backend = self._backend(['{"choice": "pushback", "reasoning": "price objection",'
                                 ' "probabilities": {"yes": 0.1, "pushback": 0.7,'
                                 ' "interested": 0.2}}'])
        out = backend.decide(choice_request(options=["yes", "pushback", "interested"]))
        assert out.source is DecisionSource.GEMINI
        assert out.choice == "pushback"
        assert out.confidence == pytest.approx(0.7)
        assert out.degraded is False
        assert out.raw is not None and out.raw["_distribution"] == "model_supplied"

    def test_a_label_without_a_distribution_is_degraded(self) -> None:
        """A fabricated uniform distribution must be labelled, never shipped quietly."""
        backend = self._backend(['{"choice": "pushback", "reasoning": "price objection"}'])
        out = backend.decide(choice_request(options=["yes", "pushback", "interested"]))
        assert out.choice == "pushback"
        assert out.probabilities["pushback"] == pytest.approx(1.0)
        assert out.probabilities["yes"] == pytest.approx(0.0, abs=1e-6)
        assert abs(sum(out.probabilities.values()) - 1.0) <= 1e-4
        assert out.degraded is True
        assert out.needs_escalation is True
        assert out.raw is not None
        assert out.raw["_distribution"] == "synthesised_uniform"

    def test_noul_schema_and_labels(self) -> None:
        backend = self._backend(['{"yes": false, "reasoning": "no deadline mentioned",'
                                 ' "probabilities": {"true": 0.15, "false": 0.85}}'])
        out = backend.decide(noul_request())
        assert out.choice == "false"
        assert out.probabilities["false"] == pytest.approx(0.85)
        assert out.degraded is False

    def test_score_answers_the_rubric(self) -> None:
        backend = self._backend(['{"level": "major", "reasoning": "missed deliverable",'
                                 ' "probabilities": {"none": 0.05, "minor": 0.2,'
                                 ' "major": 0.75}}'])
        out = backend.decide(score_request())
        assert out.choice == "major"
        assert set(out.probabilities) == {"none", "minor", "major"}
        assert out.degraded is False

    def test_the_schema_constrains_the_answer_space(self) -> None:
        backend = self._backend(['{"choice": "pushback", "reasoning": "x"}'])
        backend.decide(choice_request(options=["yes", "pushback", "interested"]))
        config = backend._client.models.calls[0]["config"]
        assert config.response_mime_type == "application/json"
        schema = config.response_schema
        assert schema["properties"]["choice"]["enum"] == ["yes", "pushback", "interested"]
        assert "reasoning" in schema["required"]
        assert "probabilities" not in schema["required"]

    def test_a_choice_outside_the_offered_set_is_refused(self) -> None:
        backend = self._backend(['{"choice": "legal", "reasoning": "x"}'])
        with pytest.raises(DecisionFailed, match="not one of the offered"):
            backend.decide(choice_request(options=["yes", "pushback"]))

    def test_non_json_output_is_decision_failed(self) -> None:
        backend = self._backend(["The sponsor is pushing back on price."])
        with pytest.raises(DecisionFailed, match="not JSON"):
            backend.decide(choice_request())

    def test_a_fenced_code_block_is_tolerated(self) -> None:
        backend = self._backend(['```json\n{"choice": "billing", "reasoning": "x"}\n```'])
        out = backend.decide(choice_request())
        assert out.choice == "billing"

    def test_a_model_specific_client_error_uses_the_fallback_model(self) -> None:
        from google.genai.errors import ClientError

        failure = ClientError(404, {"error": {"message": "model not found"}})
        backend = self._backend([failure,
                                 '{"choice": "billing", "reasoning": "x"}'],
                                gemini_model="gemini-9-ultra",
                                gemini_fallback_model="gemini-2.5-flash")
        out = backend.decide(choice_request())
        assert [call["model"] for call in backend._client.models.calls] == [
            "gemini-9-ultra", "gemini-2.5-flash"]
        assert out.model == "gemini-2.5-flash"
        assert out.choice == "billing"

    def test_a_server_error_does_not_burn_a_second_model(self) -> None:
        from google.genai.errors import ServerError

        backend = self._backend([ServerError(503, {"error": {"message": "overloaded"}})])
        with pytest.raises(DecisionUnavailable, match="failed"):
            backend.decide(choice_request())
        assert len(backend._client.models.calls) == 1

    def test_a_transport_failure_becomes_decision_unavailable(self) -> None:
        backend = self._backend([httpx.ConnectError("connection refused")])
        with pytest.raises(DecisionUnavailable, match="transport failure"):
            backend.decide(choice_request())

    def test_a_timeout_becomes_decision_unavailable(self) -> None:
        backend = self._backend([httpx.ReadTimeout("too slow")])
        with pytest.raises(DecisionUnavailable, match="timed out"):
            backend.decide(choice_request())

    def test_an_empty_response_is_decision_failed(self) -> None:
        backend = self._backend([""])
        with pytest.raises(DecisionFailed, match="no text"):
            backend.decide(choice_request())

    def test_reasoning_is_recorded_in_raw(self) -> None:
        backend = self._backend(['{"choice": "billing", "reasoning": "invoice dispute"}'])
        out = backend.decide(choice_request())
        assert out.raw is not None
        assert out.raw["_reasoning"] == "invoice dispute"


# ==============================================================================
# rules
# ==============================================================================
class TestRulesBackend:
    def test_always_available(self) -> None:
        ok, reason = RulesBackend(Settings()).available()
        assert ok is True
        assert "ceiling" in reason

    # --------------------------------------------------------- the regression
    def test_yesterday_we_thought_the_price_was_too_high_is_pushback(self) -> None:
        """The prototype bug, pinned.

        The old router tested ``"yes"`` before pushback. This sentence is a
        retrospective price complaint: routed to assent it would send the thread
        into contract signing. It must classify as pushback.
        """
        backend = RulesBackend(Settings())
        sentence = "yesterday we thought the price was too high"
        out = backend.decide(intent_request(sentence))
        assert out.choice == Intent.PUSHBACK.value
        assert out.choice != Intent.YES.value
        assert out.choice not in ("sign", "contract", "yes")

    @pytest.mark.parametrize("sentence,expected", [
        ("yesterday we thought the price was too high", Intent.PUSHBACK),
        ("we cannot commit to that budget", Intent.NO),
        ("Yes, sign the MoU", Intent.YES),
        ("Sounds interesting, please share the deck", Intent.INTERESTED),
        ("No thanks, not interested at this time", Intent.NO),
        ("We are reviewing internally, no decision yet", Intent.NEUTRAL),
        ("", Intent.UNKNOWN),
    ])
    def test_intent_delegates_to_the_frozen_helper(self, sentence: str,
                                                   expected: Intent) -> None:
        from core.protocols import classify_intent_fallback

        out = RulesBackend(Settings()).decide(intent_request(sentence))
        assert out.choice == classify_intent_fallback(sentence).value == expected.value

    def test_intent_decision_is_not_degraded_but_is_low_confidence(self) -> None:
        out = RulesBackend(Settings()).decide(intent_request("the price was too high"))
        assert out.source is DecisionSource.RULES
        assert out.degraded is False
        assert out.confidence <= CONFIDENCE_CEILING
        assert out.confidence < Settings().confidence_threshold

    # ------------------------------------------------------------- the ceiling
    def test_confidence_never_exceeds_the_ceiling(self) -> None:
        backend = RulesBackend(Settings())
        requests = [
            intent_request("yesterday we thought the price was too high"),
            choice_request(options=["proceed", "escalate", "hold"]),
            score_request(rubric=["none", "low", "medium", "high", "critical"],
                          state={"signals": ["critical"]}),
            noul_request(question="Is this request urgent?",
                         state={"text": "urgent, asap, today"}),
        ]
        for request in requests:
            out = backend.decide(request)
            assert out.confidence <= CONFIDENCE_CEILING + 1e-9, request.question
            assert max(out.probabilities.values()) <= CONFIDENCE_CEILING + 1e-9

    def test_the_ceiling_sits_in_the_escalation_band(self) -> None:
        """Above ``needs_escalation``'s 0.5, below ``confidence_threshold``'s 0.62."""
        settings = Settings()
        assert 0.5 < CONFIDENCE_CEILING < settings.confidence_threshold

    def test_a_custom_ceiling_is_honoured(self) -> None:
        backend = RulesBackend(Settings(), ceiling=0.3)
        out = backend.decide(intent_request("too expensive"))
        assert out.confidence <= 0.3 + 1e-9

    def test_every_rules_distribution_sums_to_one(self) -> None:
        backend = RulesBackend(Settings())
        for request in (intent_request("too expensive"), score_request(),
                        noul_request(question="Is this risky?")):
            out = backend.decide(request)
            assert abs(sum(out.probabilities.values()) - 1.0) <= 1e-4
            assert out.choice in out.probabilities
            assert out.confidence <= out.probabilities[out.choice] + 1e-6

    def test_rules_never_touch_the_network(self) -> None:
        """The autouse ``no_network`` fixture makes this a real assertion."""
        out = RulesBackend(Settings()).decide(intent_request("the price was too high"))
        assert out.source is DecisionSource.RULES
        assert out.model == "rules-v1"

    # ------------------------------------------------------------- derivations
    def test_score_answer_is_peaked_on_the_signalled_level(self) -> None:
        out = RulesBackend(Settings()).decide(
            score_request(state={"signals": ["critical", "reputational"]},
                          rubric=["none", "low", "medium", "high", "critical"]))
        assert out.choice == "critical"
        assert out.probabilities["critical"] == max(out.probabilities.values())
        assert out.raw is not None and out.raw["_method"] == "score:signals"

    def test_score_answer_without_signals_is_the_lowest_ambiguous_level(self) -> None:
        out = RulesBackend(Settings()).decide(score_request(state={}))
        assert out.choice == "minor"
        assert out.raw is not None and out.raw["_method"] == "score:no-signal"

    def test_score_answer_honours_an_explicit_ordinal(self) -> None:
        out = RulesBackend(Settings()).decide(score_request(state={"severity": 2}))
        assert out.choice == "major"

    def test_score_answer_honours_level_flags(self) -> None:
        out = RulesBackend(Settings()).decide(score_request(state={"major": True}))
        assert out.choice == "major"

    def test_escalation_verdict_is_driven_by_blocking_signals(self) -> None:
        backend = RulesBackend(Settings())
        request = choice_request(options=["proceed", "escalate"],
                                 state={"blocking": True})
        out = backend.decide(request)
        assert out.choice == "escalate"
        assert out.raw is not None
        assert any("blocking" in reason
                   for reason in out.raw["_signals"]["escalation_reasons"])

    def test_escalation_verdict_proceeds_without_them(self) -> None:
        out = RulesBackend(Settings()).decide(
            choice_request(options=["proceed", "escalate"], state={"blocking": False}))
        assert out.choice == "proceed"

    def test_explicit_option_signals_are_normalised_across_options(self) -> None:
        out = RulesBackend(Settings()).decide(choice_request(
            options=["billing", "technical"],
            state={"option_signals": {"billing": 3, "technical": 1}}))
        assert out.choice == "billing"
        assert out.raw is not None and out.raw["_method"] == "choice:option_signals"

    def test_option_signals_with_no_evidence_refuse_rather_than_guess(self) -> None:
        with pytest.raises(DecisionFailed, match="no positive evidence"):
            RulesBackend(Settings()).decide(choice_request(
                state={"option_signals": {"billing": 0, "technical": 0}}))

    def test_unknown_choice_falls_back_to_documented_lexical_overlap(self) -> None:
        out = RulesBackend(Settings()).decide(choice_request(
            options=["northern-region", "southern-region"],
            state={"note": "the sponsor is based in the northern region"}))
        assert out.choice == "northern-region"
        assert out.raw is not None and out.raw["_method"] == "choice:lexical"

    def test_the_derivation_is_recorded_for_review(self) -> None:
        out = RulesBackend(Settings()).decide(intent_request("too expensive"))
        assert out.raw is not None
        assert out.raw["_method"] == "choice:intent"
        assert out.raw["_signals"]["intent"] == Intent.PUSHBACK.value
        assert "not calibrated" in out.raw["_calibration"]


# ==============================================================================
# registry
# ==============================================================================
class TestDecisionRegistry:
    def test_head_answers_and_is_not_degraded(self) -> None:
        clef = StubBackend("clef")
        registry = DecisionRegistry(Settings(decision_backend="clef"),
                                    clef=clef, gemini=StubBackend("gemini"))
        out = registry.decide(choice_request())
        assert out.source is DecisionSource.CLEF
        assert out.degraded is False
        assert clef.calls == 1

    def test_falls_from_clef_to_gemini_and_says_so(self) -> None:
        registry = DecisionRegistry(
            Settings(decision_backend="clef"),
            clef=StubBackend("clef", error=DecisionUnavailable("no llama-server")),
            gemini=StubBackend("gemini"), rules=StubBackend("rules"))
        out = registry.decide(choice_request())
        assert out.source is DecisionSource.GEMINI
        assert out.degraded is True
        assert out.needs_escalation is True
        assert out.raw is not None
        assert out.raw["_degradation"]["configured_head"] == "clef"
        assert out.raw["_degradation"]["answered_by"] == "gemini"

    def test_falls_all_the_way_to_rules_and_never_fabricates_a_source(self) -> None:
        registry = DecisionRegistry(
            Settings(decision_backend="clef"),
            clef=StubBackend("clef", error=DecisionUnavailable("llama-server is not running")),
            gemini=StubBackend("gemini", error=DecisionFailed("garbage")),
            rules=StubBackend("rules"))
        out = registry.decide(choice_request())
        assert out.source is DecisionSource.RULES
        assert out.degraded is True
        tried = out.raw["_degradation"]["tried"]
        assert [entry["backend"] for entry in tried] == ["clef", "gemini"]
        assert "llama-server is not running" in tried[0]["error"]
        assert "garbage" in tried[1]["error"]

    def test_an_unavailable_backend_is_skipped_without_calling_it(self) -> None:
        clef = StubBackend("clef", available=False)
        registry = DecisionRegistry(Settings(decision_backend="clef"), clef=clef,
                                    gemini=StubBackend("gemini"))
        out = registry.decide(choice_request())
        assert out.source is DecisionSource.GEMINI
        assert clef.calls == 0, "an unavailable backend must not be called"

    def test_degradation_is_logged_structurally(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = DecisionRegistry(
            Settings(decision_backend="clef"),
            clef=StubBackend("clef", error=DecisionUnavailable("llama-server is not running")),
            gemini=StubBackend("gemini"), rules=StubBackend("rules"))
        with caplog.at_level("WARNING", logger="paytriq.decision"):
            registry.decide(choice_request(request_id="req_logged"))
        records = [r for r in caplog.records if "decision_degraded" in r.getMessage()]
        assert records, "a degradation must be logged"
        payload = json.loads(records[0].getMessage().split(" ", 1)[1])
        assert payload["event"] == "decision_degraded"
        assert payload["request_id"] == "req_logged"
        assert payload["fallback_used"] == "gemini"
        assert payload["configured_head"] == "clef"
        assert "llama-server is not running" in payload["tried"][0]["error"]
        assert payload["source"] == "gemini"
        assert records[0].decision_degradation == payload

    def test_a_clean_decision_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = DecisionRegistry(Settings(decision_backend="clef"),
                                    clef=StubBackend("clef"))
        with caplog.at_level("WARNING", logger="paytriq.decision"):
            registry.decide(choice_request())
        assert not [r for r in caplog.records if "decision_degraded" in r.getMessage()]

    def test_an_explicit_backend_is_the_head_and_keeps_capability_order_after_it(self) -> None:
        registry = DecisionRegistry(Settings(decision_backend="gemini"),
                                    clef=StubBackend("clef"), gemini=StubBackend("gemini"))
        assert [b.name for b in registry.chain()] == ["gemini", "clef", "rules"]
        out = registry.decide(choice_request())
        assert out.source is DecisionSource.GEMINI
        assert out.degraded is False

    def test_auto_prefers_clef_when_it_is_configured(self) -> None:
        registry = DecisionRegistry(Settings(decision_backend="auto",
                                             clef_backend="ollama"),
                                    clef=StubBackend("clef"), gemini=StubBackend("gemini"))
        assert [b.name for b in registry.chain()] == ["clef", "gemini", "rules"]

    def test_offline_mode_truncates_the_chain_to_rules(self) -> None:
        clef = StubBackend("clef")
        registry = DecisionRegistry(Settings(run_mode=RunMode.OFFLINE,
                                             decision_backend="clef"),
                                    clef=clef, gemini=StubBackend("gemini"))
        assert [b.name for b in registry.chain()] == ["rules"]
        out = registry.decide(choice_request())
        assert out.source is DecisionSource.RULES
        assert out.degraded is False, "rules as the intended primary is not a degradation"
        assert clef.calls == 0, "offline mode must not probe a network backend"

    def test_a_rules_decision_that_is_the_intended_backend_is_not_degraded(self) -> None:
        registry = DecisionRegistry(Settings(decision_backend="rules"),
                                    rules=RulesBackend(Settings()))
        out = registry.decide(choice_request(options=["proceed", "escalate"],
                                             state={"blocking": True}))
        assert out.source is DecisionSource.RULES
        assert out.degraded is False
        assert out.confidence <= CONFIDENCE_CEILING

    def test_a_total_failure_raises_rather_than_inventing_a_decision(self) -> None:
        registry = DecisionRegistry(
            Settings(run_mode=RunMode.OFFLINE),
            rules=StubBackend("rules", error=DecisionFailed("no evidence in the state")))
        with pytest.raises(DecisionUnavailable, match="every backend in the chain"):
            registry.decide(choice_request())

    def test_end_to_end_through_a_fake_clef_server(self) -> None:
        """The whole chain, wired the way the graph wires it."""
        settings = Settings(decision_backend="clef", clef_backend="ollama",
                            clef_base_url="http://localhost:8080")
        registry = build_with_clef(settings, fake_systemone())
        out = registry.decide(score_request())
        assert out.source is DecisionSource.CLEF
        assert out.choice == "major"
        assert out.degraded is False

    def test_end_to_end_degrades_to_rules_when_clef_is_down(self) -> None:
        def refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        settings = Settings(decision_backend="clef", clef_backend="ollama",
                            clef_base_url="http://localhost:8080")
        registry = build_with_clef(settings, refused)
        out = registry.decide(choice_request(options=["proceed", "escalate"],
                                             state={"blocking": True}))
        assert out.source is DecisionSource.RULES
        assert out.degraded is True
        assert out.raw is not None and "_degradation" in out.raw

    def test_health_aggregates_every_backend(self) -> None:
        registry = DecisionRegistry(
            Settings(decision_backend="clef"),
            clef=StubBackend("clef", available=False),
            gemini=StubBackend("gemini"), rules=StubBackend("rules"))
        report = registry.health()
        assert report["chain"] == ["clef", "gemini", "rules"]
        by_name = {entry["name"]: entry for entry in report["backends"]}
        assert by_name["clef"]["available"] is False
        assert by_name["gemini"]["in_chain"] is True
        assert by_name["rules"]["configured"] is False
        assert by_name["rules"]["available"] is True
        assert "gemini_key" in report["settings"]

    def test_health_can_skip_probing(self) -> None:
        clef = StubBackend("clef")
        registry = DecisionRegistry(Settings(decision_backend="clef"), clef=clef)
        report = registry.health(probe=False)
        assert report["chain"] == ["clef", "gemini", "rules"]
        assert clef.calls == 0

    def test_describe_lists_the_chain_and_availability(self) -> None:
        registry = DecisionRegistry(
            Settings(decision_backend="clef"),
            clef=StubBackend("clef", available=False),
            gemini=StubBackend("gemini"), rules=StubBackend("rules"))
        described = registry.describe()
        assert described["order"] == ["clef", "gemini", "rules"]
        assert [entry["role"] for entry in described["chain"]] == [
            "primary", "fallback", "fallback"]
        assert described["first_available"] == "gemini"
        assert described["configured"] == "clef"

    def test_registry_is_a_decision_backend(self) -> None:
        registry = DecisionRegistry(
            Settings(decision_backend="rules"),
            clef=StubBackend("clef"), gemini=StubBackend("gemini"),
            rules=StubBackend("rules"))
        assert isinstance(registry, DecisionBackend)
        ok, reason = registry.available()
        assert ok and "chain available" in reason

    def test_build_registry_wires_all_three_backends(self) -> None:
        """Construction must not touch the network -- only ``available()`` probes."""
        registry = build_registry(Settings(decision_backend="rules"))
        assert isinstance(registry, DecisionRegistry)
        assert sorted(registry._backends) == ["clef", "gemini", "rules"]
        assert isinstance(registry._backends["clef"], ClefBackend)
        assert isinstance(registry._backends["gemini"], GeminiBackend)
        assert isinstance(registry._backends["rules"], RulesBackend)


def build_with_clef(settings: Settings,
                    handler: Callable[[httpx.Request], httpx.Response]) -> DecisionRegistry:
    """Wire a registry whose clef backend is a mock transport."""
    transport = LlamaCppTransport(settings.clef_base_url, model_label=settings.clef_model,
                                  timeout_s=2, max_retries=0, backoff_s=0.0,
                                  transport=httpx.MockTransport(handler))
    return DecisionRegistry(settings, clef=ClefBackend(settings,
                                                       transports={"ollama": transport},
                                                       backoff_s=0.0))


# ==============================================================================
# package surface
# ==============================================================================
def test_public_surface_is_importable_and_documented() -> None:
    for name in decision.__all__:
        assert hasattr(decision, name), name
    for module in ("base", "clef", "gemini", "registry", "rules"):
        assert (Path(decision.__file__).parent / f"{module}.py").is_file()
