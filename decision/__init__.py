"""Paytriq decision layer: calibrated, always-sourced decisions.

Three backends behind one protocol
----------------------------------
``ClefBackend``   a System One **decision model** (llama.cpp locally, or Cloudflare
                  Workers AI). Closed questions answered with a calibrated
                  distribution. Endpoint: ``POST /v1/systemone`` -- not
                  ``/v1/chat/completions``, and never logprobs.
``GeminiBackend`` a generative model with JSON-schema structured output, for
                  judgements that need reasoning over text.
``RulesBackend``  deterministic, offline, confidence-capped at 0.55. The floor.

``DecisionRegistry`` chains them (clef -> gemini -> rules), falls back on
failure, and records on every ``Decision`` which one actually answered.

The invariant this package exists to protect: **a decision always names its
source, and a probability is either calibrated or absent.** Everything in
``base.py`` -- the clamping, the renormalisation, the refusal to invent a
distribution -- is in service of those two sentences.

Usage::

    from decision import build_registry
    decide = build_registry().decide        # this is what goes into AgentContext
"""
from __future__ import annotations

from .base import (
    NOUL_LABELS,
    PROBE_ENDPOINT,
    ProbeResult,
    SystemOneClient,
    build_response_key,
    choice_from,
    criteria_for_choice,
    logger,
    normalise_probabilities,
    require_answers,
    score_probabilities,
    state_text,
)
from .clef import (
    DEFAULT_LLAMA_CPP_BASE_URL,
    SYSTEMONE_PATH,
    WORKERS_AI_MODEL_PATH,
    ClefBackend,
    ClefTransport,
    LlamaCppTransport,
    WorkersAiTransport,
    build_body,
)
from .gemini import GeminiBackend
from .registry import CHAIN_ORDER, BackendStatus, DecisionRegistry, build_registry
from .rules import CONFIDENCE_CEILING, INTENT_VALUES, RulesBackend, classify_from_state

__all__ = [
    "__version__",
    # registry (the entry point agents use)
    "DecisionRegistry", "BackendStatus", "build_registry", "CHAIN_ORDER",
    # backends
    "ClefBackend", "GeminiBackend", "RulesBackend",
    # clef transports
    "ClefTransport", "LlamaCppTransport", "WorkersAiTransport",
    "SYSTEMONE_PATH", "DEFAULT_LLAMA_CPP_BASE_URL", "WORKERS_AI_MODEL_PATH",
    "PROBE_ENDPOINT", "build_body",
    # shared parsing
    "SystemOneClient", "ProbeResult",
    "normalise_probabilities", "choice_from", "score_probabilities",
    "build_response_key", "state_text", "criteria_for_choice",
    "require_answers", "NOUL_LABELS",
    # rules
    "CONFIDENCE_CEILING", "INTENT_VALUES", "classify_from_state",
    "logger",
]

__version__ = "0.1.0"
