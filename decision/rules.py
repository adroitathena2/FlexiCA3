"""Deterministic rules backend: the answer when there is no model.

This is the floor the whole system stands on. If it is wrong in a way that is
*visible*, the run degrades honestly; if it silently invented an answer, the
trace would be unauditable.

Three design commitments
------------------------

**1. Intent classification is delegated, never reimplemented.**
:func:`core.protocols.classify_intent_fallback` tests refusal and pushback
*before* assent. The previous prototype tested ``"yes"`` first, so the sentence
"yesterday we thought the price was too high" -- which contains no ``yes`` but
does contain "we thought", a near-miss for assent -- was classified as
acceptance and routed into contract signing. Delegating means this module cannot
reintroduce that ordering bug, because it does not own the ordering.
``test_decision.py`` pins the sentence so a future refactor cannot silently
reorder the checks.

**2. Every distribution is derived from the state, and the derivation is
recorded.** No keyword list invents a confidence. Each heuristic reads explicit
signals out of ``DecisionRequest.state`` and puts its reasoning in
``raw["_method"]`` and ``raw["_signals"]``, so a reviewer can see *why* the
number came out where it did and disagree with it.

**3. The confidence ceiling is 0.55, and that is the point.**
A rule has no calibration curve. It cannot know that a keyword it did find is
worth 0.62 and it cannot know that the absence of a keyword means 0.05. Any
number it produced above its evidence would be a lie with extra decimal places.
0.55 sits deliberately in a specific band:

* above :attr:`~core.schemas.Decision.needs_escalation`'s 0.5 cut-off, so a
  rules answer does not automatically halt the pipeline -- an offline demo must
  still be able to run end to end;
* below ``settings.confidence_threshold`` (0.62), so the arbiter's confidence
  gate treats it as low-confidence and routes it to escalation or a human rather
  than acting on it.

A ceiling of 1.0 would let a keyword match outrank a real model's careful
judgement. A ceiling of 0.3 would make offline runs unusable. 0.55 is the widest
honest band.
"""
from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from core.config import Settings, get_settings
from core.errors import DecisionFailed
from core.protocols import classify_intent_fallback
from core.schemas import Decision, DecisionRequest, DecisionSource, Intent, QuestionType

from .base import (
    NOUL_LABELS,
    annotate_raw,
    logger,
    make_decision,
    normalise_probabilities,
    snippet,
)

__all__ = [
    "RulesBackend",
    "CONFIDENCE_CEILING",
    "INTENT_VALUES",
    "DecisionShapeError",
    "classify_from_state",
]

#: Hard ceiling on any confidence this backend produces. See the module docstring
#: for why 0.55 and not 1.0.
CONFIDENCE_CEILING = 0.55

#: ``Intent`` values, used to recognise an intent question by its option set.
INTENT_VALUES: tuple[str, ...] = tuple(i.value for i in Intent)


class DecisionShapeError(DecisionFailed):
    """The state cannot support an answer to an otherwise well-formed question.

    A subclass of ``DecisionFailed`` so the registry's fallback chain treats it
    as "this backend could not answer" -- correctly, because a rules backend that
    answers anyway would be guessing, and a guess from the floor of the stack
    would be the least defensible decision in the whole system.
    """

#: State keys that plausibly hold the sponsor's own words. Checked in order.
_TEXT_KEYS: tuple[str, ...] = (
    "text", "reply_text", "reply", "message", "body", "content",
    "sponsor_reply", "email_body", "inbound",
)

#: Text keys for the noul heuristic ("is this urgent?", "is this risky?").
CONFIDENCE_HINTS: dict[str, tuple[str, ...]] = {
    "urgent": ("urgent", "asap", "immediately", "right away", "today",
               "deadline", "time-sensitive", "time sensitive", "quickly"),
    "risky": ("fraud", "lawsuit", "scam", "complaint", "regulator",
              "penalty", "breach", "non-compliance", "non compliance"),
    "blocking": ("blocking", "breach", "unfulfilled", "missing evidence",
                 "no evidence", "violation"),
    "ready": ("signed", "approved", "confirmed", "agreed", "go ahead",
              "proceed", "ready"),
    "confident": ("confirmed", "approved", "agreed", "signed"),
}

#: Generic score signals, checked when no question-specific table applies.
_GENERIC_LEVEL_HINTS: dict[str, tuple[str, ...]] = {
    "none": ("none", "no issue", "clean", "nothing", "compliant", "fulfilled"),
    "low": ("minor", "low", "small", "slight", "minor issue"),
    "medium": ("moderate", "medium", "partial", "some", "pending", "unclear"),
    "high": ("major", "high", "severe", "significant", "largely"),
    "critical": ("critical", "blocking", "fatal", "severe breach", "reputational"),
}

#: Option sets the choice heuristics understand, as normalised lowercase.
_OPTION_SIGNALS: dict[str, tuple[str, ...]] = {
    "escalate": ("escalate", "escalation", "human", "arbiter"),
    "hold": ("hold", "pause", "wait", "defer", "stop"),
    "proceed": ("proceed", "approve", "send", "continue", "accept", "sign"),
    "flag": ("flag", "risk", "violation", "block", "breach"),
}

#: Peaked-shape weights for a score answer: the centred level, its neighbours,
#: and the tail. A rubric answer is an assertion that the state matches one level
#: better than the others, so the honest shape is a peak, not a plateau.
_PEAK_WEIGHTS = (1.0, 0.45, 0.18, 0.08)


def _as_text(state: Mapping[str, Any]) -> str:
    """Concatenate every string leaf in the state, lowercased.

    Deliberately crude. The state is agent-supplied structured data whose shape
    is not this module's to know; flattening it to text means a heuristic
    written for one agent's state keys still works for another's, and the
    ceiling keeps the crudeness from mattering.
    """
    parts: list[str] = []
    for key, value in state.items():
        if key in ("criteria", "option_descriptions"):
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            parts.extend(v for v in value.values() if isinstance(v, str))
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if isinstance(v, str))
    return " ".join(parts).lower()


def _find_text(state: Mapping[str, Any]) -> str:
    """Best guess at the free text a question is about."""
    for key in _TEXT_KEYS:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _as_text(state)


def _peak(centre: int, size: int) -> dict[int, float]:
    """Weights for a distribution peaked at ``centre`` over ``size`` levels."""
    return {
        index: _PEAK_WEIGHTS[min(abs(index - centre), len(_PEAK_WEIGHTS) - 1)]
        for index in range(size)
    }


def _normalise_peak(weights: Mapping[str, float], ceiling: float) -> dict[str, float]:
    """Renormalise to sum 1 while forcing every entry at or below ``ceiling``.

    Scaling a normalised distribution down does **not** work: the sum has to
    stay 1, so dividing by the total undoes the scale and the peak lands back
    where it started. The honest construction is to *flatten* the peak toward
    uniform until it fits::

        beta = (p_max - ceiling) / (p_max - 1/n)
        q    = (1 - beta) * p + beta * (1/n)

    which sums to 1 and satisfies ``max(q) == ceiling`` exactly. A peak that had
    to be flattened is also a peak the evidence did not earn, so this reads
    correctly: the more confident the heuristic looked, the more the ceiling
    pulls it back toward "no idea".
    """
    keys = list(weights)
    if not keys:
        raise DecisionShapeError("rules: cannot build a distribution from no options")
    n = len(keys)
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise DecisionShapeError(
            f"rules: every signal weight was zero ({sorted(weights)}); refusing to "
            "invent a peak"
        )
    p = {k: max(0.0, float(weights[k])) / total for k in keys}
    p_max = max(p.values())
    uniform = 1.0 / n

    if p_max <= uniform:
        # Already flat; nothing for the ceiling to do.
        return normalise_probabilities(p)

    floor = max(uniform, ceiling)
    if p_max <= floor:
        return normalise_probabilities(p)

    beta = min(1.0, max(0.0, (p_max - floor) / (p_max - uniform)))
    return normalise_probabilities(
        {k: (1.0 - beta) * p[k] + beta * uniform for k in keys}
    )


def classify_from_state(state: Mapping[str, Any]) -> tuple[Intent, str]:
    """Classify intent by delegating to the frozen ``core`` helper.

    Returns ``(intent, text_used)``. The ordering of the underlying checks lives
    in :func:`core.protocols.classify_intent_fallback` and is not restated here.
    """
    text = _find_text(state)
    if not text.strip():
        return Intent.UNKNOWN, ""
    return classify_intent_fallback(text), text


# ==============================================================================
# backend
# ==============================================================================
class RulesBackend:
    """Deterministic, offline, always-available decision backend.

    ``available()`` returns ``True`` unconditionally: there is nothing to probe,
    no key to read and no socket to open. That is what makes it the right last
    link in the registry's chain.
    """

    name = "rules"
    model = "rules-v1"

    def __init__(self, settings: Settings | None = None, *,
                 ceiling: float = CONFIDENCE_CEILING) -> None:
        self.settings = settings or get_settings()
        self.ceiling = max(0.0, min(1.0, float(ceiling)))
        self._calls = 0
        self._last_error = ""
        self._last_latency_ms = 0.0

    # ---------------------------------------------------------------- protocol
    def available(self) -> tuple[bool, str]:
        return True, f"deterministic rules backend; confidence ceiling {self.ceiling:.2f}"

    def health(self) -> dict[str, Any]:
        """Diagnostics for the status panel.

        ``latency_ms`` is the last decision's wall time -- microseconds in
        practice, which is the point: the rules backend is free, and a status
        panel showing 0.08 ms says something true.
        """
        return {
            "backend": self.name,
            "available": True,
            "model": self.model,
            "reason": "deterministic; no network, no credentials, no probe",
            "latency_ms": round(self._last_latency_ms, 3),
            "last_error": self._last_error,
            "calls": self._calls,
            "confidence_ceiling": self.ceiling,
            "confidence_threshold": self.settings.confidence_threshold,
        }

    def decide(self, request: DecisionRequest) -> Decision:
        """Answer from ``request.state`` alone, capped at ``self.ceiling``."""
        started = time.perf_counter()
        self._calls += 1
        try:
            choice, probabilities, method, signals = self._answer(request)
        except DecisionShapeError as exc:
            self._last_error = str(exc)
            raise
        probabilities = _normalise_peak(probabilities, self.ceiling)
        confidence = probabilities[choice]
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self._last_latency_ms = latency_ms
        self._last_error = ""
        logger.debug(
            "decision.rules %s answered %s via %s in %.1fms (confidence=%.3f)",
            request.request_id, choice, method, latency_ms, confidence,
        )
        return make_decision(
            request_id=request.request_id,
            question=request.question,
            choice=choice,
            probabilities=probabilities,
            confidence=confidence,
            source=DecisionSource.RULES,
            model=self.model,
            latency_ms=latency_ms,
            degraded=False,
            raw=annotate_raw(
                None,
                method=method,
                signals=signals,
                ceiling=self.ceiling,
                calibration=(
                    "deterministic rule heuristic; the probabilities are derived "
                    "from explicit signals in the request state and are not "
                    "calibrated against any held-out data"
                ),
            ),
        )

    # ------------------------------------------------------------- the routing
    def _answer(self, request: DecisionRequest,
                ) -> tuple[str, dict[str, float], str, dict[str, Any]]:
        options = [str(o) for o in request.options]
        if request.question_type is QuestionType.NOUL:
            return self._answer_noul(request)
        if request.question_type is QuestionType.SCORE:
            return self._answer_score(request)
        if request.question_type is QuestionType.CHOICE:
            return self._answer_choice(request, options)
        raise DecisionShapeError(  # pragma: no cover - closed enum
            f"rules: unsupported question type {request.question_type!r}"
        )

    # ------------------------------------------------------------------- noul
    def _answer_noul(self, request: DecisionRequest,
                     ) -> tuple[str, dict[str, float], str, dict[str, Any]]:
        """Two-way answer from a signal table keyed on what was asked about.

        The question text is matched against :data:`CONFIDENCE_HINTS`; an
        unmatched question falls back to a neutral 50/50 rather than a guess,
        because a rule that invents a direction for a question it has no signals
        for is worse than one that says "no idea".
        """
        text = _find_text(request.state)
        key = self._match_hint(request.question)
        hints = CONFIDENCE_HINTS.get(key, ())
        hits = sorted({hint for hint in hints if hint in text})
        if hits:
            # More distinct signals -> a stronger answer, but the ceiling in
            # decide() flattens anything this over-claims anyway.
            # Base 0.60 (not 0.62) so a single weak signal sits distinctly
            # below confidence_threshold 0.62: with strict ``<`` an exact
            # 0.62 tie would resolve, which a one-keyword hit has not earned.
            p_true = min(0.95, 0.60 + 0.08 * (len(hits) - 1))
            weights = {NOUL_LABELS[0]: p_true, NOUL_LABELS[1]: 1.0 - p_true}
            return NOUL_LABELS[0], weights, f"noul:{key}", {
                "hint_key": key, "hits": hits,
            }
        neutral = {NOUL_LABELS[0]: 0.5, NOUL_LABELS[1]: 0.5}
        return NOUL_LABELS[0], neutral, "noul:no-signal", {
            "hint_key": key, "hits": [],
        }

    @staticmethod
    def _match_hint(question: str) -> str:
        lowered = question.lower()
        for key in CONFIDENCE_HINTS:
            if key in lowered:
                return key
        return ""

    # ------------------------------------------------------------------ score
    def _answer_score(self, request: DecisionRequest,
                      ) -> tuple[str, dict[str, float], str, dict[str, Any]]:
        """Peaked distribution over rubric levels, centred on the best-matching one.

        Signals, in priority order:

        1. ``state["score_hints"]`` -- a list of strings naming levels.
        2. ``state["signals"]`` -- free text matched against each level's hints.
        3. ``state["<rubric level>"]`` truthy flags -- e.g.
           ``{"major": true}``.
        4. ``state["severity"]``/``state["score"]`` -- an ordinal or a level name.

        With no signal at all the centre is level 1: the second-lowest on the
        rubric. That encodes "something needs looking at" without asserting a
        severity nobody stated.
        """
        levels = [str(level) for level in request.rubric]
        state = request.state
        hits: dict[str, Any] = {}

        named = state.get("score_hints") or state.get("signals")
        if isinstance(named, (list, tuple)):
            hits["signals"] = [str(s) for s in named]

        flags = {level: bool(state[level]) for level in levels if level in state}
        if flags:
            hits["level_flags"] = flags

        ordinal = state.get("severity", state.get("score"))
        if isinstance(ordinal, str):
            for _index, level in enumerate(levels):
                if level.lower() == ordinal.strip().lower():
                    hits["named_level"] = level
                    break
            else:
                if ordinal.strip().isdigit():
                    index = int(ordinal.strip())
                    if 0 <= index < len(levels):
                        hits["ordinal"] = index
        elif isinstance(ordinal, bool):
            pass
        elif isinstance(ordinal, int) and 0 <= ordinal < len(levels):
            hits["ordinal"] = ordinal

        text = _as_text(state)

        centre = 1 if len(levels) > 1 else 0
        matched: list[str] = []

        if "ordinal" in hits:
            centre = int(hits["ordinal"])
            method = "score:ordinal"
        elif "named_level" in hits:
            centre = levels.index(str(hits["named_level"]))
            method = "score:named-level"
        elif any(flags.values()):
            flagged = [level for level, on in flags.items() if on]
            matched = flagged
            centre = max(levels.index(level) for level in flagged)
            method = "score:level-flags"
        elif isinstance(named, (list, tuple)):
            blob = " ".join(str(s).lower() for s in named)
            matched = [level for level in levels
                       if level.lower() in blob or _level_hit(level, blob)]
            centre = max((levels.index(level) for level in matched), default=centre)
            method = "score:signals"
        else:
            scored = {
                level: sum(1 for hint in _GENERIC_LEVEL_HINTS.get(level.lower(), ())
                           if hint in text)
                for level in levels
            }
            matched = [level for level, count in scored.items() if count]
            centre = max(range(len(levels)), key=lambda i: (scored[levels[i]], -i))
            if not matched:
                centre = 1 if len(levels) > 1 else 0
                method = "score:no-signal"
            else:
                method = "score:text-hints"

        weights = _peak(centre, len(levels))
        probabilities = {levels[index]: weight for index, weight in weights.items()}
        return levels[centre], probabilities, method, hits or {"matched": matched}

    # ----------------------------------------------------------------- choice
    def _answer_choice(self, request: DecisionRequest, options: Sequence[str],
                       ) -> tuple[str, dict[str, float], str, dict[str, Any]]:
        """Distribution over the offered options, from explicit state signals.

        Documented routing, in priority order:

        ``intent``   the option set contains ``Intent`` values and the state
                     holds sponsor text -> :func:`classify_from_state`, peaked on
                     the classified label. Every other label shares a flat tail.
        ``option``   ``state["option_signals"]`` maps an option label to evidence
                     (a number, or a list of strings); normalised across options.
        ``verdict``  the option set is made of action words
                     (``escalate``/``hold``/``proceed``/``flag``); answered from
                     blocking risk flags, escalated disputes and low confidence
                     in the state.
        ``lexical``  nothing structured matched -> lexical overlap between each
                     option label and the state text, with a smoothing floor so
                     the answer is a distribution rather than a hard 1.0.
        """
        state = request.state
        lowered = {option.lower(): option for option in options}

        # --- intent ---------------------------------------------------------
        if set(lowered) & set(INTENT_VALUES):
            intent, text = classify_from_state(state)
            label = lowered.get(intent.value)
            if label is not None:
                weights = {option: (1.0 if option == label else 0.22) for option in options}
                return label, weights, "choice:intent", {
                    "intent": intent.value,
                    "text_excerpt": snippet(text, 120),
                    "tail_weight": 0.22,
                }

        # --- explicit per-option evidence ----------------------------------
        supplied = state.get("option_signals")
        if isinstance(supplied, Mapping):
            usable = {
                lowered[str(key).lower()]: value
                for key, value in supplied.items()
                if str(key).lower() in lowered
            }
            if usable:
                weights: dict[str, float] = {}
                for option in options:
                    value = usable.get(option, 0)
                    if isinstance(value, bool) or value is None:
                        weights[option] = 1.0 if value is True else 0.0
                    elif isinstance(value, (int, float)):
                        weights[option] = max(0.0, float(value))
                    elif isinstance(value, (list, tuple)):
                        # A list of evidence items counts as evidence items.
                        weights[option] = float(len(value))
                    elif isinstance(value, str):
                        # A string is a phrase: count its whitespace-separated
                        # words rather than its characters.
                        weights[option] = float(len(value.split()))
                    else:
                        weights[option] = 0.0
                if sum(weights.values()) <= 0:
                    raise DecisionShapeError(
                        f"rules: state['option_signals'] for {request.request_id} "
                        f"carried no positive evidence for any of {options}"
                    )
                chosen = max(weights, key=lambda o: (weights[o], -options.index(o)))
                return chosen, weights, "choice:option_signals", {
                    "signals": {k: snippet(v, 60) for k, v in usable.items()}
                }

        # --- action-word verdicts ------------------------------------------
        if set(lowered) & set(_OPTION_SIGNALS):
            escalate, evidence = self._escalation_evidence(state)
            chosen_label = "escalate" if escalate else "proceed"
            chosen = lowered.get(chosen_label)
            if chosen is None:
                chosen = options[0]
            weights = {option: (1.0 if option == chosen else 0.25) for option in options}
            return chosen, weights, "choice:verdict", evidence

        # --- lexical overlap -----------------------------------------------
        text = _as_text(state)
        words = {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2}
        weights = {}
        for option in options:
            tokens = {w for w in re.findall(r"[a-z0-9]+", option.lower()) if len(w) > 2}
            weights[option] = 1.0 + len(tokens & words)
        chosen = max(weights, key=lambda o: (weights[o], -options.index(o)))
        return chosen, weights, "choice:lexical", {
            "overlap": {k: int(v - 1.0) for k, v in weights.items()},
            "state_tokens": len(words),
        }

    @staticmethod
    def _escalation_evidence(state: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Does the state argue for escalating? Explicit, countable signals only."""
        reasons: list[str] = []

        blocking = state.get("blocking")
        if isinstance(blocking, bool) and blocking:
            reasons.append("blocking risk flag present")

        flags = state.get("risk_flags") or state.get("flags")
        if isinstance(flags, (list, tuple)):
            blocking_flags = [
                f for f in flags
                if isinstance(f, Mapping) and str(f.get("severity", "")).lower() == "blocking"
            ]
            if blocking_flags:
                reasons.append(f"{len(blocking_flags)} blocking risk flag(s)")

        disputes = state.get("disputes") or state.get("debates")
        if isinstance(disputes, (list, tuple)):
            escalated = [
                d for d in disputes
                if isinstance(d, Mapping) and str(d.get("status", "")).lower() == "escalated"
            ]
            if escalated:
                reasons.append(f"{len(escalated)} escalated dispute(s)")

        for key in ("confidence", "decision_confidence", "last_confidence"):
            value = state.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < _escalation_threshold():
                    reasons.append(f"{key}={float(value):.2f} below the escalation threshold")
                # Only the first confidence-shaped key is consulted: two of them
                # disagreeing means the state is malformed, and picking the
                # lower one would be inventing evidence.
                break

        human = state.get("requires_human") or state.get("human_in_the_loop")
        if human is True:
            reasons.append("state explicitly requests a human")

        return bool(reasons), {"escalation_reasons": reasons}


def _level_hit(level: str, blob: str) -> bool:
    """Does the generic hint table have anything for this rubric level?"""
    return any(hint in blob for hint in _GENERIC_LEVEL_HINTS.get(level.lower(), ()))


def _escalation_threshold() -> float:
    """The configured escalation threshold, read lazily.

    Read through :func:`core.config.get_settings` at call time rather than at
    import time, so a test (or an environment override) is honoured.
    """
    return get_settings().escalation_threshold
