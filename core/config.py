"""Configuration. One source of truth, read from the environment.

Two rules:

1. **No secret ever has a default.** A missing key means the corresponding
   backend is unavailable, and the run degrades *visibly* rather than silently
   pretending to have a model.
2. **Every degradation is recordable.** ``decision_backend`` names the intended
   backend; ``run_mode`` names what actually happened. A trace comparing the two
   tells you exactly which calls were real.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .errors import ConfigError
from .schemas import RunMode

__all__ = ["Settings", "get_settings", "reset_settings_cache", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parent.parent

BackendName = Literal["auto", "clef", "gemini", "rules"]


class Settings:
    """Plain, explicit settings object (no pydantic dependency at import time)."""

    def __init__(self, **overrides: object) -> None:
        # ---- identity ------------------------------------------------------
        self.app_name: str = os.getenv("APP_NAME", "paytriq")
        self.app_version: str = os.getenv("APP_VERSION", "0.1.0")
        # Provenance, never invented: GIT_COMMIT/CODE_SHA256 injected at build
        # time (Docker ARG, CI GITHUB_SHA, Render dashboard). Fall back to the
        # platform's own names (RENDER_GIT_COMMIT, GITHUB_SHA) so a build that
        # forgot the mirror still records the real SHA. A zero-commit checkout
        # records "uncommitted-working-tree" explicitly (see
        # scripts/capture_canonical_trace.py); "unknown" means genuinely unset.
        self.git_commit: str = (
            os.getenv("GIT_COMMIT", "").strip()
            or os.getenv("RENDER_GIT_COMMIT", "").strip()
            or os.getenv("GITHUB_SHA", "").strip()
            or "unknown"
        )
        self.code_sha256: str = (
            os.getenv("CODE_SHA256", "").strip()
            or os.getenv("RENDER_GIT_COMMIT", "").strip()
            or os.getenv("GITHUB_SHA", "").strip()
            or "unknown"
        )
        self.environment: str = os.getenv("ENVIRONMENT", "local")

        # ---- run mode ------------------------------------------------------
        raw_mode = os.getenv("RUN_MODE", "").strip().lower()
        self.run_mode: RunMode = _parse_run_mode(raw_mode) if raw_mode else RunMode.LIVE

        # ---- generative model (Gemini) -------------------------------------
        self.gemini_api_key: str | None = _blank_to_none(os.getenv("GEMINI_API_KEY"))
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.gemini_fallback_model: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
        self.gemini_timeout_s: float = float(os.getenv("GEMINI_TIMEOUT_S", "45"))

        # ---- decision model (Clef, System One) ------------------------------
        #: Local transport is **llama.cpp's `llama-server`**, defaulting to 18781.
        #: Deliberately NOT Ollama: the official Ollama `clef-flash` distribution
        #: is broken on Windows (ollama/ollama#18769 -- a 32-bit file-offset
        #: truncation in `ifstream` loads backbone bytes as decision-head
        #: weights, yielding non-finite logits). Fix PR #18777 was unmerged as
        #: of 2026-10-04. `decision.clef` still rewrites a loopback :11434 to
        #: :18781 with a warning, so a stale .env cannot silently break a demo.
        self.clef_base_url: str = os.getenv("CLEF_BASE_URL", "http://localhost:18781")
        self.clef_model: str = os.getenv("CLEF_MODEL", "clef-flash")
        self.clef_api_key: str | None = _blank_to_none(os.getenv("CLEF_API_KEY"))
        #: Workers AI account id, used when ``CLEF_BACKEND=workers_ai``.
        self.clef_account_id: str | None = _blank_to_none(os.getenv("CLEF_ACCOUNT_ID"))
        self.clef_backend: Literal["ollama", "workers_ai", "auto"] = os.getenv(
            "CLEF_BACKEND", "ollama"
        ).strip().lower() or "ollama"
        self.clef_timeout_s: float = float(os.getenv("CLEF_TIMEOUT_S", "5"))
        self.clef_health_ttl_s: float = float(os.getenv("CLEF_HEALTH_TTL_S", "30"))

        # ---- decision layer selection --------------------------------------
        self.decision_backend: BackendName = os.getenv(
            "DECISION_BACKEND", "auto"
        ).strip().lower() or "auto"
        #: Below this, the arbiter escalates instead of guessing.
        self.confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.62"))
        #: Below this, escalate to a human rather than to a generative model.
        self.escalation_threshold: float = float(os.getenv("ESCALATION_THRESHOLD", "0.40"))
        #: Retry once on a transient backend error before degrading.
        self.decision_max_retries: int = int(os.getenv("DECISION_MAX_RETRIES", "1"))

        # ---- agent budgets -------------------------------------------------
        self.agent_step_budget: int = int(os.getenv("AGENT_STEP_BUDGET", "6"))
        self.agent_deadline_s: float = float(os.getenv("AGENT_DEADLINE_S", "60"))
        self.max_debate_rounds: int = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))
        self.max_replans: int = int(os.getenv("MAX_REPLANS", "2"))
        self.max_auction_rounds: int = int(os.getenv("MAX_AUCTION_ROUNDS", "3"))

        # ---- tools ---------------------------------------------------------
        self.tools_enabled: bool = _as_bool(os.getenv("TOOLS_ENABLED"), True)
        self.tool_timeout_s: float = float(os.getenv("TOOL_TIMEOUT_S", "20"))
        #: Live calls are opt-in. With this off, tools serve recorded/seeded data
        #: and mark every result ``degraded=True``.
        self.tools_live: bool = _as_bool(os.getenv("TOOLS_LIVE"), False)
        self.overpass_url: str = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
        self.nominatim_url: str = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
        self.resend_api_key: str | None = _blank_to_none(os.getenv("RESEND_API_KEY"))
        self.email_from: str = os.getenv("EMAIL_FROM", "paytriq@example.invalid")
        self.map_user_agent: str = os.getenv(
            "MAP_USER_AGENT", "paytriq-research/0.1 (student project)"
        )

        # ---- paths ---------------------------------------------------------
        self.repo_root: Path = REPO_ROOT
        self.traces_dir: Path = Path(os.getenv("TRACES_DIR", str(REPO_ROOT / "traces")))
        self.artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(REPO_ROOT / "artifacts")))
        self.golden_dir: Path = Path(os.getenv("GOLDEN_DIR", str(REPO_ROOT / "tests" / "golden")))
        self.replay_path: Path = Path(
            os.getenv("REPLAY_PATH", str(REPO_ROOT / "tests" / "golden" / "replay.jsonl"))
        )

        # ---- human-in-the-loop --------------------------------------------
        #: When false, gates auto-resolve with ``approve`` for unattended demo runs.
        #: Every auto-resolution is still written to the trace with
        #: ``decided_by="auto"`` so it is never mistaken for a human decision.
        self.human_gates_interactive: bool = _as_bool(
            os.getenv("HUMAN_GATES_INTERACTIVE"), True
        )
        self.auto_approve_outcome: str = os.getenv("AUTO_APPROVE_OUTCOME", "approve")

        # ---- apply overrides last -----------------------------------------
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise ConfigError(f"unknown setting: {key!r}")
            setattr(self, key, value)

        self._validate()

    # ------------------------------------------------------------------ helpers
    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_clef(self) -> bool:
        if self.clef_backend == "workers_ai":
            return bool(self.clef_account_id)
        return True  # ollama needs no credential; availability is probed at runtime

    @property
    def is_offline(self) -> bool:
        return self.run_mode is RunMode.OFFLINE

    def effective_backend(self) -> BackendName:
        """Resolve ``auto`` to a concrete backend given what is configured."""
        if self.decision_backend != "auto":
            return self.decision_backend
        if self.has_clef:
            return "clef"
        if self.has_gemini:
            return "gemini"
        return "rules"

    def redacted(self) -> dict[str, object]:
        """Config snapshot safe to print in a demo or write into a trace."""
        def mask(v: str | None) -> str | None:
            if not v:
                return None
            return f"{v[:4]}...{v[-2:]}" if len(v) > 8 else "***"

        return {
            "app_version": self.app_version,
            "git_commit": self.git_commit[:12],
            "run_mode": self.run_mode.value,
            "decision_backend": self.decision_backend,
            "effective_backend": self.effective_backend(),
            "clef_backend": self.clef_backend,
            "clef_model": self.clef_model,
            "gemini_model": self.gemini_model if self.has_gemini else None,
            "gemini_key": mask(self.gemini_api_key),
            "clef_account": mask(self.clef_account_id),
            "tools_live": self.tools_live,
            "confidence_threshold": self.confidence_threshold,
            "max_debate_rounds": self.max_debate_rounds,
            "max_replans": self.max_replans,
        }

    def _validate(self) -> None:
        if not 0.0 <= self.escalation_threshold <= self.confidence_threshold <= 1.0:
            raise ConfigError(
                "thresholds must satisfy 0 <= escalation <= confidence <= 1, got "
                f"escalation={self.escalation_threshold} confidence={self.confidence_threshold}"
            )
        if self.decision_backend not in ("auto", "clef", "gemini", "rules"):
            raise ConfigError(f"unknown decision_backend: {self.decision_backend!r}")
        if self.clef_backend not in ("ollama", "workers_ai", "auto"):
            raise ConfigError(f"unknown clef_backend: {self.clef_backend!r}")
        if self.decision_backend == "clef" and not self.has_clef:
            raise ConfigError(
                "DECISION_BACKEND=clef but no Clef credential is configured "
                "(set CLEF_ACCOUNT_ID for workers_ai)"
            )
        if self.run_mode is RunMode.LIVE and not (self.has_gemini or self.has_clef):
            # Not fatal: the run degrades to rules and says so in every trace.
            pass
        for p in (self.traces_dir, self.artifacts_dir, self.golden_dir):
            p.parent.mkdir(parents=True, exist_ok=True)


def _blank_to_none(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    return v or None


def _as_bool(v: str | None, default: bool) -> bool:
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _parse_run_mode(raw: str) -> RunMode:
    raw = raw.strip().lower()
    if raw in ("live", "online"):
        return RunMode.LIVE
    if raw in ("replay", "recorded"):
        return RunMode.REPLAY
    if raw in ("offline", "rules", "offline-rules"):
        return RunMode.OFFLINE
    raise ConfigError(f"RUN_MODE must be live|replay|offline, got {raw!r}")


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings(**overrides: object) -> Settings:
    """Process-wide settings. Pass overrides for tests; they bypass the cache."""
    if overrides:
        return Settings(**overrides)
    return _cached_settings()


def reset_settings_cache() -> None:
    _cached_settings.cache_clear()
