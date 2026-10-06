"""Image analysis through Gemini vision. No model, no answer.

The prototype's vision tool was::

    h = int(hashlib.md5(str(image_path)).hexdigest()[:4], 16)
    return {"logo_count": (h % 5) + 1, "footfall_estimate": 500 + (h % 40) * 100}

It was called "Gemini Vision" in the demo. It was a filename hash. It produced
logo counts and footfall estimates that were then multiplied into ROI figures,
and every one of those numbers was fiction dressed as an observation.

So this module has exactly one code path: a real ``google-genai`` request with
the image bytes attached. If the image is missing, or the package is not
installed, or there is no API key, or ``settings.tools_live`` is off, the answer
is :meth:`ToolResult.unavailable` with a reason. There is no hash, no fallback,
no "deterministic mock", and no path that returns a count.

Honesty about model output
--------------------------
Even a successful call produces a *model-generated description*, not a
measurement. It is returned with ``verification="model-generated"``, a note
saying so, and the ``sha256`` of the exact bytes sent, so a reader can prove
which image the description belongs to.
:class:`~tools.browser.VerifyEvidenceTool` caps its confidence when it leans on
these findings, because a vision model can miss text and can misread it.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.protocols import ToolResult
from core.schemas import DecisionSource

from .base import BaseTool, annotate, guarded, redact, tool

__all__ = ["AnalyseEvidenceTool", "VISION_PROMPT", "MAX_IMAGE_BYTES"]

#: 8 MB. Gemini rejects larger inline payloads, and an unbounded read of an
#: arbitrary file is how a tool turns into a memory problem.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: Extension -> MIME type, because the API refuses unknown types and
#: ``mimetypes`` guesses inconsistently across platforms for the formats that
#: actually turn up as evidence (``.webp``, ``.heic``).
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".pdf": "application/pdf",
}

#: The prompt. Asks for observation, not inference: the previous failure was not
#: a bad model, it was a model asked for a number it cannot measure.
VISION_PROMPT = """\
You are inspecting a single image as compliance evidence for an event sponsorship.

Report ONLY what is visibly present in the image. Do not infer, estimate, or fill
in values you cannot read. If something is absent, say it is absent. If the image
is not legible, say so and use empty lists.

Return a single JSON object, no prose outside it, with this exact shape:

{
  "image_legible": true,
  "image_legible_reason": "one sentence",
  "observations": [
    {
      "label": "logo | banner | booth | product | signage | text | people | other",
      "text": "the exact text visible on this item, transcribed",
      "present": true,
      "confidence": 0.0,
      "note": "where in the image it appears"
    }
  ],
  "counts": {"visible_logos": 0, "people_visible": 0},
  "limitations": ["anything that prevents a confident reading"]
}

Populate "counts" only by actually counting visible items. If you cannot count
reliably, use -1 and say why in "limitations".
"""


@tool("analyse_evidence",
      "Analyse a real image file with Gemini vision (google-genai) and return "
      "structured observations. Requires GEMINI_API_KEY and TOOLS_LIVE=true. "
      "With no model there is no analysis and no fallback — the tool reports "
      "unavailable rather than inventing counts.",
      mode="live", backend="gemini-vision")
class AnalyseEvidenceTool(BaseTool):
    """Describe what is in an image. Never guess what is not there."""

    def current_mode(self) -> str:
        if self.live and self.settings.gemini_api_key and _genai_available():
            return "live"
        return "unavailable"

    def available(self) -> tuple[bool, str]:
        if not self.settings.tools_enabled:
            return False, "tools are disabled (settings.tools_enabled=False)"
        if not _genai_available():
            return False, ("google-genai is not installed; image analysis has no "
                           "fallback and will not be approximated")
        if not self.settings.gemini_api_key:
            return False, "GEMINI_API_KEY is not set; image analysis is unavailable"
        if not self.live:
            return False, ("live tool calls are off (TOOLS_LIVE=false); image "
                           "analysis is unavailable and will not be faked")
        return True, (f"live Gemini vision via {self.settings.gemini_model!r} "
                      f"(findings will be labelled model-generated)")

    # ------------------------------------------------------------------- public
    def run(self, image_path: str | None = None, *,
            prompt: str | None = None,
            max_bytes: int = MAX_IMAGE_BYTES) -> ToolResult:
        """Analyse ``image_path``.

        :param image_path: a real image on disk. Required — there is no URL
            fetch here, because a tool that both fetched and judged the image
            could report "no image" without anyone noticing.
        :param prompt: overrides the default observation prompt.
        """
        if not image_path:
            return ToolResult.unavailable(
                reason=(f"{self.name}: no image supplied. Vision analysis requires a "
                        f"real image file; there is no synthetic or estimated "
                        f"fallback, so no findings are returned."),
                source=f"{self.name}:input")

        path = Path(image_path)
        if not path.is_absolute():
            path = Path(self.settings.artifacts_dir) / path
        if not path.exists():
            return ToolResult.unavailable(
                reason=(f"{self.name}: image not found at {path}. Nothing was "
                        f"analysed, so nothing is reported."),
                source=f"{self.name}:input")
        if not path.is_file():
            return ToolResult.unavailable(
                reason=f"{self.name}: {path} is not a file; nothing was analysed",
                source=f"{self.name}:input")
        try:
            blob = path.read_bytes()
        except OSError as exc:
            return ToolResult.unavailable(
                reason=redact(f"{self.name}: could not read {path} — "
                              f"{type(exc).__name__}: {exc}", self.secrets),
                source=f"{self.name}:input")
        if not blob:
            return ToolResult.unavailable(
                reason=f"{self.name}: {path} is empty (0 bytes); nothing was analysed",
                source=f"{self.name}:input")
        limit = max(1024, int(max_bytes or MAX_IMAGE_BYTES))
        if len(blob) > limit:
            return ToolResult.unavailable(
                reason=(f"{self.name}: image is {len(blob)} bytes, over the "
                        f"{limit}-byte limit; nothing was analysed"),
                source=f"{self.name}:input")

        mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
        if mime is None:
            guessed, _ = mimetypes.guess_type(str(path))
            mime = guessed or ""
        if not mime.startswith(("image/", "application/pdf")):
            return ToolResult.unavailable(
                reason=(f"{self.name}: {path.name} is {mime or 'an unknown type'}, "
                        f"which is not an analysable image; nothing was analysed"),
                source=f"{self.name}:input")

        ok, why = self.available()
        if not ok:
            return ToolResult.unavailable(
                reason=(f"{self.name}: {why}. The image was read "
                        f"({len(blob)} bytes, sha256={_digest(blob)[:16]}...) but "
                        f"not analysed, and no substitute result is produced."),
                source=f"{self.name}:gated")

        call = guarded(self._live_analyse, timeout=self.settings.gemini_timeout_s,
                       tool_name=self.name, secrets=self.secrets,
                       errors=_genai_errors())
        result = call(blob, mime, prompt or VISION_PROMPT, path)
        if not result.ok:
            return result
        payload = result.data
        return annotate(
            result,
            source=f"{self.name}:{payload['model']}",
            reason=(f"Gemini vision ({payload['model']}) described the image. These "
                    f"are model-generated observations of a real image "
                    f"(sha256={payload['image_sha256'][:16]}...); they are not "
                    f"independent measurements and are not treated as such."),
        )

    # -------------------------------------------------------------------- live
    def _live_analyse(self, blob: bytes, mime: str, prompt: str,
                      path: Path) -> dict[str, Any]:
        """One Gemini call with the image inline. Raises for :func:`guarded`."""
        genai_types = _genai_types()
        client = _genai_client(self.settings.gemini_api_key or "")
        part = genai_types.Part.from_bytes(data=blob, mime_type=mime)
        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=[genai_types.Content(parts=[part], role="user"),
                      prompt],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        raw = getattr(response, "text", "") or ""
        if not raw.strip():
            raise ValueError("Gemini returned an empty body; no observation can "
                             "be claimed from it")
        parsed = _parse_json_object(raw)
        return {
            "image_path": str(path),
            "image_bytes": len(blob),
            "image_mime_type": mime,
            "image_sha256": _digest(blob),
            "model": self.settings.gemini_model,
            "analysed_at": _now_iso(),
            "image_legible": bool(parsed.get("image_legible")),
            "image_legible_reason": str(parsed.get("image_legible_reason") or ""),
            "observations": _clean_observations(parsed.get("observations")),
            "counts": _clean_counts(parsed.get("counts")),
            "limitations": _string_list(parsed.get("limitations")),
            "verification": "model-generated",
            "source": DecisionSource.GEMINI.value,
            "note": ("Model-generated description of the image above. Usable as a "
                     "hint, never as a measurement: a vision model can miss text "
                     "and can misread it. Not independently verifiable."),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AnalyseEvidenceTool mode={self.current_mode()!r}>"


# ------------------------------------------------------------------- internals
def _now_iso() -> str:
    from core.ids import utcnow

    return utcnow().isoformat()


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _genai_types() -> Any:
    from google.genai import types

    return types


def _genai_client(api_key: str) -> Any:
    """Build a ``google-genai`` client.

    Imported inside the call so that the absence of the package surfaces as an
    ``ImportError`` the caller already gated on, rather than at module import.
    """
    from google import genai

    if not api_key:
        raise ValueError("no Gemini API key configured")
    return genai.Client(api_key=api_key)


def _genai_available() -> bool:
    try:
        import google.genai  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _genai_errors() -> tuple[type[BaseException], ...]:
    """``google-genai`` error types, resolved lazily like reportlab's."""
    if not _genai_available():
        return ()
    try:
        from google.genai import errors as genai_errors
    except (ImportError, ModuleNotFoundError, AttributeError):
        return ()
    collected: list[type[BaseException]] = []
    for name in ("APIError", "ClientError", "ServerError",
                 "UnknownApiResponseError"):
        candidate = getattr(genai_errors, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            collected.append(candidate)
    return tuple(collected)


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the model's JSON reply, tolerating a fenced code block.

    A parse failure raises, and :func:`tools.base.guarded` turns it into
    ``FAILED``. It must not fall back to an empty interpretation: "the model
    said something I could not read" is not "the image showed nothing".
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"model reply is not a JSON object: {text[:120]!r}")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise TypeError(f"model returned {type(payload).__name__}, expected an object")
    return payload


def _clean_observations(raw: Any) -> list[dict[str, Any]]:
    """Normalise observations, dropping anything that is not a real sighting.

    ``present`` is coerced to a strict bool and a missing text value becomes an
    empty string rather than a placeholder — a fabricated transcription would be
    worse than a blank one, because it would read as something read off the
    image.
    """
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        label = str(item.get("label") or "other").strip().lower()
        try:
            confidence = float(item.get("confidence"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            confidence = 0.0
        cleaned.append({
            "label": label,
            "text": text,
            "present": bool(item.get("present")),
            "confidence": round(min(1.0, max(0.0, confidence)), 3),
            "note": str(item.get("note") or ""),
        })
    return cleaned


def _clean_counts(raw: Any) -> dict[str, int]:
    """Keep only counts the model actually produced.

    ``-1`` is the model's explicit "I cannot count this". It is preserved as a
    negative number rather than being coerced to zero, because zero is a count
    and "cannot count" is not a count. That distinction is precisely what the
    prototype's ``(h % 40) * 100`` footfall estimate erased.
    """
    if not isinstance(raw, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw if str(item).strip()]


def observed_terms(vision: Mapping[str, Any]) -> Sequence[str]:  # pragma: no cover
    """Promise terms the vision model claims to have seen (helper for callers)."""
    terms: list[str] = []
    for obs in _clean_observations(vision.get("observations")):
        if obs["present"] and obs["text"]:
            terms.append(obs["text"])
    return terms
