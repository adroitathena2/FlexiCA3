"""Fetch pages, and decide whether evidence actually shows a promise was kept.

Two tools, one idea: **evidence has to be inspected, not pattern-matched.**

The prototype's compliance check was
``found = any(keyword in url.lower() for keyword in promise_words) or "sponsor" in url``. It
returned ``found=True`` for any URL containing ``/sponsor``, for any promise. It
also accepted a *filename* as proof. So a compliance agent built on it reported
"promise fulfilled" from a string it had been handed, and the resulting MoU
status, audit finding and ROI report were all fictional.

:class:`VerifyEvidenceTool` therefore has two hard rules:

1. **A URL is never searched.** It is recorded as provenance and nothing else.
   The only corpora that can satisfy a promise are (a) page body text that was
   actually fetched, and (b) findings from a real vision model call on a real
   image. Both are labelled as to their origin, and the vision path is capped at
   a lower confidence because a model's description is not an independent check.
2. **Absence of evidence is reported as absence.** With no fetched text and no
   model findings, the verdict is ``cannot-verify`` — not ``fulfilled`` and not
   ``not-fulfilled``. Distinguishing "we did not look" from "we looked and it
   was not there" is the entire value of a compliance step.
"""
from __future__ import annotations

import hashlib
import html as html_module
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.protocols import ToolResult

from . import fixtures
from .base import (
    BaseTool,
    ToolSignal,
    annotate,
    fixture_result,
    guarded,
    http_client,
    http_result,
    tool,
)

__all__ = ["FetchPageTool", "VerifyEvidenceTool", "VERDICTS", "extract_text"]

#: The only three verdicts this tool may emit.
VERDICTS = ("fulfilled", "not-fulfilled", "cannot-verify")

#: Words that carry no evidentiary weight in a promise.
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "with", "for", "of", "on", "in", "at", "to", "by",
    "from", "will", "shall", "is", "are", "be", "been", "being", "per", "our",
    "your", "we", "us", "they", "their", "it", "its", "as", "that", "this",
    "provide", "provides", "provided", "including", "include", "includes",
    "logo", "placement", "main", "stage", "banner", "booth",
})

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|template|svg)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_END_RE = re.compile(
    r"</(p|div|br|li|tr|h[1-6]|section|article|td|th|ul|ol|table|header|footer)\s*>",
    re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\u00a0]+")
_BLANKS_RE = re.compile(r"\n{3,}")

#: Refuse non-HTTP schemes outright: ``file://`` would turn a URL into a local
#: file read, which is both a traversal risk and nonsense for a brand website.
_ALLOWED_SCHEMES = ("http", "https")

#: A body shorter than this cannot substantiate anything, so a "fulfilled"
#: verdict on it would be an accident of vocabulary.
_MIN_EVIDENCE_CHARS = 200


def extract_text(raw_html: str) -> str:
    """Normalise an HTML body to readable text.

    Deliberately regex-based rather than BeautifulSoup: the browser is a tool in
    an agent framework, and requiring a parser dependency for "drop the tags"
    would make the fallback path depend on an optional package too. Block-level
    closers become newlines so the text keeps its structure, and entity escapes
    are decoded last so ``&amp;`` does not survive into evidence matching.

    No claim is made that this is spec-compliant HTML parsing; it is a
    conservative stripper that cannot invent text.
    """
    if not raw_html:
        return ""
    text = _COMMENT_RE.sub(" ", raw_html)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html_module.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKS_RE.sub("\n\n", text).strip()


@tool("fetch_page",
      "Fetch a URL over HTTP(S) and return its visible text plus the final URL "
      "after redirects. Optional Playwright screenshot only when the package is "
      "importable and live calls are enabled.",
      mode="live", backend="http")
class FetchPageTool(BaseTool):
    """Retrieve page text that :class:`VerifyEvidenceTool` can actually inspect.

    In live mode a plain ``httpx`` GET is the default. Playwright is used only
    when the caller explicitly asks for a screenshot *and* the package is
    importable *and* ``settings.tools_live`` is on. A screenshot path is reported
    only after the file has been written and its size verified on disk — the
    prototype logged "playwright fetched {url}" without ever writing an image.
    """

    def current_mode(self) -> str:
        return "live" if self.live else "fixture"

    def available(self) -> tuple[bool, str]:
        if not self.settings.tools_enabled:
            return False, "tools are disabled (settings.tools_enabled=False)"
        if self.live:
            return True, (f"live HTTP fetch via {self.settings.map_user_agent!r}"
                          + ("; Playwright available" if _playwright_available()
                             else "; Playwright not installed"))
        return True, ("fixture mode (TOOLS_LIVE=false): only URLs seeded in "
                      "tools/fixtures.py can be fetched")

    def run(self, url: str = "", *, want_screenshot: bool = False,
            max_chars: int = 40_000) -> ToolResult:
        """Fetch ``url``.

        :param want_screenshot: attempt a Playwright capture. Ignored (and
            reported as ignored) unless every precondition holds.
        :param max_chars: truncation limit for the returned text. Truncation is
            reported in ``data["truncated"]`` so a caller never mistakes a
            clipped page for a whole one.
        """
        target = (url or "").strip()
        if not target:
            return ToolResult.failed(reason=f"{self.name}: empty url",
                                     source=f"{self.name}:input")
        parsed = urllib.parse.urlparse(target)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
            return ToolResult.failed(
                reason=(f"{self.name}: refusing {target!r}; only "
                        f"{'/'.join(_ALLOWED_SCHEMES)} URLs with a host are "
                        f"fetched (a file:// or bare path would be a local read, "
                        f"not a website)"),
                source=f"{self.name}:input")

        if not self.live:
            text = fixtures.fixture_page_text(target)
            if text is None:
                return ToolResult.unavailable(
                    reason=(f"{self.name}: fixture mode has no seeded page for "
                            f"{target}. Live fetching is off (TOOLS_LIVE=false), "
                            f"so its contents cannot be reported honestly."),
                    source=f"{self.name}:fixture")
            trimmed = self._truncate(text, max_chars)
            return fixture_result(
                {"url": target, "final_url": target, "text": trimmed,
                 "char_count": len(trimmed), "truncated": len(trimmed) < len(text),
                 "status_code": None, "screenshot_path": None,
                 "note": fixtures.FIXTURE_NOTE},
                "page text")

        call = guarded(self._live_fetch, timeout=self.timeout,
                       tool_name=self.name, secrets=self.secrets)
        result = call(target, max(200, int(max_chars or 200)))
        if not result.ok:
            return result
        # Mutate ``result.data`` in place rather than a copy: a caller reading
        # ``result.data["screenshot_path"]`` must see the *actual* outcome, and
        # the naive "copy, edit the copy" version silently loses the note.
        data: dict[str, Any] = result.data

        if want_screenshot:
            shot = self._capture_screenshot(data["final_url"])
            if shot.ok:
                data["screenshot_path"] = shot.data
            else:
                # Explicitly recorded as a non-event: no path, and the reason.
                # The page fetch itself still succeeded, so the result stays OK,
                # but it is marked degraded because a requested artefact is absent.
                data["screenshot_path"] = None
                data["screenshot_note"] = shot.reason
                annotate(result, degraded=True,
                         reason=(f"{self.name}: page text fetched, but the "
                                 f"requested screenshot was not produced — "
                                 f"{shot.reason}"))
        return annotate(result, source=f"{self.name}:http",
                        evidence_url=data["final_url"])

    # -------------------------------------------------------------------- live
    def _live_fetch(self, url: str, max_chars: int) -> dict[str, Any]:
        headers = {
            "User-Agent": self.settings.map_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en",
        }
        with http_client(timeout=self.timeout, headers=headers,
                         settings=self.settings) as client:
            response = client.get(url)
            final_url = str(response.url)
            status = int(response.status_code)
            content_type = str(response.headers.get("content-type", ""))
            checked = http_result((status, response.content),
                                  evidence_url=final_url, settings=self.settings,
                                  tool_name=self.name)
        if not checked.ok:
            raise ToolSignal(checked)

        body = checked.data[1].decode("utf-8", errors="replace")
        text = extract_text(body)
        trimmed = self._truncate(text, max_chars)
        return {
            "url": url,
            "final_url": final_url,
            "text": trimmed,
            "char_count": len(trimmed),
            "truncated": len(trimmed) < len(text),
            "status_code": status,
            "content_type": content_type,
            "screenshot_path": None,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    # -------------------------------------------------------------- screenshot
    def _capture_screenshot(self, url: str) -> ToolResult:
        """Playwright capture, or an honest refusal.

        Returns ``ToolResult.success({"path": ...})`` only once the file's size
        has been confirmed on disk. Any other outcome is a reason string, and
        the caller records ``screenshot_path=None``.
        """
        if not _playwright_available():
            # Gated through ``_playwright_available`` (not the raw import) so a
            # test can force this refusal without a browser binary present.
            return ToolResult.unavailable(
                reason=("playwright is not installed; no screenshot was taken and "
                        "none is claimed"))
        sync_playwright = _playwright_module()
        if sync_playwright is None:
            return ToolResult.unavailable(reason="playwright import failed")

        out_dir = Path(self.settings.artifacts_dir) / "evidence"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        out_path = out_dir / f"shot_{digest}.png"
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import TimeoutError as PlaywrightTimeout
            playwright_errors: tuple[type[BaseException], ...] = (
                PlaywrightError, PlaywrightTimeout)
        except (ImportError, ModuleNotFoundError):
            return ToolResult.unavailable(reason="playwright API not importable")

        capture = guarded(self._pw_capture, timeout=self.timeout,
                          tool_name=f"{self.name}.screenshot",
                          secrets=self.secrets, errors=playwright_errors)
        result = capture(sync_playwright, url, out_path)
        if not result.ok:
            return result

        written: Path = result.data
        try:
            size = written.stat().st_size
        except OSError:
            size = 0
        if size <= 0:
            # The critical check: never report a screenshot that is not on disk.
            return ToolResult.failed(
                reason=(f"{self.name}: playwright reported success but "
                        f"{out_path} is missing or empty; no screenshot claimed"),
                source=f"{self.name}.screenshot")
        return ToolResult.success(
            {"path": str(written), "bytes": size, "url": url},
            source=f"{self.name}.screenshot", evidence_url=url)

    def _pw_capture(self, sync_playwright: Any, url: str,
                    out_path: Path) -> Path:
        """Blocking Playwright capture. Raises for :func:`guarded` to convert."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as driver:
            browser = driver.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(url, timeout=int(self.timeout * 1000))
                page.screenshot(path=str(out_path), full_page=False)
            finally:
                browser.close()
        return out_path

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        limit = max(200, int(max_chars or 200))
        return text if len(text) <= limit else text[:limit]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FetchPageTool mode={self.current_mode()!r}>"


def _playwright_module() -> Any | None:
    """Import ``sync_playwright`` lazily, or ``None`` if it is absent.

    Lazy because Playwright pulls in a browser binary; importing it at module
    scope would make an optional, heavy dependency a hard one.
    """
    try:
        from playwright.sync_api import sync_playwright
    except (ImportError, ModuleNotFoundError):
        return None
    return sync_playwright


def _playwright_available() -> bool:
    return _playwright_module() is not None


# ===================================================================== verify
@tool("verify_evidence",
      "Decide whether fetched page text and/or real vision-model findings show "
      "that a promised deliverable was actually delivered. Returns fulfilled / "
      "not-fulfilled / cannot-verify with a reason. A URL or filename is never "
      "treated as evidence.",
      mode="deterministic", backend="rules")
class VerifyEvidenceTool(BaseTool):
    """Compliance check that inspects evidence instead of matching strings.

    Why the three-way verdict
    -------------------------
    ``fulfilled`` and ``not-fulfilled`` are both claims about the world; they
    need evidence. ``cannot-verify`` is the honest answer whenever the evidence
    cannot be inspected, and it is the answer the prototype could not produce.
    Callers (A5 compliance, A6 audit) must treat ``fulfilled=False`` with
    ``verdict="cannot-verify"`` as *unknown*, not as a breach — reporting
    "unfulfilled" for something nobody checked is the same sin as reporting
    "fulfilled".

    What counts as evidence
    -----------------------
    * ``page_text`` — the body returned by :class:`FetchPageTool`. Checked by
      distinct-term coverage of the promise.
    * ``vision`` — findings from :class:`~tools.vision.AnalyseEvidenceTool`.
      Usable, but capped in confidence and always labelled ``model-generated``.

    What does not count, ever
    -------------------------
    * the URL. ``https://x/sponsor/logo-partnership`` fulfils nothing. Searching
      it is the specific bug this tool exists to not have.
    * a filename, a path, a promise repeated in a caption.
    * the absence of a contradiction.
    """

    def current_mode(self) -> str:
        return "deterministic"

    def available(self) -> tuple[bool, str]:
        if not self.settings.tools_enabled:
            return False, "tools are disabled (settings.tools_enabled=False)"
        return True, ("deterministic evidence matcher: needs no backend and no "
                      "credentials; returns cannot-verify when given no evidence")

    def run(self, promise: str,
            page_text: str | None = None,
            evidence_url: str | None = None,
            image_path: str | None = None,
            vision: Mapping[str, Any] | None = None,
            *,
            required_terms: int = 2,
            coverage_threshold: float = 0.6) -> ToolResult:
        """Verify one ``promise``.

        :param promise: the commitment as written, e.g. "main-stage banner
            placement and a leaflet insert".
        :param page_text: body text **already fetched** by another tool. Never
            fetched here, so this tool makes no network call and cannot be
            tricked into reporting a fetch it did not do.
        :param evidence_url: recorded as provenance. Never searched.
        :param image_path: an image file that was *not* analysed. Its presence
            alone proves nothing; an image with no model findings yields
            ``cannot-verify`` for any promise that is not also in the text.
        :param vision: structured findings from a real vision call.
        :param required_terms: distinct promise terms that must appear.
        :param coverage_threshold: fraction of promise terms that must appear
            before "fulfilled" is allowed.
        """
        text = (promise or "").strip()
        if not text:
            return ToolResult.failed(reason=f"{self.name}: empty promise",
                                     source=f"{self.name}:input")

        terms = _promise_terms(text)
        if not terms:
            return ToolResult.unavailable(
                reason=(f"{self.name}: promise {text!r} contains no content term "
                        f"of 4+ characters, so no evidence can be matched against "
                        f"it. Refusing to guess what was promised."),
                source=f"{self.name}:input")

        # ---- collect the two admissible corpora ---------------------------
        corpus_text = (page_text or "").strip()
        text_source: str | None = None
        if len(corpus_text) >= _MIN_EVIDENCE_CHARS:
            text_source = "fetched-page-text"
        elif corpus_text:
            text_source = "fetched-page-text(too-short-to-substantiate)"

        vision_terms: set[str] = set()
        vision_note = ""
        vision_usable = False
        if isinstance(vision, Mapping):
            observations = vision.get("observations")
            if isinstance(observations, list) and observations:
                for obs in observations:
                    if not isinstance(obs, Mapping) or not obs.get("present"):
                        continue  # only things the model says it actually saw
                    for token in _tokens(str(obs.get("text") or "")):
                        if token in terms:
                            vision_terms.add(token)
                vision_usable = True
                vision_note = str(vision.get("note") or
                                  "model-generated observation; not independently verified")

        image_present = bool(image_path)
        provenance = {
            "promise": text,
            "evidence_url": evidence_url,
            "image_path": image_path if image_present else None,
            "image_inspected": bool(vision_usable),
            "page_text_chars": len(corpus_text),
            "page_text_source": text_source,
        }

        # ---- no admissible evidence at all --------------------------------
        usable_text = text_source == "fetched-page-text"
        if not usable_text and not vision_usable:
            return ToolResult.unavailable(
                reason=(
                    f"cannot-verify: {text!r} was not checked against anything. "
                    f"page_text={len(corpus_text)} chars"
                    + (" (below the 200-char floor for substantive evidence)"
                       if corpus_text else " (none supplied)")
                    + (f", image_path={image_path!r} was supplied but never "
                       f"analysed, and an unanalysed image is not evidence"
                       if image_present else "")
                    + ". A URL is recorded as provenance and is deliberately "
                      "not searched. Run fetch_page, then pass its text here; "
                      "or run analyse_evidence on the image first."
                ),
                source=f"{self.name}:no-evidence")

        # ---- text evidence ------------------------------------------------
        lowered = corpus_text.lower()
        text_hits = sorted({t for t in terms if _term_present(t, lowered)})
        coverage = len(text_hits) / len(terms)
        vision_hits = sorted(vision_terms)
        combined = sorted(set(text_hits) | set(vision_hits))

        satisfied = (
            usable_text
            and len(text_hits) >= max(1, required_terms)
            and coverage >= coverage_threshold
        )
        # Note what is deliberately absent: there is no branch in which vision
        # findings alone can set ``satisfied``. A model's description of an image
        # corroborates a text finding; it never becomes the finding. If there is
        # no fetched text, the best available verdict is ``cannot-verify``.

        if satisfied:
            matched = set(text_hits) | set(vision_hits)
            confidence = min(0.95, 0.55 + 0.4 * (len(matched) / len(terms)))
            if vision_usable:
                # A model description cannot be as strong as fetched text.
                confidence = min(confidence, 0.65)
            verdict = "fulfilled"
        elif usable_text:
            verdict = "not-fulfilled"
            confidence = round(min(0.8, 0.4 + 0.4 * coverage), 3)
        else:
            verdict = "cannot-verify"
            confidence = 0.0

        data: dict[str, Any] = {
            "verdict": verdict,
            "fulfilled": True if verdict == "fulfilled" else
                         (False if verdict == "not-fulfilled" else None),
            "promise": text,
            "terms": sorted(terms),
            "matched_terms": combined,
            "text_matched_terms": text_hits,
            "vision_matched_terms": vision_hits,
            "coverage": round(coverage, 3),
            "confidence": round(confidence, 3),
            "evidence_used": ([] if not usable_text else ["page_text"])
                             + (["vision:model-generated"] if vision_usable else []),
            "provenance": provenance,
            "reason": _explain(verdict, text, terms, text_hits, vision_hits, coverage),
            "note": vision_note,
            "checked_urls": [],
            "integrity_note": (
                "The evidence URL was recorded but never searched: matching a "
                "promise's words inside a URL or filename is not verification."
            ),
        }
        result = ToolResult.success(data, source=f"{self.name}:evidence",
                                    evidence_url=evidence_url)
        if verdict == "cannot-verify":
            annotate(result, degraded=True,
                     reason=f"{self.name}: {data['reason']}")
        return result


# ------------------------------------------------------------------- internals
def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of 2+ characters, in order."""
    return [t for t in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower())
            if len(t) >= 2]


def _promise_terms(promise: str) -> list[str]:
    """Content terms a promise should be evidenced by.

    Short words and pure filler are dropped, but note what is *kept*: brand
    names, numbers, and material nouns. Dropping "logo" and "placement" from
    the list is a deliberate narrowing — those are the words a template page
    carries whether or not anything was delivered.
    """
    seen: list[str] = []
    for token in _tokens(promise):
        if len(token) < 4 or token in _STOPWORDS or token in seen:
            continue
        seen.append(token)
    return seen


def _term_present(term: str, lowered_corpus: str) -> bool:
    """Whole-token containment, with a simple stem tolerance.

    Word-boundary matching prevents "stand" from matching "understand", which is
    the same class of error as matching a substring of a URL. The stem fallback
    (terms of 5+ characters also match with a trailing ``s``/``ed``/``ing``) is
    what lets "leaflet" match "leaflets" without making "leafe" match.
    """
    if re.search(rf"\b{re.escape(term)}\b", lowered_corpus):
        return True
    if len(term) < 5:
        return False
    return bool(re.search(rf"\b{re.escape(term)}(?:s|es|ed|ing)?\b", lowered_corpus))


def _explain(verdict: str, promise: str, terms: Sequence[str],
             text_hits: Sequence[str], vision_hits: Sequence[str],
             coverage: float) -> str:
    """One sentence saying exactly what was and was not found."""
    found = sorted(set(text_hits) | set(vision_hits))
    if verdict == "fulfilled":
        return (f"{promise!r} is supported by inspected evidence: "
                f"{len(found)}/{len(terms)} promise terms "
                f"({', '.join(found)}) appear in the evidence body.")
    if verdict == "not-fulfilled":
        missing = [t for t in terms if t not in found]
        return (f"{promise!r} is not supported by the inspected evidence: only "
                f"{len(found)}/{len(terms)} promise terms appear "
                f"({', '.join(found) or 'none'}); missing {', '.join(missing)}.")
    if vision_hits:
        return (f"{promise!r} cannot be verified from a model-generated "
                f"description alone: {', '.join(vision_hits)} was described by "
                f"the vision model, but no fetched text corroborates it.")
    return (f"{promise!r} cannot be verified: the evidence text is too short or "
            f"does not contain at least 2 distinct promise terms "
            f"(coverage {coverage:.2f} < 0.60). Not enough to call this either way.")


def sha256_text(text: str) -> str:  # pragma: no cover - provenance helper
    """Stable digest of evidence text, for trace-side comparison."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
