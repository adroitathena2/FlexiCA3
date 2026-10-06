"""Outbound email: real transport or an honest local spool. Never a fake send.

The rule this module exists to enforce
--------------------------------------
``ToolStatus.OK`` from :meth:`SendEmailTool.run` means **a transport accepted the
message**. Nothing else. The prototype's mailer appended to an in-memory list,
returned ``{"status": "sent"}``, and A3 wrote that into a thread record whose
``delivered`` flag then read ``True`` — an email that had never left the
process. Worse, when a brand had no address it invented
``partnerships@{brand}.com``, a domain that does not exist, and reported a
successful delivery to it.

So:

* local mode returns ``ok=False``, ``status=ToolStatus.UNAVAILABLE``, and the
  spooled path. Nothing in the pipeline can read that as a send.
* even a Resend 2xx is reported as *accepted*, not *delivered*. Resend returning
  an id means it queued the message; bounce and inbox arrival are separate
  events this tool never observes.
* recipients are validated and **refused**, never repaired. There is no code
  path that derives an address from a brand name.
* an idempotency key makes a retry safe: the same key never produces a second
  send, and a repeat call reports the original record instead.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.ids import new_id, utcnow
from core.protocols import ToolResult
from core.schemas import ToolStatus

from .base import (
    BaseTool,
    ToolSignal,
    annotate,
    guarded,
    http_client,
    http_result,
    redact,
    shaped_email,
    split_emails,
    tool,
    valid_email,
)

__all__ = ["SendEmailTool", "RESEND_ENDPOINT", "outbox_dir", "key_of"]

#: Resend's HTTP API. Not configurable: it is not a thing you self-host, and a
#: setting that could redirect mail to another host is a setting nobody audits.
RESEND_ENDPOINT = "https://api.resend.com/emails"

_MAX_BODY_CHARS = 200_000


def outbox_dir(settings: Any) -> Path:
    """Where local spooled messages land. Created on demand, never at import."""
    return Path(settings.artifacts_dir) / "outbox"


def key_of(record: dict[str, Any]) -> str:
    """Filesystem-safe rendering of an idempotency key.

    Keys come from caller code (``event_id:brand:day1``), so they may contain
    path separators. Stripping them prevents a key like ``../../etc/passwd`` from
    writing outside the outbox.
    """
    raw = str(record.get("key") or "message")
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw)
    return safe[:120] or "message"


@tool("send_email",
      "Send an email through the Resend API when RESEND_API_KEY is set and live "
      "tool calls are enabled. Otherwise spool it to artifacts/outbox and report "
      "UNAVAILABLE — a spooled message is never reported as sent.",
      mode="live", backend="resend")
class SendEmailTool(BaseTool):
    """Send one message, or record that it was not sent.

    Return-value contract, which A3 and the trace both depend on:

    ==========================  ============  ==============================
    outcome                     ``status``    ``data["sent"]``
    ==========================  ============  ==============================
    Resend accepted it          ``OK``        ``True`` (+ provider id)
    Spooled locally             ``UNAVAILABLE``  ``False`` (+ spool path)
    Malformed recipient         ``FAILED``    ``None``
    Reserved-TLD recipient live ``FAILED``    ``None``
    Duplicate idempotency key   ``CACHED``    ``False`` (original record)
    ==========================  ============  ==============================

    Reserved-TLD recipients (fixture ``.invalid`` contacts) are refused only
    for live sends; offline they are spooled as drafts (``UNAVAILABLE``, NOT
    sent) so the demo leaves an auditable draft instead of a refusal.
    """

    def current_mode(self) -> str:
        if self.settings.resend_api_key and self.live:
            return "live"
        return "local-outbox"

    def available(self) -> tuple[bool, str]:
        if not self.settings.tools_enabled:
            return False, "tools are disabled (settings.tools_enabled=False)"
        if self.live and self.settings.resend_api_key:
            return True, f"live delivery via Resend as {self.settings.email_from!r}"
        if self.live and not self.settings.resend_api_key:
            return False, "RESEND_API_KEY is not set; mail can only be spooled"
        return True, ("local outbox (TOOLS_LIVE=false): messages are written to "
                      "artifacts/outbox and reported as NOT sent")

    # ------------------------------------------------------------------- public
    def run(self, to: str = "", subject: str = "", body: str = "", *,
            from_address: str | None = None,
            idempotency_key: str | None = None,
            brand: str = "",
            event_id: str = "",
            attachments: Sequence[str] | None = None,
            reply_to: str | None = None) -> ToolResult:
        """Send ``body`` to ``to``.

        Every argument has a default because ``Tool.run(**kwargs)`` may be
        called with an incomplete payload; a missing argument becomes an honest
        ``FAILED`` with a reason, never a ``TypeError`` escaping into an agent.

        :param to: one address, or several separated by commas/semicolons. Every
            one must be well-shaped; reserved-TLD addresses are refused for
            live sends but spooled as offline drafts (reported NOT sent),
            because a half-sent sponsor pitch is worse than none.
        :param idempotency_key: stable key (e.g. ``f"{event_id}:{brand}:day1"``).
            Repeat calls with the same key return the first record instead of
            sending twice.
        :param attachments: paths to local files. A missing file fails the call.
        """
        recipients = split_emails(to)
        if not recipients:
            return ToolResult.failed(
                reason=(f"{self.name}: no recipient given. Addresses are never "
                        f"inferred from a brand name or a website."),
                source=f"{self.name}:input")
        malformed = [r for r in recipients if not shaped_email(r)]
        if malformed:
            # No repair path here and none should ever be added: guessing an
            # address creates mail that bounces, or worse, reaches someone
            # unrelated. Malformed addresses cannot even be drafted.
            return ToolResult.failed(
                reason=(f"{self.name}: refusing to send to {malformed!r}. Not a "
                        f"valid address shape (missing '@tld' or malformed). "
                        f"Addresses are never inferred from a brand name."),
                source=f"{self.name}:input")
        unroutable = [r for r in recipients if not valid_email(r)]
        if unroutable and self.live and self.settings.resend_api_key:
            # Live sends to reserved TLDs would bounce. Refuse, as before.
            return ToolResult.failed(
                reason=(f"{self.name}: refusing to send to {unroutable!r}. Not a "
                        f"routable address (reserved documentation domain or "
                        f"TLD). Addresses are never inferred from a brand name."),
                source=f"{self.name}:input")
        # Offline, well-shaped but unroutable recipients (fixture ``.invalid``
        # contacts) fall through to the spool below, which reports UNAVAILABLE /
        # NOT sent. Refusing to write a draft on routability grounds would break
        # the offline demo without protecting anyone -- exactly like the sender
        # handling below. The unroutable fact travels in the record.
        recipient_routable = not unroutable
        subject_line = (subject or "").strip()
        if not subject_line:
            return ToolResult.failed(reason=f"{self.name}: empty subject",
                                     source=f"{self.name}:input")
        text = (body or "")
        if len(text) > _MAX_BODY_CHARS:
            return ToolResult.failed(
                reason=(f"{self.name}: body is {len(text)} chars, over the "
                        f"{_MAX_BODY_CHARS} limit"),
                source=f"{self.name}:input")

        sender = (from_address or self.settings.email_from or "").strip()
        if not sender:
            return ToolResult.failed(
                reason=f"{self.name}: no sender configured; set EMAIL_FROM",
                source=f"{self.name}:input")
        if reply_to and not shaped_email(reply_to):
            return ToolResult.failed(
                reason=f"{self.name}: invalid reply_to {reply_to!r}",
                source=f"{self.name}:input")
        if reply_to and not valid_email(reply_to) and self.live and self.settings.resend_api_key:
            return ToolResult.failed(
                reason=f"{self.name}: invalid reply_to {reply_to!r} (not routable)",
                source=f"{self.name}:input")

        # The sender is only checked for routability when a message will
        # actually leave the process. ``EMAIL_FROM`` ships as
        # ``paytriq@example.invalid`` precisely so nothing is sent by accident,
        # and refusing to write a *draft* on that basis would break the offline
        # demo without protecting anyone. In live mode it is fatal, because a
        # reserved-TLD sender is guaranteed to bounce.
        sender_routable = valid_email(sender)
        if not sender_routable and self.live and self.settings.resend_api_key:
            return ToolResult.failed(
                reason=(f"{self.name}: configured sender {sender!r} is not a "
                        f"routable address, so a real send would bounce. Set "
                        f"EMAIL_FROM to a domain that receives mail."),
                source=f"{self.name}:input")

        files, file_error = self._read_attachments(attachments)
        if file_error is not None:
            return file_error

        key = (idempotency_key or "").strip() or new_id("msg")
        record: dict[str, Any] = {
            "key": key,
            "to": recipients,
            "from": sender,
            "subject": subject_line,
            "body": text,
            "brand": brand,
            "event_id": event_id,
            "attachments": [str(p) for p in files],
            "body_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "at": utcnow().isoformat(),
            "sender_routable": sender_routable,
            "recipient_routable": recipient_routable,
        }

        if self.live and self.settings.resend_api_key:
            return self._send_via_resend(record, reply_to)
        return self._spool(record)

    # ------------------------------------------------------------- local spool
    def _spool(self, record: dict[str, Any]) -> ToolResult:
        """Write the message to disk and report, unambiguously, that it was not sent.

        Status is ``UNAVAILABLE`` rather than ``OK`` and ``ok`` is ``False``,
        even though the message *was* durably recorded. That is deliberate: the
        caller asked for a send, and no send happened. Hand-constructing the
        ``ToolResult`` instead of using :meth:`ToolResult.unavailable` is the one
        place this module bypasses a factory, because that factory discards
        ``data`` and the spool path is exactly what the operator needs.
        """
        directory = outbox_dir(self.settings)
        record["transport"] = "local-outbox"
        record["sent"] = False
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{key_of(record)}.json"
            existing = _read_record(path)
            if existing is not None:
                result = ToolResult.cached(
                    {**existing, "outbox_path": str(path),
                     "sent": False, "duplicate": True},
                    source=f"{self.name}:outbox")
                return annotate(result, reason=(
                    f"{self.name}: idempotency key {record['key']!r} was already "
                    f"spooled at {path}; returning the original record instead of "
                    f"writing a second one. No transport was involved."))
            path.write_text(json.dumps(record, indent=2, default=str),
                            encoding="utf-8")
        except OSError as exc:
            return ToolResult.unavailable(
                reason=redact(f"{self.name}: could not write the local outbox at "
                              f"{directory} — {type(exc).__name__}: {exc}",
                              self.secrets),
                source=f"{self.name}:outbox")

        routable_note = ""
        if record.get("recipient_routable") is False:
            routable_note = " Recipients are well-shaped but not routable (reserved TLD); even with live transport this draft could not be delivered without a real address."
        return ToolResult(
            ok=False,
            data={
                "sent": False,
                "transport": "local-outbox",
                "outbox_path": str(path),
                "idempotency_key": record["key"],
                "to": record["to"],
                "subject": record["subject"],
                "body_sha256": record["body_sha256"],
                "accepted_by_provider": False,
                "sender_routable": record["sender_routable"],
                "recipient_routable": record.get("recipient_routable", True),
            },
            status=ToolStatus.UNAVAILABLE,
            source=f"{self.name}:outbox",
            degraded=True,
            reason=(f"NOT SENT — no transport accepted this message. Spooled to "
                    f"{path} because live calls are off "
                    f"(TOOLS_LIVE=false, RESEND_API_KEY unset). The file is a "
                    f"draft on disk, not a delivery.{routable_note}"),
        )

    # -------------------------------------------------------------------- live
    def _send_via_resend(self, record: dict[str, Any],
                         reply_to: str | None) -> ToolResult:
        """POST to Resend. ``OK`` means Resend accepted it, nothing stronger."""
        call = guarded(self._post_resend, timeout=self.timeout,
                       tool_name=self.name, secrets=self.secrets)
        result = call(record, reply_to)
        if not result.ok:
            return result
        provider = result.data
        return annotate(result, source=f"{self.name}:resend",
                        reason=(
                            f"accepted by Resend (id {provider.get('id', 'unknown')}). "
                            f"Acceptance is not delivery: inbox arrival and bounces "
                            f"are not observed by this tool."))

    def _post_resend(self, record: dict[str, Any],
                     reply_to: str | None) -> dict[str, Any]:
        api_key = self.settings.resend_api_key or ""
        if not api_key:
            raise ValueError("RESEND_API_KEY disappeared between gating and send")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Resend honours this header for 24h, which is what makes a retry
            # after a timeout safe instead of a duplicate pitch to a sponsor.
            "Idempotency-Key": str(record["key"]),
        }
        body: dict[str, Any] = {
            "from": record["from"],
            "to": record["to"],
            "subject": record["subject"],
            "text": record["body"],
        }
        if reply_to:
            body["reply_to"] = reply_to
        if record["attachments"]:
            body["attachments"] = [
                {
                    "filename": Path(p).name,
                    "content": base64.b64encode(
                        Path(p).read_bytes()).decode("ascii"),
                }
                for p in record["attachments"]
            ]
        with http_client(timeout=self.timeout, headers=headers,
                         settings=self.settings) as client:
            response = client.post(RESEND_ENDPOINT,
                                   content=json.dumps(body).encode("utf-8"))
            status = int(response.status_code)
            content = response.content
        checked = http_result((status, content), evidence_url=RESEND_ENDPOINT,
                              settings=self.settings, tool_name=self.name)
        if not checked.ok:
            # 5xx/429 stay UNAVAILABLE, a rejected envelope stays FAILED.
            raise ToolSignal(checked)
        try:
            payload = json.loads(checked.data[1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A 2xx with an unreadable body: the send may have gone through, so
            # this must not be retried blindly, and it must not be called OK
            # either. Reported as failed with the ambiguity stated.
            raise ValueError(
                "Resend returned HTTP 2xx with a body that is not JSON; the "
                "message may or may not have been queued, so this is not "
                "reported as sent") from None
        if not isinstance(payload, dict):
            raise TypeError(f"Resend returned {type(payload).__name__}, expected an object")
        return {"id": str(payload.get("id") or ""), "provider": "resend",
                "raw_keys": sorted(payload)[:8]}

    # ----------------------------------------------------------------- helpers
    def _read_attachments(
        self, paths: Sequence[str] | None,
    ) -> tuple[list[Path], ToolResult | None]:
        """Resolve attachment paths, or explain which one is unusable."""
        resolved: list[Path] = []
        for raw in paths or []:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(self.settings.artifacts_dir) / path
            if not path.exists():
                return [], ToolResult.failed(
                    reason=f"{self.name}: attachment not found: {path}",
                    source=f"{self.name}:input")
            if not path.is_file():
                return [], ToolResult.failed(
                    reason=f"{self.name}: attachment is not a file: {path}",
                    source=f"{self.name}:input")
            resolved.append(path)
        return resolved, None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SendEmailTool mode={self.current_mode()!r} live={self.live}>"


# ------------------------------------------------------------------- internals
def _read_record(path: Path) -> dict[str, Any] | None:
    """Load a previously spooled record, or ``None`` if absent/unreadable."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def guess_mime(path: Path) -> str:  # pragma: no cover - used when building .eml
    """Best-effort MIME type for an attachment path."""
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"
