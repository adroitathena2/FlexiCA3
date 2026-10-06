"""Recorded model outputs, keyed by a hash of the request that produced them.

The one rule this module exists to enforce
------------------------------------------
**Replay returns recorded bytes. It never synthesises plausible text.**

That sentence is the line between an honest offline demo and a fabrication, and
the difference is not subtle. A replayed run is a *performance* of a previous
run: the model was really called, at a real time, with a real answer, and what
you are looking at is that answer. It is legitimate evidence of what the system
does — provided the recording is real.

The moment a cache miss falls back to "here is something that reads like a
model would have said", the output stops being evidence of anything and becomes
a prop. Nobody reading the trace can tell, because the schema, the ids, the
provenance fields and the confidence numbers all still look correct. So:

* :meth:`JsonlReplayStore.get` returns ``None`` on a miss. Always. Never a
  default, never a template, never a "close enough" string.
* :meth:`JsonlReplayStore.stats` exposes ``misses`` so a caller can *see* that it
  is asking for things that were never recorded, rather than quietly receiving
  invented text.
* Every call is attributed to a ``DecisionSource``. ``REPLAY`` means "this was
  recorded"; it is never used to mean "this was made up".

Canonical keying
----------------
:meth:`JsonlReplayStore.key` hashes the SHA-256 of the canonical JSON encoding of
``{"namespace": ..., "payload": ...}`` — sorted keys, no insignificant
whitespace, UTF-8. Canonical form matters because a replay key has to be stable
across processes, platforms and Python versions; the same logical request must
hit the same line or the demo silently stops being deterministic.

The namespace is inside the hashed envelope deliberately. Two subsystems that
ask the same question of different models (a Gemini call and a Clef call with an
identical prompt) must not collide, because they did not receive identical
answers.

Storage is JSONL, append-only, one entry per line, same discipline as the trace:
never rewrite a recording. A recording that has been overwritten is a recording
nobody can audit. Re-recording an existing key appends a new line and is counted
as a conflict; the most recent entry wins on read, which is the only rule that
lets a demo be re-run without a stale-cache bug.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from enum import Enum
from pathlib import Path
from typing import Any

from core.errors import ReplayIntegrityError

__all__ = ["JsonlReplayStore", "canonical_json"]

logger = logging.getLogger("paytriq.observability")


def _canonical_default(value: Any) -> Any:
    """Make an arbitrary payload hashable *without inventing meaning*.

    Anything exotic is reduced to its ``repr``, which is deterministic for a
    given object but obviously not its data. That is on purpose: a caller who
    hashes a non-canonical payload should get a key that will never match again
    rather than one that collides with a different payload.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "backslashreplace")
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_default(v) for v in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return repr(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON text for hashing.

    ``sort_keys`` plus fixed separators and ``ensure_ascii=False`` (hashed as
    UTF-8, not as an escape sequence) means the encoding does not depend on dict
    insertion order, on Python version, or on the machine's default encoding.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_default,
    )


class JsonlReplayStore:
    """A JSONL-backed implementation of the ``core.protocols.ReplayStore`` protocol.

    Concurrency: a single :class:`threading.RLock` guards the file handle and
    the in-memory index, because the obvious failure mode for a shared replay
    file is two threads appending at once and producing one spliced line.
    Reads go through the index, so ``get`` is O(1) after the first call.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self._path = Path(path)
        self._read_only = read_only
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._raw: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        self._conflicts = 0
        self._writes = 0
        self._handle: Any | None = None

    # ------------------------------------------------------------- properties
    @property
    def path(self) -> Path:
        return self._path

    @property
    def read_only(self) -> bool:
        return self._read_only

    # -------------------------------------------------------------------- key
    @staticmethod
    def key(namespace: str, payload: Any) -> str:
        """SHA-256 of the canonical JSON of ``{"namespace", "payload"}``.

        A staticmethod with no instance state, so a caller can compute the key
        it *should* have and compare it against what it got — that comparison is
        how a replay-hit rate is audited rather than trusted.
        """
        envelope = canonical_json({"namespace": str(namespace), "payload": payload})
        return hashlib.sha256(envelope.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------- load
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._path.exists():
                self._read_file()
            self._loaded = True

    def _read_file(self) -> None:
        with self._path.open("r", encoding="utf-8", newline="") as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    entry = json.loads(text)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ReplayIntegrityError(
                        f"{self._path}:{lineno} is not valid JSON: {exc}. A replay "
                        f"file that cannot be parsed must not be presented as real."
                    ) from exc
                if not isinstance(entry, dict) or "key" not in entry:
                    raise ReplayIntegrityError(
                        f"{self._path}:{lineno} is not a replay entry (no 'key')."
                    )
                key = str(entry["key"])
                if key in self._index:
                    self._conflicts += 1
                    logger.warning(
                        "%s: key %s re-recorded at line %d; most recent wins",
                        self._path.name, key[:12], lineno,
                    )
                self._index[key] = entry
                self._raw[key] = text

    def _append_line(self, entry: Mapping[str, Any]) -> None:
        if self._read_only:
            raise ReplayIntegrityError(
                f"{self._path} is open read-only; refusing to record into it"
            )
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
        with self._lock:
            if self._handle is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self._path.open("a", encoding="utf-8", newline="\n")
            try:
                self._handle.write(line)
                self._handle.write("\n")
                # Flushed on every write: a demo that crashes mid-run must still
                # have the recordings it genuinely made.
                self._handle.flush()
            except OSError as exc:
                logger.error("cannot append to replay file %s: %s", self._path, exc)
                raise ReplayIntegrityError(f"cannot write {self._path}: {exc}") from exc

    # --------------------------------------------------------- ReplayStore API
    def get(self, key: str) -> dict[str, Any] | None:
        """Return the recorded output for ``key``, or ``None``.

        ``None`` means "not recorded". It does **not** mean "here is something
        reasonable instead" — see the module docstring. Callers are expected to
        turn a miss into a ``DecisionSource.RULES`` fallback or an honest
        ``DecisionUnavailable``, both of which are visible in the trace.
        """
        self._ensure_loaded()
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                self._misses += 1
                logger.info("replay miss for key %s (nothing recorded; "
                            "caller must degrade rather than invent)", key[:12])
                return None
            self._hits += 1
            output = entry.get("output")
            if not isinstance(output, dict):
                # Corrupt or hand-edited entry. Refusing is the whole point: the
                # alternative is a plausible-looking object of unknown origin.
                raise ReplayIntegrityError(
                    f"replay entry {key[:12]} has a non-object output "
                    f"({type(output).__name__}); refusing to return it"
                )
            # Deep-copied so a caller cannot mutate the stored recording in
            # memory and make a later get() return something that was never
            # recorded.
            return deepcopy(output)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store ``value`` under ``key``.

        No verification that the key matches the payload that produced it — the
        caller owns that mapping — but the mismatch is recorded in ``stats()``
        under ``unverifiable_entries`` when :meth:`record` was not used.
        """
        if not isinstance(value, dict):
            raise ReplayIntegrityError(
                f"replay values must be JSON objects, got {type(value).__name__}"
            )
        entry = {
            "key": key,
            "namespace": "",
            "payload": None,
            "output": value,
            "source": "put",
            "at": datetime.now(UTC).isoformat(),
        }
        self._store(key, entry)

    # ---------------------------------------------------------------- record
    def record(self, namespace: str, payload: Any, output: dict[str, Any],
               *, model: str = "", run_id: str = "", event_id: str = "",
               source: str = "replay", at: datetime | None = None) -> str:
        """Record a genuine model output and return its key.

        This is the *only* supported way to populate a store in a live run,
        because it stores the request alongside the response. A recording with
        no request attached is an answer nobody can check was asked for.

        Args:
            namespace: subsystem, e.g. ``"clef.decide"`` or ``"gemini.chat"``.
            payload: the request, exactly as sent. Hashed into the key.
            output: the response, exactly as received. Stored verbatim.
            model: model id, for auditing which backend produced it.
            run_id / event_id: provenance so a recording can be tied back.
            source: ``DecisionSource`` value; ``"live"`` for a genuine call.
            at: call time; defaults to now. Injected by tests for determinism.

        Returns:
            The replay key, so the caller can assert that a later ``get`` hits.
        """
        if not isinstance(output, dict):
            raise ReplayIntegrityError(
                f"recorded outputs must be JSON objects, got {type(output).__name__}; "
                f"a model response that is not an object cannot be replayed honestly"
            )
        stamp = at or datetime.now(UTC)
        key = self.key(namespace, payload)
        entry = {
            "key": key,
            "namespace": str(namespace),
            "payload": payload,
            "output": output,
            "model": model,
            "source": source,
            "run_id": run_id,
            "event_id": event_id,
            "at": stamp.isoformat() if isinstance(stamp, datetime) else str(stamp),
        }
        self._store(key, entry)
        return key

    def _store(self, key: str, entry: dict[str, Any]) -> None:
        self._append_line(entry)
        with self._lock:
            if key in self._index:
                self._conflicts += 1
            self._index[key] = entry
            self._raw[key] = json.dumps(entry, sort_keys=True, ensure_ascii=False,
                                        default=str)
            self._writes += 1
            self._loaded = True

    # --------------------------------------------------------------- read side
    def has(self, key: str) -> bool:
        """True when ``key`` was recorded. Does not count as a hit or a miss."""
        self._ensure_loaded()
        with self._lock:
            return key in self._index

    def get_raw(self, key: str) -> str | None:
        """The exact JSON text stored for ``key``.

        For auditing: a caller that suspects the replay path is altering data can
        compare ``get_raw`` byte-for-byte against what the provider returned.
        """
        self._ensure_loaded()
        with self._lock:
            return self._raw.get(key)

    def entries(self) -> list[dict[str, Any]]:
        """Every stored entry, in first-seen order (recording order)."""
        self._ensure_loaded()
        with self._lock:
            return [deepcopy(entry) for entry in self._index.values()]

    def keys(self) -> list[str]:
        self._ensure_loaded()
        with self._lock:
            return list(self._index)

    def stats(self) -> dict[str, Any]:
        """Honest counters, including the ones that look bad.

        ``misses`` is the number this module would most like an assessor to
        look at. A high miss count means the demo is not actually replaying; it
        means the run needed outputs that were never recorded.
        """
        self._ensure_loaded()
        with self._lock:
            namespaces = sorted({
                str(entry.get("namespace") or "")
                for entry in self._index.values()
            } - {""})
            lookups = self._hits + self._misses
            return {
                "path": str(self._path),
                "exists": self._path.exists(),
                "read_only": self._read_only,
                "entry_count": len(self._index),
                "namespaces": namespaces,
                "bytes": self._path.stat().st_size if self._path.exists() else 0,
                "writes": self._writes,
                "hits": self._hits,
                "misses": self._misses,
                "lookups": lookups,
                "hit_rate": round(self._hits / lookups, 4) if lookups else 0.0,
                "conflicts": self._conflicts,
                # ``unverifiable_entries`` counts entries written via put() with no
                # payload: present in the file but impossible to reproduce.
                "unverifiable_entries": sum(
                    1 for entry in self._index.values()
                    if entry.get("payload") is None
                ),
            }

    def close(self) -> None:
        """Flush and close the append handle. Safe to call more than once."""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except OSError as exc:
                    logger.error("cannot close replay file %s: %s", self._path, exc)
                finally:
                    self._handle = None

    def flush(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                except OSError as exc:
                    logger.error("cannot flush replay file %s: %s", self._path, exc)

    def __enter__(self) -> JsonlReplayStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<JsonlReplayStore {self._path.name} entries={len(self._index)}>"
