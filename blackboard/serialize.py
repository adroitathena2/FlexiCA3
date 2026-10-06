"""Snapshot, restore and dependency rendering for the Paytriq blackboard.

The blackboard is the audit trail, so being able to *re-open* it after the run
is not a convenience -- it is what makes the run inspectable at all. This module
is the read-only half of that promise:

``snapshot(board)`` / ``restore(data)``
    Full JSON-safe state, in and out of a dict. Round-trip fidelity is exact:
    entry ids, sequence numbers, timestamps, citation edges and typed models all
    survive, so a restored board is indistinguishable from the original.
``to_jsonl(board)`` / ``from_jsonl(text)``
    One entry per line, for streaming into a database, diffing two runs, or
    appending a run to a durable log. Same schema as the snapshot entries, so
    the two formats never disagree.
``render_tree(board)``
    Human-readable derivation view derived from ``refs``. Given an audit question
    -- "why did A7 rule that way?" -- this prints the decision with everything it
    was built on indented beneath it. No model, no inference: the view *is* the
    citation graph.

Nothing here mutates an existing board. Every function takes one as input and
returns plain data, or a *new* board (``restore`` / ``from_jsonl``), which is the
only way an entry id or a sequence number is ever reproduced rather than freshly
assigned.

Why JSON-safe conversion is strict
----------------------------------
Unsupported value types raise :class:`~core.errors.BlackboardError` naming the
entry and the type, rather than being coerced with ``str()``. A snapshot whose
evidence URL has quietly become ``"<object at 0x7f...>"`` is worse than no
snapshot: it looks like data. Failing here, with the offending entry id, means
the fault is attributable to the agent that posted it.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from core.errors import BlackboardError, ZoneNotFound
from core.ids import utcnow
from core.protocols import BoardEntry
from core.schemas import AgentId, DecisionSource

from .board import InMemoryBlackboard
from .zones import KIND_MODELS
from .zones import zone as resolve_zone

__all__ = [
    "SNAPSHOT_KIND",
    "SNAPSHOT_VERSION",
    "snapshot",
    "restore",
    "to_jsonl",
    "from_jsonl",
    "render_tree",
    "entry_to_dict",
    "entry_from_dict",
]

#: Identifies the payload format, so a restore never guesses.
SNAPSHOT_KIND = "paytriq.blackboard.snapshot"
SNAPSHOT_VERSION = 1

#: Entry fields a restore cannot do without.
_REQUIRED_FIELDS = ("entry_id", "zone", "kind", "author", "seq")

#: JSON-native types passed through untouched.
_JSON_SCALARS = (str, int, float, bool, type(None))


def _jsonable(value: Any, *, where: str) -> Any:
    """Recursively convert ``value`` into JSON-safe data, or raise.

    Accepted: ``str/int/float/bool/None``, ``list``, ``dict`` with string keys,
    ``datetime``/``date`` (ISO 8601, which every ``core.schemas`` model parses
    back), ``Enum`` (its value), and ``BaseModel`` (``model_dump(mode="json")``).

    Everything else -- a ``set``, an arbitrary object, a ``Decimal`` -- raises.
    ``Decimal`` is not accepted on purpose: it is not in ``core.schemas`` today,
    and guessing a rounding rule for it here would be inventing data.
    """
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value, where=where)
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"), where=where)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, where=where) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BlackboardError(
                    f"{where}: JSON object keys must be strings, got "
                    f"{type(key).__name__}: {key!r}"
                )
            out[key] = _jsonable(item, where=where)
        return out
    raise BlackboardError(
        f"{where}: cannot serialise {type(value).__name__} to JSON; post the "
        f"value as a model (see blackboard.zones.KIND_MODELS) or a JSON-native "
        "type instead"
    )


def entry_to_dict(entry: BoardEntry, model_kind: str | None = None) -> dict[str, Any]:
    """One entry as a JSON-safe dict, optionally naming its typed model."""
    where = f"entry {entry.entry_id}"
    out: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "zone": entry.zone,
        "kind": entry.kind,
        "author": entry.author.value,
        "seq": entry.seq,
        "at": entry.at,
        "confidence": entry.confidence,
        "source": entry.source.value,
        "refs": [str(ref) for ref in entry.refs],
        "payload": _jsonable(entry.payload, where=f"{where} payload"),
    }
    if model_kind is not None:
        out["model_kind"] = model_kind
    return out


def entry_from_dict(raw: Any, *, origin: str = "entry") -> tuple[BoardEntry, BaseModel | None]:
    """Inverse of :func:`entry_to_dict`.

    Returns the entry plus its re-validated typed model when ``model_kind`` names
    a registry model and the payload validates. A ``model_kind`` that does not
    validate is an error, not a warning: a snapshot claiming an entry is a valid
    ``Offer`` and storing something else is exactly the incompatible-schema
    failure this package exists to prevent.

    Raises :class:`~core.errors.BlackboardError` on malformed input.
    """
    if not isinstance(raw, dict):
        raise BlackboardError(f"{origin}: expected a JSON object, got {type(raw).__name__}")
    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise BlackboardError(f"{origin}: missing required field(s) {missing}")
    author_value = raw["author"]
    try:
        author = author_value if isinstance(author_value, AgentId) else AgentId(author_value)
    except ValueError as exc:
        raise BlackboardError(f"{origin}: unknown author {author_value!r}") from exc
    source_value = raw.get("source", DecisionSource.RULES.value)
    try:
        source = (source_value if isinstance(source_value, DecisionSource)
                  else DecisionSource(source_value))
    except ValueError as exc:
        raise BlackboardError(
            f"{origin}: unknown decision source {source_value!r}"
        ) from exc
    refs = raw.get("refs", [])
    if not isinstance(refs, list):
        raise BlackboardError(f"{origin}: refs must be a list, got {type(refs).__name__}")
    try:
        entry = BoardEntry(
            entry_id=str(raw["entry_id"]),
            zone=str(raw["zone"]),
            kind=str(raw["kind"]),
            author=author,
            payload=dict(raw.get("payload") or {}),
            refs=[str(ref) for ref in refs],
            confidence=float(raw.get("confidence", 1.0)),
            source=source,
            seq=int(raw["seq"]),
            at=str(raw.get("at", "")),
        )
    except (TypeError, ValueError) as exc:
        raise BlackboardError(f"{origin}: malformed entry: {exc}") from exc

    model: BaseModel | None = None
    model_kind = raw.get("model_kind")
    if model_kind is not None:
        model_cls = KIND_MODELS.get(str(model_kind))
        if model_cls is None:
            known = sorted(KIND_MODELS)
            raise BlackboardError(
                f"{origin}: unknown model_kind {model_kind!r}; known kinds: {known}"
            )
        model = _revalidate(model_cls, entry)
    return entry, model


def _revalidate(model_cls: type[BaseModel], entry: BoardEntry) -> BaseModel:
    try:
        return model_cls.model_validate(entry.payload)
    except ValidationError as exc:
        raise BlackboardError(
            f"entry {entry.entry_id} claims model_kind {model_cls.__name__!r} "
            f"but its payload does not validate: {exc}"
        ) from exc


# ============================================================================ #
# dict <-> board
# ============================================================================ #
def snapshot(board: InMemoryBlackboard) -> dict[str, Any]:
    """Full JSON-safe state of a board.

    Includes the zone registry order, the statistics at capture time and every
    entry in order with its ``model_kind`` where one is stored. ``captured_at``
    records when the snapshot was taken, not when the entries were posted, so a
    snapshot is distinguishable from the board it came from at a glance.
    """
    entries: list[dict[str, Any]] = []
    for entry in board.history():
        model_kind = _model_kind_for(board, entry.entry_id)
        entries.append(entry_to_dict(entry, model_kind))
    return {
        "kind": SNAPSHOT_KIND,
        "version": SNAPSHOT_VERSION,
        "captured_at": utcnow().isoformat(),
        "zones": board.zones(),
        "stats": board.stats(),
        "entries": entries,
    }


def _model_kind_for(board: InMemoryBlackboard, entry_id: str) -> str | None:
    """Kind name of the typed model stored against an entry, if any."""
    return board.model_kind_of(entry_id)


def restore(data: dict[str, Any]) -> InMemoryBlackboard:
    """Rebuild a board from :func:`snapshot` output (or any equivalent dict).

    Entries are re-appended in order through the board's internal ingestion path,
    which re-validates zone, kind, author, refs, confidence and source but keeps
    the original ``entry_id``, ``seq`` and ``at``. That is what makes the
    round-trip lossless: a restored board answers ``refs_of``, ``history`` and
    ``latest_model`` exactly as the original did, and posting after a restore
    continues the sequence rather than restarting it.

    Raises :class:`~core.errors.BlackboardError` on anything malformed, with the
    index of the offending entry.
    """
    if not isinstance(data, dict):
        raise BlackboardError(
            f"snapshot must be a dict, got {type(data).__name__}"
        )
    declared_kind = data.get("kind")
    if declared_kind is not None and declared_kind != SNAPSHOT_KIND:
        raise BlackboardError(
            f"expected a {SNAPSHOT_KIND!r} payload, got {declared_kind!r}"
        )
    version = data.get("version")
    if version is not None and version != SNAPSHOT_VERSION:
        raise BlackboardError(
            f"unsupported snapshot version {version!r}; this build reads "
            f"version {SNAPSHOT_VERSION}"
        )
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise BlackboardError(
            f"snapshot entries must be a list, got {type(raw_entries).__name__}"
        )
    board = InMemoryBlackboard()
    for index, raw in enumerate(raw_entries):
        try:
            entry, model = entry_from_dict(raw, origin=f"entries[{index}]")
            board._ingest(entry, model=model)
        except BlackboardError as exc:
            raise BlackboardError(f"cannot restore entries[{index}]: {exc}") from exc
    declared_zones = data.get("zones")
    if isinstance(declared_zones, list):
        for name in declared_zones:
            if not isinstance(name, str):
                raise BlackboardError(
                    f"snapshot zones must be strings, got {type(name).__name__}"
                )
            try:
                resolve_zone(name)
            except ZoneNotFound as exc:
                raise BlackboardError(
                    f"snapshot references zone {name!r} that is not registered "
                    "in this build"
                ) from exc
    return board


# ============================================================================ #
# jsonl
# ============================================================================ #
def to_jsonl(board: InMemoryBlackboard) -> str:
    """One JSON object per entry, chronologically, newline-terminated.

    Deterministic: keys are sorted and non-ASCII is preserved, so two snapshots
    of the same board diff cleanly and ``sha256`` of the output is stable.
    """
    lines: list[str] = []
    for entry in board.history():
        record = entry_to_dict(entry, _model_kind_for(board, entry.entry_id))
        lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return "".join(line + "\n" for line in lines)


def from_jsonl(text: str) -> InMemoryBlackboard:
    """Rebuild a board from :func:`to_jsonl` output.

    Returns a board (not a list) so that a replayed run can be read and extended
    with exactly the same API as the original. Blank lines are skipped; any other
    malformed line raises :class:`~core.errors.BlackboardError` naming the 1-based
    line number, because a truncated log that silently restores 41 of 42 entries
    is a corrupted run presented as a whole one.
    """
    if not isinstance(text, str):
        raise BlackboardError(f"jsonl input must be str, got {type(text).__name__}")
    board = InMemoryBlackboard()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BlackboardError(
                f"jsonl line {line_number} is not valid JSON: {exc}"
            ) from exc
        try:
            entry, model = entry_from_dict(raw, origin=f"line {line_number}")
            board._ingest(entry, model=model)
        except BlackboardError as exc:
            raise BlackboardError(f"cannot restore line {line_number}: {exc}") from exc
    return board


# ============================================================================ #
# rendering
# ============================================================================ #
def _line(entry: BoardEntry, seq_width: int, zone_width: int,
          kind_width: int, cites: str) -> str:
    """One rendered entry, e.g. ``#12  [contracts      ] mou         A4  c=1.00``."""
    seq = f"#{entry.seq:<{seq_width}}"
    zone = f"[{entry.zone:<{zone_width}}]"
    kind = f"{entry.kind:<{kind_width}}"
    tail = f"c={entry.confidence:.2f} src={entry.source.value}"
    return f"{seq} {zone} {kind} {entry.author.value:<3} {tail}{cites}"


def _depths(entries: list[BoardEntry]) -> dict[str, int]:
    """Derivation depth of every entry: 0 for a root, else 1 + deepest ancestor.

    Depth is well defined by induction over ``seq``: an entry can only cite
    entries that already exist, so every reference points backwards and one
    forward pass suffices. A reference that does not resolve (a risk-flag id, a
    URL) contributes nothing -- there is no ancestor to measure -- and a
    defensive guard keeps a hand-edited snapshot from looping forever.
    """
    seq_of = {entry.entry_id: entry.seq for entry in entries}
    depth: dict[str, int] = {}
    for entry in sorted(entries, key=lambda e: e.seq):
        ancestors = [
            depth[ref] for ref in entry.refs
            if ref in seq_of and seq_of[ref] < entry.seq
        ]
        depth[entry.entry_id] = 1 + max(ancestors) if ancestors else 0
    return depth


def render_tree(board: InMemoryBlackboard) -> str:
    """Render the citation graph as a depth-indented derivation view.

    The question this exists to answer is "why was that decision taken?". Every
    entry is printed once, indented by its distance from the earliest evidence,
    with the sequence numbers it explicitly cites listed as ``cites #a #b``.
    Walking up the page from a ruling therefore walks back through the exact
    chain of reasoning that produced it.

    Depth-ordered rather than a drawn tree on purpose. A drawn tree degenerates
    quickly on a real board: an entry that cites three predecessors which share
    an ancestor forces the shared node to be drawn twice, and the drawing
    collapses into a thicket of "already shown" markers. The ladder prints each
    entry exactly once, keeps the branching explicit in the ``cites`` list, and
    stays readable under the dense cross-referencing a real run produces.

    Purely derived. Nothing is invented about *why* an entry exists beyond the
    citations its author recorded.
    """
    entries = board.history()
    if not entries:
        return "Paytriq blackboard: empty (no entries posted)\n"

    seq_of = {entry.entry_id: entry.seq for entry in entries}
    depth = _depths(entries)
    ordered = sorted(entries, key=lambda e: (depth[e.entry_id], e.seq))

    seq_width = max(len(f"#{e.seq}") for e in entries)
    zone_width = max(len(z) for z in board.zones())
    kind_width = max(len(e.kind) for e in entries)

    edge_count = sum(1 for e in entries for ref in e.refs if ref in seq_of)
    roots = sum(1 for e in entries if depth[e.entry_id] == 0)

    out: list[str] = [
        f"Paytriq blackboard derivation view: {len(entries)} entries, "
        f"{edge_count} citation edges, {roots} root"
        f"{'' if roots == 1 else 's'}",
        "indent = derivation depth; `cites` = the board entries each one rests on",
        "",
    ]
    for entry in ordered:
        cited = [f"#{seq_of[ref]}" for ref in entry.refs if ref in seq_of]
        cites = ("  cites " + " ".join(cited)) if cited else ""
        out.append(
            "  " * depth[entry.entry_id]
            + _line(entry, seq_width, zone_width, kind_width, cites)
        )
    return "\n".join(out) + "\n"
