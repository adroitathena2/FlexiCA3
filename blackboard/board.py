"""In-memory, typed, append-only blackboard.

This is the coordination substrate for the whole of Paytriq. Agents never call
each other. They read the board, form a view, and post the consequences; the
board is the only shared state, and it is the reason the run can be audited
afterwards.

The architecture is classical:

* B. Hayes-Roth, *A blackboard architecture for control*, Artificial Intelligence
  26(3):251-321, 1985. The system is a set of independent knowledge sources
  working through a common, incrementally-updated data structure; the control
  mechanism is selection of the next knowledge source to apply, and the data
  structure is partitioned into typed regions with different acceptance
  conditions.
* B. Nii, *Blackboard Systems*, Stanford CS-TR-86-1123, 1986. Formalises the
  division into sub-solutions, the notion of an application-specific scheduler,
  and the requirement that the knowledge base be inspectable and testable.

Three properties are load-bearing here. The first two are enforced rather than
documented-and-hoped-for:

**Append-only.** There is no ``update``, no ``delete``, no ``clear``. Revisions
are new entries (``Offer.version``, a second ``MoU``, a superseding
``RiskFlag``), so the trail of how a decision was reached survives the run. On
top of that, entries are copied on the way in and on the way out: a caller that
mutates the dict it posted, or the entry object it read back, cannot reach
history. The audit trail is therefore genuinely immutable rather than immutably
*intended*. The single other write path is the private ``_ingest`` used by
``blackboard.serialize.restore``: a snapshot cannot be reloaded losslessly
through ``post``, which by design assigns its own sequence numbers, ids and
timestamps. ``_ingest`` re-checks every validation ``post`` does and rejects
out-of-order sequences and duplicate ids.

**Typed.** :meth:`InMemoryBlackboard.post_model` stores a validated
``core.schemas`` model alongside its dict form. When two agents both claim to
hold "an offer", the board does not have to pick a representation -- there is
one, and :meth:`get_model` hands it back as an ``Offer``. This is the direct
answer to the "two implementations with incompatible schemas" failure mode that
made the previous prototype's artefacts unverifiable. Note what typing does
*not* claim: it does not stop two agents from posting two offers. It guarantees
they are the same shape, separately versioned and separately readable.

**Justified.** Every entry may cite earlier entries in ``refs``. Those citation
edges are what turn a pile of records into an argument:
:meth:`refs_of` walks the chain and ``blackboard.serialize.render_tree`` prints
it, so "why did we do this?" is answerable from the board alone.

Thread safety: the API layer posts from request handlers while an SSE stream
reads, so every mutation and every read of internal state happens under one
re-entrant lock. Sequence numbers are assigned inside that lock, which is what
makes them monotonic under concurrency.

Import rule: ``blackboard`` depends on ``core`` only.
"""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from core.errors import BlackboardError, SchemaError
from core.ids import new_id, utcnow
from core.protocols import BoardEntry
from core.schemas import AgentId, DecisionSource

from .zones import (
    AUTHOR_FIELD,
    CONFIDENCE_FIELD,
    KIND_MODELS,
    MODEL_KINDS,
    SOURCE_FIELD,
    ZONE_NAMES,
    Zone,
    validate_post,
)
from .zones import zone as resolve_zone

__all__ = ["InMemoryBlackboard"]


def _normalise(token: str) -> str:
    """Same identifier normalisation as the zone registry."""
    return (token or "").strip().lower()


def _copy_entry(entry: BoardEntry) -> BoardEntry:
    """Deep copy of an entry, so a reader cannot mutate stored history."""
    return BoardEntry(
        entry_id=entry.entry_id,
        zone=entry.zone,
        kind=entry.kind,
        author=entry.author,
        payload=deepcopy(entry.payload),
        refs=list(entry.refs),
        confidence=entry.confidence,
        source=entry.source,
        seq=entry.seq,
        at=entry.at,
    )


def _coerce_source(source: Any) -> DecisionSource:
    """Accept a :class:`DecisionSource` or its string value; reject anything else.

    Normalising here means ``read()`` consumers never have to defend against a
    bare string that slipped past the Protocol's type hint.
    """
    if isinstance(source, DecisionSource):
        return source
    if isinstance(source, str):
        try:
            return DecisionSource(source.strip().lower())
        except ValueError as exc:
            valid = [s.value for s in DecisionSource]
            raise BlackboardError(
                f"unknown decision source {source!r}; expected one of {valid}"
            ) from exc
    raise BlackboardError(
        f"source must be a DecisionSource or its value, got "
        f"{type(source).__name__}"
    )


def _check_confidence(confidence: float) -> float:
    """Confidence is a probability. Refuse anything outside ``[0, 1]``.

    A confidence of ``82`` or ``-1`` would be arithmetically convenient and
    meaningless; the blackboard is the last place that can catch it before it is
    quoted in a demo.
    """
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise BlackboardError(
            f"confidence must be a number in [0, 1], got "
            f"{type(confidence).__name__}: {confidence!r}"
        )
    value = float(confidence)
    if not 0.0 <= value <= 1.0:
        raise BlackboardError(f"confidence must be within [0, 1], got {value}")
    return value


def _coerce_refs(refs: list[str] | None) -> list[str]:
    """Validate a citation list. References are stored verbatim, never rewritten.

    An entry may legitimately cite something that is not a board entry -- a risk
    flag id or an evidence URL is exactly what ``Dispute.evidence`` and
    ``RiskFlag.evidence`` are for -- so nothing is filtered here. Traversal in
    :meth:`InMemoryBlackboard.refs_of` is what resolves what resolves.
    """
    if refs is None:
        return []
    if isinstance(refs, (str, bytes)) or not isinstance(refs, (list, tuple)):
        raise BlackboardError(
            f"refs must be a list of entry ids, got {type(refs).__name__}"
        )
    out: list[str] = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, str) or not ref.strip():
            raise BlackboardError(
                f"refs[{index}] must be a non-empty string entry id, got {ref!r}"
            )
        out.append(ref.strip())
    return out


class InMemoryBlackboard:
    """Concrete :class:`~core.protocols.Blackboard`.

    Implements the Protocol exactly as declared in ``core.protocols`` and adds
    typed storage, reference traversal and statistics on top. Everything the
    Protocol promises, plus:

    ``post_model(zone, obj, ...)``
        Post a validated ``core.schemas`` model; keeps the typed object.
    ``get_model(entry_id, model_cls)``
        Rebuild a typed artefact from an entry (stored model or raw payload).
    ``latest_model(zone, model_cls, kind=None)``
        Most recent typed artefact of that class in a zone.
    ``refs_of(entry_id, transitive=True)``
        Citation chain of an entry.
    ``stats()``
        Entry counts per zone, author and kind.
    ``entry(entry_id)``
        One entry by id, without scanning ``history()``.
    ``model_kind_of(entry_id)``
        Which typed model (if any) an entry keeps.
    """

    __slots__ = ("_lock", "_entries", "_index", "_models", "_next_seq")

    def __init__(self) -> None:
        # Re-entrant because ``post_model`` composes ``post`` and then indexes the
        # model under the same lock without a nesting hazard.
        self._lock = threading.RLock()
        self._entries: list[BoardEntry] = []
        self._index: dict[str, BoardEntry] = {}
        self._models: dict[str, BaseModel] = {}
        self._next_seq: int = 1

    # ===================================================================== write
    def post(self, zone: str, kind: str, author: AgentId,
             payload: dict[str, Any], *,
             refs: list[str] | None = None, confidence: float = 1.0,
             source: DecisionSource = DecisionSource.RULES) -> BoardEntry:
        """Append one entry and return a copy of it.

        The **only** write path on the board. Nothing is ever mutated or removed,
        so a superseded artefact is a later entry, not an edit of an earlier one.

        ``payload`` is stored verbatim -- this method never adds, renames,
        defaults or infers a field. A board entry is exactly what the agent said
        it was, plus the board's own bookkeeping (``seq``, ``entry_id``, ``at``).

        Raises :class:`~core.errors.ZoneNotFound` for an unknown zone and
        :class:`~core.errors.BlackboardError` for a disallowed kind, a
        non-dict payload, out-of-range confidence, an unknown decision source or
        a non-``AgentId`` author. Nothing is written when validation fails.

        A zone's declared owner is not a write restriction -- see
        :class:`blackboard.zones.Zone` -- but any post by a non-owner is counted
        in ``stats()["foreign_posts"]``.
        """
        zone_def = validate_post(zone, kind, author)
        normalised_kind = _normalise(kind)
        if not isinstance(payload, dict):
            raise BlackboardError(
                f"payload must be a dict, got {type(payload).__name__}"
            )
        checked_confidence = _check_confidence(confidence)
        checked_source = _coerce_source(source)
        # Copied on the way in: a caller reusing its payload dict afterwards must
        # not be able to rewrite history.
        body = deepcopy(payload)
        checked_refs = _coerce_refs(refs)

        with self._lock:
            seq = self._next_seq
            self._next_seq = seq + 1
            entry = BoardEntry(
                entry_id=new_id("be"),
                zone=zone_def.name,
                kind=normalised_kind,
                author=author,
                payload=body,
                refs=checked_refs,
                confidence=checked_confidence,
                source=checked_source,
                seq=seq,
                at=utcnow().isoformat(),
            )
            self._entries.append(entry)
            self._index[entry.entry_id] = entry
        return _copy_entry(entry)

    def post_model(self, zone: str, obj: BaseModel, *,
                   author: AgentId | None = None,
                   refs: list[str] | None = None,
                   confidence: float | None = None,
                   source: DecisionSource | None = None) -> BoardEntry:
        """Post a validated ``core.schemas`` model, keeping it typed.

        The kind is derived from the model's class via the registry, so a
        ``BrandLead`` posted into ``opportunities`` becomes a ``brand_lead``
        entry and nothing else can become that kind by accident. The instance is
        re-validated on the way in -- an artefact that has drifted (pydantic
        models here use ``validate_assignment``, but ``object.__setattr__`` and
        restored payloads can still smuggle an invalid one past construction) is
        rejected with :class:`~core.errors.SchemaError` instead of becoming a
        fact of record.

        Omitted metadata is resolved in this order, and never invented:

        * ``author`` -- the explicit argument, else the model's own author field
          (``Lesson.author``, ``Handoff.from_agent``, ``RiskFlag.raised_by``,
          ``Bid.agent``, ``Dispute.claimant``), else the zone's declared
          owner (:attr:`blackboard.zones.Zone.authoritative_agent`).
        * ``confidence`` / ``source`` -- the explicit argument, else the model's
          own ``confidence`` / ``decision_source`` field, else the Protocol
          defaults.

        ``refs`` is **never** inferred. A model's ``evidence`` list holds risk
        flag ids and URLs, not board entry ids, and quietly promoting one of
        those into a citation edge would put a dangling reference into the
        dependency graph. Callers cite explicitly.

        Raises :class:`~core.errors.BlackboardError` when ``obj`` is not a
        registry-known model or no author can be resolved, and
        :class:`~core.errors.SchemaError` when it fails validation.
        """
        if not isinstance(obj, BaseModel):
            raise BlackboardError(
                f"post_model needs a core.schemas model instance, got "
                f"{type(obj).__name__}"
            )
        model_cls = type(obj)
        kind = MODEL_KINDS.get(model_cls)
        if kind is None:
            known = sorted(m.__name__ for m in KIND_MODELS.values())
            raise BlackboardError(
                f"{model_cls.__name__} has no blackboard kind; post it as a raw "
                f"payload instead. Typed kinds: {known}"
            )
        zone_def = resolve_zone(zone)

        resolved_author = self._resolve_author(obj, model_cls, author, zone_def)
        data = obj.model_dump(mode="json")
        try:
            validated = model_cls.model_validate(data)
        except ValidationError as exc:
            raise SchemaError(
                f"{model_cls.__name__} posted to {zone_def.name!r} failed "
                f"re-validation: {exc}"
            ) from exc

        checked_refs = _coerce_refs(refs)
        checked_confidence = _check_confidence(
            confidence if confidence is not None
            else self._field_or_none(model_cls, obj, CONFIDENCE_FIELD, 1.0)
        )
        checked_source = _coerce_source(
            source if source is not None
            else self._field_or_none(model_cls, obj, SOURCE_FIELD,
                                     DecisionSource.RULES)
        )

        entry = self.post(zone_def.name, kind, resolved_author, data,
                          refs=checked_refs, confidence=checked_confidence,
                          source=checked_source)
        with self._lock:
            self._models[entry.entry_id] = validated
        return entry

    # ====================================================================== read
    def read(self, zone: str, *, kind: str | None = None,
             limit: int | None = None) -> list[BoardEntry]:
        """Entries in a zone, oldest first.

        ``limit`` keeps the **most recent** ``limit`` entries and still returns
        them chronologically -- the useful order for "what is the state of
        play". ``limit=0`` returns nothing. Raises
        :class:`~core.errors.ZoneNotFound` for an unknown zone, so a typo is
        loud rather than indistinguishable from an empty zone.
        """
        zone_def = resolve_zone(zone)
        wanted_kind = _normalise(kind) if kind is not None else None
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise BlackboardError(
                    f"limit must be an int or None, got {type(limit).__name__}"
                )
            if limit < 0:
                raise BlackboardError(f"limit must be >= 0, got {limit}")
        with self._lock:
            matches = [
                entry for entry in self._entries
                if entry.zone == zone_def.name
                and (wanted_kind is None or entry.kind == wanted_kind)
            ]
        if limit is not None:
            matches = matches[len(matches) - limit:] if limit else []
        return [_copy_entry(entry) for entry in matches]

    def latest(self, zone: str, kind: str | None = None) -> BoardEntry | None:
        """Most recent matching entry, or ``None`` when the zone has none."""
        zone_def = resolve_zone(zone)
        wanted_kind = _normalise(kind) if kind is not None else None
        with self._lock:
            for entry in reversed(self._entries):
                if entry.zone != zone_def.name:
                    continue
                if wanted_kind is not None and entry.kind != wanted_kind:
                    continue
                return _copy_entry(entry)
        return None

    def zones(self) -> list[str]:
        """Every registered zone name, whether or not it holds an entry.

        The registry, not the data: a caller asking "what may I read?" must see
        the zones that are still empty, or it will never learn that
        ``audit`` exists until after the thing it exists to catch.
        """
        return list(ZONE_NAMES)

    def history(self) -> list[BoardEntry]:
        """Every entry ever posted, in posting order. The audit trail."""
        with self._lock:
            return [_copy_entry(entry) for entry in self._entries]

    def entry(self, entry_id: str) -> BoardEntry:
        """One entry by id.

        Raises :class:`~core.errors.BlackboardError` for an unknown id: citing a
        non-existent entry is the one error that would make a justification
        chain unfalsifiable, so it is never silently tolerated.
        """
        return _copy_entry(self._require(entry_id))

    # =============================================================== typed reads
    def get_model(self, entry_id: str, model_cls: type[BaseModel]) -> BaseModel:
        """Rebuild a typed artefact from an entry.

        Returns the model kept by :meth:`post_model` when there is one, otherwise
        validates the raw payload against ``model_cls``. The second path is what
        makes this more than a convenience: a caller that has a dict on the board
        can still demand that it *be* an ``Offer``, and get
        :class:`~core.errors.SchemaError` if it is not.
        """
        with self._lock:
            found = self._require(entry_id)
            stored = self._models.get(entry_id)
            payload = deepcopy(found.payload)
            kind = found.kind
        if stored is not None and isinstance(stored, model_cls):
            return stored.model_copy(deep=True)
        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            raise SchemaError(
                f"entry {entry_id} (kind={kind!r}) is not a valid "
                f"{model_cls.__name__}: {exc}"
            ) from exc

    def latest_model(self, zone: str, model_cls: type[BaseModel],
                     kind: str | None = None) -> BaseModel | None:
        """Most recent typed artefact of ``model_cls`` in a zone, or ``None``.

        Candidates are selected by kind -- explicitly, or from the registry when
        ``model_cls`` names a known kind -- and then validated. A candidate that
        fails validation raises :class:`~core.errors.SchemaError` rather than
        being skipped, because silently reaching further back for a valid-looking
        older offer is precisely how a stale price gets quoted as current.
        """
        zone_def = resolve_zone(zone)
        wanted_kind = _normalise(kind) if kind is not None else (
            MODEL_KINDS.get(model_cls)
        )
        with self._lock:
            candidates = [
                entry for entry in reversed(self._entries)
                if entry.zone == zone_def.name
                and (wanted_kind is None or entry.kind == wanted_kind)
            ]
        if not candidates:
            return None
        return self.get_model(candidates[0].entry_id, model_cls)

    # ============================================================ reference graph
    def model_kind_of(self, entry_id: str) -> str | None:
        """Kind name of the typed model held by an entry, or ``None`` if raw.

        Lets a serializer record whether an entry is reconstructible without
        guessing: a snapshot that says which entries are typed can re-check that
        claim on the way back in.
        """
        with self._lock:
            self._require(entry_id)
            stored = self._models.get(entry_id)
        return MODEL_KINDS.get(type(stored)) if stored is not None else None

    def refs_of(self, entry_id: str, *, transitive: bool = True) -> list[str]:
        """Citation chain of an entry.

        With ``transitive=True`` (the default) this is the full ancestry, walked
        breadth-first from the closest precedent outwards, deduplicated, and
        excluding the entry itself. That is how an agent justifies a claim by
        citing board entries: post ``refs=board.refs_of(evidence_entry_id)`` and
        the new entry inherits the whole chain of reasoning, not just the last
        hop.

        References that do not resolve to an entry (a risk flag id, a URL) are
        omitted here -- there is nothing to walk -- while remaining recorded
        verbatim on the entry itself.

        Raises :class:`~core.errors.BlackboardError` for an unknown id.
        """
        with self._lock:
            start = self._require(entry_id)
            if not transitive:
                return list(start.refs)
            seen: set[str] = {entry_id}
            ordered: list[str] = []
            frontier = [entry_id]
            while frontier:
                current = frontier.pop(0)
                node = self._index.get(current)
                if node is None:
                    continue
                for ref in node.refs:
                    if ref in seen:
                        continue
                    seen.add(ref)
                    parent = self._index.get(ref)
                    if parent is None:
                        # A non-entry citation. Recorded, not traversable.
                        continue
                    ordered.append(ref)
                    frontier.append(ref)
            return ordered

    # ======================================================================= stats
    def stats(self) -> dict[str, Any]:
        """Entry counts per zone, per author and per kind, plus citation edges.

        Every registered zone appears, including empty ones: a dashboard that
        hides the zones nobody wrote to is the dashboard that misses the run that
        only reached discovery.

        ``foreign_posts`` counts posts made by someone other than the zone's
        declared owner. Those posts are accepted -- cross-zone work is real, and
        ``core.schemas`` fixes no author per artefact -- but they are reported
        rather than hidden, so "A6 raised a risk flag" and "A5 wrote into the
        audit zone" are visible in the run statistics instead of being invisible
        in the middle of the board.
        """
        from .zones import ZONES as _ZONES

        with self._lock:
            zone_counts: dict[str, int] = {name: 0 for name in ZONE_NAMES}
            author_counts: dict[str, int] = {}
            kind_counts: dict[str, int] = {}
            foreign_counts: dict[str, int] = {name: 0 for name in ZONE_NAMES}
            edge_count = 0
            for entry in self._entries:
                zone_counts[entry.zone] = zone_counts.get(entry.zone, 0) + 1
                author_counts[entry.author.value] = (
                    author_counts.get(entry.author.value, 0) + 1
                )
                kind_counts[entry.kind] = kind_counts.get(entry.kind, 0) + 1
                edge_count += len(entry.refs)
                owner = getattr(_ZONES.get(entry.zone), "authoritative_agent", None)
                if owner is not None and entry.author is not owner:
                    foreign_counts[entry.zone] = (
                        foreign_counts.get(entry.zone, 0) + 1
                    )
            foreign_total = sum(foreign_counts.values())
            return {
                "entries": len(self._entries),
                "typed_entries": len(self._models),
                "zones": zone_counts,
                "authors": author_counts,
                "kinds": kind_counts,
                "ref_edges": edge_count,
                "foreign_posts": {k: v for k, v in foreign_counts.items() if v},
                "foreign_post_total": foreign_total,
            }

    # ==================================================================== helpers
    def _require(self, entry_id: str) -> BoardEntry:
        """The stored entry for ``entry_id``, or raise. Caller holds the lock.

        Every by-id accessor funnels through here so that "unknown entry_id"
        reads identically everywhere -- it is the message a reader debugging a
        justification chain will be staring at.
        """
        found = self._index.get(entry_id)
        if found is None:
            raise BlackboardError(
                f"unknown entry_id {entry_id!r}; the board holds "
                f"{len(self._entries)} entries"
            )
        return found

    @staticmethod
    def _resolve_author(obj: BaseModel, model_cls: type[BaseModel],
                        author: AgentId | None, zone_def: Zone) -> AgentId:
        """Pick the entry author: explicit, then the model's field, then the zone."""
        if author is not None:
            if not isinstance(author, AgentId):
                raise BlackboardError(
                    f"author must be a core.schemas.AgentId, got "
                    f"{type(author).__name__}: {author!r}"
                )
            return author
        field = AUTHOR_FIELD.get(model_cls)
        if field is not None:
            candidate = getattr(obj, field, None)
            if isinstance(candidate, AgentId):
                return candidate
        if zone_def.authoritative_agent is not None:
            return zone_def.authoritative_agent
        raise BlackboardError(
            f"author is required for {model_cls.__name__} in zone "
            f"{zone_def.name!r}: the zone declares no owner and the model "
            "carries no author field"
        )

    @staticmethod
    def _field_or_none(model_cls: type[BaseModel], obj: BaseModel,
                       table: dict[type[BaseModel], str], default: Any) -> Any:
        """Read an optional model field named in one of the registry tables."""
        field = table.get(model_cls)
        if field is None:
            return default
        value = getattr(obj, field, None)
        return default if value is None else value

    # ---------------------------------------------------------------- ingestion
    def _ingest(self, entry: BoardEntry, *,
                model: BaseModel | None = None) -> BoardEntry:
        """Re-append a reconstructed entry verbatim (used by ``serialize.restore``).

        This is the only other write path and it is deliberately *not* public:
        a restore must be able to reproduce sequence numbers, ids and timestamps
        exactly, which means bypassing id assignment. Everything that makes an
        entry trustworthy is still re-checked -- zone, kind, author, refs,
        confidence, source -- so a corrupted snapshot fails loudly instead of
        loading a plausible-looking board.
        """
        validate_post(entry.zone, entry.kind, entry.author)
        if not isinstance(entry.entry_id, str) or not entry.entry_id.strip():
            raise BlackboardError("a restored entry needs a non-empty entry_id")
        if isinstance(entry.seq, bool) or not isinstance(entry.seq, int) or entry.seq < 1:
            raise BlackboardError(
                f"entry {entry.entry_id!r} has a non-positive or non-int seq: "
                f"{entry.seq!r}"
            )
        if not isinstance(entry.payload, dict):
            raise BlackboardError(
                f"entry {entry.entry_id!r} payload must be a dict, got "
                f"{type(entry.payload).__name__}"
            )
        confidence = _check_confidence(entry.confidence)
        source = _coerce_source(entry.source)
        refs = _coerce_refs(entry.refs)

        restored = BoardEntry(
            entry_id=entry.entry_id.strip(),
            zone=_normalise(entry.zone),
            kind=_normalise(entry.kind),
            author=entry.author,
            payload=deepcopy(entry.payload),
            refs=refs,
            confidence=confidence,
            source=source,
            seq=entry.seq,
            at=entry.at or utcnow().isoformat(),
        )
        with self._lock:
            if restored.entry_id in self._index:
                raise BlackboardError(
                    f"duplicate entry_id {restored.entry_id!r} in snapshot"
                )
            if self._entries and restored.seq <= self._entries[-1].seq:
                raise BlackboardError(
                    f"snapshot entries are not strictly increasing in seq: "
                    f"{restored.entry_id} has seq={restored.seq} after "
                    f"seq={self._entries[-1].seq}"
                )
            self._entries.append(restored)
            self._index[restored.entry_id] = restored
            self._next_seq = max(self._next_seq, restored.seq + 1)
            if model is not None:
                self._models[restored.entry_id] = model
        return _copy_entry(restored)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        with self._lock:
            return (f"<InMemoryBlackboard entries={len(self._entries)} "
                    f"typed={len(self._models)} next_seq={self._next_seq}>")
