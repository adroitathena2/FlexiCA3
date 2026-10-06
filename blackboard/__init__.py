"""Paytriq blackboard: the typed, append-only shared workspace.

Every agent in the system communicates through this package and nothing else.
Two agents never call each other, never share memory, and never read each
other's files. They post findings and read findings. That is the whole
coordination model, and it is the classical one:

* B. Hayes-Roth, *A blackboard architecture for control*, Artificial Intelligence
  26(3):251-321, 1985 -- independent knowledge sources operating through a
  common, incrementally-updated data structure, with a partitioned knowledge
  base.
* B. Nii, *Blackboard Systems*, Stanford CS-TR-86-1123, 1986 -- the requirements
  such a structure must meet for the system to be testable: inspectable,
  incremental, and never silently discarding what it has already learned.

Three modules:

``zones``
    The registry of typed regions: which kinds each accepts, and which agent owns
    it.
``board``
    :class:`InMemoryBlackboard`, implementing ``core.protocols.Blackboard``.
``serialize``
    Snapshot / restore / JSONL / derivation-view rendering for replay and audit.

Usage::

    board = InMemoryBlackboard()
    entry = board.post_model("opportunities", lead)          # typed storage
    offer = board.post_model("offers", Offer(...),
                             refs=[entry.entry_id])          # justified
    board.latest_model("offers", Offer)                       # read back typed

One thing callers must know up front: **each zone declares an owner, and posts by
someone other than the owner are counted, not rejected.** ``risk_flags`` is owned
by A5, ``offers`` by A2, and so on; cross-zone work is real (A5 writes its own
``audit_finding`` entries into ``audit``), and ``core.schemas`` fixes no author
per artefact, so rejecting a non-owner would break legitimate work and guard
against nothing -- the kind-to-model binding already guarantees one definition
per artefact. What a non-owner post *does* do is appear in
``stats()["foreign_posts"]``, so the anomaly is visible rather than buried.

Every zone has a closed set of accepted kinds. If you need one this registry does
not have, call :func:`register_zone` rather than smuggling a payload into a
neighbouring zone -- the kind check exists precisely to catch that.

Import rule: ``blackboard`` depends on ``core`` only (never on ``agents``,
``graph`` or ``api``), which is what keeps the dependency graph acyclic and lets
this package be tested with no wiring at all.
"""
from __future__ import annotations

from .board import InMemoryBlackboard
from .serialize import (
    SNAPSHOT_KIND,
    SNAPSHOT_VERSION,
    entry_from_dict,
    entry_to_dict,
    from_jsonl,
    render_tree,
    restore,
    snapshot,
    to_jsonl,
)
from .zones import (
    AUTHOR_FIELD,
    CONFIDENCE_FIELD,
    KIND_MODELS,
    MODEL_KINDS,
    REGISTERED_ZONES,
    SOURCE_FIELD,
    UNTYPED_KINDS,
    ZONE_NAMES,
    ZONES,
    Zone,
    register_zone,
    validate_post,
    zone,
    zones_all,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # zones
    "Zone", "ZONES", "ZONE_NAMES", "REGISTERED_ZONES", "KIND_MODELS",
    "MODEL_KINDS", "UNTYPED_KINDS", "AUTHOR_FIELD", "CONFIDENCE_FIELD",
    "SOURCE_FIELD", "zone", "zones_all", "register_zone", "validate_post",
    # board
    "InMemoryBlackboard",
    # serialization
    "SNAPSHOT_KIND", "SNAPSHOT_VERSION", "snapshot", "restore",
    "to_jsonl", "from_jsonl", "render_tree", "entry_to_dict", "entry_from_dict",
]
