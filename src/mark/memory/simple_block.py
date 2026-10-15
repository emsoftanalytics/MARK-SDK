# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""SimpleBlock: a named, tag-scoped memory container for the simple API.

A block is not a stored entity: writes are fragments of the reserved simple
agent tagged ``block:<label>``, and the block is reconstructed from those
tags. ``MemoryRecord`` is the backward-compatible view returned to callers.
"""
from __future__ import annotations

from typing import Any

from mark.kinds import DEFAULT_CONFIDENCE, DEFAULT_IMPORTANCE, normalize_kind
from mark.memory.record import MemoryRecord, MemoryState


def _fragment_to_record(frag: Any, block_id: str) -> MemoryRecord:
    """Convert a MemoryFragment to a MemoryRecord for backward-compat API surface.

    The record state is always RAW: the simple API does not expose the
    fragment lifecycle, only its content and scores.
    """
    return MemoryRecord(
        id         = frag.id,
        block_id   = block_id,
        content    = frag.content,
        importance = frag.importance,
        confidence = frag.confidence,
        state      = MemoryState.RAW,
        source     = frag.source or "local",
        metadata   = dict(frag.metadata),
        created_at = frag.created_at,
    )


class SimpleBlock:
    """
    A named memory container in the simple API.

    Writes are stored as MemoryFragments tagged with the block label,
    so they can be scoped during retrieval with blocks=[...].
    """

    def __init__(self, label: str, mark_memory: Any, *, kind: str = "custom",
                 description: str = "") -> None:
        self._label  = label
        self._memory = mark_memory
        self._kind = kind or "custom"
        self._description = description
        # A block kind that names a memory kind ("preference", "todos"...) is
        # applied to every write; any other kind is a descriptive label only.
        try:
            self._memory_kind = normalize_kind(self._kind) if self._kind != "custom" else None
        except ValueError:
            self._memory_kind = None

    @property
    def kind(self) -> str:
        """The block's kind label (``"custom"`` unless given to ``block()``)."""
        return self._kind

    @property
    def label(self) -> str:
        """Return the display label."""
        return self._label

    @property
    def id(self) -> str:
        """Block identifier — equals the label in the simple API."""
        return self._label

    def write(
        self,
        content:    str,
        *,
        importance: float = DEFAULT_IMPORTANCE,
        confidence: float = DEFAULT_CONFIDENCE,
        metadata:   dict[str, Any] | None = None,
        source:     str = "local",
        state:      MemoryState = MemoryState.RAW,
    ) -> MemoryRecord:
        """Write content to this block. Returns a MemoryRecord.

        Defaults match every other write API (importance 0.5, confidence 1.0);
        invalid input raises :class:`mark.errors.InvalidInputError`.
        """
        meta = dict(metadata or {})
        if self._kind != "custom":
            meta.setdefault("block_kind", self._kind)
        frag_id = self._memory.remember(
            content,
            kind       = self._memory_kind,
            importance = importance,
            confidence = confidence,
            tags       = [f"block:{self._label}"],
            source     = source,
            metadata   = meta,
        )
        # Fetch the stored fragment so created_at matches what is persisted
        stored = self._memory._store.get(frag_id)
        if stored is not None:
            return _fragment_to_record(stored, self._label)

        # Fallback — should not happen, but never raise on write
        from datetime import datetime, timezone
        return MemoryRecord(
            id         = frag_id,
            block_id   = self._label,
            content    = content,
            importance = importance,
            confidence = confidence,
            state      = state,
            source     = source,
            metadata   = metadata or {},
            created_at = datetime.now(timezone.utc),
        )

    def records(self) -> list[MemoryRecord]:
        """Return all records stored in this block."""
        return [
            _fragment_to_record(f, self._label)
            for f in self._memory.list(limit=None, tags=[f"block:{self._label}"])
        ]

    def to_dict(self) -> dict[str, Any]:
        """Describe this block as a plain dictionary (records not included)."""
        return {"id": self._label, "label": self._label, "kind": self._kind,
                "description": self._description}

