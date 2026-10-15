# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Revising memories without losing history: supersede, update, resolve.

Memories are never edited in place. A revision writes a *new* memory and
retires the old one, so "what was it originally?" stays answerable:

* the old memory gets the tag ``mark:superseded`` (retrieval demotes it) and
  metadata ``lifecycle``, ``superseded_by`` and ``superseded_at``;
* the new memory carries ``supersedes`` (the previous id) and ``history``
  (every earlier id, oldest first), and inherits kind, tags and session.

``resolve`` closes a TODO/bug: tags ``mark:resolved`` + ``mark:superseded``
and metadata ``lifecycle="resolved"``/``resolved_at``. The ``lifecycle``
metadata key is the memory's lifecycle state (``MemoryState`` is left alone,
because retired memories must stay retrievable for history questions).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mark.kinds import KIND_TAG_PREFIX, TAG_RESOLVED, TAG_SUPERSEDED, kind_of
from mark.memory.facade.core import MemoryCore
from mark.types import MemoryFragment

#: Metadata keys owned by the lifecycle; never copied onto a new revision.
_LIFECYCLE_KEYS = frozenset({
    "lifecycle", "superseded_by", "superseded_at", "resolved_at", "resolution",
    "supersedes", "history", "content_hash",
})


class RevisionsMixin(MemoryCore):
    """supersede / update / resolve for this agent's memories."""

    def _require(self, memory_id: str) -> MemoryFragment:
        fragment = self.get(memory_id)  # type: ignore[attr-defined]
        if fragment is None:
            from mark.errors import MemoryNotFoundError

            raise MemoryNotFoundError(memory_id)
        return fragment

    def supersede(self, memory_id: str, text: str, **options: Any) -> str:
        """Replace memory *memory_id* with *text*; returns the new memory's id.

        The new memory inherits the old one's kind, tags, session and
        importance (override any of them with keyword arguments accepted by
        :meth:`remember`). The old one stays stored, tagged
        ``mark:superseded``.
        """
        old = self._require(memory_id)
        history = [*_as_list(old.metadata.get("history")), old.id]
        metadata = {k: v for k, v in old.metadata.items() if k not in _LIFECYCLE_KEYS
                    and k not in {"occurred_at", "access_count"}}
        metadata.update(options.pop("metadata", None) or {})
        metadata.update(supersedes=old.id, history=history)
        inherited_tags = [t for t in old.tags
                          if not t.startswith(KIND_TAG_PREFIX) and not t.startswith("mark:")]
        options.setdefault("kind", kind_of(old.tags))
        options["tags"] = list(dict.fromkeys([*inherited_tags, *(options.get("tags") or [])]))
        options.setdefault("session_id", old.session_id)
        options.setdefault("importance", old.importance)
        options.setdefault("source", old.source)
        new_id = self.remember(text, metadata=metadata, **options)  # type: ignore[attr-defined]
        self._retire(old, lifecycle="superseded", superseded_by=new_id)
        return new_id

    def update(self, memory_id: str, text: str, **options: Any) -> str:
        """Edit a memory: :meth:`supersede` it with *text*, keeping the id history.

        Returns the new id; ``get(new_id).metadata["history"]`` lists every
        earlier id of this memory, oldest first.
        """
        return self.supersede(memory_id, text, **options)

    def resolve(self, memory_id: str, *, note: str | None = None) -> str:
        """Mark a memory (typically a TODO or bug) as done; returns its id.

        Resolved memories leave ``list(kind=..., open=True)`` and are demoted
        in recall, but stay stored. Raises
        :class:`mark.errors.MemoryNotFoundError` for an unknown id.
        """
        fragment = self._require(memory_id)
        extra = {"resolved_at": _now()}
        if note:
            extra["resolution"] = note
        self._retire(fragment, lifecycle="resolved", tags=(TAG_RESOLVED, TAG_SUPERSEDED), **extra)
        return fragment.id

    def _retire(
        self, fragment: MemoryFragment, *, lifecycle: str,
        tags: tuple[str, ...] = (TAG_SUPERSEDED,), **metadata: Any,
    ) -> None:
        """Persist the lifecycle tags/metadata on *fragment* (state is unchanged)."""
        meta = {**fragment.metadata, "lifecycle": lifecycle, **metadata}
        if lifecycle == "superseded":
            meta.setdefault("superseded_at", _now())
        new_tags = list(dict.fromkeys([*fragment.tags, *tags]))
        self._store.store(fragment.model_copy(update={"metadata": meta, "tags": new_tags,
                                                      "updated_at": datetime.now(timezone.utc)}))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []
