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

History never forks. Revising a memory that was already superseded follows
``superseded_by`` to the newest version and revises that one (with a
``UserWarning`` saying so); :meth:`RevisionsMixin.current_version` returns
that id. ``resolve`` only accepts TODOs and bugs (other memories are replaced
with ``supersede`` or deleted with ``forget``); resolving a resolved memory
is a no-op that returns its id.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

from mark.kinds import KIND_TAG_PREFIX, TAG_RESOLVED, TAG_SUPERSEDED, kind_of
from mark.memory.facade.core import MemoryCore
from mark.types import MemoryFragment

#: Kinds ``resolve()`` closes.
RESOLVABLE_KINDS = ("todo", "bug")

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
        ``mark:superseded``. When *memory_id* was already superseded, its
        newest version is replaced instead (a ``UserWarning`` says so), so
        there is only ever one current version.
        """
        old = self._current(self._require(memory_id), "superseding")
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
        """Mark a TODO or bug as done; returns the id of the memory resolved.

        Resolved memories leave ``list(kind=..., open=True)`` and are demoted
        in recall, but stay stored. Only TODOs and bugs (``kind="todo"`` /
        ``"bug"`` or a leading ``TODO:``/``BUG:`` label) can be resolved;
        anything else raises :class:`mark.errors.InvalidInputError` (use
        :meth:`supersede` to replace it or ``forget`` to delete it).
        Resolving an already resolved memory changes nothing and returns its
        id; a superseded TODO resolves its newest version (with a
        ``UserWarning``). Unknown ids raise
        :class:`mark.errors.MemoryNotFoundError`.
        """
        fragment = self._require(memory_id)
        if TAG_RESOLVED in fragment.tags:
            return fragment.id  # already resolved: idempotent, keeps the first resolution
        fragment = self._current(fragment, "resolving")
        if TAG_RESOLVED in fragment.tags:
            return fragment.id
        kind = kind_of(fragment)
        if kind not in RESOLVABLE_KINDS:
            from mark.errors import InvalidInputError

            what = f"a {kind}" if kind else "a note without a kind"
            raise InvalidInputError(
                f"memory {fragment.id[:8]} is {what}; resolve() closes TODOs and bugs only. "
                "Use supersede() to replace it with the current text, or forget() to delete it")
        extra = {"resolved_at": _now()}
        if note:
            extra["resolution"] = note
        self._retire(fragment, lifecycle="resolved", tags=(TAG_RESOLVED, TAG_SUPERSEDED), **extra)
        return fragment.id

    def current_version(self, memory_id: str) -> str:
        """The id of the newest version of *memory_id* (itself when never superseded)."""
        return self._newest(self._require(memory_id)).id

    def _newest(self, fragment: MemoryFragment) -> MemoryFragment:
        """Follow ``superseded_by`` links to the newest version that still exists."""
        seen = {fragment.id}
        while True:
            nxt = fragment.metadata.get("superseded_by")
            newer = self.get(str(nxt)) if nxt and str(nxt) not in seen else None  # type: ignore[attr-defined]
            if newer is None:
                return fragment
            seen.add(newer.id)
            fragment = newer

    def _current(self, fragment: MemoryFragment, doing: str) -> MemoryFragment:
        """*fragment*, or its newest version (with a warning) when it was superseded."""
        newest = self._newest(fragment)
        if newest.id != fragment.id:
            warnings.warn(
                f"memory {fragment.id[:8]} was already superseded by {newest.id[:8]}; "
                f"{doing} the current version {newest.id[:8]} instead", UserWarning, stacklevel=3)
        return newest

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
