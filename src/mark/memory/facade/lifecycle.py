# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Fragment lifecycle for MarkMemory: get, list, count, forget, promote, clear, vacuum.

Deleting operations remove the retrieval index entry *before* the stored
row, so a fragment can never be retrieved after it has been forgotten.
``forget`` cascades to derived facts (see
:mod:`mark.store._sqlite.lifecycle`); revisions (supersede / resolve /
update) live in :mod:`mark.memory.facade.revisions`.
"""
from __future__ import annotations

import warnings
from typing import Any

from mark.kinds import TAG_RESOLVED, TAG_SUPERSEDED, kind_tag, normalize_kind
from mark.memory.facade.core import MemoryCore
from mark.types import MemoryFragment, MemoryState

#: list() page size when the caller does not choose one. A sentinel (not a
#: plain 100) lets list() warn only when the *default* silently truncated.
_DEFAULT_LIMIT = 100


class _Default(int):
    """An int that remembers it was not passed explicitly."""


_UNSET_LIMIT = _Default(_DEFAULT_LIMIT)


class LifecycleMixin(MemoryCore):
    """Inspection and maintenance of this agent's stored fragments."""

    def get(self, memory_id: str) -> MemoryFragment | None:
        """The memory with this id (any state), or None when it does not exist here."""
        fragment = self._store.get(memory_id)
        if fragment is None or fragment.agent_id != self.agent_id:
            return None
        return fragment

    def list(
        self,
        states: list[MemoryState] | None = None,
        limit: int | None = _UNSET_LIMIT,
        *,
        offset: int = 0,
        kind: str | None = None,
        open: bool | None = None,  # noqa: A002 - mirrors `mark list --open`
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryFragment]:
        """This agent's memories, most important (then newest) first.

        Page with ``limit``/``offset`` (``limit=None`` returns everything).
        ``kind`` keeps one kind (``"todo"``...); ``open=True`` drops resolved
        and superseded memories, ``open=False`` keeps only those. When the
        default page size cuts the listing short a ``UserWarning`` says how
        many exist, so nothing is ever capped silently.
        """
        filters = self._list_filters(states, kind, open, tags, session_id)
        page = self._store.page_fragments(self.agent_id, limit=limit, offset=offset, **filters)
        if isinstance(limit, _Default) and len(page) == limit:
            total = self._store.count_fragments(self.agent_id, **filters)
            if total > offset + len(page):
                warnings.warn(
                    f"list() returned the first {len(page)} of {total} memories; "
                    "pass limit=None for all, or page with limit=/offset=",
                    UserWarning, stacklevel=2,
                )
        return page

    def count(
        self,
        *,
        kind: str | None = None,
        open: bool | None = None,  # noqa: A002
        states: list[MemoryState] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> int:
        """How many memories ``list(...)`` with the same filters would return in total."""
        return self._store.count_fragments(
            self.agent_id, **self._list_filters(states, kind, open, tags, session_id),
        )

    @staticmethod
    def _list_filters(
        states: Any, kind: str | None, open_: bool | None, tags: list[str] | None, session_id: str | None,
    ) -> dict[str, Any]:
        wanted = list(tags or [])
        kind = normalize_kind(kind)
        if kind:
            wanted.append(kind_tag(kind))
        exclude: list[str] = []
        if open_ is True:
            exclude = [TAG_RESOLVED, TAG_SUPERSEDED]
        filters: dict[str, Any] = {"states": states, "tags": wanted or None,
                                   "exclude_tags": exclude or None, "session_id": session_id}
        if open_ is False:
            # Closed = retired. resolve() tags both mark:resolved and
            # mark:superseded, so one tag test covers resolved and replaced.
            filters["tags"] = [*wanted, TAG_SUPERSEDED]
        return filters

    def list_sessions(self) -> list[str]:
        """Return all session_ids used by this agent, ordered by most recent activity."""
        return self._store.list_sessions(self.agent_id)

    def list_by_session(
        self,
        session_id: str,
        *,
        states: list[MemoryState] | None = None,
        limit: int | None = 200,
    ) -> list[MemoryFragment]:
        """Return all fragments for a specific session of this agent."""
        return self._store.list_by_agent(
            self.agent_id,
            states=states,
            session_id=session_id,
            limit=limit,
        )

    def forget(
        self,
        memory_id: str | None = None,
        *,
        match: str | None = None,
        secure: bool = False,
    ) -> int:
        """Forget one memory by id, or every memory whose text contains *match*.

        Returns how many memories were forgotten (0 when nothing matched).
        Facts extracted from a forgotten memory are retracted, so they leave
        current values and timelines. ``secure=True`` also overwrites the
        text, hard-deletes those facts and zeroes freed pages; call
        :meth:`vacuum` afterwards to purge the file and its WAL.
        ``match`` is a case-insensitive literal substring, never a semantic
        search, so it only ever removes memories that contain those words.
        """
        if (memory_id is None) == (match is None):
            from mark.errors import InvalidInputError

            raise InvalidInputError("forget() needs exactly one of memory_id or match=")
        if memory_id is not None:
            ids = [memory_id] if self.get(memory_id) is not None else []
        else:
            needle = str(match).casefold().strip()
            if not needle:
                from mark.errors import InvalidInputError

                raise InvalidInputError("forget(match=...) needs non-empty text")
            ids = [f.id for f in self._store.page_fragments(self.agent_id, limit=None)
                   if needle in f.content.casefold()]
        forgotten = 0
        for fragment_id in ids:
            self._pipeline._index.remove(fragment_id)
            forgotten += bool(self._store.forget_fragment(fragment_id, secure=secure))
        return forgotten

    def vacuum(self) -> None:
        """Purge deleted content from the database file and its WAL (see ``forget(secure=True)``)."""
        self._store.vacuum()

    def promote(self, fragment_id: str) -> None:
        """Promote a fragment toward PROMOTED state.

        Walks the allowed state machine one step at a time
        (UNVERIFIED -> VERIFIED -> PROMOTED); states that cannot be promoted
        are written back unchanged. Raises KeyError for unknown ids.
        """
        fragment = self._store.get(fragment_id)
        if fragment is None:
            raise KeyError(fragment_id)
        current = fragment
        if current.state == MemoryState.UNVERIFIED:
            current = current.transition_to(MemoryState.VERIFIED)
        if current.state == MemoryState.VERIFIED:
            current = current.transition_to(MemoryState.PROMOTED)
        self._store.store(current)

    def clear(self) -> int:
        """Delete all of this agent's fragments and index entries."""
        fragments = self._store.list_by_agent(self.agent_id, limit=None)
        for fragment in fragments:
            self._pipeline._index.remove(fragment.id)
        self._store.clear(self.agent_id)
        return len(fragments)
