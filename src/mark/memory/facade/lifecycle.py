# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Fragment lifecycle for MarkMemory: list, forget, promote, clear.

Deleting operations remove the retrieval index entry *before* the stored
row, so a fragment can never be retrieved after it has been forgotten.
"""
from __future__ import annotations

from mark.memory.facade.core import MemoryCore
from mark.types import MemoryFragment, MemoryState


class LifecycleMixin(MemoryCore):
    """Inspection and maintenance of this agent's stored fragments."""

    def list(
        self,
        states: list[MemoryState] | None = None,
        limit: int = 100,
    ) -> list[MemoryFragment]:
        """Return this agent's fragments, optionally filtered by state."""
        return self._store.list_by_agent(self.agent_id, states=states, limit=limit)

    def list_sessions(self) -> list[str]:
        """Return all session_ids used by this agent, ordered by most recent activity."""
        return self._store.list_sessions(self.agent_id)

    def list_by_session(
        self,
        session_id: str,
        *,
        states: list[MemoryState] | None = None,
        limit: int = 200,
    ) -> list[MemoryFragment]:
        """Return all fragments for a specific session of this agent."""
        return self._store.list_by_agent(
            self.agent_id,
            states=states,
            session_id=session_id,
            limit=limit,
        )

    def forget(self, fragment_id: str) -> bool:
        """Delete a fragment by id."""
        self._pipeline._index.remove(fragment_id)
        return self._store.delete(fragment_id)

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
