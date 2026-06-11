# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# SessionMemory — scoped context for a named session/scene/episode.
#
# "Scene" and "Episode" are media terms. The underlying pattern is general:
# any long-running project where observations are grouped into named
# sub-contexts benefits from session-scoped helpers.
#
#   Media:        scene-04, episode-03
#   Research:     experiment-run-12, trial-session-7
#   Engineering:  incident-2024-11-14, deploy-v2.3
#   Game:         dungeon-level-3, boss-fight-1
"""Session-scoped memory helper for episodic workflows."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.memory.mark_memory import MarkMemory
    from mark.memory.observe import ObserveResult
    from mark.types import MemoryFragment


class SessionMemory:
    """Scoped memory for a named session context.

    Groups observations, recalls, and entity facts under a specific
    session_id. Supports hierarchical session_ids with prefix recall.

    Usage::

        scene = mark.runtime.memory("agent").session("season-01/ep-02/scene-04")
        scene.observe("Elena enters the North Warehouse wearing the red scarf.")
        results = scene.recall("Where is Elena?")

        # Recall across all of episode 2:
        ep_results = scene.recall_prefix("season-01/ep-02/")
    """

    def __init__(self, session_id: str, mark_memory: "MarkMemory") -> None:
        self._session_id = session_id
        self._mem        = mark_memory

    @property
    def session_id(self) -> str:
        """Return the session id."""
        return self._session_id

    def observe(
        self,
        text:        str,
        *,
        importance:  float = 0.5,
        tags:        list[str] | None = None,
        source:      str | None = None,
        metadata:    dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> "ObserveResult":
        """Store an observation in this session, auto-extracting entities."""
        return self._mem.observe(
            text,
            session_id  = self._session_id,
            importance  = importance,
            tags        = tags,
            source      = source,
            metadata    = metadata,
            memory_type = memory_type or "scene",
        )

    def recall(self, query: str, *, top_k: int = 5):
        """Retrieve observations from this exact session."""
        from mark.intelligence import RetrievalPolicy
        return self._mem.retrieve_sync(
            query,
            policy     = RetrievalPolicy.BALANCED,
            session_id = self._session_id,
        )

    def recall_prefix(self, prefix: str, query: str, *, top_k: int = 10):
        """Retrieve across all sessions matching a prefix.

        Example: session.recall_prefix("season-01/ep-02/", "Where is Elena?")
        returns observations from all scenes within episode 2.
        """
        from mark.intelligence import RetrievalPolicy
        return self._mem.retrieve_sync(
            query,
            policy         = RetrievalPolicy.BALANCED,
            session_prefix = prefix,
        )

    def list_observations(self, *, limit: int = 200) -> list["MemoryFragment"]:
        """Return all fragments recorded in this session."""
        return self._mem.list_by_session(self._session_id, limit=limit)
