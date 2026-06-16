# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# WorkingMemoryManager — short-term memory with TTL expiry.
#
# Biological analogy: the hippocampus holds recent experiences in
# fast-access working memory before consolidation into the cortex (LTM).
#
# HOOK_CONSOLIDATION can trigger LLM-backed promotion of working memory to LTM
# with quality scoring and summarisation.
"""TTL-bound working memory manager."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier


class WorkingMemoryManager:
    """
    Manages short-term working memory fragments with TTL expiry.

    Working memory fragments:
    - Have a ttl_seconds set (expire automatically)
    - Use MemoryTier.WORKING
    - Are automatically excluded from retrieval after expiry

    Typical usage:
        wm = WorkingMemoryManager(store, agent_id="director")
        frag_id = wm.store("Scene 47: Elena enters warehouse", ttl_seconds=3600)
        active = wm.active()   # returns non-expired fragments
        wm.expire()            # removes expired fragments from store

    Media agent usage:
        - Store current scene context as working memory (expires after scene)
        - Store in-frame object state (expires after shot)
        - Use consolidation.py to promote scene summaries to episodic LTM
    """

    def __init__(self, store: Any, agent_id: str,
                 session_id: Optional[str] = None) -> None:
        self._store      = store
        self._agent_id   = agent_id
        self._session_id = session_id

    def store(self, content: str, *, ttl_seconds: int = 3600,
              importance: float = 0.6, tags: Optional[List[str]] = None,
              metadata: Optional[dict] = None) -> str:
        """Store a working memory fragment. Returns fragment id."""
        fragment = MemoryFragment(
            content     = content,
            agent_id    = self._agent_id,
            scope       = MemoryScope.AGENT,
            tier        = MemoryTier.WORKING,
            state       = MemoryState.UNVERIFIED,
            ttl_seconds = ttl_seconds,
            importance  = importance,
            session_id  = self._session_id,
            tags        = ["working_memory", *(tags or [])],
            metadata    = metadata or {},
        )
        return self._store.store(fragment)

    def active(self) -> List[MemoryFragment]:
        """Return working memory fragments that have not yet expired."""
        from mark.types import MemoryTier as MT
        fragments = self._store.list_by_agent(
            self._agent_id,
            states=[MemoryState.UNVERIFIED, MemoryState.VERIFIED],
        )
        now = datetime.now(timezone.utc)
        return [
            f for f in fragments
            if f.tier == MT.WORKING and not f.is_expired(now=now)
        ]

    def expire(self) -> int:
        """Delete all expired working memory fragments. Returns count deleted."""
        from mark.types import MemoryTier as MT
        fragments = self._store.list_by_agent(self._agent_id)
        now = datetime.now(timezone.utc)
        count = 0
        for f in fragments:
            if f.tier == MT.WORKING and f.is_expired(now=now):
                self._store.delete(f.id)
                count += 1
        return count
