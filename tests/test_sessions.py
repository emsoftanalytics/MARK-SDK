# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Session-aware retrieval at the store level: exact/prefix session filters, tag
and tier filters, and session listing by recency.
"""
from __future__ import annotations

from pathlib import Path

from mark.types import MemoryFragment, MemoryTier


# ---------------------------------------------------------------------------
# Store-level session filter
# ---------------------------------------------------------------------------

def test_list_by_agent_session_exact() -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryState

    store = LocalMemoryStore()
    for ep in ["ep-01", "ep-02", "ep-03"]:
        store.store(MemoryFragment(
            content=f"Fact from {ep}.",
            agent_id="agent",
            state=MemoryState.UNVERIFIED,
            session_id=ep,
        ))

    ep1 = store.list_by_agent("agent", session_id="ep-01")
    assert len(ep1) == 1
    assert ep1[0].session_id == "ep-01"


def test_list_by_agent_session_prefix() -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryState

    store = LocalMemoryStore()
    sessions = ["s01/ep01", "s01/ep02", "s02/ep01"]
    for s in sessions:
        store.store(MemoryFragment(
            content=f"Fact for {s}.",
            agent_id="agent",
            state=MemoryState.UNVERIFIED,
            session_id=s,
        ))

    s1_frags = store.list_by_agent("agent", session_prefix="s01/")
    assert len(s1_frags) == 2
    assert all(f.session_id.startswith("s01/") for f in s1_frags)


def test_list_by_agent_tag_filter() -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryState

    store = LocalMemoryStore()
    store.store(MemoryFragment(
        content="Elena wears a red scarf.",
        agent_id="a",
        state=MemoryState.UNVERIFIED,
        tags=["character:elena", "wardrobe"],
    ))
    store.store(MemoryFragment(
        content="Unrelated fact.",
        agent_id="a",
        state=MemoryState.UNVERIFIED,
        tags=["location:warehouse"],
    ))

    frags = store.list_by_agent("a", tags=["character:elena", "wardrobe"])
    assert len(frags) == 1
    assert "Elena" in frags[0].content


def test_list_by_agent_tier_filter() -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryState

    store = LocalMemoryStore()
    store.store(MemoryFragment(
        content="Working memory note.",
        agent_id="a",
        state=MemoryState.UNVERIFIED,
        tier=MemoryTier.WORKING,
        ttl_seconds=3600,
    ))
    store.store(MemoryFragment(
        content="Long-term fact.",
        agent_id="a",
        state=MemoryState.UNVERIFIED,
        tier=MemoryTier.EPISODIC,
    ))

    working = store.list_by_agent("a", tier=MemoryTier.WORKING)
    assert len(working) == 1
    assert working[0].tier == MemoryTier.WORKING

    episodic = store.list_by_agent("a", tier=MemoryTier.EPISODIC)
    assert len(episodic) == 1
    assert episodic[0].tier == MemoryTier.EPISODIC


def test_list_sessions_ordered_by_recency(tmp_path: Path) -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryState
    from datetime import datetime, timezone, timedelta

    store = LocalMemoryStore(tmp_path / "s.db")
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i, ep in enumerate(["ep-01", "ep-02", "ep-03"]):
        frag = MemoryFragment(
            content=f"Content for {ep}.",
            agent_id="director",
            state=MemoryState.UNVERIFIED,
            session_id=ep,
            created_at=base + timedelta(days=i),
        )
        store.store(frag)

    sessions = store.list_sessions("director")
    assert sessions == ["ep-03", "ep-02", "ep-01"]  # most recent first
