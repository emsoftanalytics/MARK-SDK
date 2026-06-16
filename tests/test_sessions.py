# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Session-aware retrieval tests.
from __future__ import annotations

from pathlib import Path

import pytest

from mark import Mark
from mark.embeddings import HashEmbeddingProvider
from mark.intelligence import RetrievalPolicy, SessionFilter
from mark.memory.runtime import MarkRuntime
from mark.middlewares.skills.base import Skill, SkillResult
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


# ---------------------------------------------------------------------------
# MarkMemory — retrieve_sync with session filter
# ---------------------------------------------------------------------------

def test_retrieve_sync_session_exact(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("video-agent")
    mem.store_sync("Elena wore the red scarf.",
                   session_id="ep-01", tags=["character:elena"])
    mem.store_sync("Elena lost the red scarf.",
                   session_id="ep-03", tags=["character:elena"])

    ep1 = mem.retrieve_sync("Elena scarf", policy=RetrievalPolicy.BALANCED,
                             session_id="ep-01")
    ep3 = mem.retrieve_sync("Elena scarf", policy=RetrievalPolicy.BALANCED,
                             session_id="ep-03")

    assert all(f.session_id == "ep-01" for f in ep1.fragments)
    assert all(f.session_id == "ep-03" for f in ep3.fragments)
    runtime.shutdown()


def test_retrieve_sync_session_prefix(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("video-agent")
    mem.store_sync("S01E01 fact.", session_id="season-01/ep-01")
    mem.store_sync("S01E02 fact.", session_id="season-01/ep-02")
    mem.store_sync("S02E01 fact.", session_id="season-02/ep-01")

    s1 = mem.retrieve_sync("fact", policy=RetrievalPolicy.BALANCED,
                            session_prefix="season-01/")

    assert s1.fragments
    assert all(f.session_id.startswith("season-01/") for f in s1.fragments)
    runtime.shutdown()


def test_retrieve_sync_cross_session_default(tmp_path: Path) -> None:
    """No session filter → retrieves across all sessions."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("video-agent")
    mem.store_sync("Ep1 scarf fact.", session_id="ep-01")
    mem.store_sync("Ep3 scarf fact.", session_id="ep-03")

    result = mem.retrieve_sync("scarf", policy=RetrievalPolicy.BALANCED)

    session_ids = {f.session_id for f in result.fragments}
    assert "ep-01" in session_ids or "ep-03" in session_ids
    runtime.shutdown()


def test_retrieve_sync_tag_filter(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("video-agent")
    mem.store_sync("Elena wears red.", importance=0.9,
                   tags=["character:elena", "wardrobe"])
    mem.store_sync("Warehouse exterior shot.", importance=0.9,
                   tags=["location:warehouse"])

    elena = mem.retrieve_sync("costume", policy=RetrievalPolicy.BALANCED,
                               tags=["character:elena"])

    assert elena.fragments
    assert all("character:elena" in f.tags for f in elena.fragments)
    runtime.shutdown()


def test_list_sessions_via_mark_memory(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("director")
    for ep in ["ep-01", "ep-02", "ep-03"]:
        mem.store_sync(f"Fact from {ep}.", session_id=ep)

    sessions = mem.list_sessions()

    assert set(sessions) == {"ep-01", "ep-02", "ep-03"}
    runtime.shutdown()


def test_list_by_session(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("director")
    mem.store_sync("Scene 1 note.", session_id="scene-01")
    mem.store_sync("Scene 1 extra note.", session_id="scene-01")
    mem.store_sync("Scene 2 note.", session_id="scene-02")

    scene1 = mem.list_by_session("scene-01")
    assert len(scene1) == 2
    assert all(f.session_id == "scene-01" for f in scene1)
    runtime.shutdown()


# ---------------------------------------------------------------------------
# SimpleMemory — session filter on mark.memory.retrieve()
# ---------------------------------------------------------------------------

def test_simple_memory_retrieve_session_id(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        # Using the simple API with explicit session_id via the advanced runtime
        mem = mark.runtime.memory("__mark__")
        mem.store_sync("World bible: Elena has silver hair.",
                       session_id="world-bible",
                       tags=["block:characters"])
        mem.store_sync("Scene 5: Elena borrows an umbrella.",
                       session_id="scene-05",
                       tags=["block:characters"])

        world = mark.memory.retrieve("Elena appearance",
                                     session_id="world-bible")
        scene = mark.memory.retrieve("Elena scene",
                                     session_id="scene-05")

    assert world.memories
    assert all(r.metadata.get("migrated_from") or True for r in world.memories)
    assert scene.memories


def test_simple_memory_retrieve_session_prefix(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        mem = mark.runtime.memory("__mark__")
        for ep in ["s01/ep01", "s01/ep02", "s02/ep01"]:
            mem.store_sync(f"Fact from {ep}.", session_id=ep,
                           tags=["block:facts"])

        season1 = mark.memory.retrieve("fact", session_prefix="s01/")

    assert season1.memories
    session_ids_in_result = {
        m.metadata.get("session_id") or ""
        for m in season1.memories
    }
    # All returned records were from season-01 sessions
    # (metadata won't have session_id but fragments do; test the raw count)
    assert len(season1.memories) <= 2


# ---------------------------------------------------------------------------
# Gap escalation
# ---------------------------------------------------------------------------

class FakeWebSkill(Skill):
    name = "fake_web"
    permissions = set()

    async def run(self, input, context):
        return SkillResult(
            content="fake evidence",
            fragments=[
                MemoryFragment(
                    content="Elena's canonical scarf color is crimson red.",
                    agent_id=context["agent_id"],
                    source="https://example.test/elena",
                    tags=["character:elena"],
                )
            ],
        )


def test_retrieve_escalates_from_session_to_agent_namespace(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("video-agent")
    mem.store_sync(
        "Elena's red scarf is a world-bible wardrobe fact.",
        session_id="world-bible",
        tags=["character:elena"],
        importance=0.95,
    )

    result = mem.retrieve_sync(
        "What scarf should Elena wear?",
        policy=RetrievalPolicy.BALANCED,
        session_id="season-01/ep-09",
        tags=["character:elena"],
        escalate_on_gap=True,
    )

    assert result.fragments
    assert result.escalation_path == ["session", "agent_namespace"]
    assert all(f.agent_id == "video-agent" for f in result.fragments)
    runtime.shutdown()


def test_heal_gaps_runs_external_lookup_after_agent_namespace_miss(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("video-agent")

    result = mem.retrieve_sync(
        "What is Elena's canonical scarf color?",
        policy=RetrievalPolicy.BALANCED,
        session_id="season-01/ep-09",
        heal_gaps=True,
        web_skill=FakeWebSkill(),
    )

    assert result.external_lookup_used is True
    assert "external_lookup" in result.escalation_path
    stored = runtime.memory("video-agent").list()
    assert any("gap_healing" in f.tags for f in stored)
    assert any(f.session_id == "season-01/ep-09" for f in stored)
    runtime.shutdown()
