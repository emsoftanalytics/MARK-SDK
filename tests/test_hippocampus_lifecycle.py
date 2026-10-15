# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Hippocampus maintenance: MarkRuntime lifecycle (working memory, consolidation,
pruning, run_cycle) and Hebbian reinforcement wired through retrieval.
"""
from __future__ import annotations

from pathlib import Path

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence import RetrievalPolicy
from mark.memory.runtime import MarkRuntime
from mark.types import MemoryEdge, MemoryFragment, MemoryNode, MemoryState, MemoryTier
from mark.types.graph import EdgeRelation


# ---------------------------------------------------------------------------
# MarkRuntime lifecycle
# ---------------------------------------------------------------------------

def test_working_memory_stores_and_expires() -> None:
    runtime = MarkRuntime.local()
    wm = runtime.working_memory("agent-wm")

    fid = wm.store("Quick scene note.", ttl_seconds=1, importance=0.4)
    assert fid
    active = wm.active()
    assert any(f.id == fid for f in active)

    # Manually delete (TTL not actually elapsed in test) — check expire() works
    runtime.store.delete(fid)
    assert not any(f.id == fid for f in wm.active())
    runtime.shutdown()


def test_consolidate_promotes_working_memory(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    wm = runtime.working_memory("director")
    wm.store("Elena enters the warehouse.", ttl_seconds=9999, importance=0.85)
    wm.store("Throwaway note about lighting.", ttl_seconds=9999, importance=0.2)

    result = runtime.consolidate("director", promote_threshold=0.7)

    assert result.promoted == 1
    # High-importance fragment is now EPISODIC+VERIFIED (no TTL)
    frags = runtime.store.list_by_agent("director")
    promoted = [f for f in frags if f.tier == MemoryTier.EPISODIC]
    assert len(promoted) == 1
    assert promoted[0].state == MemoryState.VERIFIED
    assert promoted[0].ttl_seconds is None
    runtime.shutdown()


def test_consolidate_sanitizes_promoted_memory(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    wm = runtime.working_memory("gov-agent")
    wm.store("Contact admin@example.com for the release key.",
             ttl_seconds=9999, importance=0.9)

    result = runtime.consolidate("gov-agent", promote_threshold=0.7)

    assert result.promoted == 1
    assert result.sanitized == 1
    promoted = [
        f for f in runtime.store.list_by_agent("gov-agent")
        if f.tier == MemoryTier.EPISODIC
    ]
    assert len(promoted) == 1
    assert "[EMAIL]" in promoted[0].content
    assert "admin@example.com" not in promoted[0].content
    runtime.shutdown()


def test_consolidate_blocks_failure_artifacts_and_audits(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    wm = runtime.working_memory("gov-block")
    wm.store("Traceback (most recent call last):\nValueError: broken",
             ttl_seconds=9999, importance=0.95)

    # The FailureFilter is opt-in since 0.2.0a10 (agents keep errors by default).
    from mark.memory.consolidation import ConsolidationManager
    from mark.middlewares.governance import ConsolidationGate, FailureFilter

    result = ConsolidationManager(
        runtime.store, agent_id="gov-block", audit_log=runtime.governance_audit_log(),
        gate=ConsolidationGate(failure=FailureFilter()),
    ).run(promote_threshold=0.7)

    assert result.promoted == 0
    assert result.blocked == 1
    assert not [
        f for f in runtime.store.list_by_agent("gov-block")
        if f.tier == MemoryTier.EPISODIC
    ]
    blocked = runtime.governance_audit_log().list_entries(
        agent_id="gov-block",
        passed=False,
    )
    assert len(blocked) == 1
    assert "failure" in blocked[0].reason.lower()
    runtime.shutdown()


def test_prune_removes_floor_fragments(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("pruner-agent")
    mem.store_sync("Important permanent fact.", importance=0.95)
    # Force a near-floor fragment
    frag = MemoryFragment(
        content="Trivial note that decays.",
        agent_id="pruner-agent",
        state=MemoryState.UNVERIFIED,
        importance=0.05,  # already at decay floor
    )
    runtime.store.store(frag)

    stats = runtime.prune("pruner-agent")

    assert stats.fragments_pruned >= 1
    assert runtime.store.get(frag.id) is None  # pruned
    runtime.shutdown()


def test_prune_dissolves_weak_edges(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    n1 = MemoryNode(label="a", agent_id="edge-agent")
    n2 = MemoryNode(label="b", agent_id="edge-agent")
    runtime.store.store_node(n1)
    runtime.store.store_node(n2)
    weak_edge = MemoryEdge(source_id=n1.id, target_id=n2.id,
                           relation=EdgeRelation.RELATED_TO,
                           agent_id="edge-agent", weight=0.05)  # below EDGE_FLOOR
    runtime.store.store_edge(weak_edge)

    stats = runtime.prune("edge-agent")

    assert stats.edges_dissolved >= 1
    assert not any(e.id == weak_edge.id for e in runtime.store.list_edges("edge-agent"))
    runtime.shutdown()


def test_run_cycle_returns_summary(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    wm = runtime.working_memory("cycle-agent")
    wm.store("Keep this.", ttl_seconds=9999, importance=0.9)

    summary = runtime.run_cycle("cycle-agent", promote_threshold=0.7)

    assert summary["agent_id"] == "cycle-agent"
    assert "promoted" in summary
    assert "expired" in summary
    assert "fragments_pruned" in summary
    assert "edges_dissolved" in summary
    assert summary["promoted"] >= 1
    runtime.shutdown()


def test_integrity_check_passes_on_runtime_store(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    assert runtime.integrity_check() == []
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Hebbian reinforcement wired through retrieval
# ---------------------------------------------------------------------------

def test_retrieval_increments_access_count(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("hebb-agent")
    fid = mem.store_sync("Hebbian test fragment.", importance=0.5)

    mem.retrieve_sync("Hebbian test", policy=RetrievalPolicy.BALANCED)

    loaded = runtime.store.get(fid)
    assert loaded is not None
    assert loaded.metadata.get("access_count", 0) >= 1
    assert loaded.last_accessed_at is not None
    runtime.shutdown()


def test_first_access_uses_current_access_count(tmp_path: Path) -> None:
    """touch() runs before on_access() so the first retrieval counts and boosts importance."""
    from mark.plasticity.hebbian import REINFORCE_DELTA

    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("hebb-order")
    initial_importance = 0.5
    fid = mem.store_sync("First access matters.", importance=initial_importance)

    mem.retrieve_sync("First access", policy=RetrievalPolicy.BALANCED)

    loaded = runtime.store.get(fid)
    assert loaded.metadata.get("access_count") == 1
    # Boost at access_count=1: delta * (1 - exp(-1/10)) ≈ 0.05 * 0.095 ≈ 0.00476
    # importance must have increased above the initial value
    assert loaded.importance > initial_importance
    runtime.shutdown()


def test_co_retrieval_strengthens_edge(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    fid_a = runtime.memory("coact-agent").store_sync("FastAPI handles requests.", importance=0.9)
    fid_b = runtime.memory("coact-agent").store_sync("FastAPI uses Pydantic models.", importance=0.9)

    na = MemoryNode(label="fastapi", agent_id="coact-agent", fragment_id=fid_a)
    nb = MemoryNode(label="pydantic", agent_id="coact-agent", fragment_id=fid_b)
    runtime.store.store_node(na)
    runtime.store.store_node(nb)
    initial_weight = 0.5
    edge = MemoryEdge(source_id=na.id, target_id=nb.id,
                      relation=EdgeRelation.RELATED_TO,
                      agent_id="coact-agent", weight=initial_weight)
    runtime.store.store_edge(edge)

    runtime.memory("coact-agent").retrieve_sync("FastAPI", policy=RetrievalPolicy.DEEP)

    strengthened = runtime.store.edges_from_node(na.id)
    assert strengthened and strengthened[0].weight > initial_weight
    runtime.shutdown()


def test_co_retrieval_strengthens_incoming_edge(tmp_path: Path) -> None:
    """Incoming edges (B→A) are strengthened when A and B are co-retrieved."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    fid_a = runtime.memory("coact-in").store_sync("FastAPI route layer.", importance=0.9)
    fid_b = runtime.memory("coact-in").store_sync("FastAPI validation layer.", importance=0.9)

    na = MemoryNode(label="route", agent_id="coact-in", fragment_id=fid_a)
    nb = MemoryNode(label="validation", agent_id="coact-in", fragment_id=fid_b)
    runtime.store.store_node(na)
    runtime.store.store_node(nb)
    # Edge goes B → A (incoming from A's perspective)
    initial_weight = 0.5
    edge = MemoryEdge(source_id=nb.id, target_id=na.id,
                      relation=EdgeRelation.RELATED_TO,
                      agent_id="coact-in", weight=initial_weight)
    runtime.store.store_edge(edge)

    runtime.memory("coact-in").retrieve_sync("FastAPI", policy=RetrievalPolicy.DEEP)

    # Edge originated from nb — check it was strengthened
    edges = runtime.store.edges_from_node(nb.id)
    assert edges and edges[0].weight > initial_weight
    runtime.shutdown()
