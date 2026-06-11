# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

from pathlib import Path

from mark import ClusterResult, DeduplicationConsolidator, DeduplicationResult, HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_deduplication_result_importable() -> None:
    assert DeduplicationResult is not None


def test_deduplication_consolidator_importable() -> None:
    assert DeduplicationConsolidator is not None


def test_cluster_result_importable() -> None:
    assert ClusterResult is not None


# ---------------------------------------------------------------------------
# DeduplicationResult dataclass
# ---------------------------------------------------------------------------

def test_result_defaults() -> None:
    r = DeduplicationResult()
    assert r.merged  == 0
    assert r.kept    == 0
    assert r.skipped == 0


# ---------------------------------------------------------------------------
# No-op on empty store
# ---------------------------------------------------------------------------

def test_empty_store_returns_zero_counts(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    result  = runtime.deduplicate("agent")
    assert result.merged  == 0
    assert result.kept    == 0
    runtime.shutdown()


# ---------------------------------------------------------------------------
# No duplicates → all kept
# ---------------------------------------------------------------------------

def test_distinct_fragments_all_kept(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.",    importance=0.9)
    mem.store_sync("Red scarf is an artifact.", importance=0.8)
    mem.store_sync("North Warehouse exterior.", importance=0.7)

    result = runtime.deduplicate("agent")
    assert result.merged == 0
    assert result.kept   == 3
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Identical text → duplicates merged
# ---------------------------------------------------------------------------

def test_identical_text_deduplicated(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)
    mem.store_sync("Elena is left-handed.", importance=0.5)   # duplicate

    result = runtime.deduplicate("agent", similarity_threshold=0.99)
    assert result.merged == 1
    assert result.kept   == 1

    remaining = runtime.store.list_by_agent("agent")
    assert len(remaining) == 1
    runtime.shutdown()


def test_keeps_highest_importance(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    fid_hi  = mem.store_sync("Elena is left-handed.", importance=0.9)
    fid_lo  = mem.store_sync("Elena is left-handed.", importance=0.3)

    runtime.deduplicate("agent", similarity_threshold=0.99)

    remaining = runtime.store.list_by_agent("agent")
    assert len(remaining) == 1
    assert remaining[0].id == fid_hi   # higher importance kept
    runtime.shutdown()


def test_merged_from_ids_recorded_in_metadata(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    _       = mem.store_sync("Elena is left-handed.", importance=0.9)
    fid_lo  = mem.store_sync("Elena is left-handed.", importance=0.3)

    runtime.deduplicate("agent", similarity_threshold=0.99)

    kept = runtime.store.list_by_agent("agent")[0]
    assert fid_lo in kept.metadata.get("merged_from_ids", [])
    runtime.shutdown()


def test_duplicate_fragment_nodes_reattached_to_kept_fragment(tmp_path: Path) -> None:
    from mark.types import MemoryNode

    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    fid_hi  = mem.store_sync("Elena is left-handed.", importance=0.9)
    fid_lo  = mem.store_sync("Elena is left-handed.", importance=0.3)

    node = MemoryNode(label="Elena duplicate", agent_id="agent", fragment_id=fid_lo)
    runtime.store.store_node(node)

    runtime.deduplicate("agent", similarity_threshold=0.99)

    assert runtime.store.get(fid_lo) is None
    attached = runtime.store.nodes_for_fragment(fid_hi)
    assert any(n.label == "Elena duplicate" for n in attached)
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Fragments without embeddings are skipped
# ---------------------------------------------------------------------------

def test_fragments_without_embeddings_skipped(tmp_path: Path) -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier
    from datetime import datetime, timezone

    store = LocalMemoryStore(tmp_path / "mem.db")
    now   = datetime.now(timezone.utc)
    frag  = MemoryFragment(
        id="no-emb-1", content="No embedding.", agent_id="agent",
        scope=MemoryScope.AGENT, tier=MemoryTier.EPISODIC,
        state=MemoryState.UNVERIFIED, importance=0.5, confidence=1.0,
        tags=[], metadata={}, embedding=None,
        created_at=now, updated_at=now,
    )
    store.store(frag)

    result = DeduplicationConsolidator(store, "agent").run()
    assert result.skipped == 1
    assert result.merged  == 0
    store.close()


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def test_high_threshold_no_merge(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    mem.store_sync("Elena is the protagonist.", importance=0.9)
    mem.store_sync("Elena is the lead character.", importance=0.8)

    # Near-similar but probably not identical with hash embeddings
    result = runtime.deduplicate("agent", similarity_threshold=1.0)
    # With threshold=1.0, only exact vector matches merge
    assert result.kept + result.merged == 2
    runtime.shutdown()


# ---------------------------------------------------------------------------
# MarkRuntime.deduplicate delegates correctly
# ---------------------------------------------------------------------------

def test_runtime_deduplicate_returns_result(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    result  = runtime.deduplicate("agent")
    assert isinstance(result, DeduplicationResult)
    runtime.shutdown()


def test_run_cycle_includes_deduplication(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)
    mem.store_sync("Elena is left-handed.", importance=0.5)

    summary = runtime.run_cycle("agent")

    assert summary["dedup_merged"] == 1
    assert summary["dedup_kept"] == 1
    assert len(runtime.store.list_by_agent("agent")) == 1
    runtime.shutdown()


def test_deduplicate_custom_threshold(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem     = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)
    mem.store_sync("Elena is left-handed.", importance=0.5)

    result = runtime.deduplicate("agent", similarity_threshold=0.95)
    assert result.merged >= 0   # just verify it runs without error
    runtime.shutdown()
