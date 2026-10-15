# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Maintenance pruner: skips protected fragments, idempotent decay, Hebbian
parameter, sealed edges, uncapped sweeps, and timezone-naive datetimes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mark.memory.block_chain import BlockChain
from mark.memory.block_graph import BlockGraph
from mark.memory.working import WorkingMemoryManager
from mark.plasticity import EdgeCoActivation, ExponentialDecay, MemoryPruner, TemporalScope
from mark.types import BlockLink, MemoryBlock, MemoryEdge, MemoryNode, MemoryState, MemoryTier
from mark.types.graph import EdgeRelation

from _helpers_maintenance import AGENT, _frag, _sealed_block_with, store


# ── item 10: pruner ───────────────────────────────────────────────────────────

def _old(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def test_pruner_skips_quarantined_contradicted_and_sealed(store) -> None:
    quarantined = _frag("q", importance=0.06, state=MemoryState.QUARANTINED, created_at=_old(9999))
    contradicted = _frag("c", importance=0.06, state=MemoryState.CONTRADICTED, created_at=_old(9999))
    sealed = _frag("s", importance=0.06, created_at=_old(9999))
    doomed = _frag("d", importance=0.06, created_at=_old(9999))
    store.store(quarantined)
    store.store(contradicted)
    store.store(doomed)
    _sealed_block_with(store, sealed)

    stats = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM)).run(AGENT)

    assert store.get(quarantined.id) is not None
    assert store.get(contradicted.id) is not None
    assert store.get(sealed.id) is not None
    assert store.get(doomed.id) is None
    assert stats.fragments_pruned == 1
    assert stats.protected == 3


def test_pruner_is_idempotent_at_same_instant(store) -> None:
    frag = _frag("Decaying fact.", importance=0.8, created_at=_old(48))
    store.store(frag)
    pruner = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM))
    now = datetime.now(timezone.utc)

    first = pruner.run(AGENT, now=now)
    after_first = store.get(frag.id).importance
    second = pruner.run(AGENT, now=now)

    assert first.importance_updated == 1
    assert second.importance_updated == 0
    assert store.get(frag.id).importance == pytest.approx(after_first)
    assert after_first == pytest.approx(0.4, rel=1e-3)
    meta = store.get(frag.id).metadata["mark_decay"]
    assert meta["base_importance"] == pytest.approx(0.8)


def test_pruner_decay_composes_over_runs(store) -> None:
    frag = _frag("Decaying fact.", importance=0.8, created_at=_old(48))
    store.store(frag)
    pruner = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM))
    now = datetime.now(timezone.utc)
    pruner.run(AGENT, now=now)
    pruner.run(AGENT, now=now + timedelta(hours=48))
    # Two half-lives in total: 0.8 → 0.2 (not 0.8 → 0.4 → 0.1 from re-applying age).
    assert store.get(frag.id).importance == pytest.approx(0.2, rel=1e-3)


def test_pruner_honours_hebbian_parameter(store) -> None:
    class AlwaysDissolve:
        def should_dissolve(self, weight: float) -> bool:
            return True

    a = MemoryNode(label="A", agent_id=AGENT)
    b = MemoryNode(label="B", agent_id=AGENT)
    store.store_node(a)
    store.store_node(b)
    store.store_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=EdgeRelation.RELATED_TO,
                                agent_id=AGENT, weight=0.9))

    assert MemoryPruner(store).run(AGENT).edges_dissolved == 0
    assert MemoryPruner(store, hebbian=AlwaysDissolve()).run(AGENT).edges_dissolved == 1
    with pytest.raises(TypeError):
        MemoryPruner(store, hebbian=object())
    assert isinstance(MemoryPruner(store, hebbian=EdgeCoActivation())._coact, EdgeCoActivation)


def test_pruner_never_dissolves_sealed_edges(store) -> None:
    graph = BlockGraph(store, AGENT)
    block = graph.create_block("edges")
    a = MemoryNode(label="A", agent_id=AGENT)
    b = MemoryNode(label="B", agent_id=AGENT)
    store.store_node(a)
    store.store_node(b)
    graph.add_nodes(block.id, [a.id, b.id])
    graph.add_edge(block.id, a.id, b.id, weight=0.01)
    BlockChain(store, AGENT).seal(block.id)

    stats = MemoryPruner(store).run(AGENT)
    assert stats.edges_dissolved == 0
    assert BlockChain(store, AGENT).verify(block.id).valid


def test_maintenance_has_no_200_row_cap(store) -> None:
    for i in range(230):
        store.store(_frag(f"expired note {i}", tier=MemoryTier.WORKING, ttl_seconds=1,
                          created_at=_old(1)))
    assert WorkingMemoryManager(store, AGENT).expire() == 230

    for i in range(230):
        store.store(_frag(f"old note {i}", importance=0.06, created_at=_old(9999)))
    stats = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM)).run(AGENT)
    assert stats.fragments_pruned == 230


def test_mark_memory_clear_has_no_cap(tmp_path) -> None:
    from mark.embeddings import HashEmbeddingProvider
    from mark.memory.runtime import MarkRuntime

    runtime = MarkRuntime.local(store_path=tmp_path / "m.db", embedder=HashEmbeddingProvider(dim=16))
    mem = runtime.memory("bulk")
    for i in range(210):
        mem.store_sync(f"fact number {i}")
    assert mem.clear() == 210
    runtime.shutdown()


# ── item 10: timezone-naive datetimes ─────────────────────────────────────────

def test_naive_datetimes_are_coerced_to_utc() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    frag = _frag("naive", created_at=naive, updated_at=naive, last_accessed_at=naive, ttl_seconds=60)
    assert frag.created_at.tzinfo is not None
    assert frag.updated_at.tzinfo is not None
    assert frag.last_accessed_at.tzinfo is not None
    assert frag.is_expired() is True
    assert frag.is_expired(now=datetime(2024, 1, 1, 12, 0, 30)) is False

    assert MemoryNode(label="n", created_at=naive).created_at.tzinfo is not None
    assert MemoryEdge(source_id="a", target_id="b", relation=EdgeRelation.RELATED_TO,
                      agent_id=AGENT, created_at=naive).created_at.tzinfo is not None
    assert BlockLink(source_block_id="a", target_block_id="b", agent_id=AGENT,
                     created_at=naive).created_at.tzinfo is not None
    block = MemoryBlock(name="b", agent_id=AGENT, created_at=naive, sealed_at=naive)
    assert block.created_at.tzinfo is not None and block.sealed_at.tzinfo is not None


def test_decay_handles_naive_datetimes() -> None:
    class Fake:
        importance = 0.8
        state = MemoryState.UNVERIFIED
        created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48)  # naive
        metadata: dict = {}

    decay = ExponentialDecay(TemporalScope.SHORT_TERM)
    assert decay.apply(Fake()) == pytest.approx(0.4, rel=1e-2)
    assert decay.apply(Fake(), now=datetime.now(timezone.utc).replace(tzinfo=None)) == pytest.approx(0.4, rel=1e-2)
