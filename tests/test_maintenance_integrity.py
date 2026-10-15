# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Maintenance jobs respect sealed blocks, audit evidence, and caller filters."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from mark.intelligence import BUILTIN_POLICIES, RetrievalPolicy, RetrievalResult
from mark.memory.block_chain import BlockChain, sealed_membership
from mark.memory.block_graph import BlockGraph
from mark.memory.consolidation import ConsolidationManager
from mark.memory.deduplication import DeduplicationConsolidator
from mark.memory.mark_memory import MarkMemory
from mark.memory.working import WorkingMemoryManager
from mark.middlewares.skills import Skill, SkillResult
from mark.plasticity import EdgeCoActivation, ExponentialDecay, MemoryPruner, TemporalScope
from mark.store import LocalMemoryStore
from mark.types import (
    BlockLink,
    MemoryBlock,
    MemoryEdge,
    MemoryFragment,
    MemoryNode,
    MemoryState,
    MemoryTier,
)
from mark.types.gap import GapReport
from mark.types.graph import EdgeRelation, NodeType

AGENT = "maint-agent"
EMB = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture()
def store():
    s = LocalMemoryStore(":memory:")
    yield s
    s.close()


def _frag(content: str, **kw) -> MemoryFragment:
    return MemoryFragment(content=content, agent_id=AGENT, **kw)


def _sealed_block_with(store, *fragments: MemoryFragment, name: str = "sealed") -> MemoryBlock:
    graph = BlockGraph(store, AGENT)
    block = graph.create_block(name)
    for fragment in fragments:
        store.store(fragment)
        graph.add_fragment(block.id, fragment.id)
    return BlockChain(store, AGENT).seal(block.id)


# ── item 9: sealed blocks ─────────────────────────────────────────────────────

def test_dedup_never_deletes_sealed_member(store) -> None:
    sealed_frag = _frag("Elena wears the red scarf.", importance=0.2, embedding=EMB)
    block = _sealed_block_with(store, sealed_frag)
    free_dup = _frag("Elena wears the red scarf!", importance=0.9, embedding=EMB)
    store.store(free_dup)

    result = DeduplicationConsolidator(store, AGENT, similarity_threshold=0.9).run()

    assert store.get(sealed_frag.id) is not None
    assert result.merged == 0
    assert BlockChain(store, AGENT).verify(block.id).valid


def test_dedup_never_repoints_sealed_node(store) -> None:
    graph = BlockGraph(store, AGENT)
    keep = _frag("Marco owns a lamp.", importance=0.9, embedding=EMB)
    dup = _frag("Marco owns a lamp!", importance=0.1, embedding=EMB)
    store.store(keep)
    store.store(dup)
    node = MemoryNode(label="Lamp", node_type=NodeType.OBJECT, agent_id=AGENT, fragment_id=dup.id)
    store.store_node(node)
    block = graph.create_block("nodes")
    graph.add_node(block.id, node.id)
    BlockChain(store, AGENT).seal(block.id)

    DeduplicationConsolidator(store, AGENT, similarity_threshold=0.9).run()

    assert store.get(dup.id) is not None
    assert store.get_node(node.id).fragment_id == dup.id
    assert BlockChain(store, AGENT).verify(block.id).valid


def test_dedup_still_merges_unsealed_duplicates(store) -> None:
    a = _frag("Free fact.", importance=0.9, embedding=EMB)
    b = _frag("Free fact!", importance=0.1, embedding=EMB)
    store.store(a)
    store.store(b)
    result = DeduplicationConsolidator(store, AGENT, similarity_threshold=0.9).run()
    assert result.merged == 1
    assert store.get(b.id) is None


def test_consolidation_does_not_rewrite_sealed_member(store) -> None:
    working = _frag(
        "Contact elena@example.com about the scarf.",
        tier=MemoryTier.WORKING, ttl_seconds=3600, importance=0.9,
    )
    block = _sealed_block_with(store, working)

    result = ConsolidationManager(store, AGENT, promote_threshold=0.5).run()

    after = store.get(working.id)
    assert after.content == working.content
    assert after.tier == MemoryTier.WORKING
    assert result.promoted == 0
    assert BlockChain(store, AGENT).verify(block.id).valid


def test_block_graph_refuses_moving_fragment_out_of_sealed_block(store) -> None:
    fragment = _frag("Sealed evidence.")
    _sealed_block_with(store, fragment)
    graph = BlockGraph(store, AGENT)
    other = graph.create_block("open")
    with pytest.raises(ValueError, match="cannot be moved"):
        graph.add_fragment(other.id, fragment.id)
    assert store.get(fragment.id).block_id != other.id


def test_sealed_membership_lists_members(store) -> None:
    fragment = _frag("Sealed.")
    block = _sealed_block_with(store, fragment)
    membership = sealed_membership(store, AGENT)
    assert block.id in membership.block_ids
    assert membership.protects_fragment(store.get(fragment.id))


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


# ── item 15: retrieve retries keep caller filters ─────────────────────────────

class _RecordingPipeline:
    """Pipeline double: always reports a gap, records every SessionFilter."""

    def __init__(self, store) -> None:
        self.calls: list[tuple[object, bool]] = []
        self._store = store
        self._plugins: dict = {}

    def retrieve(self, query, agent_id, policy, *, session_filter=None, compress=False, expand=False):
        self.calls.append((session_filter, expand))
        return RetrievalResult(
            query=query,
            fragments=[],
            scores=[],
            policy_used=policy,
            spec_used=BUILTIN_POLICIES[policy.value],
            gap_report=GapReport.from_score(0.0, 0, 0.5),
        )

    def index_fragment(self, fragment):
        self._store.store(fragment)
        return fragment


class _EvidenceSkill(Skill):
    name = "fake_web"
    permissions: set[str] = set()

    async def run(self, input, context):
        return SkillResult(
            content="evidence",
            fragments=[MemoryFragment(content="The vault code is blue.", agent_id=context["agent_id"],
                                      source="https://example.test/vault")],
        )


def test_retrieve_escalation_and_heal_retries_preserve_filters(store) -> None:
    pipeline = _RecordingPipeline(store)
    with ThreadPoolExecutor(max_workers=1) as executor:
        mem = MarkMemory(agent_id=AGENT, store=store, pipeline=pipeline,
                         embedder=None, executor=executor)  # type: ignore[arg-type]
        mem.retrieve_sync(
            "What is the vault code?",
            RetrievalPolicy.BALANCED,
            session_id="ep-1",
            tags=["vault"],
            block_id="block-123",
            heal_gaps=True,
            web_skill=_EvidenceSkill(),
            expand=True,
        )

    # initial + agent escalation + session retry + agent retry
    assert len(pipeline.calls) == 4
    for session_filter, expand in pipeline.calls:
        assert session_filter.block_id == "block-123"
        assert session_filter.tags == ["vault"]
        assert expand is True
    sessions = [sf.session_id for sf, _ in pipeline.calls]
    assert sessions == ["ep-1", None, "ep-1", None]
