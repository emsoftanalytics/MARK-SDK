# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Maintenance jobs respect sealed blocks (dedup, consolidation, block graph) and
retrieve/heal retries keep the caller's filters.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from mark.intelligence import BUILTIN_POLICIES, RetrievalPolicy, RetrievalResult
from mark.memory.block_chain import BlockChain, sealed_membership
from mark.memory.block_graph import BlockGraph
from mark.memory.consolidation import ConsolidationManager
from mark.memory.deduplication import DeduplicationConsolidator
from mark.memory.mark_memory import MarkMemory
from mark.middlewares.skills import Skill, SkillResult
from mark.types import MemoryFragment, MemoryNode, MemoryTier
from mark.types.gap import GapReport
from mark.types.graph import NodeType

from _helpers_maintenance import AGENT, _frag, _sealed_block_with, store


EMB = [1.0, 0.0, 0.0, 0.0]


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
