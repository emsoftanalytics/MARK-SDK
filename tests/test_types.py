from datetime import datetime, timedelta, timezone

import pytest

from mark.types import (
    EdgeRelation,
    GapReport,
    GapSeverity,
    MemoryBlockModel,
    MemoryDocument,
    MemoryEdge,
    MemoryFragment,
    MemoryNode,
    MemoryScope,
    MemoryState,
    MemoryTier,
    NodeType,
    SourceAttribution,
    SourceCredibility,
)


def test_memory_fragment_state_transitions_and_ttl():
    fragment = MemoryFragment(
        content="The API uses FastAPI.",
        agent_id="agent-1",
        ttl_seconds=10,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=20),
    )

    assert fragment.is_working_memory
    assert fragment.is_expired()

    unverified = fragment.transition_to(MemoryState.UNVERIFIED)
    verified = unverified.transition_to(MemoryState.VERIFIED)

    assert verified.state == MemoryState.VERIFIED
    with pytest.raises(ValueError):
        verified.transition_to(MemoryState.RAW)


def test_memory_block_model_groups_retrievable_fragments():
    block = MemoryBlockModel(name="project", agent_id="agent-1")
    raw = MemoryFragment(content="Raw candidate", agent_id="agent-1")
    verified = MemoryFragment(
        content="Verified fact",
        agent_id="agent-1",
        state=MemoryState.VERIFIED,
        tier=MemoryTier.SEMANTIC,
    )

    block.add(raw)
    block.add(verified)

    assert block.stats()["raw"] == 1
    assert block.stats()["verified"] == 1
    assert block.retrievable()[0].content == "Verified fact"
    assert block.fragments[0].block_id == block.id


def test_source_attribution_and_gap_report():
    attribution = SourceAttribution(
        source_url="https://example.com/docs",
        source_title="Example Docs",
        credibility=SourceCredibility.VERIFIED,
        via="document",
    )
    gap = GapReport.from_score(top_score=0.1, result_count=1, threshold=0.8)

    assert attribution.trust_score == 0.90
    assert "Example Docs" in attribution.cite()
    assert gap.severity == GapSeverity.HIGH
    assert gap.should_search


def test_document_node_and_edge_graph_primitives():
    fragment = MemoryFragment(content="Postgres stores task records.", agent_id="agent-1")
    document = MemoryDocument(title="Architecture", content="Postgres stores task records.", agent_id="agent-1")
    document.register_chunks([fragment])
    node_a = MemoryNode(label="Postgres", node_type=NodeType.ENTITY, fragment_id=fragment.id)
    node_b = MemoryNode(label="Task records", node_type=NodeType.CONCEPT, document_id=document.id)
    edge = MemoryEdge(
        source_id=node_a.id,
        target_id=node_b.id,
        relation=EdgeRelation.SUPPORTS,
        agent_id="agent-1",
    )

    assert document.fragment_ids == [fragment.id]
    assert node_a.summary().startswith("[NODE|entity")
    assert edge.is_expansive()
    assert not edge.is_contradiction()
    assert MemoryScope.AGENT.value == "agent"
