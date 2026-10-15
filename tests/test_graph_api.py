# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Graph API: media NodeType/EdgeRelation values, store node/edge additions, and
MarkMemory node/edge CRUD.
"""
from __future__ import annotations

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.types.graph import EdgeRelation, NodeType


# ---------------------------------------------------------------------------
# NodeType and EdgeRelation additions
# ---------------------------------------------------------------------------

def test_media_node_types_exist() -> None:
    assert NodeType.CHARACTER.value == "character"
    assert NodeType.OBJECT.value    == "object"
    assert NodeType.LOCATION.value  == "location"
    assert NodeType.SCENE.value     == "scene"


def test_media_edge_relations_exist() -> None:
    assert EdgeRelation.OWNS.value       == "owns"
    assert EdgeRelation.APPEARS_IN.value == "appears_in"
    assert EdgeRelation.LOCATED_AT.value == "located_at"
    assert EdgeRelation.SYMBOLIZES.value == "symbolizes"


def test_media_edge_relations_are_expansive() -> None:
    assert EdgeRelation.OWNS.is_expansive
    assert EdgeRelation.APPEARS_IN.is_expansive
    assert EdgeRelation.LOCATED_AT.is_expansive
    assert not EdgeRelation.SYMBOLIZES.is_expansive  # thematic only — not traversed


# ---------------------------------------------------------------------------
# Store additions
# ---------------------------------------------------------------------------

def test_get_node_by_label() -> None:
    from mark.store import LocalMemoryStore
    from mark.types.graph import MemoryNode

    store = LocalMemoryStore()
    node  = MemoryNode(label="Elena", agent_id="a", node_type=NodeType.CHARACTER)
    store.store_node(node)

    found = store.get_node_by_label("Elena", "a")
    assert found is not None
    assert found.id == node.id
    assert found.label == "Elena"

    missing = store.get_node_by_label("NonExistent", "a")
    assert missing is None


def test_delete_edges_between() -> None:
    from mark.store import LocalMemoryStore
    from mark.types.graph import MemoryEdge, MemoryNode

    store = LocalMemoryStore()
    n1 = MemoryNode(label="Elena", agent_id="a", node_type=NodeType.CHARACTER)
    n2 = MemoryNode(label="Red Scarf", agent_id="a", node_type=NodeType.OBJECT)
    store.store_node(n1)
    store.store_node(n2)

    e1 = MemoryEdge(source_id=n1.id, target_id=n2.id,
                    relation=EdgeRelation.OWNS, agent_id="a", weight=0.9)
    e2 = MemoryEdge(source_id=n1.id, target_id=n2.id,
                    relation=EdgeRelation.MENTIONS, agent_id="a", weight=0.3)
    store.store_edge(e1)
    store.store_edge(e2)

    # Delete only OWNS
    deleted = store.delete_edges_between(n1.id, n2.id, EdgeRelation.OWNS)
    assert deleted == 1
    remaining = store.edges_from_node(n1.id)
    assert len(remaining) == 1
    assert remaining[0].relation == EdgeRelation.MENTIONS

    # Delete all remaining
    deleted_all = store.delete_edges_between(n1.id, n2.id)
    assert deleted_all == 1
    assert store.edges_from_node(n1.id) == []


# ---------------------------------------------------------------------------
# MarkMemory graph API
# ---------------------------------------------------------------------------

def test_node_get_or_create() -> None:
    runtime = MarkRuntime.local()
    mem = runtime.memory("agent")

    n1 = mem.node("Elena", NodeType.CHARACTER)
    n2 = mem.node("Elena", NodeType.CHARACTER)  # should return existing

    assert n1.id == n2.id
    assert n1.label == "Elena"
    assert n1.node_type == NodeType.CHARACTER
    runtime.shutdown()


def test_node_get_or_create_from_string_type() -> None:
    runtime = MarkRuntime.local()
    mem = runtime.memory("agent")

    n = mem.node("Red Scarf", "object")
    assert n.node_type == NodeType.OBJECT
    runtime.shutdown()


def test_attach_node_links_fragment() -> None:
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    fid = mem.store_sync("Elena wears a red scarf.")

    node = mem.attach_node(fid, "Elena", NodeType.CHARACTER)

    assert node.fragment_id == fid
    nodes_for_frag = runtime.store.nodes_for_fragment(fid)
    assert any(n.id == node.id for n in nodes_for_frag)
    runtime.shutdown()


def test_link_nodes_creates_edge() -> None:
    runtime = MarkRuntime.local()
    mem = runtime.memory("agent")

    edge = mem.link_nodes("Elena", "Red Scarf", EdgeRelation.OWNS, weight=0.9)

    assert edge.relation == EdgeRelation.OWNS
    assert abs(edge.weight - 0.9) < 1e-6
    # Nodes were created implicitly
    assert runtime.store.get_node_by_label("Elena", "agent") is not None
    assert runtime.store.get_node_by_label("Red Scarf", "agent") is not None
    runtime.shutdown()


def test_link_nodes_from_string_relation() -> None:
    runtime = MarkRuntime.local()
    mem = runtime.memory("agent")

    edge = mem.link_nodes("Elena", "North Warehouse", "located_at")
    assert edge.relation == EdgeRelation.LOCATED_AT
    runtime.shutdown()


def test_unlink_removes_edges() -> None:
    runtime = MarkRuntime.local()
    mem = runtime.memory("agent")

    mem.link_nodes("Elena", "Red Scarf", EdgeRelation.OWNS)
    mem.link_nodes("Elena", "Red Scarf", EdgeRelation.MENTIONS)

    removed = mem.unlink("Elena", "Red Scarf", EdgeRelation.OWNS)
    assert removed == 1

    removed_all = mem.unlink("Elena", "Red Scarf")
    assert removed_all == 1  # MENTIONS still present

    removed_none = mem.unlink("Elena", "Red Scarf")
    assert removed_none == 0
    runtime.shutdown()


def test_unlink_nonexistent_nodes_returns_zero() -> None:
    runtime = MarkRuntime.local()
    mem = runtime.memory("agent")

    count = mem.unlink("Ghost", "Nobody")
    assert count == 0
    runtime.shutdown()
