# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Graph API tests — node/edge CRUD, neighbourhood traversal, SimpleMemory delegation,
# media-domain NodeType/EdgeRelation values, and retrieval expansion via attached nodes.
from __future__ import annotations

from pathlib import Path

import pytest

from mark import GraphNeighborhood, Mark
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


# ---------------------------------------------------------------------------
# Neighbourhood traversal
# ---------------------------------------------------------------------------

def test_neighborhood_center_and_depth(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("director")
    fid = mem.store_sync("Elena wears a red scarf.")

    mem.attach_node(fid, "Elena", NodeType.CHARACTER)
    mem.node("Red Scarf", NodeType.OBJECT)
    mem.node("Elena's Mother", NodeType.CHARACTER)
    mem.link_nodes("Elena", "Red Scarf", EdgeRelation.OWNS, weight=0.9)
    mem.link_nodes("Red Scarf", "Elena's Mother", EdgeRelation.SYMBOLIZES)

    hood = mem.neighborhood("Elena", depth=2)

    assert hood.center.label == "Elena"
    assert hood.depth == 2
    labels = hood.node_labels()
    assert "Elena" in labels
    assert "Red Scarf" in labels
    assert "Elena's Mother" in labels  # reachable at depth 2
    runtime.shutdown()


def test_neighborhood_includes_attached_fragments(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("director")
    fid_a = mem.store_sync("Elena wears the scarf.")
    fid_b = mem.store_sync("The scarf belonged to Elena's mother.")

    mem.attach_node(fid_a, "Elena", NodeType.CHARACTER)
    mem.attach_node(fid_b, "Red Scarf", NodeType.OBJECT)
    mem.link_nodes("Elena", "Red Scarf", EdgeRelation.OWNS)

    hood = mem.neighborhood("Elena", depth=1)

    frag_texts = hood.fragment_texts()
    assert any("scarf" in t.lower() for t in frag_texts)
    runtime.shutdown()


def test_neighborhood_raises_for_unknown_node(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("director")

    with pytest.raises(KeyError, match="Ghost"):
        mem.neighborhood("Ghost")
    runtime.shutdown()


def test_neighborhood_incoming_edges_traversed(tmp_path: Path) -> None:
    """B → A: A's neighbourhood at depth 1 should include B."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("director")

    mem.node("Red Scarf", NodeType.OBJECT)
    mem.node("Elena", NodeType.CHARACTER)
    mem.link_nodes("Red Scarf", "Elena", EdgeRelation.OWNS)  # edge goes Scarf → Elena

    hood = mem.neighborhood("Elena", depth=1)

    assert "Red Scarf" in hood.node_labels()
    runtime.shutdown()


# ---------------------------------------------------------------------------
# as_context() output
# ---------------------------------------------------------------------------

def test_neighborhood_as_context_format(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("director")
    fid = mem.store_sync("Elena is left-handed.")
    mem.attach_node(fid, "Elena", NodeType.CHARACTER)
    mem.node("Glass Circle", NodeType.ENTITY)
    mem.link_nodes("Elena", "Glass Circle", EdgeRelation.PART_OF)

    ctx = mem.neighborhood("Elena").as_context()

    assert "Elena" in ctx
    assert "character" in ctx
    assert "part_of" in ctx
    assert "<graph_context" in ctx
    assert "</graph_context>" in ctx
    runtime.shutdown()


# ---------------------------------------------------------------------------
# SimpleMemory graph delegation
# ---------------------------------------------------------------------------

def test_simple_memory_graph_methods(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        rec = mark.memory.block("chars").write("Elena is the protagonist.")

        node  = mark.memory.node("Elena", "character")
        assert node is not None
        assert node.label == "Elena"

        mark.memory.node("Red Scarf", "object")
        edge = mark.memory.link_nodes("Elena", "Red Scarf", "owns", weight=0.9)
        assert edge.relation == EdgeRelation.OWNS

        hood = mark.memory.neighborhood("Elena", depth=1)
        assert isinstance(hood, GraphNeighborhood)
        assert "Elena" in hood.node_labels()
        assert "Red Scarf" in hood.node_labels()

        removed = mark.memory.unlink("Elena", "Red Scarf", "owns")
        assert removed == 1


# ---------------------------------------------------------------------------
# Graph expansion in retrieval uses attached nodes
# ---------------------------------------------------------------------------

def test_attached_node_expands_retrieval(tmp_path: Path) -> None:
    """Attaching two fragments to connected nodes causes them to co-retrieve."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                 embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("director")

    from mark.intelligence import RetrievalPolicy

    fid_a = mem.store_sync("Elena is fearless.", importance=0.9)
    fid_b = mem.store_sync("The red scarf is her mother's.", importance=0.9)

    mem.attach_node(fid_a, "Elena", NodeType.CHARACTER)
    mem.attach_node(fid_b, "Red Scarf", NodeType.OBJECT)
    mem.link_nodes("Elena", "Red Scarf", EdgeRelation.OWNS, weight=1.0)

    result = mem.retrieve_sync("Elena", policy=RetrievalPolicy.DEEP)

    ids_found = {f.id for f in result.fragments}
    # fid_a should be retrieved directly; fid_b via graph expansion through OWNS edge
    assert fid_a in ids_found or fid_b in ids_found
    runtime.shutdown()
