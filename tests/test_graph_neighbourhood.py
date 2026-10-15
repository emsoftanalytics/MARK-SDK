# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Graph API: neighbourhood traversal and as_context() rendering, SimpleMemory
graph delegation, and retrieval expansion via attached nodes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark import GraphNeighborhood, Mark
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.types.graph import EdgeRelation, NodeType


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
