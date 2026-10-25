"""Open relation vocabulary: EdgeRelation accepts any relation name (B7).

Known relations stay enum members; custom ones ("manages") round-trip
through the store and are traversed by graph expansion.
"""
from __future__ import annotations

import pickle

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.graph_expansion import GraphExpander
from mark.memory.runtime import MarkRuntime
from mark.store import LocalMemoryStore
from mark.types.graph import EdgeRelation, MemoryEdge, MemoryNode


def test_known_relations_are_unchanged() -> None:
    assert EdgeRelation("owns") is EdgeRelation.OWNS
    assert EdgeRelation("OWNS") is EdgeRelation.OWNS
    assert EdgeRelation.CONTRADICTS.value == "contradicts"
    assert EdgeRelation.RELATED_TO in list(EdgeRelation)


def test_custom_relation_is_accepted_and_normalised() -> None:
    manages = EdgeRelation("manages")
    assert manages.value == "manages"
    assert manages == "manages"
    assert EdgeRelation("Manages") is manages
    assert EdgeRelation("reports to").value == "reports_to"
    assert manages.is_expansive
    assert manages not in list(EdgeRelation)  # the known vocabulary is not polluted
    assert pickle.loads(pickle.dumps(manages)) is manages


@pytest.mark.parametrize("bad", ["", "   ", "x" * 80, "drop;table", 7])
def test_invalid_relation_names_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        EdgeRelation(bad)


def test_custom_relation_round_trips_through_the_store() -> None:
    store = LocalMemoryStore()
    a, b = MemoryNode(label="Priya", agent_id="a"), MemoryNode(label="Tomasz", agent_id="a")
    store.store_node(a)
    store.store_node(b)
    store.store_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation="manages", agent_id="a"))
    [edge] = store.edges_from_node(a.id)
    assert edge.relation.value == "manages"


def test_link_nodes_with_custom_relation_and_expansion() -> None:
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    try:
        mem = runtime.memory("org")
        edge = mem.link_nodes("Priya", "Tomasz", relation="manages")
        assert edge.relation.value == "manages"
        store = runtime.store
        f1 = mem.store_sync("Priya runs the payments group.")
        f2 = mem.store_sync("Tomasz is on call this week.")
        priya = store.get_node_by_label("Priya", "org")
        tomasz = store.get_node_by_label("Tomasz", "org")
        store.store_node(priya.model_copy(update={"fragment_id": f1}))
        store.store_node(tomasz.model_copy(update={"fragment_id": f2}))
        scores = GraphExpander(store).expand({f1: 1.0}, "org", max_hops=1)
        assert scores.get(f2, 0.0) > 0.0
    finally:
        runtime.shutdown()
