# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""mark.observe() with an LLM: node/edge creation from extraction, fail-open on
empty or malformed output, configure_llm() late binding, and Mark.local(llm=...).
"""
from __future__ import annotations

from pathlib import Path

from mark import Mark, ObserveMiddleware
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.types.graph import EdgeRelation, NodeType

from _helpers_observe import _ELENA_JSON_RESPONSE, _EMPTY_RESPONSE, _FakeLLM, _MALFORMED_RESPONSE


# ---------------------------------------------------------------------------
# observe() with LLM — extraction path
# ---------------------------------------------------------------------------

def test_observe_with_llm_creates_nodes_and_edges(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32),
                                llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe(
        "Elena enters the North Warehouse wearing the red scarf.",
        session_id="season-01/ep-02/scene-04",
    )

    assert result.inferred is True
    labels = {n.label for n in result.nodes}
    # Deterministic + LLM combined; all three key entities must be present
    assert "Elena" in labels
    assert "Red Scarf" in labels
    assert "North Warehouse" in labels
    assert len(result.edges) >= 3   # deterministic + LLM may create more
    runtime.shutdown()


def test_observe_nodes_have_correct_types(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe("Elena enters the warehouse.")
    node_map = {n.label: n for n in result.nodes}

    assert node_map["Elena"].node_type == NodeType.CHARACTER
    assert node_map["Red Scarf"].node_type == NodeType.OBJECT
    assert node_map["North Warehouse"].node_type == NodeType.LOCATION
    runtime.shutdown()


def test_observe_edge_relations_correct(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe("Elena enters the warehouse.")
    rels = {e.relation for e in result.edges}

    assert EdgeRelation.OWNS in rels
    assert EdgeRelation.LOCATED_AT in rels
    assert EdgeRelation.APPEARS_IN in rels
    runtime.shutdown()


def test_observe_fragment_attached_to_nodes(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32),
                                llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe("Elena enters the warehouse.")
    frag_id = result.fragment_id

    # Elena node should be attached to the fragment
    elena_node = next((n for n in result.nodes if n.label == "Elena"), None)
    assert elena_node is not None
    assert elena_node.fragment_id == frag_id
    runtime.shutdown()


def test_observe_with_empty_extraction_stores_fragment(tmp_path: Path) -> None:
    """Plain text with no entities + LLM returning NONE → inferred=False."""
    llm = _FakeLLM(_EMPTY_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe("The sky is blue.")
    assert result.fragment_id
    assert result.inferred is False   # neither deterministic nor LLM found anything
    assert result.nodes == []
    assert result.edges == []
    runtime.shutdown()


def test_observe_with_malformed_llm_output_stores_fragment(tmp_path: Path) -> None:
    llm = _FakeLLM(_MALFORMED_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe("Chaos reigns.")
    assert result.fragment_id           # write succeeded
    assert result.inferred is False     # malformed output → no entities
    runtime.shutdown()


# ---------------------------------------------------------------------------
# configure_llm() — late binding
# ---------------------------------------------------------------------------

def test_configure_llm_enables_extraction(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("director")

    # Without LLM — no extraction
    r1 = mem.observe("Elena walks in.")
    assert r1.inferred is False

    # Bind LLM and observe again
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime.configure_llm(llm)
    runtime.use(ObserveMiddleware())
    mem2 = runtime.memory("director")  # new MarkMemory picks up the LLM
    r2 = mem2.observe("Elena enters the warehouse.")
    assert r2.inferred is True
    runtime.shutdown()


def test_mark_configure_llm_propagates_to_simple_memory(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path, middleware=[ObserveMiddleware()]) as mark:
        llm = _FakeLLM(_ELENA_JSON_RESPONSE)
        mark.configure_llm(llm)

        result = mark.memory.observe(
            "Elena enters the North Warehouse.",
            session_id="ep-01",
        )
        assert result.inferred is True
        assert any(n.label == "Elena" for n in result.nodes)


# ---------------------------------------------------------------------------
# Mark.local(llm=...) constructor wiring
# ---------------------------------------------------------------------------

def test_mark_local_with_llm(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    with Mark.local(project_path=tmp_path, llm=llm,
                    middleware=[ObserveMiddleware()]) as mark:
        result = mark.memory.observe("Elena wears the red scarf.")
        assert result.inferred is True
        assert len(result.nodes) == 3
