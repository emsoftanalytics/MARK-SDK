# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Integration: observe() uses LLM structured extraction to create typed nodes,
edges and tags, falling back to deterministic extraction on malformed output.
"""
from __future__ import annotations

from pathlib import Path

from mark import ObserveMiddleware
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.types.graph import EdgeRelation, NodeType

from _helpers_structured_extractor import _FakeLLM, _VALID_JSON


# ---------------------------------------------------------------------------
# Integration: observe() uses structured extraction
# ---------------------------------------------------------------------------

def test_observe_with_llm_creates_structured_nodes(tmp_path: Path) -> None:
    llm = _FakeLLM(_VALID_JSON)
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        llm=llm,
        middleware=[ObserveMiddleware()],
    )
    mem = runtime.memory("agent")

    result = mem.observe(
        "Elena enters the North Warehouse wearing the red scarf.",
        session_id="ep-01",
    )
    assert result.inferred is True
    labels = {n.label for n in result.nodes}
    assert "Elena" in labels
    assert "Red Scarf" in labels
    assert "North Warehouse" in labels
    runtime.shutdown()


def test_observe_llm_tags_stored_on_fragment(tmp_path: Path) -> None:
    """LLM-derived tags are applied to the fragment at write time (merged before store)."""
    llm = _FakeLLM(_VALID_JSON)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("agent")

    result = mem.observe("Some observation text.")
    stored = runtime.store.get(result.fragment_id)
    assert stored is not None
    tag_set = set(stored.tags)
    # Tags from LLM JSON response are on the fragment
    assert "character:elena" in tag_set
    assert "object:red-scarf" in tag_set
    assert "location:north-warehouse" in tag_set
    runtime.shutdown()


def test_observe_malformed_llm_falls_back_to_deterministic(tmp_path: Path) -> None:
    """Malformed LLM output → structured extraction returns empty → deterministic only."""
    llm = _FakeLLM("not json")
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("agent")

    result = mem.observe(
        "Elena enters the North Warehouse.",
        session_id="ep-01",
    )
    # Deterministic still finds Elena and North Warehouse
    labels = {n.label for n in result.nodes}
    assert "Elena" in labels or "North Warehouse" in labels
    assert result.fragment_id
    runtime.shutdown()


def test_observe_node_types_from_structured_extraction(tmp_path: Path) -> None:
    llm = _FakeLLM(_VALID_JSON)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("agent")

    result = mem.observe("Elena enters the warehouse.")
    node_map = {n.label: n for n in result.nodes}

    assert node_map["Elena"].node_type == NodeType.CHARACTER
    assert node_map["Red Scarf"].node_type == NodeType.OBJECT
    assert node_map["North Warehouse"].node_type == NodeType.LOCATION
    runtime.shutdown()


def test_observe_edge_relations_from_structured_extraction(tmp_path: Path) -> None:
    llm = _FakeLLM(_VALID_JSON)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("agent")

    result = mem.observe("Elena enters the warehouse.")
    rels = {e.relation for e in result.edges}

    assert EdgeRelation.OWNS in rels
    assert EdgeRelation.LOCATED_AT in rels
    runtime.shutdown()
