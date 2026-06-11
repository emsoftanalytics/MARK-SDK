# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Tests for: ExtractedEntity, ExtractedRelation, StructuredExtraction,
# _parse_structured_response, LLMStructuredExtractor, ExtractionMerger,
# and observe() integration with structured extraction.
from __future__ import annotations

from pathlib import Path

import pytest

from mark import (
    DeterministicExtractor,
    ExtractionMerger,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    LLMStructuredExtractor,
    Mark,
    StructuredExtraction,
)
from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.extractor import _parse_structured_response
from mark.memory.runtime import MarkRuntime
from mark.types.graph import EdgeRelation, NodeType


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_extracted_entity_valid() -> None:
    e = ExtractedEntity(type="character", label="Elena", confidence=0.9)
    assert e.type == "character"
    assert e.label == "Elena"
    assert e.confidence == 0.9


def test_extracted_entity_default_confidence() -> None:
    e = ExtractedEntity(type="object", label="Sword")
    assert e.confidence == 1.0


def test_extracted_entity_rejects_invalid_type() -> None:
    with pytest.raises(Exception):
        ExtractedEntity(type="vehicle", label="Car")  # type: ignore[arg-type]


def test_extracted_entity_confidence_clamped() -> None:
    with pytest.raises(Exception):
        ExtractedEntity(type="character", label="X", confidence=1.5)
    with pytest.raises(Exception):
        ExtractedEntity(type="character", label="X", confidence=-0.1)


def test_extracted_relation_valid() -> None:
    r = ExtractedRelation(source="Elena", relation="owns", target="Sword", confidence=0.85)
    assert r.relation == "owns"


def test_extracted_relation_rejects_invalid_relation() -> None:
    with pytest.raises(Exception):
        ExtractedRelation(source="A", relation="destroys", target="B")  # type: ignore[arg-type]


def test_structured_extraction_defaults() -> None:
    s = StructuredExtraction()
    assert s.entities == []
    assert s.relations == []
    assert s.tags == []
    assert s.summary is None


def test_structured_extraction_extra_fields_ignored() -> None:
    s = StructuredExtraction.model_validate({"entities": [], "unknown_field": "ignored"})
    assert s.entities == []


# ---------------------------------------------------------------------------
# _parse_structured_response
# ---------------------------------------------------------------------------

_VALID_JSON = """{
  "entities": [
    {"type": "character", "label": "Elena",          "confidence": 0.95},
    {"type": "object",    "label": "Red Scarf",      "confidence": 0.90},
    {"type": "location",  "label": "North Warehouse","confidence": 0.90}
  ],
  "relations": [
    {"source": "Elena", "relation": "owns",       "target": "Red Scarf",      "confidence": 0.90},
    {"source": "Elena", "relation": "located_at", "target": "North Warehouse","confidence": 0.90}
  ],
  "tags": ["character:elena", "object:red-scarf", "location:north-warehouse"],
  "summary": "Elena enters the North Warehouse."
}"""

_FENCED_JSON = "```json\n" + _VALID_JSON + "\n```"
_FENCED_NO_LANG = "```\n" + _VALID_JSON + "\n```"
_EMPTY_JSON = '{"entities": [], "relations": [], "tags": [], "summary": null}'
_MALFORMED = "I have no idea what format you want."


def test_parse_valid_json() -> None:
    result = _parse_structured_response(_VALID_JSON)
    assert len(result.entities) == 3
    assert len(result.relations) == 2
    assert "character:elena" in result.tags


def test_parse_fenced_json_lang() -> None:
    result = _parse_structured_response(_FENCED_JSON)
    assert len(result.entities) == 3


def test_parse_fenced_json_no_lang() -> None:
    result = _parse_structured_response(_FENCED_NO_LANG)
    assert len(result.entities) == 3


def test_parse_empty_json() -> None:
    result = _parse_structured_response(_EMPTY_JSON)
    assert result.entities == []
    assert result.relations == []


def test_parse_malformed_returns_empty() -> None:
    result = _parse_structured_response(_MALFORMED)
    assert isinstance(result, StructuredExtraction)
    assert result.entities == []


def test_parse_entity_labels_preserved() -> None:
    result = _parse_structured_response(_VALID_JSON)
    labels = {e.label for e in result.entities}
    assert "Elena" in labels
    assert "Red Scarf" in labels
    assert "North Warehouse" in labels


def test_parse_entity_types_correct() -> None:
    result = _parse_structured_response(_VALID_JSON)
    types = {e.label: e.type for e in result.entities}
    assert types["Elena"] == "character"
    assert types["Red Scarf"] == "object"
    assert types["North Warehouse"] == "location"


# ---------------------------------------------------------------------------
# LLMStructuredExtractor
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


def test_llm_extractor_returns_structured_extraction() -> None:
    llm = _FakeLLM(_VALID_JSON)
    extractor = LLMStructuredExtractor(llm)
    result = extractor.extract_structured("Elena enters the warehouse.")
    assert isinstance(result, StructuredExtraction)
    assert len(result.entities) == 3


def test_llm_extractor_calls_llm_once() -> None:
    llm = _FakeLLM(_VALID_JSON)
    extractor = LLMStructuredExtractor(llm)
    extractor.extract_structured("some text")
    assert len(llm.prompts) == 1


def test_llm_extractor_prompt_contains_text() -> None:
    llm = _FakeLLM(_VALID_JSON)
    extractor = LLMStructuredExtractor(llm)
    extractor.extract_structured("Elena enters the North Warehouse.")
    assert "Elena enters the North Warehouse." in llm.prompts[0]


def test_llm_extractor_filters_by_confidence() -> None:
    low_conf_json = """{
  "entities": [
    {"type": "character", "label": "Elena",   "confidence": 0.9},
    {"type": "character", "label": "Unknown", "confidence": 0.1}
  ],
  "relations": [],
  "tags": ["character:elena", "character:unknown"]
}"""
    llm = _FakeLLM(low_conf_json)
    extractor = LLMStructuredExtractor(llm, min_confidence=0.5)
    result = extractor.extract_structured("text")
    labels = {e.label for e in result.entities}
    assert "Elena" in labels
    assert "Unknown" not in labels
    assert "character:elena" in result.tags
    assert "character:unknown" not in result.tags


def test_llm_extractor_fail_open_on_exception() -> None:
    class BrokenLLM:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("model unavailable")

    extractor = LLMStructuredExtractor(BrokenLLM())
    result = extractor.extract_structured("text")
    assert isinstance(result, StructuredExtraction)
    assert result.entities == []


def test_llm_extractor_handles_malformed_output() -> None:
    llm = _FakeLLM("Not JSON at all.")
    extractor = LLMStructuredExtractor(llm)
    result = extractor.extract_structured("text")
    assert result.entities == []


# ---------------------------------------------------------------------------
# ExtractionMerger
# ---------------------------------------------------------------------------

def _make_det(entities=None, relations=None, tags=None) -> ExtractionResult:
    return ExtractionResult(
        entities  = entities  or [],
        relations = relations or [],
        tags      = tags      or [],
    )


def _make_struct(**kwargs) -> StructuredExtraction:
    return StructuredExtraction(**kwargs)


def test_merger_deterministic_only() -> None:
    det = _make_det(
        entities=[("character", "Elena")],
        tags=["character:elena"],
    )
    merged = ExtractionMerger().merge(det, StructuredExtraction())
    assert merged.entities == [("character", "Elena")]
    assert merged.tags == ["character:elena"]


def test_merger_adds_new_llm_entities() -> None:
    det = _make_det(entities=[("character", "Elena")], tags=["character:elena"])
    struct = _make_struct(
        entities=[
            ExtractedEntity(type="location", label="North Warehouse"),
        ],
    )
    merged = ExtractionMerger().merge(det, struct)
    labels = [label for _, label in merged.entities]
    assert "Elena" in labels
    assert "North Warehouse" in labels


def test_merger_deduplicates_by_label() -> None:
    det    = _make_det(entities=[("character", "Elena")], tags=["character:elena"])
    struct = _make_struct(entities=[ExtractedEntity(type="character", label="Elena")])
    merged = ExtractionMerger().merge(det, struct)
    # Elena appears only once
    elena_count = sum(1 for _, label in merged.entities if label == "Elena")
    assert elena_count == 1


def test_merger_deduplicates_case_insensitive() -> None:
    det    = _make_det(entities=[("character", "Elena")])
    struct = _make_struct(entities=[ExtractedEntity(type="character", label="ELENA")])
    merged = ExtractionMerger().merge(det, struct)
    # Only one Elena (case-insensitive)
    assert len(merged.entities) == 1


def test_merger_merges_relations_without_duplicates() -> None:
    det    = _make_det(relations=[("Elena", "located_at", "Warehouse")])
    struct = _make_struct(
        entities=[],
        relations=[
            ExtractedRelation(source="Elena", relation="located_at", target="Warehouse"),
            ExtractedRelation(source="Elena", relation="owns",       target="Red Scarf"),
        ],
    )
    merged = ExtractionMerger().merge(det, struct)
    # located_at should appear only once; owns should appear once
    located_at_count = sum(1 for _, rel, _ in merged.relations if rel == "located_at")
    owns_count       = sum(1 for _, rel, _ in merged.relations if rel == "owns")
    assert located_at_count == 1
    assert owns_count == 1


def test_merger_generates_tag_for_new_entity() -> None:
    det    = _make_det()
    struct = _make_struct(entities=[ExtractedEntity(type="location", label="North Warehouse")])
    merged = ExtractionMerger().merge(det, struct)
    assert "location:north-warehouse" in merged.tags


def test_merger_merges_tags() -> None:
    det    = _make_det(tags=["character:elena"])
    struct = _make_struct(tags=["location:north-warehouse"])
    merged = ExtractionMerger().merge(det, struct)
    assert "character:elena"       in merged.tags
    assert "location:north-warehouse" in merged.tags


def test_merger_no_duplicate_tags() -> None:
    det    = _make_det(tags=["character:elena"])
    struct = _make_struct(
        entities=[ExtractedEntity(type="character", label="Elena")],
        tags=["character:elena"],
    )
    merged = ExtractionMerger().merge(det, struct)
    assert merged.tags.count("character:elena") == 1


# ---------------------------------------------------------------------------
# Integration: observe() uses structured extraction
# ---------------------------------------------------------------------------

def test_observe_with_llm_creates_structured_nodes(tmp_path: Path) -> None:
    llm = _FakeLLM(_VALID_JSON)
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        llm=llm,
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
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
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
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
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
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
    mem = runtime.memory("agent")

    result = mem.observe("Elena enters the warehouse.")
    node_map = {n.label: n for n in result.nodes}

    assert node_map["Elena"].node_type == NodeType.CHARACTER
    assert node_map["Red Scarf"].node_type == NodeType.OBJECT
    assert node_map["North Warehouse"].node_type == NodeType.LOCATION
    runtime.shutdown()


def test_observe_edge_relations_from_structured_extraction(tmp_path: Path) -> None:
    llm = _FakeLLM(_VALID_JSON)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
    mem = runtime.memory("agent")

    result = mem.observe("Elena enters the warehouse.")
    rels = {e.relation for e in result.edges}

    assert EdgeRelation.OWNS in rels
    assert EdgeRelation.LOCATED_AT in rels
    runtime.shutdown()
