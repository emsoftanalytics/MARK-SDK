# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Structured extraction: LLMStructuredExtractor (prompting, confidence filtering,
fail-open) and ExtractionMerger (dedup and merge of entities, relations, tags).
"""
from __future__ import annotations

from mark import (
    ExtractionMerger,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    LLMStructuredExtractor,
    StructuredExtraction,
)

from _helpers_structured_extractor import _FakeLLM, _VALID_JSON


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
