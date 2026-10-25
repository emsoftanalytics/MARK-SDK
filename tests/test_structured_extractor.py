# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Structured extraction: ExtractedEntity / ExtractedRelation / StructuredExtraction
schema validation and _parse_structured_response parsing.
"""
from __future__ import annotations

import pytest

from mark import ExtractedEntity, ExtractedRelation, StructuredExtraction
from mark.intelligence.extractor import _parse_structured_response

from _helpers_structured_extractor import _VALID_JSON


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
