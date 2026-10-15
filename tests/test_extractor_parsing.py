# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Robust LLM response parsing, sentence-scoped relations, and logged LLM failures."""
from __future__ import annotations

import logging

import pytest

from mark.intelligence.compressor import LLMContextualCompressor
from mark.intelligence.extractor import (
    DeterministicExtractor,
    LLMStructuredExtractor,
    _parse_structured_response,
)
from mark.intelligence.query_expander import LLMQueryExpander
from mark.types import MemoryFragment

_BODY = (
    '{"entities": [{"type": "character", "label": "Elena", "confidence": 0.9},'
    ' {"type": "location", "label": "North Warehouse", "confidence": 0.8}],'
    ' "relations": [{"source": "Elena", "relation": "located_at", "target": "North Warehouse"}],'
    ' "tags": ["character:elena"], "summary": "Elena arrives."}'
)


@pytest.mark.parametrize(
    "raw",
    [
        _BODY,
        "Sure! Here is the extraction:\n```json\n" + _BODY + "\n```",
        "Here you go: " + _BODY,
        _BODY + "\n\nLet me know if you need anything else {or not}.",
        "Thinking... {not json at all} then the answer " + _BODY + " done.",
        "```\n" + _BODY + "\n```\ntrailing prose",
    ],
)
def test_parse_tolerates_prose_and_fences(raw: str) -> None:
    result = _parse_structured_response(raw)
    assert [e.label for e in result.entities] == ["Elena", "North Warehouse"]
    assert len(result.relations) == 1
    assert result.summary == "Elena arrives."


def test_one_invalid_entity_type_does_not_discard_the_rest() -> None:
    raw = (
        '{"entities": [{"type": "spaceship", "label": "Nostromo"},'
        ' {"type": "character", "label": "Ripley"}], "relations": []}'
    )
    result = _parse_structured_response(raw)
    assert [e.label for e in result.entities] == ["Ripley"]


def test_types_are_lowercased_and_aliased() -> None:
    raw = (
        '{"entities": [{"type": "Person", "label": "Elena"},'
        ' {"type": "PLACE", "label": "Dock"}, {"type": "thing", "label": "Lamp"}],'
        ' "relations": [{"source": "Elena", "relation": "Located In", "target": "Dock"},'
        ' {"source": "Elena", "relation": "has", "target": "Lamp"},'
        ' {"source": "Elena", "relation": "teleports", "target": "Mars"}]}'
    )
    result = _parse_structured_response(raw)
    assert {(e.type, e.label) for e in result.entities} == {
        ("character", "Elena"), ("location", "Dock"), ("object", "Lamp"),
    }
    assert {(r.relation, r.target) for r in result.relations} == {("located_at", "Dock"), ("owns", "Lamp")}


def test_null_and_bad_confidence_default_or_clamp() -> None:
    raw = (
        '{"entities": [{"type": "character", "label": "A", "confidence": null},'
        ' {"type": "character", "label": "B", "confidence": "0.4"},'
        ' {"type": "character", "label": "C", "confidence": 7},'
        ' {"type": "character", "label": "D", "confidence": "high"}]}'
    )
    conf = {e.label: e.confidence for e in _parse_structured_response(raw).entities}
    assert conf == {"A": 1.0, "B": 0.4, "C": 1.0, "D": 1.0}


def test_items_missing_labels_are_skipped() -> None:
    raw = '{"entities": [{"type": "character"}, "Elena", {"type": "character", "label": "  "},' \
          ' {"type": "object", "label": "Key"}], "relations": [{"source": "", "relation": "owns"}]}'
    result = _parse_structured_response(raw)
    assert [e.label for e in result.entities] == ["Key"]
    assert result.relations == []


def test_braces_inside_strings_are_handled() -> None:
    raw = 'prefix {"entities": [{"type": "concept", "label": "set {x}"}], "summary": "a } b"} tail'
    result = _parse_structured_response(raw)
    assert result.entities[0].label == "set {x}"
    assert result.summary == "a } b"


def test_unparseable_response_logs_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="mark.intelligence.extractor"):
        result = _parse_structured_response("no json here")
    assert result.entities == []
    assert any("no JSON object" in rec.message for rec in caplog.records)


def test_llm_extractor_parses_messy_output() -> None:
    class ChattyLLM:
        def complete(self, prompt: str) -> str:
            return "Of course!\n```json\n" + _BODY + "\n```\nAnything else?"

    result = LLMStructuredExtractor(ChattyLLM()).extract_structured("Elena arrives at the North Warehouse.")
    assert {e.label for e in result.entities} == {"Elena", "North Warehouse"}


# ── item 12: LLM helpers log failures ─────────────────────────────────────────

class _BrokenLLM:
    def complete(self, prompt: str) -> str:
        raise ConnectionError("model offline")


def test_llm_helpers_log_failures_with_exception_type(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert LLMQueryExpander(_BrokenLLM()).expand("q") == ["q"]
        frag = MemoryFragment(content="keep me", agent_id="a")
        assert LLMContextualCompressor(_BrokenLLM()).compress("q", [frag]) == [frag]
        assert LLMStructuredExtractor(_BrokenLLM()).extract_structured("t").entities == []
    messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert sum("ConnectionError" in m for m in messages) == 3


# ── item 11: deterministic relations are sentence-scoped ──────────────────────

def test_relations_only_link_within_the_same_sentence() -> None:
    text = "Elena holds a lantern. Marco enters the Vault."
    rels = set(DeterministicExtractor().extract(text).relations)
    assert ("Elena", "owns", "Lantern") in rels
    assert ("Marco", "located_at", "Vault") in rels
    assert ("Elena", "located_at", "Vault") not in rels
    assert ("Marco", "owns", "Lantern") not in rels


def test_pronoun_sentence_is_not_linked_to_earlier_subject() -> None:
    text = "Elena enters the Vault. She carries a heavy box."
    rels = DeterministicExtractor().extract(text).relations
    assert not any(rel == "owns" for _, rel, _ in rels)


@pytest.mark.parametrize(
    "text",
    [
        "Elena used to live here.",
        "Marco takes care of the garden.",
        "The festival took place downtown.",
        "Elena took part in the race.",
        "Marco held on tightly.",
    ],
)
def test_idioms_do_not_create_objects(text: str) -> None:
    result = DeterministicExtractor().extract(text)
    assert not any(t == "object" for t, _ in result.entities), result.entities
    assert not any(rel == "owns" for _, rel, _ in result.relations)


def test_regular_ownership_still_extracted() -> None:
    result = DeterministicExtractor().extract(
        "Elena enters the North Warehouse wearing the red scarf.", session_id="s/scene-4",
    )
    rels = set(result.relations)
    assert ("Elena", "located_at", "North Warehouse") in rels
    assert ("Elena", "owns", "Red Scarf") in rels
    assert ("Elena", "appears_in", "scene-4") in rels
