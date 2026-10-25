# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""mark.observe() basics: LLMProvider protocol check, extraction-parser unit
tests, and observe() without an LLM (deterministic extraction).
"""
from __future__ import annotations

from pathlib import Path

from mark import LLMProvider, ObserveMiddleware, ObserveResult
from mark.embeddings import HashEmbeddingProvider
from mark.memory.observe import _parse_extraction, extract_entities
from mark.memory.runtime import MarkRuntime

from _helpers_observe import _EMPTY_RESPONSE, _FakeLLM, _MALFORMED_RESPONSE


_ELENA_RESPONSE = """\
ENTITIES:
character: Elena
object: Red Scarf
location: North Warehouse

RELATIONS:
Elena -> owns -> Red Scarf
Elena -> located_at -> North Warehouse
Red Scarf -> appears_in -> North Warehouse
"""


# ---------------------------------------------------------------------------
# LLMProvider protocol check
# ---------------------------------------------------------------------------

def test_llm_provider_protocol_satisfied() -> None:
    llm = _FakeLLM(_ELENA_RESPONSE)
    assert isinstance(llm, LLMProvider)


def test_non_llm_does_not_satisfy_protocol() -> None:
    class NotAnLLM:
        pass
    assert not isinstance(NotAnLLM(), LLMProvider)


# ---------------------------------------------------------------------------
# Extraction parser unit tests
# ---------------------------------------------------------------------------

def test_parse_extraction_entities_and_relations() -> None:
    entities, relations = _parse_extraction(_ELENA_RESPONSE)

    assert ("character", "Elena") in entities
    assert ("object", "Red Scarf") in entities
    assert ("location", "North Warehouse") in entities

    assert ("Elena", "owns", "Red Scarf") in relations
    assert ("Elena", "located_at", "North Warehouse") in relations
    assert ("Red Scarf", "appears_in", "North Warehouse") in relations


def test_parse_extraction_none_sections() -> None:
    entities, relations = _parse_extraction(_EMPTY_RESPONSE)
    assert entities == []
    assert relations == []


def test_parse_extraction_malformed_returns_empty() -> None:
    entities, relations = _parse_extraction(_MALFORMED_RESPONSE)
    assert isinstance(entities, list)
    assert isinstance(relations, list)


def test_extract_entities_calls_llm() -> None:
    llm = _FakeLLM(_ELENA_RESPONSE)
    entities, relations = extract_entities("Elena enters the warehouse.", llm)
    assert len(llm.calls) == 1
    assert len(entities) == 3
    assert len(relations) == 3


def test_extract_entities_llm_exception_returns_empty() -> None:
    class BrokenLLM:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("model unavailable")

    entities, relations = extract_entities("some text", BrokenLLM())
    assert entities == []
    assert relations == []


# ---------------------------------------------------------------------------
# observe() without LLM — deterministic extraction runs automatically
# ---------------------------------------------------------------------------

def test_observe_without_llm_stores_fragment(tmp_path: Path) -> None:
    """Deterministic extractor runs even without an LLM configured."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32),
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe(
        "Elena enters the North Warehouse.",
        session_id="season-01/ep-02/scene-04",
    )

    assert isinstance(result, ObserveResult)
    assert result.fragment_id
    # Deterministic extractor finds: scene-04 (session), North Warehouse (location), Elena (subject)
    assert result.inferred is True
    assert result.session_id == "season-01/ep-02/scene-04"
    labels = {n.label for n in result.nodes}
    assert "Elena" in labels or "North Warehouse" in labels   # at least one found

    # Auto-tags should be applied to the stored fragment
    stored = runtime.store.get(result.fragment_id)
    assert stored is not None
    assert stored.content == "Elena enters the North Warehouse."
    assert any(t.startswith("scene:") or t.startswith("location:") for t in stored.tags)
    runtime.shutdown()


def test_observe_plain_text_no_entities_inferred_false(tmp_path: Path) -> None:
    """Text with no extractable entities gives inferred=False without LLM."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")

    result = mem.observe("The sky is blue.")
    assert result.fragment_id
    assert result.inferred is False
    assert result.nodes == []
    assert result.edges == []
    runtime.shutdown()
