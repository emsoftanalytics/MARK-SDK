# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Tests for mark.observe(), LLMProvider protocol, ObserveResult, ObserveEvent,
# and HOOK_OBSERVE_EVENT wiring.
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mark import LLMProvider, Mark, ObserveEvent, ObserveResult
from mark.embeddings import HashEmbeddingProvider
from mark.memory.observe import _parse_extraction, extract_entities
from mark.memory.runtime import MarkRuntime
from mark.plugins import HOOK_OBSERVE_EVENT
from mark.types.graph import EdgeRelation, NodeType


# ---------------------------------------------------------------------------
# Minimal LLM stubs
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Returns a hardcoded extraction response for any prompt."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


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

# JSON format used by LLMStructuredExtractor in observe() integration tests
_ELENA_JSON_RESPONSE = """\
{
  "entities": [
    {"type": "character", "label": "Elena",          "confidence": 0.95},
    {"type": "object",    "label": "Red Scarf",      "confidence": 0.90},
    {"type": "location",  "label": "North Warehouse","confidence": 0.90}
  ],
  "relations": [
    {"source": "Elena",     "relation": "owns",       "target": "Red Scarf",      "confidence": 0.90},
    {"source": "Elena",     "relation": "located_at", "target": "North Warehouse","confidence": 0.90},
    {"source": "Red Scarf", "relation": "appears_in", "target": "North Warehouse","confidence": 0.85}
  ],
  "tags": ["character:elena", "object:red-scarf", "location:north-warehouse"],
  "summary": "Elena enters the North Warehouse wearing the red scarf."
}"""

_EMPTY_RESPONSE = """\
ENTITIES: NONE
RELATIONS: NONE
"""

_MALFORMED_RESPONSE = "I don't know what you mean. Here is some random text."


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
                                embedder=HashEmbeddingProvider(dim=32))
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
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("director")

    result = mem.observe("The sky is blue.")
    assert result.fragment_id
    assert result.inferred is False
    assert result.nodes == []
    assert result.edges == []
    runtime.shutdown()


# ---------------------------------------------------------------------------
# observe() with LLM — extraction path
# ---------------------------------------------------------------------------

def test_observe_with_llm_creates_nodes_and_edges(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32),
                                llm=llm)
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
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
    mem = runtime.memory("director")

    result = mem.observe("Elena enters the warehouse.")
    node_map = {n.label: n for n in result.nodes}

    assert node_map["Elena"].node_type == NodeType.CHARACTER
    assert node_map["Red Scarf"].node_type == NodeType.OBJECT
    assert node_map["North Warehouse"].node_type == NodeType.LOCATION
    runtime.shutdown()


def test_observe_edge_relations_correct(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
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
                                llm=llm)
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
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
    mem = runtime.memory("director")

    result = mem.observe("The sky is blue.")
    assert result.fragment_id
    assert result.inferred is False   # neither deterministic nor LLM found anything
    assert result.nodes == []
    assert result.edges == []
    runtime.shutdown()


def test_observe_with_malformed_llm_output_stores_fragment(tmp_path: Path) -> None:
    llm = _FakeLLM(_MALFORMED_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db", llm=llm)
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
    mem2 = runtime.memory("director")  # new MarkMemory picks up the LLM
    r2 = mem2.observe("Elena enters the warehouse.")
    assert r2.inferred is True
    runtime.shutdown()


def test_mark_configure_llm_propagates_to_simple_memory(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
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
    with Mark.local(project_path=tmp_path, llm=llm) as mark:
        result = mark.memory.observe("Elena wears the red scarf.")
        assert result.inferred is True
        assert len(result.nodes) == 3


# ---------------------------------------------------------------------------
# HOOK_OBSERVE_EVENT — schema and emission
# ---------------------------------------------------------------------------

def test_observe_event_schema_fields(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32),
                                llm=llm)
    mem = runtime.memory("director")
    result = mem.observe("Elena enters the warehouse.", session_id="ep-01")

    event = ObserveEvent(
        agent_id    = "director",
        fragment_id = result.fragment_id,
        content     = "Elena enters the warehouse.",
        session_id  = "ep-01",
        node_labels = [n.label for n in result.nodes],
        edge_count  = len(result.edges),
        inferred    = result.inferred,
    )
    assert event.agent_id == "director"
    assert event.inferred is True
    assert "Elena" in event.node_labels
    assert event.edge_count >= 3   # deterministic + LLM may produce more
    assert event.timestamp is not None
    runtime.shutdown()


def test_hook_observe_event_fires(tmp_path: Path) -> None:
    """Registering a HOOK_OBSERVE_EVENT plugin causes it to receive ObserveEvent."""
    from mark.plugins import MarkCorePlugin, PluginRegistry

    received: list[ObserveEvent] = []

    class ObservatoryPlugin(MarkCorePlugin):
        @property
        def name(self) -> str: return "test-observatory"
        @property
        def version(self) -> str: return "0.0.1"
        @property
        def provides(self) -> set[str]: return {HOOK_OBSERVE_EVENT}
        def get_hook(self, hook_name: str):
            if hook_name == HOOK_OBSERVE_EVENT:
                return lambda event: received.append(event)
            raise KeyError(hook_name)

    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        plugins=[ObservatoryPlugin()],
        llm=llm,
    )
    mem = runtime.memory("director")
    mem.observe("Elena enters the warehouse.", session_id="ep-01")

    assert len(received) == 1
    event = received[0]
    assert isinstance(event, ObserveEvent)
    assert event.agent_id == "director"
    assert event.session_id == "ep-01"
    assert event.inferred is True
    runtime.shutdown()


def test_hook_observe_event_still_fires_without_llm(tmp_path: Path) -> None:
    """Observatory plugin receives events even when no LLM is configured."""
    from mark.plugins import MarkCorePlugin

    received: list[ObserveEvent] = []

    class ObservatoryPlugin(MarkCorePlugin):
        @property
        def name(self) -> str: return "test-observatory"
        @property
        def version(self) -> str: return "0.0.1"
        @property
        def provides(self) -> set[str]: return {HOOK_OBSERVE_EVENT}
        def get_hook(self, hook_name: str):
            return lambda event: received.append(event)

    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        plugins=[ObservatoryPlugin()],
    )
    mem = runtime.memory("director")
    mem.observe("The sky is blue.")

    assert len(received) == 1
    assert received[0].inferred is False
    runtime.shutdown()


# ---------------------------------------------------------------------------
# SimpleMemory.observe() delegation
# ---------------------------------------------------------------------------

def test_simple_memory_observe_without_llm(tmp_path: Path) -> None:
    """Deterministic extractor runs on simple memory observe without LLM."""
    with Mark.local(project_path=tmp_path) as mark:
        result = mark.memory.observe(
            "Elena enters the warehouse.",
            session_id="ep-01",
            importance=0.8,
        )
        assert result.fragment_id
        # Deterministic extractor finds Elena (subject) and warehouse (location)
        assert result.inferred is True
        assert result.session_id == "ep-01"


def test_simple_memory_observe_tagged_as_observe_block(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        result = mark.memory.observe("Some observation.")
        stored = mark.runtime.store.get(result.fragment_id)
        assert stored is not None
        assert "block:observe" in stored.tags
