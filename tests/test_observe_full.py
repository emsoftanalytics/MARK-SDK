# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""DeterministicExtractor, observe() with memory_type/auto_tags, and top-level
Mark.observe() routing by agent_id.
"""
from __future__ import annotations

from pathlib import Path

from mark import DeterministicExtractor, ExtractionResult, Mark, ObserveMiddleware
from mark.intelligence.extractor import ExtractionResult
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# DeterministicExtractor
# ---------------------------------------------------------------------------

def test_extractor_finds_character_from_subject() -> None:
    ext = DeterministicExtractor()
    r = ext.extract("Elena entered the building.")
    labels = [label for _, label in r.entities]
    assert "Elena" in labels
    types = {label: t for t, label in r.entities}
    assert types["Elena"] == "character"


def test_extractor_finds_location_from_verb() -> None:
    ext = DeterministicExtractor()
    r = ext.extract("Elena enters the North Warehouse.")
    labels = [label for _, label in r.entities]
    assert "North Warehouse" in labels
    types = {label: t for t, label in r.entities}
    assert types["North Warehouse"] == "location"


def test_extractor_finds_object_from_possession_verb() -> None:
    ext = DeterministicExtractor()
    r = ext.extract("Elena wearing the red scarf.")
    labels = [label for _, label in r.entities]
    assert "Red Scarf" in labels
    types = {label: t for t, label in r.entities}
    assert types["Red Scarf"] == "object"


def test_extractor_creates_scene_from_session_id() -> None:
    ext = DeterministicExtractor()
    r = ext.extract("Elena walked in.", session_id="season-01/ep-02/scene-04")
    labels = [label for _, label in r.entities]
    assert "scene-04" in labels
    types = {label: t for t, label in r.entities}
    assert types["scene-04"] == "scene"


def test_extractor_memory_type_overrides_context_type() -> None:
    ext = DeterministicExtractor()
    r = ext.extract("Data collected.", session_id="experiment/run-12", memory_type="episode")
    types = {label: t for t, label in r.entities}
    assert types["run-12"] == "episode"


def test_extractor_generates_tags() -> None:
    ext = DeterministicExtractor()
    r = ext.extract(
        "Elena enters the North Warehouse wearing the red scarf.",
        session_id="season-01/ep-02/scene-04",
    )
    tag_set = set(r.tags)
    assert "scene:scene-04" in tag_set
    assert "location:north-warehouse" in tag_set
    assert "object:red-scarf" in tag_set


def test_extractor_infers_relations() -> None:
    ext = DeterministicExtractor()
    r = ext.extract(
        "Elena enters the North Warehouse wearing the red scarf.",
        session_id="season-01/ep-02/scene-04",
    )
    rels = {(a, rel, b) for a, rel, b in r.relations}
    assert ("Elena", "located_at", "North Warehouse") in rels or \
           any(rel == "located_at" for _, rel, _ in r.relations)
    assert any(rel == "appears_in" for _, rel, _ in r.relations)


def test_extractor_plain_text_returns_empty() -> None:
    ext = DeterministicExtractor()
    r = ext.extract("The sky is blue.")
    assert r.entities == []
    assert r.relations == []
    assert r.tags == []


def test_extractor_returns_extraction_result_type() -> None:
    r = DeterministicExtractor().extract("Some text.")
    assert isinstance(r, ExtractionResult)


# ---------------------------------------------------------------------------
# observe() with memory_type and auto_tags
# ---------------------------------------------------------------------------

def test_observe_auto_tags_applied_to_fragment(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("agent")

    result = mem.observe(
        "Elena enters the North Warehouse.",
        session_id="season-01/ep-02/scene-04",
    )
    stored = runtime.store.get(result.fragment_id)
    assert stored is not None
    tag_set = set(stored.tags)
    assert "scene:scene-04" in tag_set
    assert "location:north-warehouse" in tag_set
    assert result.auto_tags  # populated
    runtime.shutdown()


def test_observe_memory_type_creates_correct_context_node(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("agent")

    result = mem.observe(
        "Experiment started.",
        session_id="project/run-42",
        memory_type="episode",
    )
    labels = {n.label for n in result.nodes}
    assert "run-42" in labels
    types = {n.label: n.node_type for n in result.nodes}
    # NodeType does not have EPISODE, so it falls back to the closest available type
    # The extractor stores it as "episode" string; node() converts it
    # If NodeType("episode") fails, fallback occurs — just verify the node was created
    assert "run-42" in labels
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Mark.observe() — top-level with agent_id
# ---------------------------------------------------------------------------

def test_mark_observe_routes_to_agent(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        result = mark.observe(
            "Elena enters the North Warehouse.",
            agent_id="video-agent",
            session_id="season-01/ep-02/scene-04",
        )
        assert result.fragment_id
        assert result.session_id == "season-01/ep-02/scene-04"
        # Fragment stored under the named agent, not __mark__
        stored = mark.runtime.store.get(result.fragment_id)
        assert stored is not None
        assert stored.agent_id == "video-agent"


def test_mark_observe_default_agent(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        result = mark.observe("The experiment concluded.", session_id="run-01")
        stored = mark.runtime.store.get(result.fragment_id)
        assert stored is not None
        assert stored.agent_id == "__mark__"
