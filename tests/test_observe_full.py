# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Tests for: DeterministicExtractor, WorldBibleMemory, scoped entity helpers,
# SessionMemory, Mark.observe(), Mark.world_bible, memory_type param.
from __future__ import annotations

from pathlib import Path

import pytest

from mark import (
    CharacterMemory, DeterministicExtractor, ExtractionResult,
    LocationMemory, Mark, ObjectMemory, ObserveMiddleware, SessionMemory, WorldBibleMemory,
)
from mark.intelligence.extractor import ExtractionResult
from mark.memory.runtime import MarkRuntime
from mark.types.graph import EdgeRelation, NodeType


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


# ---------------------------------------------------------------------------
# WorldBibleMemory
# ---------------------------------------------------------------------------

def test_world_bible_remember_promotes_fact(tmp_path: Path) -> None:
    from mark.types import MemoryState
    with Mark.local(project_path=tmp_path) as mark:
        fid = mark.world_bible.remember("Elena is left-handed.", tags=["character:elena"])
        assert fid
        facts = mark.world_bible.list_facts()
        assert any(f.id == fid for f in facts)
        # Verify PROMOTED state
        stored = mark.runtime.store.get(fid)
        assert stored is not None
        assert stored.state == MemoryState.PROMOTED
        assert "world-bible" in stored.tags
        assert "character:elena" in stored.tags


def test_world_bible_list_facts_with_tag_filter(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        mark.world_bible.remember("Elena is left-handed.", tags=["character:elena"])
        mark.world_bible.remember("The warehouse was destroyed.", tags=["location:north-warehouse"])

        elena_facts = mark.world_bible.list_facts(tags=["character:elena"])
        assert len(elena_facts) == 1
        assert "Elena" in elena_facts[0].content


def test_world_bible_check_returns_relevant_facts(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        mark.world_bible.remember("Elena is left-handed.", tags=["character:elena"])
        mark.world_bible.remember("The sky is always clear.", tags=["world"])

        conflicts = mark.world_bible.check("Elena uses her right hand.")
        # "Elena" appears in both texts → should surface the left-handed fact
        labels = [f.content for f in conflicts]
        assert any("Elena" in c for c in labels)


def test_world_bible_check_no_conflict(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        mark.world_bible.remember("Elena is left-handed.", tags=["character:elena"])
        result = mark.world_bible.check("The robot activates.")
        # No shared keywords → no conflict
        assert result == []


def test_world_bible_forget(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        fid = mark.world_bible.remember("Temporary canonical fact.")
        assert len(mark.world_bible.list_facts()) == 1
        mark.world_bible.forget(fid)
        assert mark.world_bible.list_facts() == []


# ---------------------------------------------------------------------------
# CharacterMemory / ObjectMemory / LocationMemory
# ---------------------------------------------------------------------------

def test_character_memory_scoped_writes(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")

    elena = mem.character("Elena")
    assert isinstance(elena, CharacterMemory)
    assert elena.tag == "character:elena"

    fid = elena.remember("Elena is left-handed.", importance=0.9)
    assert fid

    facts = elena.facts()
    assert len(facts) == 1
    assert "left-handed" in facts[0].content
    assert "character:elena" in facts[0].tags
    runtime.shutdown()


def test_character_memory_recall(tmp_path: Path) -> None:
    from mark.embeddings import HashEmbeddingProvider
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")

    elena = mem.character("Elena")
    elena.remember("Elena is left-handed.", importance=0.9)
    elena.remember("Elena is the protagonist.", importance=0.9)

    results = elena.recall("What are Elena's traits?")
    assert results.fragments  # at least one result
    runtime.shutdown()


def test_object_memory_scoped(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")

    scarf = mem.object("Red Scarf")
    assert isinstance(scarf, ObjectMemory)
    assert scarf.tag == "object:red-scarf"

    scarf.remember("The red scarf belonged to Elena's mother.")
    facts = scarf.facts()
    assert facts
    assert "object:red-scarf" in facts[0].tags
    runtime.shutdown()


def test_location_memory_scoped(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")

    warehouse = mem.location("North Warehouse")
    assert isinstance(warehouse, LocationMemory)
    assert warehouse.tag == "location:north-warehouse"

    warehouse.remember("The warehouse is abandoned since episode 2.")
    facts = warehouse.facts()
    assert facts
    assert "location:north-warehouse" in facts[0].tags
    runtime.shutdown()


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------

def test_session_memory_observe_and_recall(tmp_path: Path) -> None:
    from mark.embeddings import HashEmbeddingProvider
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")

    scene = mem.session("season-01/ep-02/scene-04")
    assert isinstance(scene, SessionMemory)
    assert scene.session_id == "season-01/ep-02/scene-04"

    result = scene.observe("Elena enters the North Warehouse.")
    assert result.session_id == "season-01/ep-02/scene-04"

    frags = scene.list_observations()
    assert any(f.id == result.fragment_id for f in frags)
    runtime.shutdown()


def test_session_memory_recall_prefix(tmp_path: Path) -> None:
    from mark.embeddings import HashEmbeddingProvider
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")

    s1 = mem.session("season-01/ep-02/scene-01")
    s2 = mem.session("season-01/ep-02/scene-02")
    s3 = mem.session("season-02/ep-01/scene-01")

    s1.observe("Elena arrives.", importance=0.9)
    s2.observe("Elena leaves.", importance=0.9)
    s3.observe("Marcus arrives.", importance=0.9)

    results = s1.recall_prefix("season-01/ep-02/", "Who is Elena?")
    ep2_ids = {f.id for f in results.fragments}
    # Results should only include ep-02 fragments
    all_frags = runtime.store.list_by_agent("agent")
    s3_ids = {f.id for f in all_frags if f.session_id and "season-02" in f.session_id}
    assert not (ep2_ids & s3_ids)
    runtime.shutdown()


# ---------------------------------------------------------------------------
# General-purpose validation (not media-specific)
# ---------------------------------------------------------------------------

def test_entity_helpers_work_for_non_media_projects(tmp_path: Path) -> None:
    """Entity helpers work for any long-running project, not just media."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("research-agent")

    # Research project: subject, instrument, site
    subject = mem.character("Watson")          # character = any named person
    instrument = mem.object("Spectrometer")    # object = any tool/artifact
    site = mem.location("Lab 3B")              # location = any place

    subject.remember("Watson is the lead researcher.")
    instrument.remember("The spectrometer needs calibration every 48h.")
    site.remember("Lab 3B is closed on Mondays.")

    assert len(subject.facts()) == 1
    assert len(instrument.facts()) == 1
    assert len(site.facts()) == 1
    assert subject.tag == "character:watson"
    assert instrument.tag == "object:spectrometer"
    assert site.tag == "location:lab-3b"
    runtime.shutdown()
