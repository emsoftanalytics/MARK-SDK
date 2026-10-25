"""Facts in recall (C6) and honest gap reports (C5)."""
from __future__ import annotations

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.types import GapSeverity


@pytest.fixture()
def runtime():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime
    runtime.shutdown()


@pytest.fixture()
def mem(runtime):
    return runtime.memory("dev")


def _attrs(result) -> list[str]:
    return sorted(f.attribute for f in result.facts)


def test_open_question_is_capped_unless_everything_is_asked(mem) -> None:
    for attr, value in [("port", "4444"), ("owner", "dana"), ("team", "platform"), ("region", "eu"), ("tier", "1")]:
        mem.facts.set("atlas-core", attr, value)
    mem.store_sync("atlas-core is the billing gateway.")
    assert len(mem.retrieve_sync("What is atlas-core?").facts) == 3
    assert len(mem.retrieve_sync("Tell me about atlas-core").facts) == 5
    assert _attrs(mem.retrieve_sync("Which port does atlas-core use?")) == ["port"]


def test_first_person_questions_use_the_user_subject(runtime, mem) -> None:
    mem.facts.set("user", "lives_in", "Porto")
    mem.facts.set("user", "employer", "Acme")
    mem.facts.set("user", "birthday", "14 March")
    mem.store_sync("I moved from Lisbon to Porto.")
    assert _attrs(mem.retrieve_sync("Where do I live now?")) == ["lives_in"]
    assert mem.retrieve_sync("What is the weather like?").facts == []
    # An "I" question naming no attribute does not dump the user's profile.
    assert mem.retrieve_sync("Do I like jazz?").facts == []
    runtime.pipeline.configure_user_subject("me")
    mem.facts.set("me", "lives_in", "Faro")
    assert [f.value for f in mem.retrieve_sync("Where do I live now?").facts] == ["Faro"]


def test_facts_attach_through_matched_memories(mem) -> None:
    mem.facts.set("tasky", "test_command", "pytest -x -q tests")
    mem.facts.set("tasky", "version", "0.2.0")
    mem.facts.set("tasky", "storage", "sqlite")
    mem.store_sync("tasky test command is now pytest -x -q tests (stop at first failure).")
    result = mem.retrieve_sync("What is the test command now?")
    assert _attrs(result) == ["test_command"]


def test_facts_by_attribute_name_without_subject(mem) -> None:
    mem.facts.set("tasky", "version", "0.2.0")
    mem.store_sync("Release notes drafted for the next sprint.")
    assert _attrs(mem.retrieve_sync("What is the current version?")) == ["version"]


def test_pronoun_subjects_from_old_extractors_are_ignored(mem) -> None:
    mem.facts.set("My", "manager", "Priya")
    mem.store_sync("I like tea.")
    assert mem.retrieve_sync("What is my bank PIN hint?").facts == []


@pytest.mark.parametrize("query, missing", [
    ("What is TaskQueue used for?", ["taskqueue"]),
    ("How is Redis configured?", ["redis"]),
    ("Where is parse_manifest defined?", ["parse manifest"]),
])
def test_absent_identifiers_and_names_are_gaps(mem, query: str, missing: list[str]) -> None:
    mem.store_sync("CONFIG: config is loaded in todo/config.py by load_config(path).")
    mem.store_sync("DECISION: plugin modules are typed with a typing.Protocol.")
    result = mem.retrieve_sync(query)
    assert result.missing_anchors == missing
    assert result.gap_report.severity == GapSeverity.HIGH


def test_exact_identifier_match_is_not_a_gap(mem) -> None:
    mem.store_sync("BUG: no such table. FIX: TodoStore.__init__ calls self._migrate() which runs the schema.")
    mem.store_sync("CONFIG: config is loaded in todo/config.py by load_config(path).")
    for query in ("What does TodoStore._migrate do?", "Where is load_config defined?"):
        result = mem.retrieve_sync(query)
        assert result.missing_anchors == []
        assert result.gap_report.severity == GapSeverity.NONE, query


def test_displayed_score_is_the_gap_score(mem) -> None:
    mem.store_sync("The deploy runbook lives in the ops wiki.")
    result = mem.retrieve_sync("Who approved the quarterly offsite budget?")
    shown = float(result.as_context(format="numbered").split("score=")[1].split(")")[0].split(",")[0])
    assert shown == pytest.approx(max(result.relevance), abs=5e-4)
    if result.gap_report.reason.startswith("Top score"):
        assert f"{shown:.3f}" in result.gap_report.reason


@pytest.mark.parametrize("query", ["", "   ", "?!", "..."])
def test_blank_queries_return_nothing(mem, query: str) -> None:
    mem.store_sync("Anything at all.")
    mem.facts.set("user", "name", "Ada")
    result = mem.retrieve_sync(query)
    assert result.fragments == [] and result.facts == []
    assert result.gap_report.severity == GapSeverity.NONE
