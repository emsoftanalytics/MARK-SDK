# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""RetrievalResult.facts holds the facts the query asks about, not every fact.

A query naming an attribute keeps only that attribute; a question whose head
noun ("Which region ...") is not among the subject's attributes returns no
facts for it; an open question ("Tell me about atlas-core") keeps them all.
"""
from __future__ import annotations

import pytest

from mark.cli import main
from mark.memory.runtime import MarkRuntime


@pytest.fixture()
def memory():
    runtime = MarkRuntime.local()
    memory = runtime.memory("ops")
    memory.facts.set("atlas-core", "port", "4444")
    memory.facts.set("atlas-core", "owner", "dana")
    memory.facts.set("atlas-core", "team", "platform")
    memory.store_sync("atlas-core moved to region eu-west.", occurred_at="2026-03-01")
    memory.store_sync("atlas-core moved to region us-east.", occurred_at="2026-04-02")
    yield memory
    runtime.shutdown()


def _attributes(memory, query: str) -> list[str]:
    return sorted(fact.attribute for fact in memory.retrieve_sync(query).facts)


@pytest.mark.parametrize("query, expected", [
    ("Which region is atlas-core in now?", []),          # no region fact: none of the others
    ("What regions does atlas-core use?", []),
    ("What is the region of atlas-core?", []),
    ("Which port does atlas-core use now?", ["port"]),
    ("What ports does atlas-core listen on?", ["port"]),  # stemmed attribute name
    ("Who owns atlas-core?", ["owner"]),
    ("Which team runs atlas-core?", ["team"]),
    ("Tell me about atlas-core", ["owner", "port", "team"]),
    ("What is atlas-core?", ["owner", "port", "team"]),
    ("What happened to atlas-core?", ["owner", "port", "team"]),
])
def test_facts_follow_the_attribute_asked(memory, query: str, expected: list[str]) -> None:
    assert _attributes(memory, query) == expected


def test_region_question_still_answers_from_memories(memory) -> None:
    result = memory.retrieve_sync("Which region does atlas-core use now?")
    assert result.facts == []
    assert "us-east" in result.top().content


def test_head_matching_a_fact_value_keeps_that_fact(memory) -> None:
    memory.facts.set("atlas-core", "lead", "platform team")
    assert "lead" in _attributes(memory, "Which team leads atlas-core?")


def test_retrieve_many_applies_the_same_selection(memory) -> None:
    results = memory.retrieve_many_sync(["Which region is atlas-core in now?", "Tell me about atlas-core"])
    assert [len(r.facts) for r in results] == [0, 3]


def test_cli_recall_shows_only_relevant_facts(tmp_path, capsys) -> None:
    db = str(tmp_path / "memory.db")
    for args in (["fact", "set", "atlas-core", "port", "4444"],
                 ["fact", "set", "atlas-core", "owner", "dana"],
                 ["store", "atlas-core moved to region us-east."]):
        assert main(["--db", db, "--no-server", *args]) == 0
    capsys.readouterr()
    assert main(["--db", db, "--no-server", "recall", "Which region is atlas-core in now?"]) == 0
    out = capsys.readouterr().out
    assert " F. " not in out and "us-east" in out
    assert main(["--db", db, "--no-server", "recall", "Which port does atlas-core use?"]) == 0
    out = capsys.readouterr().out
    assert "port = 4444" in out and "owner" not in out


@pytest.fixture()
def people():
    runtime = MarkRuntime.local()
    memory = runtime.memory("ops")
    memory.facts.set("billing", "port", "8080")
    memory.facts.set("billing", "region", "eu-west-1")
    memory.facts.set("HAL-7", "status", "open")
    memory.facts.set("HAL-7", "assignee", "Mei Lin")
    memory.facts.set("search", "manager", "Ana Costa")
    memory.facts.set("search", "lead", "Bo Kim")
    memory.facts.set("search", "port", "9200")
    memory.facts.set("Mei Lin", "team", "Search")
    memory.facts.set("Mei Lin", "location", "Berlin")
    yield memory
    runtime.shutdown()


@pytest.mark.parametrize("query, expected", [
    ("Who owns billing?", []),                        # no owner fact: not the port or region
    ("Who currently owns billing?", []),
    ("Who is assigned to HAL-7?", ["assignee"]),      # assigned -> assignee
    ("Who manages search?", ["manager"]),             # manages -> manager
    ("Who leads search now?", ["lead"]),
    ("Who is Mei Lin?", ["location", "team"]),        # open question about a person
    ("Tell me about billing", ["port", "region"]),
])
def test_who_questions_select_the_agent_noun(people, query: str, expected: list[str]) -> None:
    assert _attributes(people, query) == expected
