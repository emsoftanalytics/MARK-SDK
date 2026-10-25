# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Final a10 score display: no 1.000 for inflection-only matches, stable ties, outdated below current."""
from __future__ import annotations

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.query_understanding import _content_terms, _coverage
from mark.memory.runtime import MarkRuntime


@pytest.fixture()
def mem():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime.memory("display")
    runtime.shutdown()


def test_inflection_only_match_is_weaker_than_the_exact_word() -> None:
    terms = _content_terms("Where do I live now?")
    assert _coverage(terms, "I live in Lisbon.") == 1.0
    assert 0.0 < _coverage(terms, "In the living room the samples were compared.") < 1.0
    # Plurals and possessives are the same word.
    assert _coverage(_content_terms("dietary restrictions"), "dietary restriction: no meat") == 1.0
    assert _coverage(_content_terms("Ana's husband"), "Ana is married; her husband is Tom") == 1.0


def test_unrelated_inflection_never_shows_one(mem) -> None:
    mem.remember("I live in Lisbon.", occurred_at="2024-01-10")
    mem.remember("I moved from Lisbon to Porto.", occurred_at="2026-06-01")
    living = mem.remember("Living room: the living room walls get two coats of paint.")
    result = mem.recall("Where do I live now?")
    shown = dict(zip((f.id for f in result.fragments), result.relevance))
    assert result.fragments[0].content.startswith("I moved")
    assert shown.get(living, 0.0) < 1.0


def test_equal_scores_are_ordered_newest_first_and_stable(mem) -> None:
    old = mem.remember("The parking pass is in the blue folder.", occurred_at="2025-01-01")
    new = mem.remember("The parking pass is in the red folder.", occurred_at="2026-01-01")
    first = mem.recall("parking pass folder")
    second = mem.recall("parking pass folder")
    ids = [f.id for f in first.fragments]
    assert ids == [f.id for f in second.fragments]
    assert first.relevance[0] >= first.relevance[1]
    assert ids.index(new) < ids.index(old)


def test_displayed_scores_never_increase(mem) -> None:
    for i in range(8):
        mem.remember(f"Note {i}: the backup job runs at 0{i}:00 and writes to bucket b{i}.")
    mem.remember("The backup job was moved to 05:30.")
    result = mem.recall("When does the backup job run now?")
    assert all(a >= b for a, b in zip(result.relevance, result.relevance[1:]))


# ── 6. Outdated free text sits below the current statement ───────────────────

def test_retracted_statement_is_outdated_and_not_tied(mem) -> None:
    old = mem.remember("I drink two coffees every morning.", occurred_at="2025-01-01")
    new = mem.remember("I no longer drink coffee; I switched to green tea.", occurred_at="2026-08-01")
    result = mem.recall("Do I drink coffee?")
    shown = dict(zip((f.id for f in result.fragments), result.relevance))
    assert result.fragments[0].id == new
    assert old in result.outdated
    assert shown[old] < shown[new]


def test_memory_stating_an_earlier_fact_value_is_outdated(mem) -> None:
    old = mem.remember("spendlog default database path is ~/.spendlog.db.", occurred_at="2026-01-01")
    current = mem.remember("spendlog stores its database at ~/.local/share/spendlog/spendlog.db.",
                           occurred_at="2026-09-01")
    mem.facts.set("spendlog", "db_path", "~/.spendlog.db", valid_from="2026-01-01")
    mem.facts.set("spendlog", "db_path", "~/.local/share/spendlog/spendlog.db", valid_from="2026-09-01")
    result = mem.recall("What is the spendlog database path?")
    shown = dict(zip((f.id for f in result.fragments), result.relevance))
    assert old in result.outdated
    assert current not in result.outdated
    assert shown[old] < shown[current]


def test_a_value_still_in_the_current_list_is_not_outdated(mem) -> None:
    fix = mem.remember("FIX: spendlog export reports 'unknown exporter; available: csv' instead of a traceback.")
    mem.facts.set("spendlog", "exporters", "csv", valid_from="2026-01-01")
    mem.facts.set("spendlog", "exporters", "csv, json", valid_from="2026-09-01")
    result = mem.recall("What happens on spendlog export with an unknown exporter?")
    assert fix not in result.outdated
