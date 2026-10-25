# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Final a10 updates: unnamed changes supersede older statements; same-attribute conflicts newest first."""
from __future__ import annotations

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime

FILLER = [f"Team note {i}: the {topic} review went fine (ref {i})."
          for i, topic in enumerate(["laptop", "budget", "newsletter", "parking", "vendor"] * 12)]


@pytest.fixture()
def mem():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    memory = runtime.memory("updates")
    memory.store_many_sync(FILLER)
    yield memory
    runtime.shutdown()


@pytest.mark.parametrize("old, new, query, want", [
    ("My manager is Funke.", "I now report to Daniel.", "Who is my manager?", "Daniel"),
    ("We use Jira for tickets.", "We switched to Linear for tickets.", "What do we use for tickets?", "Linear"),
    ("The standup is at 9:00.", "Update: standup moved to 10:15.", "What time is standup?", "10:15"),
    ("I live in Kaduna.", "These days I'm based in Abuja.", "Where do I live?", "Abuja"),
])
def test_unnamed_update_supersedes_the_older_statement(mem, old: str, new: str, query: str, want: str) -> None:
    old_id = mem.remember(old, occurred_at="2025-01-01")
    mem.remember(new, occurred_at="2026-07-01")
    result = mem.recall(query)
    assert want in result.fragments[0].content
    assert old_id in result.outdated
    shown = dict(zip((f.id for f in result.fragments), result.relevance))
    assert shown[old_id] < result.relevance[0]


def test_same_attribute_conflict_is_newest_first_and_flagged(mem) -> None:
    old = mem.remember("The wifi SSID is QV-Guest.", occurred_at="2025-01-01")
    new = mem.remember("The wifi SSID is QV-Visitors.", occurred_at="2026-03-01")
    result = mem.recall("What is the wifi SSID?")
    assert result.fragments[0].id == new
    assert old in result.conflicts.get(new, [])


def test_an_update_found_far_down_does_not_lift_unrelated_results(mem) -> None:
    mem.remember("I live in Kaduna.", occurred_at="2025-02-01")
    mem.remember("These days I'm based in Abuja.", occurred_at="2026-09-01")
    result = mem.recall("Where do I live?")
    for fragment, score in zip(result.fragments, result.relevance):
        if "Team note" in fragment.content:
            assert score < 0.5
