"""Current vs stale in free text (B4): updates outrank what they supersede, "originally" reverses it."""
from __future__ import annotations

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.query_understanding import TemporalIntent
from mark.intelligence.supersession import is_retired, order_by_supersession, replaced_values, supersedes
from mark.memory.runtime import MarkRuntime
from mark.types import MemoryFragment


def _frag(text: str, when: str, **kw) -> MemoryFragment:
    return MemoryFragment(content=text, agent_id="a", metadata={"occurred_at": when}, **kw)


@pytest.fixture()
def mem():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime.memory("me")
    runtime.shutdown()


def test_replaced_values() -> None:
    assert replaced_values("I moved from Lisbon to Porto.") == ["lisbon"]
    assert "berlin" in replaced_values("Correction: my favourite city is actually Lisbon, not Berlin.")
    assert "drink coffee" in replaced_values("I no longer drink coffee; I switched to green tea.")


@pytest.mark.parametrize("new, old, expected", [
    ("I moved from Lisbon to Porto.", "I live in Lisbon.", True),
    ("Correction: my favourite city is actually Lisbon, not Berlin.", "My favourite city is Berlin.", True),
    ("I no longer drink coffee; I switched to green tea.", "I drink two coffees every morning.", True),
    ("CHANGE: tasky storage moved from JSON to SQLite.", "DECISION: tasky storage backend is JSON.", True),
    # Same subject, different attribute: a datastore change is not a port change.
    ("DATABASE: atlas-core switched its datastore to PostgreSQL 15.", "CHANGE: atlas-core moved to port 4444.", False),
    # Different subjects with the same wording.
    ("CHANGE: atlas-gate moved from port 7001 to port 7002.", "CHANGE: atlas-core moved from port 6430 to port 7001.", False),
    # Not worded as an update.
    ("I like Porto.", "I live in Lisbon.", False),
])
def test_supersedes(new: str, old: str, expected: bool) -> None:
    assert supersedes(_frag(new, "2026-06-01"), _frag(old, "2026-01-01")) is expected


def test_an_older_update_never_supersedes_a_newer_statement() -> None:
    assert not supersedes(_frag("I moved from Lisbon to Porto.", "2020-01-01"), _frag("I live in Lisbon.", "2026-01-01"))


def test_a_replaced_value_named_by_the_query_does_not_count() -> None:
    new, old = _frag("I moved from Lisbon to Porto.", "2026-06-01"), _frag("The Lisbon flat rent is 900.", "2026-01-01")
    assert not supersedes(new, old, "How much is the rent in Lisbon?")


def test_ordering_swaps_only_superseding_pairs() -> None:
    old, other, new = (_frag("I live in Lisbon.", "2024-01-10"), _frag("I like tea.", "2025-01-01"),
                       _frag("I moved from Lisbon to Porto.", "2026-06-01"))
    frags, scores, _ = order_by_supersession("Where do I live?", TemporalIntent.NONE, [old, other, new],
                                             [0.9, 0.8, 0.7], [0.9, 0.8, 0.7])
    assert [f.content for f in frags] == [new.content, old.content, other.content]
    assert scores == [0.9, 0.8, 0.7]  # permuted, never invented
    frags, _, _ = order_by_supersession("Where did I live originally?", TemporalIntent.ORIGINAL,
                                        [new, old], [0.9, 0.8], [0.9, 0.8])
    assert frags[0] is old


def test_retired_markers() -> None:
    assert is_retired(MemoryFragment(content="x", agent_id="a", tags=["mark:superseded"]))
    assert is_retired(MemoryFragment(content="x", agent_id="a", metadata={"resolved_by": "abc"}))
    assert not is_retired(MemoryFragment(content="x", agent_id="a"))


def test_moved_beats_lived_now_and_without_shared_words(mem) -> None:
    mem.store_sync("I live in Lisbon.", occurred_at="2024-01-10")
    mem.store_sync("I moved from Lisbon to Porto.", occurred_at="2026-06-01")
    for i in range(20):
        mem.store_sync(f"Filler note {i} about the garden and the city park.")
    assert "Porto" in mem.retrieve_sync("Where do I live now?").top().content
    # "city" and "live" are not in the move: it is found through "Lisbon".
    assert "Porto" in mem.retrieve_sync("Which city do I live in?").top().content


def test_correction_beats_original_and_originally_reverses(mem) -> None:
    mem.store_sync("My favourite city is Berlin.", occurred_at="2026-01-01")
    mem.store_sync("Correction: my favourite city is actually Lisbon, not Berlin.", occurred_at="2026-06-01")
    assert "Correction" in mem.retrieve_sync("favourite city").top().content
    assert mem.retrieve_sync("What was my favourite city originally?").top().content == "My favourite city is Berlin."


def test_no_longer(mem) -> None:
    mem.store_sync("I drink two coffees every morning.", occurred_at="2025-01-01")
    mem.store_sync("I no longer drink coffee; I switched to green tea.", occurred_at="2026-08-01")
    assert "green tea" in mem.retrieve_sync("What do I drink in the morning?").top().content


def test_api_superseded_memory_is_demoted_except_for_original_questions(mem) -> None:
    old = mem.store_sync("TODO: the open-connection bug in the store needs a fix.", tags=["mark:superseded"])
    mem.store_sync("TODO: add a csv export to the store.")
    assert mem.retrieve_sync("open TODO in the store").top().id != old


def test_last_and_most_recently_order_by_time(mem) -> None:
    mem.store_sync("BUG: crash on empty file.", occurred_at="2026-10-01")
    mem.store_sync("BUG: priority not persisted.", occurred_at="2026-10-09")
    mem.store_sync("BUG: config opened in text mode.", occurred_at="2026-10-05")
    assert "priority" in mem.retrieve_sync("What was the last bug?").top().content
    assert "priority" in mem.retrieve_sync("Which bug did we hit most recently?").top().content
    assert "empty file" in mem.retrieve_sync("What was the first bug?").top().content
