# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Retractions and corrections from free text reach the fact store.

"no longer" withdraws the current value only (never a newer one);
"Correction: ... actually ..." replaces a wrong value so it leaves history;
first-person facts land on the user subject; the write path reports how
many facts it extracted.
"""
from __future__ import annotations

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.fact_extraction import PatternFactExtractor
from mark.memory.runtime import MarkRuntime


def _memory(**kw):
    runtime = MarkRuntime.local(store_path=":memory:", embedder=HashEmbeddingProvider(dim=32), **kw)
    return runtime.memory("w3")


def _values(memory, subject: str, attribute: str) -> list[str]:
    return [f.value for f in memory.facts.current(subject, attribute)]


def test_no_longer_retracts_the_current_value() -> None:
    memory = _memory()
    memory.store_sync("[2026-01-01] atlas-core runs on port 8080.", extract_facts=True)
    assert _values(memory, "atlas-core", "port") == ["8080"]
    memory.store_sync("[2026-02-01] atlas-core no longer runs on port 8080.", extract_facts=True)
    assert _values(memory, "atlas-core", "port") == []
    # Re-ingesting the denial is a no-op; a newer assertion is never undone by an older denial.
    memory.store_sync("[2026-03-01] atlas-core runs on port 8080.", extract_facts=True)
    memory.store_sync("[2026-02-01] atlas-core no longer runs on port 8080.", extract_facts=True)
    assert _values(memory, "atlas-core", "port") == ["8080"]


def test_denial_of_another_value_changes_nothing() -> None:
    memory = _memory()
    memory.store_sync("atlas-core runs on port 8080.", extract_facts=True)
    memory.store_sync("atlas-core doesn't run on port 9090 anymore.", extract_facts=True)
    assert _values(memory, "atlas-core", "port") == ["8080"]


def test_correction_replaces_the_wrong_value() -> None:
    memory = _memory()
    memory.store_sync("[2026-01-01] My favourite city is Berlin.", extract_facts=True)
    memory.store_sync("[2026-02-01] Correction: my favourite city is actually Lisbon.", extract_facts=True)
    assert _values(memory, "user", "favourite city") == ["Lisbon"]
    assert [f.value for f in memory.facts.history("user", "favourite city")] == ["Lisbon"]
    (fact,) = memory.facts.current("user", "favourite city")
    assert fact.metadata.get("correction") is True


def test_first_person_history_and_runtime_user_subject() -> None:
    memory = _memory(fact_extractor=PatternFactExtractor(user_subject="me"))
    memory.store_sync("[2025-05-01] I live in Lisbon.")
    memory.store_sync("[2026-04-01] I moved to Porto.")
    assert _values(memory, "me", "location") == ["Porto"]
    assert [f.value for f in memory.facts.history("me", "location")] == ["Lisbon", "Porto"]


def test_label_kinds_are_kept_on_facts_and_count_is_reported() -> None:
    memory = _memory()
    fragment_id = memory.store_sync("DECISION: tasky storage backend is SQLite.", extract_facts=False)
    assert memory._extract_stored([fragment_id], True) == 1
    (fact,) = memory.facts.current("tasky", "storage backend")
    assert fact.metadata.get("tags") == ["decision"]
    assert memory.last_extracted == 1
