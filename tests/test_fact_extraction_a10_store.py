# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Extracted facts in the store: dating (dev M-e), multi-valued values, past values, known subjects.

The extractor proposes candidates; :mod:`mark.intelligence.fact_extraction.apply`
records them. These tests go through ``memory.remember(..., extract_facts=True)``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime


def _memory():
    runtime = MarkRuntime.local(store_path=":memory:", embedder=HashEmbeddingProvider(dim=32))
    return runtime.memory("facts")


def _say(memory, text: str, when: str | None = None) -> None:
    memory.remember(text, extract_facts=True, occurred_at=when)


def _values(facts) -> list[str]:
    return [str(f.value) for f in facts]


def test_backdated_move_beats_an_undated_earlier_statement() -> None:
    memory = _memory()
    _say(memory, "gamma-svc runs on port 9090.")  # undated: stamped "now"
    _say(memory, "gamma-svc moved from port 9090 to port 9191.", "2026-10-01")
    assert str(memory.facts.get("gamma-svc", "port").value) == "9191"
    assert _values(memory.facts.history("gamma-svc", "port")) == ["9090", "9191"]


def test_dated_later_value_is_left_alone() -> None:
    memory = _memory()
    _say(memory, "delta-svc runs on port 7000.", "2026-10-10")
    _say(memory, "delta-svc moved from port 7000 to port 7001.", "2026-10-01")
    # The 2026-10-10 statement is dated: the service is on 7000 again.
    assert str(memory.facts.get("delta-svc", "port").value) == "7000"


def test_previous_value_is_recorded_before_the_change() -> None:
    memory = _memory()
    _say(memory, "I switched from Vim to Neovim.", "2026-09-01")
    assert str(memory.facts.get("user", "editor").value) == "Neovim"
    assert _values(memory.facts.history("user", "editor")) == ["Vim", "Neovim"]


def test_moving_in_a_month_supersedes_an_undated_location() -> None:
    memory = _memory()
    _say(memory, "I live in Lagos.")
    _say(memory, "I moved to Porto in September.")
    assert str(memory.facts.get("user", "location").value) == "Porto"
    assert _values(memory.facts.history("user", "location")) == ["Lagos", "Porto"]


def test_preferences_and_lists_are_multi_valued() -> None:
    memory = _memory()
    _say(memory, "I like jazz.")
    _say(memory, "I like green tea.")
    assert sorted(_values(memory.facts.current("user", "likes"))) == ["green tea", "jazz"]
    _say(memory, "I hate jazz.")
    assert _values(memory.facts.current("user", "likes")) == ["green tea"]
    assert _values(memory.facts.current("user", "hates")) == ["jazz"]
    _say(memory, "I speak Igbo and Twi.")
    assert sorted(_values(memory.facts.current("user", "language"))) == ["Igbo", "Twi"]


def test_no_longer_ends_a_multi_valued_value_but_keeps_history() -> None:
    memory = _memory()
    _say(memory, "I drink coffee.", "2026-01-01")
    _say(memory, "I no longer drink coffee.", "2026-08-01")
    assert memory.facts.current("user", "drinks") == []
    [old] = memory.facts.history("user", "drinks")
    assert old.value == "coffee" and old.valid_to == datetime(2026, 8, 1, tzinfo=timezone.utc)


def test_used_to_is_history_not_current() -> None:
    memory = _memory()
    _say(memory, "I work at Seefar.", "2026-01-01")
    _say(memory, "I used to work at Shell.", "2026-10-01")
    assert str(memory.facts.get("user", "employer").value) == "Seefar"
    history = memory.facts.history("user", "employer")
    assert _values(history) == ["Shell", "Seefar"]
    assert history[0].valid_to is not None and history[0].valid_to <= datetime.now(timezone.utc)


def test_used_to_ends_the_current_value() -> None:
    memory = _memory()
    _say(memory, "I live in Madrid.", "2025-01-01")
    _say(memory, "I used to live in Madrid.", "2026-10-01")
    assert memory.facts.get("user", "location") is None
    [madrid] = memory.facts.history("user", "location")
    assert madrid.valid_to == datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_known_store_subject_splits_a_long_subject() -> None:
    memory = _memory()
    _say(memory, "kestrel uses SQLite.")
    _say(memory, "kestrel default database: k.db")
    assert str(memory.facts.get("kestrel", "default database").value) == "k.db"
    assert memory.facts.current("kestrel default") == []


def test_a_reversed_preference_ends_the_old_one() -> None:
    memory = _memory()
    _say(memory, "I prefer trains over planes.", "2026-01-01")
    _say(memory, "I prefer planes over trains.", "2026-06-01")
    assert _values(memory.facts.current("user", "prefers")) == ["planes over trains"]


def test_a_correction_in_a_list_keeps_the_other_values() -> None:
    memory = _memory()
    _say(memory, "I like jazz.")
    _say(memory, "Correction: I like green tea.")
    assert sorted(_values(memory.facts.current("user", "likes"))) == ["green tea", "jazz"]
