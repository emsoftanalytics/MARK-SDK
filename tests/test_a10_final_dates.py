# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Final a10 date questions: named months on every embedder, empty windows, "yesterday", "last December"."""
from __future__ import annotations

from datetime import date

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.date_parsing import find_dates
from mark.memory.runtime import MarkRuntime


@pytest.mark.parametrize("text, start, end", [
    ("last December", date(2025, 12, 1), date(2025, 12, 31)),
    ("last October", date(2025, 10, 1), date(2025, 10, 31)),
    ("last September", date(2026, 9, 1), date(2026, 9, 30)),
    ("next March", date(2027, 3, 1), date(2027, 3, 31)),
    ("next November", date(2026, 11, 1), date(2026, 11, 30)),
    ("in December", date(2026, 12, 1), date(2026, 12, 31)),
])
def test_last_and_next_month_are_relative_to_today(text: str, start: date, end: date) -> None:
    span = find_dates(text, reference=date(2026, 10, 18))[0]
    assert (span.start, span.end) == (start, end)


def _dated(mode: str):
    from mark.intelligence.retrieval_policy import FusionConfig

    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    runtime.pipeline.configure_fusion(FusionConfig(mode=mode))
    mem = runtime.memory("dates")
    year = date.today().year
    mem.store_many_sync([f"What did I do in the afternoon? I reviewed the vendor contract yesterday (ref {i})."
                         for i in range(150)])
    ids = {
        "book": mem.remember("Booked flights for the Accra conference.", occurred_at=f"{year}-08-14T10:00:00"),
        "sign": mem.remember("Signed the lease for the Ikoyi flat.",
                             occurred_at=(date.today().fromordinal(date.today().toordinal() - 1)).isoformat()),
        "drill": mem.remember("Ran the quarterly fire drill.", occurred_at=f"{year - 1}-09-03T10:00:00"),
    }
    return runtime, mem, ids


@pytest.mark.parametrize("mode", ["rrf", "score"])
def test_named_month_finds_its_memory_on_every_fusion(mode: str) -> None:
    runtime, mem, ids = _dated(mode)
    try:
        result = mem.recall("What did I book in August?")
        assert result.fragments and result.fragments[0].id == ids["book"]
        result = mem.recall("What did I sign yesterday?")
        assert result.fragments and result.fragments[0].id == ids["sign"]
    finally:
        runtime.shutdown()


def test_named_window_without_memories_says_so() -> None:
    runtime, mem, ids = _dated("score")
    try:
        result = mem.recall("What did I sign in February?")
        assert result.gap_report.severity.value in ("high", "critical")
        assert "February" in (result.gap_report.reason or "")
    finally:
        runtime.shutdown()
