# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
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
