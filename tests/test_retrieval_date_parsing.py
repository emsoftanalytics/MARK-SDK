"""Natural and relative date parsing for queries and memories (B5)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from mark.intelligence.date_parsing import find_dates, parse_when

REF = date(2026, 10, 15)  # a Thursday


def _range(text: str, **kw) -> tuple[date, date]:
    spans = find_dates(text, REF, **kw)
    assert spans, f"no date in {text!r}"
    return spans[0].start, spans[0].end


@pytest.mark.parametrize("text, start, end", [
    ("The board review is on the 3rd of May 2026.", date(2026, 5, 3), date(2026, 5, 3)),
    ("due May 3", date(2026, 5, 3), date(2026, 5, 3)),
    ("on May 3rd, 2025", date(2025, 5, 3), date(2025, 5, 3)),
    ("3 May", date(2026, 5, 3), date(2026, 5, 3)),
    ("2026-05-03", date(2026, 5, 3), date(2026, 5, 3)),
    ("Invoice due 05/03/2026 (UK format)", date(2026, 3, 5), date(2026, 3, 5)),
    ("12/25/2026", date(2026, 12, 25), date(2026, 12, 25)),   # second number > 12: month first
    ("25/12/2026", date(2026, 12, 25), date(2026, 12, 25)),
    ("yesterday", date(2026, 10, 14), date(2026, 10, 14)),
    ("tomorrow", date(2026, 10, 16), date(2026, 10, 16)),
    ("What did we talk about last Tuesday?", date(2026, 10, 13), date(2026, 10, 13)),
    ("next Tuesday", date(2026, 10, 20), date(2026, 10, 20)),
    ("next week", date(2026, 10, 19), date(2026, 10, 25)),
    ("last week", date(2026, 10, 5), date(2026, 10, 11)),
    ("last month", date(2026, 9, 1), date(2026, 9, 30)),
    ("this year", date(2026, 1, 1), date(2026, 12, 31)),
    ("Revenue target for Q3 is 4.2M", date(2026, 7, 1), date(2026, 9, 30)),
    ("Q3 2025", date(2025, 7, 1), date(2025, 9, 30)),
    ("the third-quarter sales goal", date(2026, 7, 1), date(2026, 9, 30)),
    ("3 days ago", date(2026, 10, 12), date(2026, 10, 12)),
    ("in the past 7 days", date(2026, 10, 8), date(2026, 10, 15)),
    ("What's due in March?", date(2026, 3, 1), date(2026, 3, 31)),
    ("May 2026", date(2026, 5, 1), date(2026, 5, 31)),
    ("in 2025", date(2025, 1, 1), date(2025, 12, 31)),
])
def test_expressions(text: str, start: date, end: date) -> None:
    assert _range(text) == (start, end)


def test_us_locale_switch() -> None:
    assert _range("05/03/2026", day_first=False) == (date(2026, 5, 3), date(2026, 5, 3))


@pytest.mark.parametrize("text", [
    "May I ask a question?", "Release v1.2.10 fixed it", "port 8080 on 10.0.3.17", "the march of time",
    "version 2.4.1", "ratio 3/4", "", "IP 192.168.40.12",
])
def test_non_dates(text: str) -> None:
    assert find_dates(text, REF) == []


def test_invalid_numbers_are_not_dates() -> None:
    assert find_dates("31/02/2026", REF) == []


def test_relative_dates_in_content_use_the_memorys_own_date() -> None:
    spans = find_dates("Ship the beta next Tuesday, said Ana", date(2026, 10, 8))
    assert (spans[0].start, spans[0].end) == (date(2026, 10, 13), date(2026, 10, 13))


def test_parse_when_bounds() -> None:
    assert parse_when("2026-10-05") == datetime(2026, 10, 5, tzinfo=timezone.utc)
    assert parse_when("2026-10-05T10:30:00") == datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc)
    assert parse_when("last week", REF) == datetime(2026, 10, 5, tzinfo=timezone.utc)
    end = parse_when("last week", REF, end=True)
    assert end.date() == date(2026, 10, 11) and end.hour == 23
    assert parse_when(date(2026, 1, 2)) == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert parse_when(None) is None
    with pytest.raises(ValueError):
        parse_when("whenever")
