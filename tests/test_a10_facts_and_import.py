# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Facts surface (get_all, scheduled, natural dates) and a forgiving, idempotent import_jsonl."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mark.errors import InvalidInputError
from mark.runtime import Mark


@pytest.fixture()
def mark(tmp_path: Path):
    with Mark.local(tmp_path / "a") as m:
        yield m


def test_get_all_returns_every_current_value(mark: Mark) -> None:
    facts = mark.memory.facts
    facts.add("Dr. Bello", "phone", "+234 1")
    facts.add("Dr. Bello", "phone", "+234 2")
    assert sorted(f.value for f in facts.get_all("Dr. Bello", "phone")) == ["+234 1", "+234 2"]
    facts.set("atlas", "port", 9090)
    assert [f.value for f in facts.get_all("atlas", "port")] == [9090]
    assert facts.get_all("nobody", "port") == []


def test_future_values_say_scheduled(mark: Mark) -> None:
    fact = mark.memory.facts.set("atlas", "port", 9191, valid_from="2099-01-01")
    assert fact.is_scheduled and not fact.is_current
    assert fact.as_text() == "atlas port = 9191 (scheduled from 2099-01-01)"
    assert "since" in mark.memory.facts.set("atlas", "owner", "ana").as_text()


def test_natural_dates_for_valid_from_as_of_and_occurred_at(mark: Mark) -> None:
    facts = mark.memory.facts
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    assert facts.set("atlas", "port", 1, valid_from="yesterday").valid_from.date() == yesterday
    facts.set("atlas", "port", 2, valid_from="3 May 2026")
    assert facts.as_of("atlas", "port", "4 May 2026").value == 2
    fid = mark.memory.remember("atlas restarted", occurred_at="2 days ago")
    assert mark.memory.get(fid).metadata.get("occurred_at", "")[:10] == str(yesterday - timedelta(days=1))
    for bad in (lambda: facts.set("atlas", "port", 3, valid_from="someday maybe"),
                lambda: facts.as_of("atlas", "port", "soonish"),
                lambda: mark.memory.remember("x", occurred_at="whenever")):
        with pytest.raises(InvalidInputError, match="could not read a"):
            bad()


def test_import_skips_bad_lines_with_a_report(mark: Mark, tmp_path: Path) -> None:
    mark.memory.remember("We chose SQLite.")
    mark.memory.facts.set("atlas", "port", 6, valid_from="2026-01-01")
    mark.memory.facts.set("atlas", "port", 7, valid_from="2026-01-01")   # a same-time sequence
    backup = tmp_path / "b.jsonl"
    mark.export_jsonl(backup)
    lines = backup.read_text().splitlines()
    lines.insert(2, "{not json")
    lines.insert(3, json.dumps({"type": "memory", "id": "x", "agent_id": "__mark__"}))   # no content
    lines.insert(4, json.dumps({"type": "fact", "agent_id": "__mark__", "subject": "s"}))
    backup.write_text("\n".join(lines) + "\n")
    with Mark.local(tmp_path / "b") as other:
        counts = other.import_jsonl(backup)
        assert (counts["memories"], counts["facts"], counts["invalid"]) == (1, 2, 3)
        assert len(counts["problems"]) == 3 and "line 3" in counts["problems"][0]
        again = other.import_jsonl(backup)
        assert (again["memories"], again["facts"]) == (0, 0)
        history = other.memory.facts.history("atlas", "port")
        assert [f.value for f in history] == [6, 7]                     # not 6, 7, 6, 7
