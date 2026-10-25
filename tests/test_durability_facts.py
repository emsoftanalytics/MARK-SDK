"""Fact semantics (backlog A7): scheduled values, typed values, multi-valued attributes."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mark.errors import InvalidMemoryError
from mark.memory.runtime import MarkRuntime
from mark.security import FernetEncryptionProvider


@pytest.fixture()
def memory(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "f.db")
    yield runtime.memory("a")
    runtime.shutdown()


def _days(n: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=n)


def test_future_values_are_scheduled_not_current(memory) -> None:
    memory.facts.set("atlas", "port", 4000, valid_from=_days(-10))
    memory.facts.set("atlas", "port", 5000, valid_from=_days(+10))
    assert memory.facts.get("atlas", "port").value == 4000  # the old value holds until then
    assert [f.value for f in memory.facts.current()] == [4000]
    assert [f.value for f in memory.facts.scheduled()] == [5000]
    assert [f.value for f in memory.facts.scheduled("atlas", "port")] == [5000]
    assert not memory.facts.scheduled()[0].is_current
    assert memory.facts.as_of("atlas", "port", _days(11)).value == 5000


def test_only_future_value_means_no_current_value(memory) -> None:
    memory.facts.set("launch", "date", "2031-01-01", valid_from=_days(30))
    assert memory.facts.get("launch", "date") is None and len(memory.facts.scheduled()) == 1


@pytest.mark.parametrize("value", [4444, 3.5, True, False, None, ["a", "b"], {"host": "x", "port": 1}, "4444"])
def test_values_keep_their_type(memory, value) -> None:
    fact = memory.facts.set("svc", "setting", value)
    assert fact.value == value and type(fact.value) is type(value)
    again = memory.facts.get("svc", "setting")
    assert again.value == value and type(again.value) is type(value)


def test_typed_values_round_trip_encrypted(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    runtime = MarkRuntime.local(store_path=tmp_path / "e.db", encryption=FernetEncryptionProvider(passphrase="p"))
    memory = runtime.memory("a")
    memory.facts.set("svc", "limits", {"cpu": 2, "mem": [1, 2]})
    assert memory.facts.get("svc", "limits").value == {"cpu": 2, "mem": [1, 2]}
    runtime.shutdown()


def test_legacy_string_rows_still_read_as_strings(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "l.db")
    memory = runtime.memory("a")
    memory.facts.set("svc", "port", "1")
    runtime.shutdown()
    conn = sqlite3.connect(tmp_path / "l.db")
    conn.execute("UPDATE facts SET value = CAST('8080' AS BLOB)")  # a plain string, as 0.2.0a9 wrote it
    conn.commit()
    conn.close()
    runtime = MarkRuntime.local(store_path=tmp_path / "l.db")
    assert runtime.memory("a").facts.get("svc", "port").value == "8080"
    runtime.shutdown()


def test_reasserting_a_value_is_typed(memory) -> None:
    when = _days(-1)
    memory.facts.set("svc", "port", 80, valid_from=when)
    memory.facts.set("svc", "port", 80, valid_from=when)  # no-op
    assert len(memory.facts.history("svc", "port")) == 1
    assert memory.facts.retract("svc", "port", value="80") == 1  # text match still retracts


def test_bad_fact_values_raise_mark_errors(memory) -> None:
    with pytest.raises(InvalidMemoryError):
        memory.facts.set("svc", "x", {"k": object()})
    with pytest.raises(InvalidMemoryError):
        memory.facts.set("svc", "x", "bad \ud800")
    assert memory.facts.set("svc", "when", datetime(2026, 1, 1)).value == "2026-01-01 00:00:00"  # text, as before


def test_multi_valued_attributes(memory) -> None:
    facts = memory.facts
    facts.add("Dr. Bello", "phone", "+234 1", valid_from=_days(-3))
    facts.add("Dr. Bello", "phone", "+234 2", valid_from=_days(-2))
    facts.add("Dr. Bello", "phone", "+234 2", valid_from=_days(-1))  # already current: no-op
    assert [f.value for f in facts.current("Dr. Bello", "phone")] == ["+234 1", "+234 2"]
    assert facts.get("Dr. Bello", "phone").value == "+234 2"  # latest
    assert facts.remove("Dr. Bello", "phone", "+234 1") == 1
    assert [f.value for f in facts.current("Dr. Bello", "phone")] == ["+234 2"]
    assert [f.value for f in facts.history("Dr. Bello", "phone")] == ["+234 1", "+234 2"]
    facts.set("Dr. Bello", "phone", "+234 9")  # set() replaces every value
    assert [f.value for f in facts.current("Dr. Bello", "phone")] == ["+234 9"]


def test_add_keeps_a_set_value_current(memory) -> None:
    memory.facts.set("team", "lang", "Python", valid_from=_days(-5))
    memory.facts.add("team", "lang", "Rust", valid_from=_days(-1))
    assert sorted(f.value for f in memory.facts.current("team", "lang")) == ["Python", "Rust"]
    memory.facts.set("team", "lang", "Go", valid_from=_days(-4))  # back-dated set: ends earlier values only
    assert sorted(f.value for f in memory.facts.current("team", "lang")) == ["Go", "Rust"]
