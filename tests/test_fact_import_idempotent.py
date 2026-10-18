"""N4: restoring the same export twice never duplicates fact history (store helper).

``import_jsonl`` (memory/facade/portability.py) calls ``import_fact_row``
for every exported fact row; this pins the store side of that contract.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mark.store.sqlite import LocalMemoryStore

AGENT = "a"
DAY = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _export(store: LocalMemoryStore) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in {"id", "valid_to"}} for row in store.fact_rows(AGENT)]


@pytest.mark.parametrize("encrypted", [False, True])
def test_reimporting_an_export_adds_nothing(tmp_path, encrypted) -> None:
    provider = None
    if encrypted:
        pytest.importorskip("cryptography")
        from mark.security import FernetEncryptionProvider

        provider = FernetEncryptionProvider(ephemeral=True)
    source = LocalMemoryStore(tmp_path / "s.db")
    for value in (6, 7):  # two changes recorded on the same day
        source.fact_set(AGENT, "svc", "replicas", value, valid_from=DAY)
    source.fact_set(AGENT, "svc", "replicas", 6, valid_from=DAY.replace(day=2))
    source.fact_add(AGENT, "svc", "owner", "ana", valid_from=DAY)
    rows = _export(source)
    source.close()
    target = LocalMemoryStore(tmp_path / "t.db", encryption=provider)
    assert [target.import_fact_row(AGENT, row) for row in rows] == [True] * len(rows)
    assert [target.import_fact_row(AGENT, row) for row in rows] == [False] * len(rows)
    history = [f["value"] for f in target.fact_history(AGENT, "svc", "replicas")]
    assert history == [6, 7, 6]
    assert [f["value"] for f in target.fact_current(AGENT, "svc", "replicas")] == [6]
    assert len(target.fact_rows(AGENT)) == len(rows)
    target.close()


def test_rows_without_recorded_at_match_on_the_value(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "m.db")
    row = {"subject": "api", "attribute": "port", "value": "8080", "valid_from": DAY.isoformat()}
    assert store.import_fact_row(AGENT, row) is True
    assert store.import_fact_row(AGENT, row) is False
    store.close()
