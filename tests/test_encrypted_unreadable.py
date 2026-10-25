"""M1: plaintext written into an encrypted store (by MARK 0.2.0a8 without the key) is quarantined.

a8 cannot be fixed, so the newer build cleans up when it opens the store
with the key: rows that are not valid ciphertext are quarantined and tagged
``mark:unreadable`` (facts are retracted and flagged), reported by
``integrity_check()``, and left out of exports unless explicitly included.
Fact values stored as a Python bytes repr ("b'gAAAA...'") become the token.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

pytest.importorskip("cryptography")

from mark.security import FernetEncryptionProvider  # noqa: E402
from mark.store.sqlite import LocalMemoryStore  # noqa: E402
from mark.types import MemoryFragment, MemoryState  # noqa: E402

AGENT = "a"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _provider() -> FernetEncryptionProvider:
    return FernetEncryptionProvider(passphrase="correct horse battery staple")


def _a8_rows(path) -> tuple[str, str]:
    """Write what a keyless a8 writes: a plaintext fragment row and a plaintext fact value."""
    conn = sqlite3.connect(path)
    template = conn.execute("SELECT * FROM fragments LIMIT 1").fetchone()
    names = [d[0] for d in conn.execute("SELECT * FROM fragments LIMIT 1").description]
    row = dict(zip(names, template))
    fid = str(uuid.uuid4())
    row.update(id=fid, content=b"plaintext written by a8 without the key", tags="[]")
    conn.execute(f"INSERT INTO fragments ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                 [row[n] for n in names])
    fact_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO facts (id, agent_id, subject, subject_key, attribute, attribute_key, value, valid_from,"
        " valid_to, recorded_at, retracted_at, source_fragment_id, metadata)"
        " VALUES (?, ?, 'svc', 'svc', 'port', 'port', ?, ?, NULL, ?, NULL, NULL, '{}')",
        (fact_id, AGENT, b"9090", NOW.isoformat(), NOW.isoformat()),
    )
    conn.commit()
    conn.close()
    return fid, fact_id


@pytest.fixture
def mixed_store(tmp_path):
    path = tmp_path / "enc.db"
    store = LocalMemoryStore(path, encryption=_provider())
    good = store.store(MemoryFragment(content="launch code 1234", agent_id=AGENT, state=MemoryState.UNVERIFIED))
    store.fact_set(AGENT, "api", "port", 8080, valid_from=NOW)
    store.close()
    bad, bad_fact = _a8_rows(path)
    return path, good, bad, bad_fact


def test_plaintext_rows_are_quarantined_and_reported(mixed_store, caplog) -> None:
    path, good, bad, bad_fact = mixed_store
    store = LocalMemoryStore(path, encryption=_provider())
    row = store.get(bad)
    assert row.state == MemoryState.QUARANTINED and "mark:unreadable" in row.tags
    assert store.get(bad).state == MemoryState.QUARANTINED  # reads never relabel it
    assert bad not in store.retrievable_ids(AGENT) and good in store.retrievable_ids(AGENT)
    report = store.integrity_check()
    assert any(bad in line and "unreadable" in line for line in report)
    assert any(bad_fact in line for line in report)
    assert [f["value"] for f in store.fact_current(AGENT)] == [8080]  # the plaintext fact is retracted
    assert [f.id for f in store.page_fragments(AGENT, limit=None)] == [good]  # exports skip it
    with caplog.at_level(logging.WARNING, logger="mark.store.sqlite"):
        rows = {f.id: f for f in store.page_fragments(AGENT, limit=None, include_unreadable=True)}
    assert rows[bad].content == "plaintext written by a8 without the key"
    assert "unreadable" in caplog.text
    store.close()


def test_the_scan_only_reads_rows_added_since_the_last_open(mixed_store, monkeypatch) -> None:
    path, *_ = mixed_store
    LocalMemoryStore(path, encryption=_provider()).close()
    calls: list[int] = []
    provider = _provider()
    original = provider.decrypt
    monkeypatch.setattr(provider, "decrypt", lambda data: calls.append(1) or original(data))
    LocalMemoryStore(path, encryption=provider).close()
    assert len(calls) <= 1  # the key check only; no row was re-scanned


def test_bytes_repr_fact_values_become_the_token(tmp_path) -> None:
    path = tmp_path / "enc.db"
    provider = _provider()
    store = LocalMemoryStore(path, encryption=provider)
    store.fact_set(AGENT, "api", "port", "8080", valid_from=NOW)
    store.close()
    conn = sqlite3.connect(path)
    token = bytes(conn.execute("SELECT value FROM facts").fetchone()[0])
    conn.execute("UPDATE facts SET value = ?", (repr(token),))  # the literal "b'...'" text
    conn.commit()
    conn.close()
    store = LocalMemoryStore(path, encryption=_provider())
    assert [f["value"] for f in store.fact_current(AGENT)] == ["8080"]
    store.close()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT typeof(value) FROM facts").fetchone()[0] == "blob"
    conn.close()


def test_plaintext_rows_cost_no_key_derivation(tmp_path, monkeypatch) -> None:
    """a10 re-test: each plaintext row was "decrypted", deriving a key from a
    salt of plaintext bytes (one PBKDF2 run per row: a 14 s first reopen)."""
    path = tmp_path / "enc.db"
    store = LocalMemoryStore(path, encryption=_provider())
    store.store(MemoryFragment(content="launch code 1234", agent_id=AGENT, state=MemoryState.UNVERIFIED))
    store.close()
    for _ in range(4):
        _a8_rows(path)
    provider = _provider()
    derivations: list[bytes] = []
    original = provider._derive_bytes
    monkeypatch.setattr(provider, "_derive_bytes", lambda salt: derivations.append(salt) or original(salt))
    store = LocalMemoryStore(path, encryption=provider)
    assert len(derivations) == 1          # the store's own salt (key check) only
    assert len(store.unreadable_report()) == 8   # 4 memories + 4 facts still caught
    store.close()
