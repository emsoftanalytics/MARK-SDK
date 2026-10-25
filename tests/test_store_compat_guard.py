# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Upgrade/downgrade safety: newer stores are refused, unreadable rows are isolated.

A store written by a newer MARK must not be opened (and so never migrated or
written) by this version. A row this version cannot decrypt — for example
plaintext written into an encrypted store by an older MARK — must be set
aside without breaking reads of every other memory.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mark.errors import MarkError, StoreVersionError
from mark.memory.runtime import MarkRuntime
from mark.store.sqlite import LocalMemoryStore


def _stamp(db: Path, version: int) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE schema_info SET value = ? WHERE key = 'schema_version'", (str(version),))


def test_store_from_a_newer_schema_is_refused_untouched(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    rt = MarkRuntime.local(store_path=db)
    rt.memory("a").store_sync("Keep me.")
    rt.shutdown()
    newer = LocalMemoryStore.CURRENT_SCHEMA_VERSION + 1
    _stamp(db, newer)
    before = db.read_bytes()

    with pytest.raises(StoreVersionError) as info:
        MarkRuntime.local(store_path=db)
    assert isinstance(info.value, MarkError) and isinstance(info.value, RuntimeError)
    assert f"schema version {newer}" in str(info.value) and "pip install -U mark-sdk" in str(info.value)
    with sqlite3.connect(db) as conn:
        stamp = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0]
    assert stamp == str(newer)
    assert db.read_bytes() == before


def test_current_and_older_stamps_still_open(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    MarkRuntime.local(store_path=db).shutdown()
    _stamp(db, LocalMemoryStore.CURRENT_SCHEMA_VERSION - 1)
    rt = MarkRuntime.local(store_path=db)
    rt.memory("a").store_sync("Still works.")
    assert rt.memory("a").retrieve_sync("works").fragments
    rt.shutdown()


def test_cli_reports_newer_store_cleanly(tmp_path: Path, capsys) -> None:
    from mark.cli import main

    db = tmp_path / "memory.db"
    MarkRuntime.local(store_path=db).shutdown()
    _stamp(db, LocalMemoryStore.CURRENT_SCHEMA_VERSION + 5)
    assert main(["--db", str(db), "--no-server", "stats"]) == 1
    assert "Upgrade mark-sdk" in capsys.readouterr().err


def test_undecryptable_row_is_isolated_not_fatal(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    from mark.security import FernetEncryptionProvider

    key = Fernet.generate_key().decode()
    db = tmp_path / "memory.db"
    rt = MarkRuntime.local(store_path=db, encryption=FernetEncryptionProvider(key=key))
    mem = rt.memory("a")
    good = mem.store_sync("The staging database is db-7.")
    bad = mem.store_sync("The production database is db-9.")
    rt.shutdown()
    # What an older MARK opened without the key could leave behind: plaintext bytes.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE fragments SET content = ? WHERE id = ?", (b"leaked plaintext", bad))

    rt = MarkRuntime.local(store_path=db, encryption=FernetEncryptionProvider(key=key))
    mem = rt.memory("a")
    result = mem.retrieve_sync("Which database is staging?")
    ids = [fragment.id for fragment in result.fragments]
    assert good in ids and bad not in ids
    # The unreadable row is set aside (flagged), not silently deleted.
    assert "unreadable" in rt.store.get(bad).content
    rt.shutdown()
