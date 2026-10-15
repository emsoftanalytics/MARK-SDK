# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""An encrypted store must never be opened (or written) without its key.

Covers the persisted ``encrypted_at_rest`` marker, content sniffing for stores
created before the marker existed, wrong keys, the CLI path, and the reverse
case (encrypting a plaintext store that already holds data).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from mark.cli import main  # noqa: E402
from mark.errors import (  # noqa: E402
    EncryptionKeyError,
    EncryptionRequiredError,
    MarkError,
    PlaintextStoreError,
)
from mark.memory.runtime import MarkRuntime  # noqa: E402
from mark.security import FernetEncryptionProvider  # noqa: E402
from mark.store.sqlite import LocalMemoryStore  # noqa: E402

def _key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def _encrypted_db(tmp_path: Path, key: str) -> Path:
    db = tmp_path / "memory.db"
    rt = MarkRuntime.local(store_path=db, encryption=FernetEncryptionProvider(key=key))
    rt.memory("a").store_sync("The staging database is db-7.")
    rt.shutdown()
    return db


def _plain_fragment_count(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
    finally:
        conn.close()


def test_open_encrypted_store_without_key_is_refused(tmp_path: Path) -> None:
    db = _encrypted_db(tmp_path, _key())
    with pytest.raises(EncryptionRequiredError) as info:
        MarkRuntime.local(store_path=db)
    assert isinstance(info.value, MarkError)
    assert "encryption=" in str(info.value)
    assert _plain_fragment_count(db) == 1  # nothing was written


def test_wrong_key_is_a_clear_error(tmp_path: Path) -> None:
    db = _encrypted_db(tmp_path, _key())
    with pytest.raises(EncryptionKeyError):
        LocalMemoryStore(db, encryption=FernetEncryptionProvider(key=_key()))


def test_right_key_reopens_and_reads(tmp_path: Path) -> None:
    key = _key()
    db = _encrypted_db(tmp_path, key)
    rt = MarkRuntime.local(store_path=db, encryption=FernetEncryptionProvider(key=key))
    result = rt.memory("a").retrieve_sync("staging database")
    rt.shutdown()
    assert any("db-7" in fragment.content for fragment in result.fragments)


def test_passphrase_store_wrong_passphrase(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    LocalMemoryStore(db, encryption=FernetEncryptionProvider(passphrase="right")).close()
    LocalMemoryStore(db, encryption=FernetEncryptionProvider(passphrase="right")).close()
    with pytest.raises(EncryptionKeyError):
        LocalMemoryStore(db, encryption=FernetEncryptionProvider(passphrase="wrong"))


def test_legacy_encrypted_store_without_marker_is_detected(tmp_path: Path) -> None:
    key = _key()
    db = _encrypted_db(tmp_path, key)
    conn = sqlite3.connect(db)  # simulate a store written before the marker existed
    conn.execute("DELETE FROM schema_info WHERE key IN ('encrypted_at_rest', 'encryption_check')")
    conn.commit()
    conn.close()
    with pytest.raises(EncryptionRequiredError):
        LocalMemoryStore(db)
    # Sniffing persisted the marker, and the right key still opens the store.
    store = LocalMemoryStore(db, encryption=FernetEncryptionProvider(key=key))
    assert store.get_meta("encrypted_at_rest") == "1"
    assert store.get_meta("encryption_check")
    store.close()
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM schema_info WHERE key IN ('encrypted_at_rest', 'encryption_check')")
    conn.commit()
    conn.close()
    with pytest.raises(EncryptionKeyError):
        LocalMemoryStore(db, encryption=FernetEncryptionProvider(key=_key()))


def test_plaintext_store_with_data_refuses_encryption(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    rt = MarkRuntime.local(store_path=db)
    rt.memory("a").store_sync("plain text memory")
    rt.shutdown()
    with pytest.raises(PlaintextStoreError):
        LocalMemoryStore(db, encryption=FernetEncryptionProvider(key=_key()))
    # Still a healthy plaintext store afterwards.
    store = LocalMemoryStore(db)
    assert store.get_meta("encrypted_at_rest") == "0" and store.fts_available
    store.close()


def test_empty_plaintext_store_can_become_encrypted(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    LocalMemoryStore(db).close()
    key = _key()
    store = LocalMemoryStore(db, encryption=FernetEncryptionProvider(key=key))
    assert store.get_meta("encrypted_at_rest") == "1"
    store.close()
    with pytest.raises(EncryptionRequiredError):
        LocalMemoryStore(db)


def test_cli_refuses_encrypted_store(tmp_path: Path, capsys) -> None:
    db = _encrypted_db(tmp_path, _key())
    assert main(["--db", str(db), "--no-server", "recall", "staging"]) == 1
    assert "encrypted" in capsys.readouterr().err
    assert main(["--db", str(db), "--no-server", "store", "leak me"]) == 1
    assert _plain_fragment_count(db) == 1  # no plaintext slipped into the encrypted store
