# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Opening an older encrypted store without its key refuses before any backup or migration."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.fernet import Fernet  # noqa: E402

from mark.errors import EncryptionRequiredError  # noqa: E402
from mark.memory.runtime import MarkRuntime  # noqa: E402
from mark.security import FernetEncryptionProvider  # noqa: E402


def test_keyless_open_of_older_encrypted_store_changes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    runtime = MarkRuntime.local(store_path=db, encryption=FernetEncryptionProvider(key=Fernet.generate_key().decode()))
    runtime.memory("a").remember("The vault code word is heron.")
    runtime.shutdown()
    with sqlite3.connect(db) as conn:  # pretend 0.2.0a9 wrote it (schema 5, already marked encrypted)
        conn.execute("UPDATE schema_info SET value = '5' WHERE key = 'schema_version'")
    with pytest.raises(EncryptionRequiredError):
        MarkRuntime.local(store_path=db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0] == "5"
    assert not list(tmp_path.glob("memory.db.v*-backup-*"))  # no backup taken, nothing migrated
