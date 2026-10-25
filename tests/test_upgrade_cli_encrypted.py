# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""`mark migrate` never reports an upgrade it did not do on an encrypted store."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.fernet import Fernet  # noqa: E402

from mark import cli  # noqa: E402
from mark.memory.runtime import MarkRuntime  # noqa: E402
from mark.security import FernetEncryptionProvider  # noqa: E402


def _older_encrypted_store(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    runtime = MarkRuntime.local(store_path=db, encryption=FernetEncryptionProvider(key=Fernet.generate_key().decode()))
    runtime.memory("a").remember("The vault code word is heron.")
    runtime.shutdown()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE schema_info SET value = '5' WHERE key = 'schema_version'")
    return db


def _schema(db: Path) -> str:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0]


@pytest.mark.parametrize("extra", [["--dry-run"], []])
def test_migrate_refuses_encrypted_store_and_changes_nothing(tmp_path: Path, capsys, extra: list[str]) -> None:
    db = _older_encrypted_store(tmp_path)
    assert cli.main(["--db", str(db), "migrate", *extra]) == 1
    out, err = capsys.readouterr()
    assert "upgraded" not in out and "encrypted" in err
    assert _schema(db) == "5" and not list(tmp_path.glob("memory.db.v*-backup-*"))
