# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Opening an older store backs it up first, then upgrades it (0.2.0a10).

The backup is a verified, owner-only copy next to the store (or in
``backup_dir``), named after the schema it holds, never overwriting a file.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import stat
from pathlib import Path

import pytest

from mark.store.sqlite import LocalMemoryStore
from tests._helpers_upgrade import AGENT, backups, build_store, counts, downgrade, dump, stamp

FIXTURES = Path(__file__).parent / "fixtures"
CURRENT = LocalMemoryStore.CURRENT_SCHEMA_VERSION


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize("version", [4, 5])
def test_older_store_is_backed_up_then_upgraded(tmp_path: Path, version: int, caplog) -> None:
    db = tmp_path / "memory.db"
    before = build_store(db, version=version)
    original = dump(db)
    caplog.set_level(logging.INFO, logger="mark.store.sqlite")

    store = LocalMemoryStore(db)
    assert store.schema_version() == CURRENT
    found = backups(tmp_path)
    assert len(found) == 1
    backup = found[0]
    assert backup.name.startswith(f"memory.db.v{version}-backup-") and backup.suffix == ".db"
    assert stamp(backup) == version and counts(backup) == before and dump(backup) == original
    assert counts(db) == before  # nothing lost in the migration
    meta = json.loads(store.get_meta("last_upgrade"))
    assert store.get_meta("last_upgrade_backup") == str(backup)
    assert (meta["from_schema"], meta["to_schema"], meta["backup"]) == (version, CURRENT, str(backup))
    assert meta["backup_bytes"] == backup.stat().st_size
    store.close()
    line = next(r.getMessage() for r in caplog.records if "upgraded" in r.getMessage())
    assert f"from schema {version} to {CURRENT}" in line and f"backup kept at {backup}" in line
    assert "backup " in line and "store " in line and " s" in line  # sizes and timings
    # A second open has nothing to migrate: no second backup.
    LocalMemoryStore(db).close()
    assert backups(tmp_path) == [backup]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_backup_is_owner_only_and_the_folder_is_untouched(tmp_path: Path) -> None:
    folder = tmp_path / "proj"
    folder.mkdir(mode=0o755)
    os.chmod(folder, 0o755)
    db = folder / "memory.db"
    build_store(db)
    os.chmod(db, 0o644)  # an a8 store: world-readable
    LocalMemoryStore(db).close()
    (backup,) = backups(folder)
    assert _mode(backup) == 0o600
    assert _mode(folder) == 0o755
    assert not list(folder.glob("*.db-journal"))  # the backup is one self-contained file
    assert not Path(f"{backup}-wal").exists()


def test_pre_versioned_fixture_store_is_backed_up(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    shutil.copy(FIXTURES / "store_v0.2.0a7.db", db)
    before = counts(db, ("fragments",))
    store = LocalMemoryStore(db)
    (backup,) = backups(tmp_path)
    assert ".v3-backup-" in backup.name and counts(backup, ("fragments",)) == before
    assert len(store._fetchall("SELECT id FROM fragments")) == 3
    store.close()


def test_new_and_in_memory_stores_take_no_backup(tmp_path: Path) -> None:
    LocalMemoryStore(tmp_path / "new.db").close()
    LocalMemoryStore(":memory:").close()
    assert backups(tmp_path) == []


def test_backup_can_be_switched_off(tmp_path: Path, monkeypatch) -> None:
    first, second = tmp_path / "one.db", tmp_path / "two.db"
    build_store(first)
    build_store(second)
    store = LocalMemoryStore(first, backup_on_upgrade=False)
    assert store.schema_version() == CURRENT and store.get_meta("last_upgrade_backup") is None
    assert json.loads(store.get_meta("last_upgrade"))["backup"] is None
    store.close()
    monkeypatch.setenv("MARK_BACKUP_ON_UPGRADE", "0")
    LocalMemoryStore(second).close()
    assert backups(tmp_path) == []
    assert stamp(second) == CURRENT


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_backup_dir_is_created_private(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    build_store(db)
    target = tmp_path / "safe" / "backups"
    LocalMemoryStore(db, backup_dir=target).close()
    (backup,) = backups(target)
    assert _mode(target) == 0o700 and _mode(backup) == 0o600
    assert backups(tmp_path) == []


def test_an_existing_file_is_never_overwritten(tmp_path: Path, monkeypatch) -> None:
    from mark.store._sqlite import upgrade_backup

    db = tmp_path / "memory.db"
    build_store(db)
    monkeypatch.setattr(upgrade_backup, "_timestamp", lambda: "20260101T000000Z")
    taken = tmp_path / "memory.db.v5-backup-20260101T000000Z.db"
    taken.write_bytes(b"someone else's file")
    LocalMemoryStore(db).close()
    assert taken.read_bytes() == b"someone else's file"
    assert (tmp_path / "memory.db.v5-backup-20260101T000000Z-1.db").exists()


def test_encrypted_store_backup_stays_encrypted(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    from cryptography.fernet import Fernet

    key = Fernet.generate_key()
    db = tmp_path / "memory.db"
    before = build_store(db, encryption=FernetEncryptionProvider(key))
    store = LocalMemoryStore(db, encryption=FernetEncryptionProvider(key))
    store.close()
    (backup,) = backups(tmp_path)
    raw = backup.read_bytes()
    # Memories stay ciphertext (graph-API document rows are stored as given, as in the store).
    assert b"Lisbon" not in raw and "上海".encode() not in raw and "🚀".encode() not in raw
    assert counts(backup) == before
    conn = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    contents = [bytes(row[0]) if isinstance(row[0], bytes) else row[0].encode() for row in
                conn.execute("SELECT content FROM fragments")]
    conn.close()
    assert "Ana moved to Lisbon in 2024." in {FernetEncryptionProvider(key).decrypt_str(c) for c in contents}


def test_runtime_and_mark_pass_the_options(tmp_path: Path) -> None:
    from mark.memory.runtime import MarkRuntime
    from mark.runtime import Mark

    one, two = tmp_path / "one" / "memory.db", tmp_path / "two" / "memory.db"
    one.parent.mkdir()
    two.parent.mkdir()
    build_store(one)
    build_store(two)
    MarkRuntime.local(store_path=one, backup_on_upgrade=False).shutdown()
    assert backups(one.parent) == [] and stamp(one) == CURRENT
    mark = Mark.local(two, backup_dir=tmp_path / "kept")
    mark.close()
    assert len(backups(tmp_path / "kept")) == 1


def test_runtime_reopens_upgraded_store_with_its_memories(tmp_path: Path) -> None:
    from mark.memory.runtime import MarkRuntime

    db = tmp_path / "memory.db"
    rt = MarkRuntime.local(store_path=db)
    rt.memory(AGENT).store_sync("上海仓库的库存周五盘点。")
    rt.shutdown()
    downgrade(db, 4)
    rt = MarkRuntime.local(store_path=db)
    assert len(backups(tmp_path)) == 1
    hits = rt.memory(AGENT).retrieve_sync("上海仓库").fragments
    assert hits and "上海" in hits[0].content
    rt.shutdown()
