# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""When the backup or the upgrade fails, the store ends up exactly as it was.

Every failure raises an :class:`UpgradeBackupError` subclass whose message
says why (with numbers) and what to do, and carries the numbers as
attributes for programs.
"""
from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path

import pytest

from mark.errors import (
    BackupVerificationError,
    BackupWriteError,
    InsufficientSpaceError,
    MarkError,
    UpgradeBackupError,
    UpgradeFailedError,
)
from mark.store.sqlite import LocalMemoryStore
from tests._helpers_upgrade import backups, build_store, counts, dump, stamp

Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def old(tmp_path: Path):
    db = tmp_path / "memory.db"
    before = build_store(db, version=5)
    return db, before, dump(db), db.read_bytes()


def _assert_unchanged(db: Path, original: list[str], raw: bytes | None = None) -> None:
    assert stamp(db) == 5
    assert dump(db) == original
    if raw is not None:
        assert db.read_bytes() == raw


def test_failing_migration_step_rolls_back_and_names_the_backup(old, monkeypatch) -> None:
    db, before, original, raw = old
    steps = dict(LocalMemoryStore._MIGRATIONS)
    steps[6] = [*steps[6], "SELECT no_such_function()"]
    monkeypatch.setattr(LocalMemoryStore, "_MIGRATIONS", steps)
    with pytest.raises(UpgradeFailedError) as info:
        LocalMemoryStore(db)
    (backup,) = backups(db.parent)
    err = info.value
    assert isinstance(err, UpgradeBackupError) and isinstance(err, MarkError) and isinstance(err, RuntimeError)
    assert str(backup) in str(err) and "no_such_function" in str(err)
    assert "rolled back" in str(err) and "unchanged at schema 5" in str(err)
    assert (err.backup_path, err.from_version, err.to_version, err.restored) == (str(backup), 5, 6, False)
    _assert_unchanged(db, original, raw)  # byte for byte
    assert counts(backup) == before


def test_failure_after_commit_restores_from_the_backup(old, monkeypatch) -> None:
    db, before, original, _raw = old

    def broken(self, plan):
        raise RuntimeError("simulated post-commit check failure")

    monkeypatch.setattr(LocalMemoryStore, "_upgrade_confirm", broken)
    with pytest.raises(UpgradeFailedError) as info:
        LocalMemoryStore(db)
    (backup,) = backups(db.parent)
    assert info.value.restored is True
    assert "restored from the backup" in str(info.value) and str(backup) in str(info.value)
    _assert_unchanged(db, original)
    assert counts(db) == before == counts(backup)
    monkeypatch.undo()
    store = LocalMemoryStore(db)  # still a healthy, upgradable store
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION
    store.close()


def test_lost_rows_fail_verification_and_roll_back(old, monkeypatch) -> None:
    db, before, original, raw = old
    run = LocalMemoryStore._run_migrations

    def lossy(self, conn, from_version):
        run(self, conn, from_version)
        conn.execute("DELETE FROM fragments WHERE rowid = (SELECT max(rowid) FROM fragments)")

    monkeypatch.setattr(LocalMemoryStore, "_run_migrations", lossy)
    with pytest.raises(UpgradeFailedError) as info:
        LocalMemoryStore(db)
    n = before["fragments"]
    assert f"fragments: {n} rows before, {n - 1} after" in str(info.value)
    _assert_unchanged(db, original, raw)


def test_not_enough_space_refuses_with_the_numbers(old, monkeypatch) -> None:
    from mark.store._sqlite import upgrade_backup

    db, _before, original, raw = old
    monkeypatch.setattr(upgrade_backup.shutil, "disk_usage", lambda p: Usage(10**12, 10**12 - 1000, 1000))
    with pytest.raises(InsufficientSpaceError) as info:
        LocalMemoryStore(db)
    err = info.value
    size = db.stat().st_size
    assert err.store_bytes == size and err.free_bytes == 1000
    assert err.required_bytes > size and err.shortfall_bytes == err.required_bytes - 1000
    assert err.directory == str(db.parent)
    text = str(err)
    for part in ("Not enough free space", "memory.db", "store ", "WAL ", "10% margin", "free",
                 "short by", "backup_dir=", "backup_on_upgrade=False", "Your store was not changed."):
        assert part in text, part
    assert backups(db.parent) == []
    _assert_unchanged(db, original, raw)


def test_backup_that_does_not_match_is_removed(old, monkeypatch) -> None:
    from mark.store._sqlite import upgrade_backup

    db, before, original, raw = old
    real = upgrade_backup.table_counts

    def short_backup(conn):
        found = real(conn)
        if "backup-" in conn.execute("PRAGMA database_list").fetchone()[2]:
            found["fragments"] -= 1
        return found

    monkeypatch.setattr(upgrade_backup, "table_counts", short_backup)
    with pytest.raises(BackupVerificationError) as info:
        LocalMemoryStore(db)
    n = before["fragments"]
    assert f"fragments: store {n} rows, backup {n - 1}" in str(info.value)
    assert "was removed" in str(info.value) and "Your store was not changed." in str(info.value)
    assert info.value.differences == [f"fragments: store {n} rows, backup {n - 1}"]
    assert backups(db.parent) == []
    _assert_unchanged(db, original, raw)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_unwritable_backup_dir_names_the_path_and_reason(old, tmp_path: Path) -> None:
    db, _before, original, raw = old
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        with pytest.raises(BackupWriteError) as info:
            LocalMemoryStore(db, backup_dir=locked)
    finally:
        os.chmod(locked, 0o700)
    assert str(locked) in str(info.value) and "Permission denied" in str(info.value)
    assert info.value.backup_path.startswith(str(locked))
    _assert_unchanged(db, original, raw)


def test_failure_without_a_backup_still_rolls_back(old, monkeypatch) -> None:
    db, _before, original, raw = old
    steps = dict(LocalMemoryStore._MIGRATIONS)
    steps[6] = [*steps[6], "SELECT no_such_function()"]
    monkeypatch.setattr(LocalMemoryStore, "_MIGRATIONS", steps)
    with pytest.raises(UpgradeFailedError) as info:
        LocalMemoryStore(db, backup_on_upgrade=False)
    assert info.value.backup_path is None and "no backup was taken" in str(info.value).lower()
    _assert_unchanged(db, original, raw)
