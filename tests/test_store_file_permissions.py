# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Files and directories MARK creates for a store are private (POSIX only).

The ``.mark/`` directory is created 0700 and the database (plus its -wal and
-shm siblings) 0600, regardless of a permissive umask. Directories the user
already had are left alone.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mark import Mark
from mark.cli import main
from mark.store.sqlite import LocalMemoryStore

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture(autouse=True)
def _permissive_umask():
    old = os.umask(0o022)
    yield
    os.umask(old)


def _assert_private_db(db: Path) -> None:
    assert _mode(db) == 0o600
    for suffix in ("-wal", "-shm"):
        sibling = db.with_name(db.name + suffix)
        if sibling.exists():
            assert _mode(sibling) == 0o600, sibling


def test_mark_local_creates_private_dir_and_db(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        mark.memory.block("ops").write("The staging database is db-7.")
        _assert_private_db(tmp_path / ".mark" / "memory.db")
    assert _mode(tmp_path / ".mark") == 0o700


def test_existing_user_directory_is_not_chmodded(tmp_path: Path) -> None:
    folder = tmp_path / "chosen"
    folder.mkdir(mode=0o755)
    os.chmod(folder, 0o755)
    store = LocalMemoryStore(folder / "memory.db")
    store.close()
    assert _mode(folder) == 0o755
    _assert_private_db(folder / "memory.db")


def test_store_creates_missing_parent_private(tmp_path: Path) -> None:
    db = tmp_path / "new" / "memory.db"
    LocalMemoryStore(db).close()
    assert _mode(db.parent) == 0o700
    _assert_private_db(db)


def test_cli_creates_private_store(tmp_path: Path) -> None:
    db = tmp_path / ".mark" / "memory.db"
    assert main(["--db", str(db), "--no-server", "store", "hello"]) == 0
    assert _mode(db.parent) == 0o700
    _assert_private_db(db)


def test_existing_database_file_mode_is_kept(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    LocalMemoryStore(db).close()
    os.chmod(db, 0o640)  # a deliberate choice by the user
    LocalMemoryStore(db).close()
    assert _mode(db) == 0o640
