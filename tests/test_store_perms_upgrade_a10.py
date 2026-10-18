# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Stores written 0644/0775 by 0.2.0a8 are made private on open (dev re-test minor)."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mark import Mark

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_mark_folder_store_is_tightened(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        mark.memory.remember("old a8 store")
    folder, db = tmp_path / ".mark", tmp_path / ".mark" / "memory.db"
    os.chmod(folder, 0o775)
    os.chmod(db, 0o644)
    with Mark.local(tmp_path) as mark:
        assert mark.memory.count() == 1
    assert _mode(folder) == 0o700 and _mode(db) == 0o600


def test_stores_outside_a_mark_folder_keep_their_mode(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    with Mark.local(db) as mark:
        mark.memory.remember("team store")
    os.chmod(db, 0o640)
    with Mark.local(db):
        pass
    assert _mode(db) == 0o640
