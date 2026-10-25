"""N5: a store is never migrated while an older ``mark serve`` holds it open."""
from __future__ import annotations

import os
import sqlite3
import sys

import pytest

from mark.errors import MarkError
from mark.store._sqlite.server_guard import StoreInUseError, foreign_server
from mark.store.sqlite import LocalMemoryStore

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="flock-based fake server lock")


def _stamp(path, version: int | None = None) -> int:
    conn = sqlite3.connect(path)
    try:
        if version is not None:
            conn.execute("UPDATE schema_info SET value = ? WHERE key = 'schema_version'", (str(version),))
            conn.commit()
        return int(conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0])
    finally:
        conn.close()


def _fake_server_lock(db, pid: int) -> int:
    """Hold <db>.server.lock like a running ``mark serve`` with *pid* would."""
    import fcntl

    fd = os.open(f"{db}.server.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a second open file description conflicts
    os.write(fd, str(pid).encode())
    return fd


def test_migration_is_refused_while_an_older_server_holds_the_store(tmp_path) -> None:
    db = tmp_path / "m.db"
    LocalMemoryStore(db).close()
    current = _stamp(db)
    _stamp(db, current - 1)  # as an older build left it
    fd = _fake_server_lock(db, 424242)
    try:
        with pytest.raises(StoreInUseError) as info:
            LocalMemoryStore(db)
        assert isinstance(info.value, MarkError) and isinstance(info.value, RuntimeError)
        assert "424242" in str(info.value) and "mark serve stop" in str(info.value)
        assert _stamp(db) == current - 1  # nothing was migrated
    finally:
        os.close(fd)
    LocalMemoryStore(db).close()  # server gone: the upgrade runs
    assert _stamp(db) == current


def test_current_stores_open_normally_beside_a_server(tmp_path) -> None:
    db = tmp_path / "m.db"
    LocalMemoryStore(db).close()
    fd = _fake_server_lock(db, 424242)
    try:  # nothing to migrate: the lock is not even probed
        store = LocalMemoryStore(db)
        store.close()
    finally:
        os.close(fd)


def test_the_servers_own_lock_never_blocks_its_engine(tmp_path) -> None:
    db = tmp_path / "m.db"
    LocalMemoryStore(db).close()
    _stamp(db, _stamp(db) - 1)
    fd = _fake_server_lock(db, os.getpid())
    try:
        assert foreign_server(str(db)) == (False, os.getpid())
        LocalMemoryStore(db).close()
    finally:
        os.close(fd)
    assert foreign_server(str(db)) == (False, None)  # free lock: probing leaves it free
