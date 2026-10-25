"""`mark serve`: OS file lock, index warm-up and a stop request (backlog A5)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mark._cli.client import _live_server, stop_server
from mark._cli.config import _lock_path, _state_path
from mark._cli.server import _acquire_lock, _release_lock, serve
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime


def test_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    db = tmp_path / "l.db"
    lock = _acquire_lock(db)
    with pytest.raises(SystemExit, match=f"pid {os.getpid()}"):
        _acquire_lock(db)
    _release_lock(lock)
    assert not _lock_path(db).exists()
    _release_lock(_acquire_lock(db))


def test_stale_lock_file_never_blocks(tmp_path: Path) -> None:
    db = tmp_path / "l.db"
    _lock_path(db).write_text("999999")  # left by a crashed server: no OS lock held
    _release_lock(_acquire_lock(db))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_lock_of_a_killed_holder_is_free_immediately(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    code = ("import sys, time; from pathlib import Path; from mark._cli.server import _acquire_lock;"
            "_acquire_lock(Path(sys.argv[1])); print('locked', flush=True); time.sleep(60)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    holder = subprocess.Popen([sys.executable, "-c", code, str(db)], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(SystemExit, match="already running"):
            _acquire_lock(db)
        holder.send_signal(signal.SIGKILL)
        holder.wait(10)
        _release_lock(_acquire_lock(db))
    finally:
        holder.kill()
        holder.wait(10)
        holder.stdout.close()


def _start(db: Path) -> threading.Thread:
    thread = threading.Thread(target=serve, args=(db,), daemon=True)
    thread.start()
    for _ in range(200):
        if _live_server(_state_path(db)):
            return thread
        time.sleep(0.05)
    raise AssertionError("server did not start")


def test_stop_request_shuts_the_server_down(tmp_path: Path) -> None:
    db = (tmp_path / "s.db").resolve()
    assert stop_server(db) is False  # nothing running
    thread = _start(db)
    assert stop_server(db) is True
    thread.join(10)
    assert not thread.is_alive()
    assert not _state_path(db).exists() and not _lock_path(db).exists()
    _start(db)  # the database can be served again at once
    assert stop_server(db) is True


def test_warm_up_preloads_the_index_without_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "w.db"
    runtime = MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32))
    ids = runtime.memory("a").store_many_sync([f"warm memory {i}" for i in range(20)])
    runtime.memory("b").store_sync("other agent memory")
    runtime.shutdown()
    cold = MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32))
    assert len(cold.index) == 0
    assert cold.warm_up() == 21
    assert set(cold.index.ids("a")) == set(ids) and len(cold.index.ids("b")) == 1
    assert "access_count" not in cold.store.get(ids[0]).metadata
    cold.shutdown()
