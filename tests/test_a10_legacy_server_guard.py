"""a10 re-test issue 3: a8/a9 ``mark serve`` (pid lock file, no OS lock) blocks migration.

Pre-a10 servers create ``<db>.server.lock`` with O_EXCL and their pid, plus
``<db>.server.json``; the a10 flock probe found that lock "free" and
migrated the schema under the running server. Set ``MARK_A9_VENV`` to a venv
with MARK 0.2.0a9 installed to also run the end-to-end check against it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mark.store._sqlite.server_guard import StoreInUseError, foreign_server
from mark.store.sqlite import LocalMemoryStore

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX pid probing")
DEAD_PID = 2 ** 22 + 12345   # above the default pid_max: never a live process


def _old_store(tmp_path) -> tuple[Path, int]:
    db = tmp_path / "m.db"
    LocalMemoryStore(db).close()
    conn = sqlite3.connect(db)
    current = int(conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0])
    conn.execute("UPDATE schema_info SET value = ? WHERE key = 'schema_version'", (str(current - 1),))
    conn.commit()
    conn.close()
    return db, current


def _version(db) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0])
    finally:
        conn.close()


@pytest.fixture
def mark_like_process():
    """A live process whose command line mentions MARK, like ``mark serve``."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "mark", "serve"])
    yield proc.pid
    proc.kill()
    proc.wait()


def test_a_live_legacy_pid_lock_refuses_the_migration(tmp_path, mark_like_process) -> None:
    db, current = _old_store(tmp_path)
    Path(f"{db}.server.lock").write_text(str(mark_like_process))   # a9 style: no flock
    with pytest.raises(StoreInUseError) as info:
        LocalMemoryStore(db)
    assert str(mark_like_process) in str(info.value)
    assert _version(db) == current - 1


def test_stale_or_unrelated_pids_do_not_block_upgrades(tmp_path) -> None:
    db, current = _old_store(tmp_path)
    Path(f"{db}.server.lock").write_text(str(DEAD_PID))   # crashed a9 server
    Path(f"{db}.server.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": 9, "token": "t", "pid": DEAD_PID}))
    if Path("/proc/self/cmdline").exists():   # live but not MARK (recycled pid)
        unrelated = subprocess.Popen(["sleep", "30"])
        Path(f"{db}.server.lock").write_text(str(unrelated.pid))
    else:
        unrelated = None
    try:
        assert foreign_server(str(db)) == (False, None)
        LocalMemoryStore(db).close()
        assert _version(db) == current
    finally:
        if unrelated is not None:
            unrelated.kill()
            unrelated.wait()


def _hello_server(token: str) -> ThreadingHTTPServer:
    class Hello(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            nonce = self.path.partition("nonce=")[2]
            body = json.dumps({"proof": hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest(),
                               "protocol": "0.2.0a9-something"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Hello)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_server_answering_hello_refuses_the_migration(tmp_path) -> None:
    # e.g. an a9 server in a container sharing the volume: its pid means nothing here.
    db, current = _old_store(tmp_path)
    server = _hello_server("tok")
    try:
        Path(f"{db}.server.json").write_text(json.dumps(
            {"host": "127.0.0.1", "port": server.server_address[1], "token": "tok", "pid": DEAD_PID}))
        with pytest.raises(StoreInUseError):
            LocalMemoryStore(db)
        assert _version(db) == current - 1
        # A wrong token proves nothing: some other program on a reused port.
        Path(f"{db}.server.json").write_text(json.dumps(
            {"host": "127.0.0.1", "port": server.server_address[1], "token": "other", "pid": DEAD_PID}))
        assert foreign_server(str(db)) == (False, None)
    finally:
        server.shutdown()
        server.server_close()


def test_mark_serve_refuses_to_start_beside_a_legacy_server(tmp_path, mark_like_process) -> None:
    from mark._cli.server import _acquire_lock

    db = tmp_path / "m.db"
    Path(f"{db}.server.lock").write_text(str(mark_like_process))
    with pytest.raises(SystemExit, match="older MARK server"):
        _acquire_lock(db)
    assert Path(f"{db}.server.lock").read_text() == str(mark_like_process)   # left as it was


@pytest.mark.skipif(not os.environ.get("MARK_A9_VENV"), reason="set MARK_A9_VENV to an a9 venv")
def test_real_a9_server_keeps_its_schema(tmp_path) -> None:
    venv = Path(os.environ["MARK_A9_VENV"])
    db = tmp_path / "db" / "memory.db"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"MARK_EMBEDDER": "hash"}
    subprocess.run([str(venv / "bin" / "mark"), "--db", str(db), "store", "a9 memo"], env=env, check=True)
    before = _version(db)
    server = subprocess.Popen([str(venv / "bin" / "mark"), "--db", str(db), "serve"], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while not Path(f"{db}.server.json").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        with pytest.raises(StoreInUseError):
            LocalMemoryStore(db)
        assert _version(db) == before
    finally:
        server.terminate()
        server.wait(10)
