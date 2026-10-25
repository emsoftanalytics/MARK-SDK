"""N6: CLI requests in flight when ``mark serve`` dies finish in-process, without duplicates.

A fake server passes the client's authenticated handshake, then drops the
connection: either before doing anything, or after really running the
command (the write committed, the reply lost) — the two cases a kill -9
produces.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mark._cli import client
from mark._cli.config import _proof, _state_path
from mark._cli.engine import _Engine, protocol_fingerprint

AGENT = "a"


class _DyingServer:
    """Answers /hello like a genuine server; every POST optionally runs, then the socket is cut."""

    def __init__(self, db, *, commit_first: bool) -> None:
        token = secrets.token_urlsafe(16)
        outer = self
        self.posts = 0

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                nonce = self.path.partition("nonce=")[2]
                body = json.dumps({"proof": _proof(token, nonce), "protocol": protocol_fingerprint()}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                outer.posts += 1
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if commit_first:
                    engine = _Engine(db)
                    try:
                        engine.run(request["cmd"], request.get("args", {}))
                    finally:
                        engine.close()
                self.close_connection = True
                self.connection.close()  # killed: no reply at all

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        state = {"host": "127.0.0.1", "port": self.httpd.server_address[1], "token": token,
                 "pid": os.getpid(), "db": str(db)}
        _state_path(db).write_text(json.dumps(state))

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _count(db) -> int:
    engine = _Engine(db)
    try:
        return engine.runtime.store.count(AGENT)
    finally:
        engine.close()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "memory.db"
    client.execute(path, "store", {"agent": AGENT, "text": "the gateway runs on port 8443"}, use_server=False)
    return path


@pytest.mark.parametrize("commit_first", [False, True])
def test_store_is_finished_in_process_exactly_once(db, commit_first, capsys) -> None:
    server = _DyingServer(db, commit_first=commit_first)
    try:
        result = client.execute(db, "store", {"agent": AGENT, "text": "deploy key rotated on friday"})
    finally:
        server.close()
    assert server.posts == 1 and result["id"]
    assert _count(db) == 2  # never stored twice
    assert "in-process" in capsys.readouterr().err


@pytest.mark.parametrize("commit_first", [False, True])
def test_ingest_retries_with_its_idempotency_key(db, commit_first, monkeypatch) -> None:
    monkeypatch.setattr(client, "INGEST_CHUNK_BYTES", 80)  # several chunks
    lines = [f"line {i}: build {i} passed on the staging cluster" for i in range(6)]
    server = _DyingServer(db, commit_first=commit_first)
    try:
        client.execute(db, "ingest", {"agent": AGENT, "lines": lines})
    finally:
        server.close()
    assert _count(db) == 1 + len(lines)
    client.execute(db, "ingest", {"agent": AGENT, "lines": lines}, use_server=False)
    assert _count(db) == 1 + len(lines)


def test_recall_is_answered_in_process(db) -> None:
    server = _DyingServer(db, commit_first=False)
    try:
        result = client.execute(db, "recall", {"agent": AGENT, "query": "gateway port"})
    finally:
        server.close()
    assert "8443" in json.dumps(result)


def test_forget_already_done_by_the_server_reports_success(db) -> None:
    listed = client.execute(db, "list", {"agent": AGENT}, use_server=False)
    memory_id = listed["memories"][0]["id"]
    server = _DyingServer(db, commit_first=True)
    try:
        result = client.execute(db, "forget", {"agent": AGENT, "id": memory_id})
    finally:
        server.close()
    assert result["forgotten"]

