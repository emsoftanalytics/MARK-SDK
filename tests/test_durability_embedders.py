"""Network embedders retry timeouts/5xx and fail with a clear MARK error (backlog A4)."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mark.embeddings import EmbeddingHTTPError, OpenAICompatibleEmbeddingProvider
from mark.errors import MarkError
from mark.memory.runtime import MarkRuntime


class _SlowThenOk:
    """Loopback server whose first ``slow`` requests stall past the client's read timeout."""

    def __init__(self, slow: int, status: int = 200) -> None:
        self.calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                outer.calls += 1
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if outer.calls <= slow:
                    time.sleep(0.6)
                data = json.dumps({"data": [{"index": i, "embedding": [1.0, float(i)]}
                                            for i in range(len(body["input"]))]}).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_read_timeouts_are_retried() -> None:
    server = _SlowThenOk(slow=1)
    try:
        provider = OpenAICompatibleEmbeddingProvider(base_url=server.url, timeout=0.2)
        assert len(provider.embed("x")) == 2
        assert server.calls == 2
    finally:
        server.close()


def test_exhausted_retries_raise_a_mark_error_naming_the_endpoint() -> None:
    server = _SlowThenOk(slow=99)
    try:
        provider = OpenAICompatibleEmbeddingProvider(base_url=server.url, timeout=0.2, max_attempts=2)
        with pytest.raises(EmbeddingHTTPError) as info:
            provider.embed("x")
        assert isinstance(info.value, MarkError) and isinstance(info.value, RuntimeError)
        assert server.url.removesuffix("/v1") in str(info.value) and "2 attempts" in str(info.value)
    finally:
        server.close()


def test_embedder_outage_stores_the_batch_unembedded(tmp_path: Path) -> None:
    # A transient outage (timeout) must not lose writes: rows are stored with
    # no vector, found by keyword meanwhile, and embedded when the server is back.
    server = _SlowThenOk(slow=99)
    try:
        embedder = OpenAICompatibleEmbeddingProvider(base_url=server.url, timeout=0.2, max_attempts=1)
        runtime = MarkRuntime.local(store_path=tmp_path / "n.db", embedder=embedder)
        report = runtime.memory("a").store_many_sync(["one alpha", "two beta", "three gamma"])
        assert report.stored == 3 and runtime.store.count("a") == 3
        assert runtime.memory("a").recall("beta").fragments
        runtime.shutdown()
    finally:
        server.close()
