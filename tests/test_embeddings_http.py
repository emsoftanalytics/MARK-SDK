# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Embedding providers over HTTP (local fake server): OpenAI-compatible batching,
retries/backoff and errors, and Ollama endpoint fallback.
"""
from __future__ import annotations

import json
import logging
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

from mark.embeddings import (
    EmbeddingHTTPError,
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from mark.embeddings import _http

from _helpers_embeddings import _norm


# ── local fake embedding server ──────────────────────────────────────────────


class FakeServer:
    """Tiny threaded HTTP server; `handler(path, body, headers)` -> (status, payload)."""

    def __init__(self, handler: Callable[[str, dict, dict], tuple[int, Any]]) -> None:
        self.requests: list[tuple[str, dict, dict]] = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {k.lower(): v for k, v in self.headers.items()}
                outer.requests.append((self.path, body, headers))
                status, payload = handler(self.path, body, headers)
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", lambda s: sleeps.append(s))
    original = _http.post_json

    def patched(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("sleep", lambda s: sleeps.append(s))
        return original(*args, **kwargs)

    for module in ("mark.embeddings.ollama", "mark.embeddings.openai_compatible"):
        monkeypatch.setattr(sys.modules[module], "post_json", patched)
    return sleeps


def _vec(i: int) -> list[float]:
    return [float(i + 1), 2.0, 0.0]


# ── OpenAI-compatible ────────────────────────────────────────────────────────


def test_openai_compatible_batches_sorts_and_normalises(no_sleep: list[float]) -> None:
    def handler(path: str, body: dict, headers: dict) -> tuple[int, Any]:
        data = [{"index": i, "embedding": _vec(i), "object": "embedding"} for i in range(len(body["input"]))]
        return 200, {"object": "list", "data": list(reversed(data)), "model": body["model"]}

    server = FakeServer(handler)
    try:
        provider = OpenAICompatibleEmbeddingProvider(
            base_url=server.url + "/v1/", model="nomic", api_key="sk-secret", batch_size=2, dimensions=3
        )
        vectors = provider.embed_batch(["a", "b", "c"])
        assert len(vectors) == 3
        # batch_size=2: "c" is index 0 of the second request.
        for i, vec in zip([0, 1, 0], vectors):
            expected = _vec(i)
            n = _norm(expected)
            assert vec == pytest.approx([x / n for x in expected])
        assert [len(r[1]["input"]) for r in server.requests] == [2, 1]
        path, body, headers = server.requests[0]
        assert path == "/v1/embeddings"
        assert body["model"] == "nomic" and body["dimensions"] == 3
        assert headers["authorization"] == "Bearer sk-secret"
        assert provider.dim == 3
        assert provider.fingerprint == "openai-compatible:nomic:3"
        assert "sk-secret" not in repr(provider)
        assert provider.embed("z") == pytest.approx(vectors[0])
    finally:
        server.close()


def test_openai_compatible_retries_5xx_and_429_then_succeeds(no_sleep: list[float]) -> None:
    statuses = [503, 429]

    def handler(path: str, body: dict, headers: dict) -> tuple[int, Any]:
        if statuses:
            return statuses.pop(0), {"error": "busy"}
        return 200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    server = FakeServer(handler)
    try:
        provider = OpenAICompatibleEmbeddingProvider(base_url=server.url + "/v1")
        assert provider.embed("x") == [1.0, 0.0]
        assert len(server.requests) == 3
        assert len(no_sleep) == 2
    finally:
        server.close()


def test_openai_compatible_does_not_retry_4xx_and_hides_key(no_sleep: list[float], caplog) -> None:
    server = FakeServer(lambda p, b, h: (401, {"error": "bad key"}))
    try:
        provider = OpenAICompatibleEmbeddingProvider(base_url=server.url + "/v1", api_key="sk-topsecret")
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(EmbeddingHTTPError) as info:
                provider.embed("x")
        assert info.value.status == 401
        assert "401" in str(info.value)
        assert "sk-topsecret" not in str(info.value)
        assert "sk-topsecret" not in caplog.text
        assert len(server.requests) == 1
    finally:
        server.close()


def test_openai_compatible_gives_up_after_three_attempts(no_sleep: list[float]) -> None:
    server = FakeServer(lambda p, b, h: (500, {"error": "boom"}))
    try:
        with pytest.raises(EmbeddingHTTPError) as info:
            OpenAICompatibleEmbeddingProvider(base_url=server.url + "/v1").embed("x")
        assert info.value.status == 500
        assert len(server.requests) == 3
    finally:
        server.close()


def test_openai_compatible_count_mismatch_is_clear_error(no_sleep: list[float]) -> None:
    server = FakeServer(lambda p, b, h: (200, {"data": []}))
    try:
        with pytest.raises(EmbeddingHTTPError, match="0 embeddings for 1 inputs"):
            OpenAICompatibleEmbeddingProvider(base_url=server.url + "/v1").embed("x")
    finally:
        server.close()


def test_connection_refused_is_retried_with_clear_message(no_sleep: list[float]) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    provider = OpenAICompatibleEmbeddingProvider(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-x")
    with pytest.raises(EmbeddingHTTPError, match="Could not connect") as info:
        provider.embed("x")
    assert info.value.status is None
    assert "sk-x" not in str(info.value)
    assert len(no_sleep) == 2  # 3 attempts


def test_backoff_is_full_jitter_exponential() -> None:
    sleeps: list[float] = []
    with pytest.raises(EmbeddingHTTPError):
        _http.post_json(
            "http://127.0.0.1:1/x",
            {},
            max_attempts=4,
            backoff=1.0,
            max_backoff=3.0,
            sleep=sleeps.append,
            rng=lambda: 1.0,
            connect_timeout=0.5,
        )
    assert sleeps == [1.0, 2.0, 3.0]


# ── Ollama ───────────────────────────────────────────────────────────────────


def test_ollama_uses_batch_embed_endpoint(no_sleep: list[float]) -> None:
    def handler(path: str, body: dict, headers: dict) -> tuple[int, Any]:
        assert path == "/api/embed"
        return 200, {"model": body["model"], "embeddings": [_vec(i) for i in range(len(body["input"]))]}

    server = FakeServer(handler)
    try:
        provider = OllamaEmbeddingProvider(model="nomic-embed-text", base_url=server.url, batch_size=2, keep_alive="10m")
        vectors = provider.embed_batch(["a", "b", "c"])
        assert len(vectors) == 3 and _norm(vectors[2]) == pytest.approx(1.0)
        assert [r[1]["input"] for r in server.requests] == [["a", "b"], ["c"]]
        assert server.requests[0][1]["keep_alive"] == "10m"
        assert provider.dim == 3
        assert provider.fingerprint == "ollama:nomic-embed-text:3"
    finally:
        server.close()


def test_ollama_falls_back_to_legacy_endpoint_on_404(no_sleep: list[float]) -> None:
    def handler(path: str, body: dict, headers: dict) -> tuple[int, Any]:
        if path == "/api/embed":
            return 404, "404 page not found"
        return 200, {"embedding": [0.0, 3.0, 4.0]}

    server = FakeServer(handler)
    try:
        provider = OllamaEmbeddingProvider(base_url=server.url)
        assert provider.embed_batch(["a", "b"]) == [pytest.approx([0.0, 0.6, 0.8])] * 2
        assert [r[0] for r in server.requests] == ["/api/embed", "/api/embeddings", "/api/embeddings"]
        assert server.requests[1][1]["prompt"] == "a"
        provider.embed("c")  # remembers legacy mode
        assert server.requests[-1][0] == "/api/embeddings"
    finally:
        server.close()


def test_ollama_missing_model_404_is_not_masked(no_sleep: list[float]) -> None:
    server = FakeServer(lambda p, b, h: (404, {"error": "model \"nope\" not found, try pulling it first"}))
    try:
        with pytest.raises(EmbeddingHTTPError, match="not found"):
            OllamaEmbeddingProvider(model="nope", base_url=server.url).embed("x")
        assert len(server.requests) == 1
    finally:
        server.close()
