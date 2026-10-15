"""Stdlib-only JSON-over-HTTP helper with timeouts and bounded retries.

Embedding requests are pure reads, so every transient failure is retried
with exponential backoff and full jitter, up to ``max_attempts``: connection
errors, read timeouts and HTTP 429/5xx. Other HTTP errors (bad key, unknown
model) fail at once. The final error is an :class:`EmbeddingHTTPError` —
a :class:`mark.errors.MarkError` — naming the endpoint and the attempts made.
"""
from __future__ import annotations

import http.client
import json
import random
import socket
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from mark.errors import MarkError

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingHTTPError(MarkError, RuntimeError):
    """An embedding server request failed (status is None for transport errors)."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class _ConnectError(Exception):
    """Connection could not be established (always retryable)."""


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    connect_timeout: float = 5.0,
    read_timeout: float = 60.0,
    max_attempts: int = 3,
    backoff: float = 0.5,
    max_backoff: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> Any:
    """POST JSON and return the decoded JSON response.

    Retries (exponential backoff with full jitter) on connection errors, read
    timeouts and HTTP 429/5xx; any other HTTP error raises EmbeddingHTTPError
    immediately. After the last attempt the error names the endpoint and the
    number of attempts. Header values (e.g. API keys) are never included in
    error messages.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Unsupported embedding endpoint URL: {url!r}")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_headers.update(headers or {})
    port = f":{parts.port}" if parts.port else ""
    safe_url = f"{parts.scheme}://{parts.hostname}{port}{parts.path}"

    attempts = max(1, max_attempts)
    last_error: EmbeddingHTTPError | None = None
    for attempt in range(attempts):
        if attempt:
            sleep(rng() * min(max_backoff, backoff * (2 ** (attempt - 1))))
        try:
            status, text = _send(parts, path, body, request_headers, connect_timeout, read_timeout)
        except _ConnectError as exc:
            last_error = EmbeddingHTTPError(
                f"Could not connect to embedding server at {safe_url} ({exc}). "
                "Is the server running and reachable?"
            )
            continue
        except socket.timeout:
            # A slow or overloaded server (e.g. a model still loading): retry.
            last_error = EmbeddingHTTPError(
                f"Embedding server at {safe_url} did not respond within {read_timeout:g}s."
            )
            continue
        except (http.client.HTTPException, OSError) as exc:
            last_error = EmbeddingHTTPError(f"Connection to embedding server at {safe_url} failed: {exc}")
            continue
        if 200 <= status < 300:
            try:
                return json.loads(text) if text else None
            except json.JSONDecodeError as exc:
                raise EmbeddingHTTPError(
                    f"Embedding server at {safe_url} returned invalid JSON.", status=status, body=text[:500]
                ) from exc
        error = EmbeddingHTTPError(
            f"Embedding server at {safe_url} returned HTTP {status}: {text[:200].strip()}",
            status=status,
            body=text[:2000],
        )
        if status not in _RETRYABLE_STATUS:
            raise error
        last_error = error
    assert last_error is not None
    if attempts > 1:
        last_error.args = (f"{last_error} (gave up after {attempts} attempts)",)
    raise last_error


def _send(
    parts: Any,
    path: str,
    body: bytes,
    headers: dict[str, str],
    connect_timeout: float,
    read_timeout: float,
) -> tuple[int, str]:
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=connect_timeout)
    try:
        try:
            conn.connect()
        except OSError as exc:  # refused, DNS failure, connect timeout
            raise _ConnectError(str(exc) or type(exc).__name__) from exc
        if conn.sock is not None:
            conn.sock.settimeout(read_timeout)
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        text = response.read().decode("utf-8", errors="replace")
        return response.status, text
    finally:
        conn.close()


def l2_normalize(vector: list[float]) -> list[float]:
    """Return an L2-normalised copy of a vector (zero vectors unchanged)."""
    values = [float(value) for value in vector]
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        return values
    return [value / norm for value in values]
