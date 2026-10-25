"""A tiny loopback HTTP/1.1 client for talking to ``mark serve`` (standard library ``socket`` only).

Why not ``urllib``/``http.client``: importing them pulls in ``ssl`` and the
email parser, about a third of a served ``mark recall``'s start-up, for a
request that never leaves 127.0.0.1 and never uses TLS or a proxy. The server
(:mod:`mark._cli.server`) always answers with ``Content-Length`` and JSON, so
one request per connection (``Connection: close``) is all this needs.
"""
from __future__ import annotations

import json
import socket
from typing import Any


class WireError(OSError):
    """The server closed the connection or answered with something that is not HTTP."""


def request(host: str, port: int, method: str, path: str, *, body: bytes = b"",
            headers: dict[str, str] | None = None, timeout: float = 300.0) -> tuple[int, bytes]:
    """Send one request; returns ``(status, body)``. Raises ``OSError`` on network failure."""
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host}:{port}", "Connection: close",
             f"Content-Length: {len(body)}"]
    lines += [f"{name}: {value}" for name, value in (headers or {}).items()]
    payload = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall(payload)
        chunks: list[bytes] = []
        while True:
            data = conn.recv(65536)
            if not data:
                break
            chunks.append(data)
            if _complete(b"".join(chunks)):
                break
    return _parse(b"".join(chunks))


def call_json(host: str, port: int, method: str, path: str, payload: Any = None, *,
              token: str | None = None, timeout: float = 300.0) -> tuple[int, Any]:
    """``request`` with a JSON body and a JSON answer (``{}`` for an empty body)."""
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, data = request(host, port, method, path, body=body, headers=headers, timeout=timeout)
    return status, json.loads(data or b"{}")


def _split(raw: bytes) -> tuple[bytes, bytes] | None:
    head, sep, rest = raw.partition(b"\r\n\r\n")
    return (head, rest) if sep else None


def _length(head: bytes) -> int | None:
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            return int(value.strip())
    return None


def _complete(raw: bytes) -> bool:
    parts = _split(raw)
    if parts is None:
        return False
    length = _length(parts[0])
    return length is not None and len(parts[1]) >= length


def _parse(raw: bytes) -> tuple[int, bytes]:
    parts = _split(raw)
    if parts is None:
        raise WireError("connection closed before a complete HTTP response")
    head, rest = parts
    status_line = head.split(b"\r\n", 1)[0].split()
    if len(status_line) < 2 or not status_line[0].startswith(b"HTTP/") or not status_line[1].isdigit():
        raise WireError("not an HTTP response")
    length = _length(head)
    return int(status_line[1]), rest if length is None else rest[:length]
