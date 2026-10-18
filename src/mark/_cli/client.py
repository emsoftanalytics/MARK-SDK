"""CLI client: find a live ``mark serve`` process and talk to it, else run in-process.

Standard library only: when a server answers, the client never imports MARK,
and it speaks HTTP through :mod:`mark._cli.wire` (a raw loopback socket), so
it never imports ``urllib``/``http.client``/``ssl`` either (start-up time).
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mark._cli.config import INGEST_CHUNK_BYTES, _proof, _state_path
from mark._cli.engine import _Engine, protocol_fingerprint


def _live_server(state: Path) -> dict[str, Any] | None:
    """Server details when a genuine server for this database answers.

    The server must prove it holds the token (HMAC of a fresh nonce) before
    any request carrying the token is sent, so a stale state file can never
    leak the token or memory content to another process on the same port.
    """
    try:
        data = json.loads(state.read_text())
    except (OSError, ValueError):
        return None  # no state file: no server (the common, cold case)
    try:
        from mark._cli.wire import call_json

        nonce = os.urandom(16).hex()  # what secrets.token_hex does, without importing random
        status, hello = call_json(data["host"], int(data["port"]), "GET", f"/hello?nonce={nonce}", timeout=1.0)
        if status != 200:
            return None
        if not hmac.compare_digest(str(hello.get("proof", "")), _proof(data["token"], nonce)):
            return None
        if hello.get("protocol") != protocol_fingerprint():
            # A server from another build may not accept this client's
            # arguments: run in-process instead of failing mid-command.
            print("mark: server runs a different MARK build; running in-process "
                  "(restart `mark serve` to use the server)", file=sys.stderr)
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def _base_url(server: dict[str, Any]) -> str:
    host = server["host"]
    host = f"[{host}]" if ":" in host else host
    return f"http://{host}:{int(server['port'])}"


def _call_server(server: dict[str, Any], cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """POST one command to an authenticated server; raise ``RuntimeError`` on failure."""
    from mark._cli.wire import call_json

    # Loopback socket: never via a proxy, never TLS (see mark._cli.wire).
    _status, payload = call_json(server["host"], int(server["port"]), "POST", "/",
                                 {"cmd": cmd, "args": args}, token=server["token"], timeout=300)
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise RuntimeError(payload.get("error", "server error"))
    return payload["result"]


def execute(db: Path, cmd: str, args: dict[str, Any], *, use_server: bool = True) -> dict[str, Any]:
    """Run a command via a live server for *db*, else in-process."""
    server = _live_server(_state_path(db)) if use_server else None
    if server is not None:
        if cmd == "ingest":
            return _ingest_in_chunks(server, args)
        return _call_server(server, cmd, args)
    engine = _Engine(db)
    try:
        return engine.run(cmd, args)
    finally:
        engine.close()


def stop_server(db: Path, *, timeout: float = 10.0) -> bool:
    """Ask the ``mark serve`` process for *db* to stop; True once it has stopped.

    Returns False when no (genuine) server is running for *db*. The request
    is authenticated like any other, so only a process that can read the
    0600 state file can stop the server. Waits up to ``timeout`` seconds for
    the server to finish shutting down (its state file disappears).
    """
    state = _state_path(db)
    server = _live_server(state)
    if server is None:
        return False
    try:
        _call_server(server, "stop", {})
    except (OSError, RuntimeError, ValueError):
        pass  # it may close the connection while stopping; confirm below
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not state.exists() or _live_server(state) is None:
            return True
        time.sleep(0.05)
    return False


def _ingest_in_chunks(server: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Send large ingests as several requests, each well under the size cap.

    Each chunk is one transaction on the server and idempotent (see
    ``_Engine.cmd_ingest``): if the server dies mid-ingest, re-running the
    same command stores the missing lines and never duplicates the rest.
    """
    result: dict[str, Any] = {"stored": 0}
    chunk: list[str] = []
    size = 0

    def send() -> None:
        # Counts add up; other fields (e.g. current "facts") come from the last chunk.
        nonlocal result
        part = _call_server(server, "ingest", {**args, "lines": chunk})
        summed = {key: result.get(key, 0) + part[key]
                  for key in ("stored", "existing", "skipped", "extracted") if key in part}
        result = {**part, **summed}

    for line in args["lines"]:
        line_size = len(line.encode("utf-8")) + 8  # + JSON quoting/separator slack
        if chunk and size + line_size > INGEST_CHUNK_BYTES:
            send()
            chunk, size = [], 0
        chunk.append(line)
        size += line_size
    if chunk:
        send()
    return result
