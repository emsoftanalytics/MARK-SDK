"""CLI client: find a live ``mark serve`` process and talk to it, else run in-process.

Standard library only: when a server answers, the client never imports MARK.

A server that dies while a request is in flight (killed, crashed, stopped)
never surfaces as an error: the command is finished in-process instead.
Reads simply run again. Writes are retried without duplicating: ``ingest``
chunks carry their idempotency key; a ``store``/``supersede`` first looks
for the memory the server may already have written (same text, written
since the request was sent); ``forget``/``resolve`` of an id the server
already removed report success.
"""
from __future__ import annotations

import hmac
import http.client
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
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
        nonce = secrets.token_hex(16)
        url = f"{_base_url(data)}/hello?nonce={nonce}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=1.0) as response:
            hello = json.loads(response.read())
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


class _ServerGone(Exception):
    """The server stopped answering mid-request (connection reset/closed, truncated reply)."""


def _call_server(server: dict[str, Any], cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """POST one command to an authenticated server.

    Raises ``RuntimeError`` when the command failed on the server and
    ``_ServerGone`` when the server itself went away (the outcome is unknown).
    """
    request = urllib.request.Request(
        f"{_base_url(server)}/",
        data=json.dumps({"cmd": cmd, "args": args}, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {server['token']}", "Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy
    try:
        with opener.open(request, timeout=300) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read() or b"{}")
        except (OSError, ValueError, http.client.HTTPException) as gone:
            raise _ServerGone(str(gone)) from None
    except (OSError, ValueError, http.client.HTTPException) as exc:  # reset, refused, cut short
        raise _ServerGone(str(exc)) from None
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "server error"))
    return payload["result"]


def execute(db: Path, cmd: str, args: dict[str, Any], *, use_server: bool = True) -> dict[str, Any]:
    """Run a command via a live server for *db*, else in-process."""
    server = _live_server(_state_path(db)) if use_server else None
    if server is not None:
        # Anything the server may have written for this request is at least this new.
        sent = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
        try:
            if cmd == "ingest":
                return _ingest_in_chunks(server, args, db)
            return _call_server(server, cmd, args)
        except _ServerGone:
            print("mark: the server stopped during the request; finishing it in-process", file=sys.stderr)
            return _run_local(db, cmd, args, retry_since=sent)
    if cmd == "ingest":  # same chunks (hence idempotency keys) with or without a server
        return _ingest_in_chunks(None, args, db)
    return _run_local(db, cmd, args)


def _run_local(db: Path, cmd: str, args: dict[str, Any], *, retry_since: str | None = None) -> dict[str, Any]:
    """Run *cmd* in this process; ``retry_since`` marks a retry after the server died mid-request."""
    engine = _Engine(db)
    try:
        if retry_since is not None:
            done = _already_applied(engine, cmd, args, retry_since)
            if done is not None:
                return done
        try:
            return engine.run(cmd, args)
        except ValueError as exc:
            # The server removed it before it died: the request did what was asked.
            if retry_since is not None and cmd in {"forget", "resolve"} and "no memory with id" in str(exc):
                return {"forgotten": 1, "secure": bool(args.get("secure"))} if cmd == "forget" else {"id": args["id"]}
            raise
    finally:
        engine.close()


def _already_applied(engine: Any, cmd: str, args: dict[str, Any], since: str) -> dict[str, Any] | None:
    """The result of a write the dead server already committed, or None to run it now."""
    if cmd not in {"store", "remember", "supersede"}:
        return None  # reads, and writes that are idempotent by themselves
    found = engine.runtime.store.recent_duplicate(args["agent"], args["text"], since)
    if found is None:
        return None
    if cmd == "supersede":
        return {"id": found, "superseded": args["id"]}
    return {"id": found, **({"extracted": 0} if args.get("extract_facts") else {})}


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
    except (OSError, RuntimeError, ValueError, http.client.HTTPException, _ServerGone):
        pass  # it may close the connection while stopping; confirm below
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not state.exists() or _live_server(state) is None:
            return True
        time.sleep(0.05)
    return False


def _ingest_in_chunks(server: dict[str, Any] | None, args: dict[str, Any], db: Path) -> dict[str, Any]:
    """Send large ingests as several requests, each well under the size cap (in-process without *server*).

    Each chunk is one transaction on the server and idempotent (see
    ``_Engine.cmd_ingest``: the key is derived from the chunk's lines). If the
    server dies mid-ingest, this chunk and the rest are stored in-process with
    the *same* chunking, so the lines it already stored are never duplicated
    (an in-process ingest chunks identically, so re-running a file with or
    without a server never duplicates either).
    """
    result: dict[str, Any] = {"stored": 0}
    chunk: list[str] = []
    size = 0
    local: list[Any] = [] if server is not None else [_Engine(db)]  # in-process once the server is gone

    def send() -> None:
        # Counts add up; other fields (e.g. current "facts") come from the last chunk.
        nonlocal result
        request = {**args, "lines": chunk}
        if not local and server is not None:
            try:
                part = _call_server(server, "ingest", request)
            except _ServerGone:
                print("mark: the server stopped during the ingest; finishing it in-process", file=sys.stderr)
                local.append(_Engine(db))
        if local:
            part = local[0].run("ingest", request)
        result = {**part, "stored": result["stored"] + part["stored"],
                  **({"extracted": result.get("extracted", 0) + part["extracted"]} if "extracted" in part else {})}

    try:
        for line in args["lines"]:
            line_size = len(line.encode("utf-8")) + 8  # + JSON quoting/separator slack
            if chunk and size + line_size > INGEST_CHUNK_BYTES:
                send()
                chunk, size = [], 0
            chunk.append(line)
            size += line_size
        if chunk:
            send()
    finally:
        for engine in local:
            engine.close()
    return result
