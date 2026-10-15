"""CLI client: find a live ``mark serve`` process and talk to it, else run in-process.

Standard library only: when a server answers, the client never imports MARK.
"""
from __future__ import annotations

import hmac
import json
import secrets
import sys
import urllib.error
import urllib.request
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


def _call_server(server: dict[str, Any], cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """POST one command to an authenticated server; raise ``RuntimeError`` on failure."""
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
        payload = json.loads(exc.read() or b"{}")
    if not payload.get("ok"):
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


def _ingest_in_chunks(server: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Send large ingests as several requests, each well under the size cap."""
    result: dict[str, Any] = {"stored": 0}
    chunk: list[str] = []
    size = 0

    def send() -> None:
        # Counts add up; other fields (e.g. current "facts") come from the last chunk.
        nonlocal result
        part = _call_server(server, "ingest", {**args, "lines": chunk})
        result = {**part, "stored": result["stored"] + part["stored"],
                  **({"extracted": result.get("extracted", 0) + part["extracted"]} if "extracted" in part else {})}

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
