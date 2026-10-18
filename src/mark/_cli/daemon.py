"""``mark serve --detach`` and ``mark serve status``.

An agent's shell blocks on a foreground ``mark serve``, so ``--detach``
starts the server as a background process (its own session, stdin closed,
output in the 0600 log file ``X.db.server.log``), waits until it answers the
authenticated handshake, prints its pid and port and returns. The server's
pid is in its 0600 state file ``X.db.server.json`` (with host, port and
token) and in the lock file ``X.db.server.lock``; ``mark serve stop`` stops
it. ``mark serve status`` says whether a server answers for the database
(exit 0) or not (exit 1). Standard library only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mark._cli.config import _state_path

#: Seconds ``--detach`` waits for the new server to answer.
START_TIMEOUT = 60.0


def log_path(db: Path) -> Path:
    """Where a detached server writes its output."""
    return db.with_name(db.name + ".server.log")


def status(db: Path) -> tuple[bool, str]:
    """``(running, one-line message)`` for the server of *db*."""
    from mark._cli.client import _live_server

    server = _live_server(_state_path(db))
    if server is None:
        return False, f"no mark serve running for {db}"
    return True, (f"mark serve running for {db} (pid {server.get('pid')}, "
                  f"{server.get('host')}:{server.get('port')})")


def detach(db: Path, host: str, port: int) -> tuple[bool, str]:
    """Start ``mark serve`` for *db* in the background; ``(ok, message)``."""
    running, message = status(db)
    if running:
        return True, message.replace("running", "already running", 1)
    log = log_path(db)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        options = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        process = subprocess.Popen(
            [sys.executable, "-m", "mark.cli", "--db", str(db), "serve", "start", "--host", host,
             "--port", str(port)],
            stdin=subprocess.DEVNULL, stdout=fd, stderr=fd, close_fds=True, **options)
    finally:
        os.close(fd)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False, f"mark serve exited (code {process.returncode}): {_tail(log)}"
        running, message = status(db)
        if running and _state_pid(db) == process.pid:
            return True, f"{message.replace('running', 'started', 1)}; log: {log}"
        time.sleep(0.1)
    return False, f"mark serve (pid {process.pid}) did not answer within {START_TIMEOUT:g}s; see {log}"


def _state_pid(db: Path) -> int | None:
    try:
        return int(json.loads(_state_path(db).read_text())["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _tail(log: Path) -> str:
    try:
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return "(no output)"
    return lines[-1][:300] if lines else "(no output)"
