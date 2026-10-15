"""``mark serve stop`` and ``mark mcp``: commands that manage processes, not memories.

Both prefer the implementation owned by their subsystem and fall back to a
small local one, so the command line works whichever is present:

* ``serve stop`` uses ``mark._cli.client.stop_server(db)`` when available,
  else signals the pid recorded in the server's 0600 state file;
* ``mcp`` uses ``mark.adapters.mcp.run_mcp_server`` when available, else
  builds the server with ``create_mark_mcp_server_from_local`` over
  ``Mark.local(db)`` (the same database file the CLI uses) and runs it on stdio.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from mark._cli.config import _state_path


def stop_server(db: Path) -> str:
    """Stop the ``mark serve`` process of *db*; returns a one-line status message."""
    try:
        from mark._cli import client

        stop = client.stop_server  # provided by the durability workstream
    except (ImportError, AttributeError):
        return _stop_by_pid(db)
    result = stop(db)
    return result if isinstance(result, str) else (
        f"stopped mark serve for {db}" if result else f"no mark serve running for {db}")


def _stop_by_pid(db: Path, timeout: float = 5.0) -> str:
    import signal

    from mark._cli.server import _pid_alive

    state = _state_path(db)
    try:
        pid = int(json.loads(state.read_text())["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return f"no mark serve running for {db}"
    if not _pid_alive(pid):
        state.unlink(missing_ok=True)  # stale state from a crashed server
        return f"no mark serve running for {db}"
    os.kill(pid, signal.SIGTERM)  # serve installs a SIGTERM handler and cleans up
    deadline = time.monotonic() + timeout
    # Done when the process is gone or has removed its state file on shutdown
    # (a zombie child still "exists" for signal 0 until its parent reaps it).
    while time.monotonic() < deadline and _pid_alive(pid) and state.exists():
        time.sleep(0.05)
    if _pid_alive(pid) and state.exists():
        return f"sent SIGTERM to mark serve (pid {pid}); it has not exited yet"
    return f"stopped mark serve (pid {pid}) for {db}"


def run_mcp(db: Path, agent: str, *, briefing: bool = False) -> None:
    """Serve MARK over MCP (stdio) for *db*; blocks until the client disconnects."""
    try:
        from mark.adapters import mcp as adapter
    except ImportError as exc:  # pragma: no cover - adapter module is part of the package
        raise RuntimeError(str(exc)) from exc
    try:
        from mark.adapters.mcp.launcher import main as launch
    except ImportError:
        launch = None
    if launch is not None:
        launch(["--db", str(db), "--agent", agent, *(["--briefing"] if briefing else [])])
        return
    from mark.runtime import Mark

    with Mark.local(db) as mark:
        server = adapter.create_mark_mcp_server_from_local(mark, agent_id=agent, briefing=briefing)
        server.run()
