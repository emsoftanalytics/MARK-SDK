"""Detect a running ``mark serve`` from a pre-0.2.0a10 build (a8/a9).

Why a second detector: a8/a9 servers do not hold an OS lock. They create
``<db>.server.lock`` with ``O_EXCL`` and write their pid into it, and publish
``<db>.server.json`` (host, port, token, pid). The flock probe in
:mod:`~mark.store._sqlite.server_guard` finds such a lock *free*, so an a10
open migrated the schema under a live a9 server (re-test issue 3), whose
later writes then missed the a10 tables.

A legacy server counts as running when

* the lock file or the state file names a live process other than us that
  looks like MARK (on Linux ``/proc/<pid>/cmdline`` must mention "mark", so a
  recycled pid of an unrelated process does not block upgrades forever), or
* the state file's address answers the ``/hello`` handshake with a valid
  proof for its token (covers a server in another pid namespace, e.g. a
  container sharing the volume, and platforms without pid probing).

Standard library only; called only when a migration is due or a server starts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys


def legacy_server(db_path: str, *, probe_timeout: float = 1.0) -> int | None:
    """Return the pid (0 when unknown) of a live legacy server for *db_path*, else None."""
    pids: list[int] = []
    lock_pid = _read_pid_file(f"{db_path}.server.lock")
    if lock_pid is not None:
        pids.append(lock_pid)
    state = _read_state(f"{db_path}.server.json")
    state_pid = _as_pid(state.get("pid")) if state else None
    if state_pid is not None:
        pids.append(state_pid)
    for pid in pids:
        if _live_mark_process(pid):
            return pid
    if state and state_pid != os.getpid() and _answers_hello(state, probe_timeout):
        return state_pid or 0
    return None


def _read_pid_file(path: str) -> int | None:
    try:
        with open(path, "rb") as handle:
            return _as_pid(handle.read(32).decode("ascii", "replace").strip())
    except OSError:
        return None


def _read_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_pid(value: object) -> int | None:
    try:
        pid = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _live_mark_process(pid: int) -> bool:
    """True when *pid* is another live process that looks like MARK.

    Unknown liveness (Windows: signal 0 would terminate the process) is
    *not* "alive" here; the hello probe decides instead.
    """
    if pid == os.getpid() or sys.platform == "win32":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by another user
    except (OSError, ValueError):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmdline = handle.read().lower()
    except OSError:
        return True  # no /proc (macOS): a live pid in our lock file is enough
    return b"mark" in cmdline


def _answers_hello(state: dict, timeout: float) -> bool:
    """True when the state file's server proves it holds the state file's token."""
    host, port, token = state.get("host"), state.get("port"), state.get("token")
    if not isinstance(host, str) or not isinstance(token, str) or not isinstance(port, int):
        return False
    import http.client

    nonce = secrets.token_hex(16)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", f"/hello?nonce={nonce}")
        response = conn.getresponse()
        if response.status != 200:
            return False
        hello = json.loads(response.read(65536) or b"{}")
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        conn.close()
    expected = hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return isinstance(hello, dict) and hmac.compare_digest(str(hello.get("proof", "")), expected)
