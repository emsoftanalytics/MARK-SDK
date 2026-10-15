"""``mark serve``: a loopback-only HTTP server that keeps MARK warm.

Safety invariants:

* one server per database, enforced by an ``O_EXCL`` lock file holding the
  owner's pid (stale locks from dead owners are reclaimed);
* the server only binds to loopback, every POST must carry the random bearer
  token, and bodies are capped at ``MAX_REQUEST_BYTES``;
* the token is published only through a 0600 state file, and ``GET /hello``
  answers a nonce with an HMAC proof so clients can authenticate the server
  before revealing the token (see :func:`mark._cli.client._live_server`).
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
from pathlib import Path
from typing import Any

from mark._cli.config import (
    _LOOPBACK,
    MAX_REQUEST_BYTES,
    _lock_path,
    _proof,
    _state_path,
)
from mark._cli.engine import _Engine, protocol_fingerprint


def serve(db: Path, host: str = "127.0.0.1", port: int = 0, *, stop_event: Any = None) -> None:
    """Serve commands for *db* on loopback until interrupted (or ``stop_event`` is set)."""
    import signal
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if host not in _LOOPBACK:
        raise SystemExit("mark serve only binds to loopback (127.0.0.1, ::1, localhost)")
    state = _state_path(db)
    lock = _acquire_lock(db)  # one server per database, or SystemExit
    try:
        engine = _Engine(db)
    except BaseException:
        _release_lock(lock)
        raise
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep stdout clean
            return

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            # Identity handshake: prove we hold the token without revealing it.
            nonce = self.path.partition("nonce=")[2][:128]
            if not self.path.startswith("/hello") or not nonce:
                self._reply(404, {"error": "not found"})
                return
            self._reply(200, {"proof": _proof(token, nonce), "protocol": protocol_fingerprint()})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            supplied = self.headers.get("Authorization", "").encode("utf-8", "replace")
            if not hmac.compare_digest(supplied, f"Bearer {token}".encode()):
                self.close_connection = True
                self._reply(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.close_connection = True  # don't read (or wait for) an oversized body
                self._reply(413, {"error": f"request body must be 1..{MAX_REQUEST_BYTES} bytes"})
                return
            try:
                request = json.loads(self.rfile.read(length))
                self._reply(200, {"ok": True, "result": engine.run(request["cmd"], request.get("args", {}))})
            except Exception as exc:  # report, keep serving
                self._reply(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    server_class = ThreadingHTTPServer
    if ":" in host:  # IPv6 loopback
        server_class = type("MarkHTTPServerV6", (ThreadingHTTPServer,), {"address_family": socket.AF_INET6})
    server = server_class((host, port), Handler)
    server.daemon_threads = True
    _write_state(state, {"host": host, "port": server.server_address[1], "token": token, "pid": os.getpid(),
                         "db": str(db)})
    stop = threading.Event()

    def _shutdown(*_: Any) -> None:
        if not stop.is_set():
            stop.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    if stop_event is not None:
        threading.Thread(target=lambda: (stop_event.wait(), _shutdown()), daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass
    print(f"MARK serving {db} on {host}:{server.server_address[1]} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        engine.close()
        # Remove the state file only if it is still ours (never another server's).
        try:
            if json.loads(state.read_text()).get("token") == token:
                state.unlink()
        except (OSError, ValueError):
            pass
        _release_lock(lock)


def _acquire_lock(db: Path) -> Path:
    """Create the per-database server lock, clearing it if its owner is gone."""
    lock = _lock_path(db)
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                owner = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and _pid_alive(owner):
                raise SystemExit(f"a MARK server (pid {owner}) is already running for {db}")
            lock.unlink(missing_ok=True)  # stale lock from a crashed server
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        return lock
    raise SystemExit(f"could not acquire the server lock for {db}")


def _release_lock(lock: Path) -> None:
    """Remove *lock* only if this process owns it."""
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe; unknown means alive so a lock is never stolen."""
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (OSError, ValueError):  # platforms without signal-0 probing
        return True
    return True


def _write_state(path: Path, data: dict[str, Any]) -> None:
    """Atomically write the 0600 state file (created private, then renamed in)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle)
    os.replace(tmp, path)
