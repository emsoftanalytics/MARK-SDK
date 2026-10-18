"""``mark serve``: a loopback-only HTTP server that keeps MARK warm.

Safety invariants:

* one server per database, enforced by an OS file lock (``flock`` on POSIX,
  ``msvcrt.locking`` on Windows) on ``X.db.server.lock``. The kernel drops
  the lock when the process dies (even on kill -9), so there is no stale
  lock to reclaim and no check-then-create race; the file holds the pid
  only for the "already running" message;
* the index is warmed in the background as the server starts, so the first
  served recall does not pay the cold-start cost;
* an authenticated ``{"cmd": "stop"}`` request shuts the server down cleanly
  (see :func:`mark._cli.client.stop_server`);
* the server only binds to loopback, every POST must carry the random bearer
  token, and bodies are capped at ``MAX_REQUEST_BYTES``;
* the token is published only through a 0600 state file, and ``GET /hello``
  answers a nonce with an HMAC proof so clients can authenticate the server
  before revealing the token (see :func:`mark._cli.client._live_server`).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import socket
import sys
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
from mark._private_fs import ensure_private_dir

logger = logging.getLogger("mark.cli")


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
    # Warm concurrently with serving: early requests simply wait for the same index lock.
    threading.Thread(target=_warm, args=(engine,), name="mark-serve-warm", daemon=True).start()

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
                if request.get("cmd") == "stop":  # handled here: stopping is not an engine command
                    self._reply(200, {"ok": True, "result": {"stopping": True, "pid": os.getpid()}})
                    _shutdown()
                    return
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
        _save_snapshots(engine)  # the next process starts from a snapshot covering every write
        engine.close()
        # Remove the state file only if it is still ours (never another server's).
        try:
            if json.loads(state.read_text()).get("token") == token:
                state.unlink()
        except (OSError, ValueError):
            pass
        _release_lock(lock)


class _ServerLock:
    """An open, OS-locked lock file; the lock lives exactly as long as the descriptor."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self.fd = fd


def _acquire_lock(db: Path) -> _ServerLock:
    """Take the per-database server lock, or exit when another live server holds it."""
    lock = _lock_path(db)
    ensure_private_dir(lock.parent)
    for _ in range(3):
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _lock_fd(fd)
        except OSError:
            owner = _read_owner(fd)
            os.close(fd)
            raise SystemExit(f"a MARK server{f' (pid {owner})' if owner else ''} is already running for {db}")
        # The previous holder may have unlinked the file between our open and lock:
        # only a lock on the file that is still at the path counts.
        try:
            same = os.path.samestat(os.fstat(fd), os.stat(lock))
        except OSError:
            same = False
        if not same:
            os.close(fd)
            continue
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        return _ServerLock(lock, fd)
    raise SystemExit(f"could not acquire the server lock for {db}")


def _release_lock(lock: _ServerLock) -> None:
    """Remove the lock file (while still holding it, so no one else can lose it), then unlock."""
    try:
        lock.path.unlink()
    except OSError:
        pass
    try:
        os.close(lock.fd)  # closing the descriptor releases the OS lock
    except OSError:
        pass


def _lock_fd(fd: int) -> None:
    """Non-blocking exclusive lock on *fd*; raises OSError when it is held elsewhere."""
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _read_owner(fd: int) -> str:
    try:
        return os.pread(fd, 32, 0).decode("ascii", "replace").strip() if hasattr(os, "pread") else ""
    except OSError:  # e.g. the byte range is locked (Windows)
        return ""


def _warm(engine: Any) -> None:
    """Preload the index for every agent; failures only cost the speed-up."""
    warm_up = getattr(engine.runtime, "warm_up", None)
    if warm_up is None:
        return
    import time

    started = time.monotonic()
    try:
        count = warm_up()
    except Exception:  # noqa: BLE001 - e.g. shut down while warming
        logger.debug("mark serve: index warm-up skipped", exc_info=True)
        return
    print(f"MARK index warm: {count} memories in {time.monotonic() - started:.1f}s", flush=True)


def _save_snapshots(engine: Any) -> None:
    """Refresh stale vector snapshots on a graceful stop; failures only cost the next cold start."""
    save = getattr(engine.runtime, "save_vector_snapshots", None)
    if save is None:
        return
    try:
        save()
    except Exception:  # noqa: BLE001 - e.g. disk full: never block the shutdown
        logger.warning("mark serve: could not save the vector snapshot", exc_info=True)


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe; unknown means alive so a lock is never stolen."""
    if pid == os.getpid() or sys.platform == "win32":  # os.kill(pid, 0) terminates on Windows
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
    ensure_private_dir(path.parent)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle)
    os.replace(tmp, path)
