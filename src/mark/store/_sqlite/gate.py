"""Statement gate: lets ``close()`` wait for in-flight SQLite statements.

Why: the store's connections are opened with ``check_same_thread=False`` and
shared with background threads (ContextWatch prefetch, fact extraction,
servers). Closing a ``sqlite3.Connection`` while another thread is inside
``execute()``/``fetchone()`` on it frees memory that thread is still using:
the process dies with SIGSEGV (0.2.0a10 integration re-test, issue 1).

Every read and write enters the gate first. ``begin_close()`` flips the
gate to *closing* — later statements raise ``StoreClosedError`` — and waits,
bounded, for the statements already running. A statement still running when
the wait gives up keeps its connection: the gate closes that connection
itself when the statement leaves, so no connection is ever closed
mid-statement.
"""
from __future__ import annotations

import sqlite3
import threading
import time

from mark.errors import StoreClosedError


class StatementGate:
    """Counts running statements per connection; coordinates a safe close."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._closing = False
        self._active = 0
        self._busy: dict[int, list] = {}          # id(conn) -> [conn, running statements]
        self._orphans: dict[int, sqlite3.Connection] = {}  # close when their statement ends
        self._local = threading.local()            # per-thread nesting depth

    # ── statements ────────────────────────────────────────────────────────────

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def enter(self, path: str) -> None:
        """Admit one statement on this thread or raise ``StoreClosedError``.

        A thread already inside a statement (a write helper calling a read
        helper) is admitted while closing, so the outer operation can finish.
        """
        with self._cond:
            if self._closing and self._depth() == 0:
                raise StoreClosedError(path)
            self._active += 1
            self._local.depth = self._depth() + 1

    def bind(self, conn: sqlite3.Connection) -> None:
        """Record that the statement entered on this thread uses *conn*."""
        with self._cond:
            entry = self._busy.setdefault(id(conn), [conn, 0])
            entry[1] += 1

    def leave(self, conn: sqlite3.Connection | None) -> None:
        """End one statement; close *conn* now if close() left it to us."""
        orphan = None
        with self._cond:
            self._active -= 1
            self._local.depth = self._depth() - 1
            if conn is not None:
                entry = self._busy.get(id(conn))
                if entry is not None:
                    entry[1] -= 1
                    if entry[1] <= 0:
                        del self._busy[id(conn)]
                        orphan = self._orphans.pop(id(conn), None)
            self._cond.notify_all()
        if orphan is not None:
            _close_quietly(orphan)

    # ── close ─────────────────────────────────────────────────────────────────

    def begin_close(self, timeout: float = 5.0) -> None:
        """Refuse new statements and wait (bounded) for the running ones.

        Statements on the calling thread are not waited for (they cannot
        finish while we block). When a running statement outlives *timeout*
        it is interrupted (``sqlite3.Connection.interrupt`` is thread-safe)
        and given one more second before its connection is orphaned.
        """
        with self._cond:
            self._closing = True
            own = self._depth()
            deadline = time.monotonic() + max(0.0, timeout)
            self._wait_until(lambda: self._active <= own, deadline)
            if self._active > own:
                for conn, _ in list(self._busy.values()):
                    try:
                        conn.interrupt()
                    except sqlite3.Error:  # pragma: no cover - already closed
                        pass
                self._wait_until(lambda: self._active <= own, time.monotonic() + 1.0)

    def _wait_until(self, done, deadline: float) -> None:
        while not done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cond.wait(remaining)

    def close_or_orphan(self, conn: sqlite3.Connection) -> None:
        """Close *conn* unless a statement is using it; then it closes on leave()."""
        with self._cond:
            if id(conn) in self._busy:
                self._orphans[id(conn)] = conn
                return
        _close_quietly(conn)

    @property
    def closing(self) -> bool:
        return self._closing


def _close_quietly(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except sqlite3.Error:  # pragma: no cover - best effort
        pass
