"""Connection management, locking, transactions and change listeners.

Concurrency model (see also ``LocalMemoryStore``):
  - file databases: one connection per thread (WAL mode, so readers never
    block the writer); every write is serialised by the store's process-level
    re-entrant lock ``_lock``;
  - ``:memory:`` databases: a single shared connection (a second connection
    would see a different, empty database), so every access — reads
    included — is serialised by the same lock.

``_writing`` is the only way the store writes: one lock-held transaction that
commits on success and rolls back on any exception. Structural writes also
bump the persisted ``write_version`` counter inside that same transaction.

``close()`` is final: every later read or write raises
:class:`mark.errors.StoreClosedError` instead of silently reopening a
connection (which used to leak file descriptors and let a background thread
keep writing to a store its owner had shut down).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Iterator

from mark.errors import StoreClosedError
from mark.security import EncryptionProvider
from mark.store._sqlite.codec import ListenerCallback

# Keep the historical logger name so log filters keep working after the split.
logger = logging.getLogger("mark.store.sqlite")


class ConnectionMixin:
    """Per-thread connections, read/write context managers and listeners."""

    # Instance state, initialised by LocalMemoryStore.__init__.
    db_path: str
    encryption: EncryptionProvider
    _lock: threading.RLock
    _local: threading.local
    _generation: int
    _conns: list[sqlite3.Connection]
    _listeners: list[ListenerCallback]
    _shared_conn: sqlite3.Connection | None

    def close(self) -> None:
        """Close every connection this store opened, on any thread; idempotent.

        After close the store refuses further use (``StoreClosedError``).
        """
        with self._lock:
            self._closed = True
            if self._shared_conn is not None:
                self._shared_conn.close()
                self._shared_conn = None
            for conn in self._conns:
                try:
                    conn.close()
                except sqlite3.Error:  # pragma: no cover - best effort
                    pass
            self._conns.clear()
            # Invalidate thread-local handles held by other threads.
            self._generation += 1
            self._local.conn = None

    # ── listeners ─────────────────────────────────────────────────────────────

    def add_listener(self, callback: ListenerCallback) -> None:
        """Register callback(event, fragment_id) for "delete"/"unretrievable" events.

        Listener exceptions are logged and never propagate to the caller.
        """
        with self._lock:
            self._listeners.append(callback)

    def _emit(self, event: str, fragment_ids: Iterable[str]) -> None:
        """Notify listeners. Called after the write transaction has committed."""
        listeners = list(self._listeners)
        if not listeners:
            return
        for fragment_id in fragment_ids:
            for callback in listeners:
                try:
                    callback(event, fragment_id)
                except Exception:  # noqa: BLE001 - listeners must never break the store
                    logger.exception("MARK store listener failed for %s(%s)", event, fragment_id)

    # ── connections ───────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return this thread's connection, opening one if needed.

        A thread-local handle is reused only while its generation matches the
        store's; close() bumps the generation so stale handles are reopened.
        """
        if self._shared_conn is not None:
            return self._shared_conn
        # getattr: _closed is set by close(); stores built before it existed default to open.
        if getattr(self, "_closed", False) or self.db_path == ":memory:":
            raise StoreClosedError(self.db_path)
        local = self._local
        conn = getattr(local, "conn", None)
        if conn is None or getattr(local, "generation", -1) != self._generation:
            conn = self._open(self.db_path)
            with self._lock:
                if getattr(self, "_closed", False):  # closed while we were opening
                    conn.close()
                    raise StoreClosedError(self.db_path)
                self._conns.append(conn)
                local.generation = self._generation
            local.conn = conn
        return conn

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for reads; serialised when the connection is shared."""
        if self._shared_conn is not None:
            with self._lock:
                conn = self._shared_conn
                if conn is None:  # pragma: no cover - closed concurrently
                    raise StoreClosedError(self.db_path)
                yield conn
        else:
            yield self._conn()

    @contextmanager
    def _writing(self, *, structural: bool = True) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside one serialised transaction (commit or rollback).

        Structural writes (anything that can change which fragments are
        retrievable) bump the persisted ``write_version`` in the same
        transaction, so caches in this or any other process can validate
        cheaply. Access/importance/edge-weight/event writes pass
        ``structural=False``.
        """
        with self._lock:
            conn = self._conn()
            if structural:
                self._reset_write_notes()  # unknown change set unless the writer notes its ids
            try:
                yield conn
                if structural:
                    conn.execute(
                        "UPDATE schema_info SET value = CAST(value AS INTEGER) + 1"
                        " WHERE key = 'write_version'"
                    )
                    # Same transaction: which fragments changed, so id caches can
                    # apply a delta instead of rescanning (see scope_delta).
                    self._log_structural_write(conn)
            except BaseException:
                conn.rollback()
                raise
            conn.commit()

    # Overridden by ScopeCacheMixin (write log for incremental id caches).
    def _reset_write_notes(self) -> None:
        pass

    def _log_structural_write(self, conn: sqlite3.Connection) -> None:
        pass

    def _fetchall(self, sql: str, params: Iterable[object] = ()) -> list[sqlite3.Row]:
        with self._reading() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def _fetchone(self, sql: str, params: Iterable[object] = ()) -> sqlite3.Row | None:
        with self._reading() as conn:
            return conn.execute(sql, tuple(params)).fetchone()

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        """Open a connection with the store's pragmas (WAL for file databases)."""
        conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if path != ":memory:":
            ConnectionMixin._enable_wal(conn)
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-32000")
        return conn

    @staticmethod
    def _enable_wal(conn: sqlite3.Connection, attempts: int = 50) -> None:
        """Switch to WAL, retrying while another connection holds the lock.

        Several connections opening a brand-new database at once can race on
        the journal-mode switch, and SQLite reports "database is locked"
        without consulting the busy timeout, so back off and retry here.
        """
        for attempt in range(attempts):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == attempts - 1:
                    raise
                time.sleep(0.01 * (attempt + 1))
