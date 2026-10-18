"""Thread-safe SQLite store for fragments, blocks, documents, nodes, and edges.

This is the public module: it defines :class:`LocalMemoryStore`, composed from
focused mixins in :mod:`mark.store._sqlite` (one per concern), and re-exports
the module-level helpers and constants that used to live here so existing
``from mark.store.sqlite import ...`` imports keep working.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from mark._private_fs import ensure_private_db_file, ensure_private_dir
from mark.security import EncryptionProvider, NoOpEncryptionProvider
from mark.store._sqlite.blocks import BlockMixin
from mark.store._sqlite.codec import (  # noqa: F401 - re-exported for compatibility
    _FRAGMENT_COLUMNS,
    _FTS_META_KEY,
    _FTS_TABLE,
    _IDS_CACHE_MAX_AGE,
    _IDS_CACHE_SIZE,
    _RETRIEVABLE_STATES,
    _RETRIEVABLE_VALUES,
    _SQL_CHUNK,
    _TTL_LIVE_SQL,
    _UPSERT_FRAGMENT_SQL,
    ListenerCallback,
    _as_bytes,
    _chunks,
    _decode_embedding,
    _dt_to_text,
    _escape_like,
    _fact_key,
    _fts_match_expression,
    _optional_str,
    _pack_embedding,
    _parse_dt,
    _parse_optional_dt,
    _probe_fts5,
    _table_columns,
    _unpack_embedding,
)
from mark.store._sqlite.connection import ConnectionMixin
from mark.store._sqlite.document_index import DocumentIndexMixin
from mark.store._sqlite.encryption_guard import EncryptionGuardMixin
from mark.store._sqlite.events import EventMixin
from mark.store._sqlite.fact_jobs import FactJobMixin
from mark.store._sqlite.fact_import import FactImportMixin
from mark.store._sqlite.facts import FactMixin
from mark.store._sqlite.facts_multi import FactMultiMixin
from mark.store._sqlite.fragment_reads import FragmentReadMixin
from mark.store._sqlite.fragments import FragmentWriteMixin
from mark.store._sqlite.graph import GraphMixin
from mark.store._sqlite.lifecycle import LifecycleStoreMixin
from mark.store._sqlite.mentions import MentionMixin
from mark.store._sqlite.rows import RowDecodingMixin
from mark.store._sqlite.schema import SchemaMixin
from mark.store._sqlite.search import SearchMixin

logger = logging.getLogger(__name__)

__all__ = ["ListenerCallback", "LocalMemoryStore"]


class LocalMemoryStore(
    LifecycleStoreMixin,
    FragmentWriteMixin,
    FragmentReadMixin,
    SearchMixin,
    MentionMixin,
    BlockMixin,
    GraphMixin,
    DocumentIndexMixin,
    FactMixin,
    FactMultiMixin,
    FactImportMixin,
    FactJobMixin,
    EventMixin,
    RowDecodingMixin,
    SchemaMixin,
    EncryptionGuardMixin,
    ConnectionMixin,
):
    """Thread-safe SQLite store for fragments, blocks, documents, nodes, and edges.

    Concurrency model:
      - file databases use one connection per thread (WAL mode); writes are
        serialised by a process-level lock;
      - ``:memory:`` databases share a single connection, so *every* access —
        reads included — is serialised by the same re-entrant lock.

    Fragment content (and fact values) are encrypted at rest by
    ``encryption``; the FTS5 lexical index exists only when it is a no-op,
    since it would otherwise store plaintext. Opening an encrypted store
    without its key (or with a wrong one), or a plaintext store holding data
    with a key, raises a :class:`mark.errors.StoreEncryptionError`.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        encryption: EncryptionProvider | None = None,
    ) -> None:
        self.db_path = str(path)
        self.encryption = encryption or NoOpEncryptionProvider()
        # Re-entrant: write helpers may call other locked helpers.
        self._lock = threading.RLock()
        self._local = threading.local()  # per-thread connection + generation
        self._generation = 0  # bumped by close() to invalidate thread-local handles
        self._conns: list[sqlite3.Connection] = []
        self._listeners: list[ListenerCallback] = []
        # retrievable_ids() cache: filter key -> scope_delta._Entry (ids, version, expiry, refresh).
        self._ids_cache: dict = {}
        self._fts_supported = False
        self._fts_enabled = False
        self._shared_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            # Each :memory: connection is a separate database, so share one.
            self._shared_conn = self._open(":memory:")
        else:
            # Private by default: memory.db (and -wal/-shm) 0600, a new parent 0700.
            ensure_private_dir(Path(self.db_path).expanduser().parent)
            ensure_private_db_file(self.db_path)
        self._init_schema()
