"""Thread-safe SQLite store for fragments, blocks, documents, nodes, and edges."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import threading
import time
import uuid
from array import array
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from mark.security import ContentHasher, EncryptionProvider, NoOpEncryptionProvider
from mark.types import BlockLink, BlockStatus, MemoryBlock, MemoryDocument, MemoryEdge, MemoryFragment, MemoryNode
from mark.types import MemoryScope, MemoryState, MemoryTier  # MemoryTier used in list_by_agent filter
from mark.types.graph import EdgeRelation, NodeType

logger = logging.getLogger(__name__)

# States a fragment must be in to surface in retrieval.
_RETRIEVABLE_STATES: tuple[MemoryState, ...] = (
    MemoryState.UNVERIFIED,
    MemoryState.VERIFIED,
    MemoryState.PROMOTED,
)
_RETRIEVABLE_VALUES = frozenset(state.value for state in _RETRIEVABLE_STATES)

# Stay well below SQLite's default host-parameter limit (999 on old builds).
_SQL_CHUNK = 900

# A fragment row is live unless a TTL is set and has elapsed. Rows whose
# created_at cannot be parsed by SQLite are kept (fail open, like is_expired).
# retrievable_ids() cache bounds: entries are also invalidated by write_version
# and by the next TTL expiry among the agent's live fragments.
_IDS_CACHE_SIZE = 64
_IDS_CACHE_MAX_AGE = 300.0

_TTL_LIVE_SQL = (
    " AND (ttl_seconds IS NULL OR julianday(created_at) IS NULL"
    " OR (julianday('now') - julianday(created_at)) * 86400.0 <= ttl_seconds)"
)

_FRAGMENT_COLUMNS = (
    "id", "content", "block_id", "scope", "tier", "agent_id", "session_id",
    "importance", "state", "confidence", "ttl_seconds", "created_at",
    "updated_at", "last_accessed_at", "tags", "source", "metadata",
    "embedding", "contradiction_ids", "embedding_blob",
)

_UPSERT_FRAGMENT_SQL = (
    f"INSERT INTO fragments ({', '.join(_FRAGMENT_COLUMNS)}) VALUES ("
    + ", ".join(f":{name}" for name in _FRAGMENT_COLUMNS)
    + ") ON CONFLICT(id) DO UPDATE SET "
    + ", ".join(f"{name} = excluded.{name}" for name in _FRAGMENT_COLUMNS if name != "id")
)

_FTS_TABLE = "fragments_fts"
_FTS_META_KEY = "fts_backfilled"

ListenerCallback = Callable[[str, str], None]


class LocalMemoryStore:
    """Thread-safe SQLite store for fragments, blocks, documents, nodes, and edges.

    Concurrency model:
      - file databases use one connection per thread (WAL mode); writes are
        serialised by a process-level lock;
      - ``:memory:`` databases share a single connection, so *every* access —
        reads included — is serialised by the same re-entrant lock.
    """

    CURRENT_SCHEMA_VERSION = 4

    # Column additions per schema version: (table, column, declaration).
    # Applied by inspecting PRAGMA table_info, never by trusting the stamped
    # version or row counts, so partially-migrated databases converge.
    # Column order must match the _DDL_* definitions (ALTER TABLE appends).
    _COLUMN_MIGRATIONS: dict[int, list[tuple[str, str, str]]] = {
        # 1: baseline
        # 2: graph-scoped memory blocks — block provenance columns, block-scoped edges.
        2: [
            ("blocks", "session_id", "TEXT"),
            ("blocks", "status", "TEXT DEFAULT 'open'"),
            ("blocks", "content_hash", "TEXT"),
            ("blocks", "prev_block_id", "TEXT"),
            ("blocks", "prev_block_hash", "TEXT"),
            ("blocks", "sealed_at", "TEXT"),
            ("edges", "block_id", "TEXT"),
        ],
        # 3: binary float32 embeddings (JSON column kept for legacy rows).
        3: [
            ("fragments", "embedding_blob", "BLOB"),
        ],
    }

    # Non-column statements per schema version. All must be idempotent.
    _MIGRATIONS: dict[int, list[str]] = {
        2: [
            """
            CREATE TABLE IF NOT EXISTS block_nodes (
                block_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (block_id, node_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS block_links (
                id TEXT PRIMARY KEY,
                source_block_id TEXT NOT NULL,
                target_block_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                agent_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """,
        ],
        # 3: indexes are created from _INDEXES after migrations; FTS backfill
        #    is driven by the fts_backfilled meta key.
        3: [],
        # 4: bi-temporal facts (subject, attribute) -> value with validity intervals.
        4: [
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                attribute TEXT NOT NULL,
                attribute_key TEXT NOT NULL,
                value BLOB NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                recorded_at TEXT NOT NULL,
                retracted_at TEXT,
                source_fragment_id TEXT,
                metadata TEXT DEFAULT '{}'
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_fact_key ON facts(agent_id, subject_key, attribute_key, valid_from)",
            "CREATE INDEX IF NOT EXISTS idx_fact_current ON facts(agent_id, valid_to, retracted_at)",
        ],
    }

    _DDL_SCHEMA_INFO = """
        CREATE TABLE IF NOT EXISTS schema_info (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """
    _DDL_BLOCKS = """
        CREATE TABLE IF NOT EXISTS blocks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            block_type TEXT DEFAULT 'ltm',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            session_id TEXT,
            status TEXT DEFAULT 'open',
            content_hash TEXT,
            prev_block_id TEXT,
            prev_block_hash TEXT,
            sealed_at TEXT
        )
    """
    _DDL_BLOCK_NODES = """
        CREATE TABLE IF NOT EXISTS block_nodes (
            block_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY (block_id, node_id)
        )
    """
    _DDL_BLOCK_LINKS = """
        CREATE TABLE IF NOT EXISTS block_links (
            id TEXT PRIMARY KEY,
            source_block_id TEXT NOT NULL,
            target_block_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            agent_id TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _DDL_FRAGMENTS = """
        CREATE TABLE IF NOT EXISTS fragments (
            id TEXT PRIMARY KEY,
            content BLOB NOT NULL,
            block_id TEXT,
            scope TEXT NOT NULL,
            tier TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            session_id TEXT,
            importance REAL DEFAULT 0.5,
            state TEXT DEFAULT 'raw',
            confidence REAL DEFAULT 1.0,
            ttl_seconds INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT,
            tags TEXT DEFAULT '[]',
            source TEXT,
            metadata TEXT DEFAULT '{}',
            embedding TEXT,
            contradiction_ids TEXT DEFAULT '[]',
            embedding_blob BLOB
        )
    """
    _DDL_DOCUMENTS = """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            agent_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            fragment_ids TEXT DEFAULT '[]',
            tags TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _DDL_NODES = """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            label TEXT NOT NULL,
            agent_id TEXT,
            fragment_id TEXT,
            document_id TEXT,
            weight REAL DEFAULT 1.0,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _DDL_EDGES = """
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            agent_id TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            block_id TEXT
        )
    """
    _DDL_SESSION_EVENTS = """
        CREATE TABLE IF NOT EXISTS session_events (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_id TEXT,
            event_type TEXT NOT NULL,
            fragment_id TEXT,
            query TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    # Lexical channel. Kept in sync from Python (not triggers) because the
    # fragments.content column may hold ciphertext. rowid == fragments.rowid.
    _DDL_FTS = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
            content,
            tags,
            fragment_id UNINDEXED,
            agent_id UNINDEXED,
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
    """
    _INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_frag_agent   ON fragments(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_state   ON fragments(state)",
        "CREATE INDEX IF NOT EXISTS idx_frag_scope   ON fragments(scope)",
        "CREATE INDEX IF NOT EXISTS idx_frag_tier    ON fragments(tier)",
        "CREATE INDEX IF NOT EXISTS idx_frag_session ON fragments(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_block   ON fragments(block_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_agent_state_imp"
        " ON fragments(agent_id, state, importance DESC)",
        "CREATE INDEX IF NOT EXISTS idx_node_frag    ON nodes(fragment_id)",
        "CREATE INDEX IF NOT EXISTS idx_node_label_agent ON nodes(label, agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_edge_source  ON edges(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_edge_target  ON edges(target_id)",
        "CREATE INDEX IF NOT EXISTS idx_edge_block   ON edges(block_id)",
        "CREATE INDEX IF NOT EXISTS idx_sevt_agent   ON session_events(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_sevt_session ON session_events(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_blink_source ON block_links(source_block_id)",
        "CREATE INDEX IF NOT EXISTS idx_blink_target ON block_links(target_block_id)",
        "CREATE INDEX IF NOT EXISTS idx_bnode_node   ON block_nodes(node_id)",
        "CREATE INDEX IF NOT EXISTS idx_block_agent  ON blocks(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_block_status ON blocks(status)",
    ]

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        encryption: EncryptionProvider | None = None,
    ) -> None:
        self.db_path = str(path)
        self.encryption = encryption or NoOpEncryptionProvider()
        self._lock = threading.RLock()
        self._local = threading.local()
        self._generation = 0
        self._conns: list[sqlite3.Connection] = []
        self._listeners: list[ListenerCallback] = []
        self._ids_cache: dict[tuple, tuple[frozenset[str], int, float]] = {}
        self._fts_supported = False
        self._fts_enabled = False
        self._shared_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._shared_conn = self._open(":memory:")
        else:
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def close(self) -> None:
        """Close every connection this store opened, on any thread."""
        with self._lock:
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
        listeners = list(self._listeners)
        if not listeners:
            return
        for fragment_id in fragment_ids:
            for callback in listeners:
                try:
                    callback(event, fragment_id)
                except Exception:  # noqa: BLE001 - listeners must never break the store
                    logger.exception("MARK store listener failed for %s(%s)", event, fragment_id)

    # ── meta ──────────────────────────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        """Return a value from the schema_info key/value table, or None."""
        row = self._fetchone("SELECT value FROM schema_info WHERE key = ?", (key,))
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a value in the schema_info key/value table."""
        with self._writing(structural=False) as conn:
            conn.execute(
                "INSERT INTO schema_info (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    @property
    def fts_available(self) -> bool:
        """True when the FTS5 lexical channel is active.

        Requires SQLite FTS5 support AND an unencrypted store — plaintext is
        never indexed when fragment content is encrypted at rest.
        """
        return self._fts_enabled

    # ── fragments ─────────────────────────────────────────────────────────────

    def store(self, fragment: MemoryFragment) -> str:
        """Insert or update a fragment row (content encrypted at rest).

        Raises ValueError when the id already belongs to a different agent.
        """
        return self.store_many([fragment])[0]

    def store_many(self, fragments: Iterable[MemoryFragment]) -> list[str]:
        """Insert or update many fragments in one transaction (all or nothing).

        Same semantics as store() per fragment. Raises ValueError (and writes
        nothing) when any id already belongs to a different agent.
        """
        unretrievable: list[str] = []
        ids: list[str] = []
        with self._writing() as conn:
            for fragment in fragments:
                fragment_id, became_unretrievable = self._store_row(conn, fragment)
                ids.append(fragment_id)
                if became_unretrievable:
                    unretrievable.append(fragment_id)
        if unretrievable:
            self._emit("unretrievable", unretrievable)
        return ids

    def _store_row(self, conn: sqlite3.Connection, fragment: MemoryFragment) -> tuple[str, bool]:
        fragment = ContentHasher.attach(fragment, self.encryption)
        encrypted_content = self.encryption.encrypt_str(fragment.content)
        params = {
            "id": fragment.id,
            "content": encrypted_content,
            "block_id": fragment.block_id,
            "scope": fragment.scope.value,
            "tier": fragment.tier.value,
            "agent_id": fragment.agent_id,
            "session_id": fragment.session_id,
            "importance": fragment.importance,
            "state": fragment.state.value,
            "confidence": fragment.confidence,
            "ttl_seconds": fragment.ttl_seconds,
            "created_at": fragment.created_at.isoformat(),
            "updated_at": fragment.updated_at.isoformat(),
            "last_accessed_at": _dt_to_text(fragment.last_accessed_at),
            "tags": json.dumps(fragment.tags),
            "source": fragment.source,
            "metadata": json.dumps(fragment.metadata),
            # Embeddings live in embedding_blob; the JSON column is legacy-only.
            "embedding": None,
            "contradiction_ids": json.dumps(fragment.contradiction_ids),
            "embedding_blob": _pack_embedding(fragment.embedding),
        }
        existing = conn.execute(
            "SELECT rowid, agent_id, state FROM fragments WHERE id = ?", (fragment.id,)
        ).fetchone()
        if existing is not None and str(existing["agent_id"]) != fragment.agent_id:
            raise ValueError(
                f"Fragment id {fragment.id!r} already belongs to another agent; "
                "refusing to overwrite it."
            )
        cursor = conn.execute(_UPSERT_FRAGMENT_SQL, params)
        rowid = int(existing["rowid"]) if existing is not None else int(cursor.lastrowid or 0)
        became_unretrievable = (
            existing is not None
            and str(existing["state"]) in _RETRIEVABLE_VALUES
            and fragment.state.value not in _RETRIEVABLE_VALUES
        )
        if self._fts_enabled:
            self._fts_upsert(conn, rowid, fragment.id, fragment.agent_id, fragment.content, fragment.tags)
        return fragment.id, became_unretrievable

    def get(self, fragment_id: str) -> MemoryFragment | None:
        """Fetch a fragment by id, or None.

        A fragment whose content no longer matches its stored hash is returned
        as CONTRADICTED, and that state is persisted so it stops surfacing.
        """
        row = self._fetchone("SELECT * FROM fragments WHERE id = ?", (fragment_id,))
        if row is None:
            return None
        fragment, tampered = self._decode_fragment(dict(row))
        if tampered:
            self._persist_tampered([fragment.id])
        return fragment

    def get_many(self, ids: Iterable[str]) -> dict[str, MemoryFragment]:
        """Fetch many fragments by id in as few queries as possible (chunks of 900)."""
        unique = list(dict.fromkeys(ids))
        result: dict[str, MemoryFragment] = {}
        tampered_ids: list[str] = []
        for chunk in _chunks(unique, _SQL_CHUNK):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._fetchall(f"SELECT * FROM fragments WHERE id IN ({placeholders})", chunk)
            for row in rows:
                fragment, tampered = self._decode_fragment(dict(row))
                result[fragment.id] = fragment
                if tampered:
                    tampered_ids.append(fragment.id)
        if tampered_ids:
            self._persist_tampered(tampered_ids)
        return result

    def list_by_agent(
        self,
        agent_id: str,
        *,
        scope: MemoryScope | None = None,
        states: Iterable[MemoryState] | None = None,
        tier: MemoryTier | None = None,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        limit: int | None = 200,
    ) -> list[MemoryFragment]:
        """Query fragments for agent_id with optional session, tag, tier, scope, and block filters.

        session_id      — exact match on session_id column
        session_prefix  — prefix match: session_id LIKE '<prefix>%' (wildcards escaped)
        tags            — all listed tags must be present (AND semantics)
        tier            — exact match on tier column
        scope           — exact match on scope column
        block_id        — restrict to fragments belonging to one block
        block_ids       — restrict to fragments belonging to any listed block
        limit           — maximum rows; None returns every match

        Fragments belonging to a QUARANTINED block are always excluded, so a
        corrupted block is isolated without affecting the rest of the agent's
        memory.
        """
        where, params = self._fragment_filters(
            agent_id,
            scope=scope,
            states=states,
            tier=tier,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            block_id=block_id,
            block_ids=block_ids,
        )
        sql = f"SELECT * FROM fragments WHERE {where} ORDER BY importance DESC, created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._fetchall(sql, params)
        return [self._row_to_fragment(dict(row)) for row in rows]

    def retrievable_ids(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
    ) -> set[str]:
        """Return ids of every retrievable fragment matching the filters (no row limit).

        Retrievable means: state in {UNVERIFIED, VERIFIED, PROMOTED}, not a
        member of a QUARANTINED block, and not TTL-expired. Evaluated entirely
        in SQL; no fragment rows are decoded.
        """
        where, params = self._fragment_filters(
            agent_id,
            scope=scope,
            states=_RETRIEVABLE_STATES,
            tier=tier,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            block_id=block_id,
            block_ids=block_ids,
        )
        key = (where, tuple(params))
        version = self._write_version()
        cached = self._ids_cache.get(key)
        if cached is not None and cached[1] == version and time.monotonic() < cached[2]:
            return set(cached[0])
        with self._reading() as conn:
            cursor = conn.cursor()
            cursor.row_factory = None  # plain tuples: ~3x faster than sqlite3.Row
            ids = frozenset(
                row[0] for row in cursor.execute(
                    f"SELECT id FROM fragments WHERE {where}{_TTL_LIVE_SQL}", params
                )
            )
            next_expiry = cursor.execute(
                "SELECT MIN((julianday(created_at) - julianday('now')) * 86400.0 + ttl_seconds)"
                " FROM fragments WHERE agent_id = ? AND ttl_seconds IS NOT NULL"
                " AND julianday(created_at) IS NOT NULL"
                " AND (julianday('now') - julianday(created_at)) * 86400.0 <= ttl_seconds",
                (agent_id,),
            ).fetchone()[0]
        ttl_left = _IDS_CACHE_MAX_AGE if next_expiry is None else min(_IDS_CACHE_MAX_AGE, max(0.0, next_expiry))
        with self._lock:
            if len(self._ids_cache) >= _IDS_CACHE_SIZE:
                self._ids_cache.pop(next(iter(self._ids_cache)))
            self._ids_cache[key] = (ids, version, time.monotonic() + ttl_left)
        return set(ids)

    def _write_version(self) -> int:
        row = self._fetchone("SELECT value FROM schema_info WHERE key = 'write_version'")
        return int(row[0]) if row is not None else -1

    def embeddings_for_agent(self, agent_id: str) -> list[tuple[str, list[float] | None]]:
        """Return (id, embedding-or-None) for every retrievable-state fragment of an agent.

        Cheap projection: only the id and embedding columns are read.
        """
        state_placeholders = ",".join("?" for _ in _RETRIEVABLE_STATES)
        rows = self._fetchall(
            f"SELECT id, embedding_blob, embedding FROM fragments"
            f" WHERE agent_id = ? AND state IN ({state_placeholders}) ORDER BY rowid",
            [agent_id, *(state.value for state in _RETRIEVABLE_STATES)],
        )
        return [(str(row[0]), _decode_embedding(row[1], row[2])) for row in rows]

    def lexical_search(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int = 50,
        include_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 keyword search over fragment content and tags.

        Returns (fragment_id, score) pairs, best first; higher score is better.
        Returns [] when FTS5 is unavailable (or the store is encrypted), when
        the query has no usable tokens, or when SQLite rejects the query.
        Results are not state-filtered: intersect with retrievable_ids() (or
        pass it as include_ids) to restrict to retrievable fragments.
        """
        if not self._fts_enabled or limit <= 0:
            return []
        if include_ids is not None and not include_ids:
            return []
        match = _fts_match_expression(query)
        if match is None:
            return []
        sql = (
            f"SELECT fragment_id, -bm25({_FTS_TABLE}) AS score FROM {_FTS_TABLE}"
            f" WHERE {_FTS_TABLE} MATCH ? AND agent_id = ?"
        )
        params: list[object] = [match, agent_id]
        if include_ids is not None:
            # One JSON parameter filters exactly at any size. Post-filtering a
            # truncated hit list would silently drop in-scope matches whenever
            # many out-of-scope documents rank higher.
            sql += " AND fragment_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(sorted(include_ids)))
        sql += f" ORDER BY bm25({_FTS_TABLE}) LIMIT ?"
        params.append(limit)
        try:
            rows = self._fetchall(sql, params)
        except sqlite3.OperationalError:
            logger.debug("FTS query rejected: %r", match, exc_info=True)
            return []
        return [(str(row[0]), float(row[1])) for row in rows]

    def phrase_exists(self, agent_id: str, phrase: str) -> bool | None:
        """Whether *phrase* (as consecutive tokens) occurs in any of the agent's memories.

        Returns None when it cannot be answered (FTS5 unavailable, encrypted
        store, or no usable tokens), so callers can fall back to other evidence.
        """
        if not self._fts_enabled:
            return None
        tokens = re.findall(r"\w+", phrase.lower())
        if not tokens:
            return None
        match = '"' + " ".join(token.replace('"', '""') for token in tokens) + '"'
        # Only retrievable, unexpired memories count: a quarantined injection
        # attempt or an expired note must not make an unknown subject look known.
        states = ",".join("?" for _ in _RETRIEVABLE_VALUES)
        try:
            row = self._fetchone(
                f"SELECT 1 FROM {_FTS_TABLE} JOIN fragments f ON f.id = {_FTS_TABLE}.fragment_id"
                f" WHERE {_FTS_TABLE} MATCH ? AND {_FTS_TABLE}.agent_id = ?"
                f" AND f.state IN ({states}){_TTL_LIVE_SQL.replace('created_at', 'f.created_at').replace('ttl_seconds', 'f.ttl_seconds')}"
                " LIMIT 1",
                (match, agent_id, *sorted(_RETRIEVABLE_VALUES)),
            )
        except sqlite3.OperationalError:
            logger.debug("FTS phrase query rejected: %r", match, exc_info=True)
            return None
        return row is not None

    def list_sessions(self, agent_id: str) -> list[str]:
        """Return distinct session_ids for agent_id, ordered by most recent activity."""
        rows = self._fetchall(
            """
            SELECT session_id
            FROM fragments
            WHERE agent_id = ? AND session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            """,
            (agent_id,),
        )
        return [str(row[0]) for row in rows]

    def list_all(self) -> list[MemoryFragment]:
        """Return every stored fragment."""
        rows = self._fetchall("SELECT * FROM fragments")
        return [self._row_to_fragment(dict(row)) for row in rows]

    def update_state(self, fragment_id: str, new_state: MemoryState) -> None:
        """Update a fragment's lifecycle state."""
        with self._writing() as conn:
            cursor = conn.execute(
                "UPDATE fragments SET state = ?, updated_at = ? WHERE id = ?",
                (new_state.value, datetime.now(timezone.utc).isoformat(), fragment_id),
            )
            changed = cursor.rowcount > 0
        if changed and new_state.value not in _RETRIEVABLE_VALUES:
            self._emit("unretrievable", [fragment_id])

    def update_embedding(self, fragment_id: str, embedding: list[float]) -> None:
        """Persist a fragment's embedding (float32 blob)."""
        with self._writing(structural=False) as conn:
            conn.execute(
                "UPDATE fragments SET embedding_blob = ?, embedding = NULL WHERE id = ?",
                (_pack_embedding(embedding), fragment_id),
            )

    def delete(self, fragment_id: str) -> bool:
        """Delete by id; returns True when a row was removed."""
        with self._writing() as conn:
            row = conn.execute("SELECT rowid FROM fragments WHERE id = ?", (fragment_id,)).fetchone()
            if row is None:
                return False
            if self._fts_enabled:
                conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE rowid = ?", (int(row[0]),))
            conn.execute("DELETE FROM fragments WHERE id = ?", (fragment_id,))
        self._emit("delete", [fragment_id])
        return True

    def clear(self, agent_id: str | None = None) -> None:
        """Delete fragments, optionally only for one agent."""
        where, params = ("", ()) if agent_id is None else (" WHERE agent_id = ?", (agent_id,))
        with self._writing() as conn:
            ids = [str(row[0]) for row in conn.execute(f"SELECT id FROM fragments{where}", params)]
            if self._fts_enabled:
                if agent_id is None:
                    conn.execute(f"DELETE FROM {_FTS_TABLE}")
                else:
                    conn.execute(
                        f"DELETE FROM {_FTS_TABLE} WHERE rowid IN (SELECT rowid FROM fragments{where})",
                        params,
                    )
            conn.execute(f"DELETE FROM fragments{where}", params)
        self._emit("delete", ids)

    def count(self, agent_id: str) -> int:
        """Return the number of fragments stored for an agent."""
        row = self._fetchone("SELECT COUNT(*) FROM fragments WHERE agent_id = ?", (agent_id,))
        return int(row[0]) if row else 0

    def store_block(self, block: MemoryBlock) -> str:
        """Insert or replace a block row, including provenance/chain columns."""
        with self._writing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO blocks
                    (id, name, agent_id, scope, block_type, metadata, created_at,
                     session_id, status, content_hash, prev_block_id, prev_block_hash, sealed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.id,
                    block.name,
                    block.agent_id,
                    block.scope.value,
                    block.block_type,
                    json.dumps(block.metadata),
                    block.created_at.isoformat(),
                    block.session_id,
                    block.status.value,
                    block.content_hash,
                    block.prev_block_id,
                    block.prev_block_hash,
                    _dt_to_text(block.sealed_at),
                ),
            )
        return block.id

    def get_block(self, block_id: str) -> MemoryBlock | None:
        """Fetch a block by id, or None."""
        row = self._fetchone("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return None if row is None else self._row_to_block(dict(row))

    def get_block_by_name(self, name: str, agent_id: str) -> MemoryBlock | None:
        """Find an agent's block by name, or None."""
        row = self._fetchone("SELECT * FROM blocks WHERE name = ? AND agent_id = ?", (name, agent_id))
        return None if row is None else self._row_to_block(dict(row))

    def list_blocks(self, agent_id: str) -> list[MemoryBlock]:
        """Return all blocks."""
        rows = self._fetchall("SELECT * FROM blocks WHERE agent_id = ? ORDER BY created_at DESC", (agent_id,))
        return [self._row_to_block(dict(row)) for row in rows]

    def fragments_for_block(self, block_id: str) -> list[MemoryFragment]:
        """Return fragments belonging to a block."""
        rows = self._fetchall("SELECT * FROM fragments WHERE block_id = ? ORDER BY importance DESC", (block_id,))
        return [self._row_to_fragment(dict(row)) for row in rows]

    # ── graph-scoped block API ────────────────────────────────────────────────

    def assign_fragment_block(self, fragment_id: str, block_id: str | None) -> None:
        """Move a fragment into a block (or out, with block_id=None).

        Raises ValueError when the fragment currently belongs to a SEALED
        block and the move would take it out of that block.
        """
        with self._writing() as conn:
            row = conn.execute(
                """
                SELECT f.block_id AS block_id, b.status AS status
                FROM fragments f LEFT JOIN blocks b ON b.id = f.block_id
                WHERE f.id = ?
                """,
                (fragment_id,),
            ).fetchone()
            if (
                row is not None
                and row["block_id"] is not None
                and row["block_id"] != block_id
                and row["status"] == BlockStatus.SEALED.value
            ):
                raise ValueError(
                    f"Fragment {fragment_id!r} belongs to sealed block {row['block_id']!r}; "
                    "sealed blocks are immutable."
                )
            conn.execute(
                "UPDATE fragments SET block_id = ?, updated_at = ? WHERE id = ?",
                (block_id, datetime.now(timezone.utc).isoformat(), fragment_id),
            )

    def add_node_to_block(self, block_id: str, node_id: str) -> None:
        """Register a node as a member of a block (idempotent, many-to-many)."""
        with self._writing() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO block_nodes (block_id, node_id, added_at) VALUES (?, ?, ?)",
                (block_id, node_id, datetime.now(timezone.utc).isoformat()),
            )

    def remove_node_from_block(self, block_id: str, node_id: str) -> bool:
        """Remove a node's membership in a block. Returns True if removed."""
        with self._writing() as conn:
            cursor = conn.execute(
                "DELETE FROM block_nodes WHERE block_id = ? AND node_id = ?",
                (block_id, node_id),
            )
        return cursor.rowcount > 0

    def nodes_in_block(self, block_id: str) -> list[MemoryNode]:
        """Return all member nodes of a block."""
        rows = self._fetchall(
            """
            SELECT n.* FROM nodes n
            JOIN block_nodes bn ON bn.node_id = n.id
            WHERE bn.block_id = ?
            ORDER BY bn.added_at
            """,
            (block_id,),
        )
        return [self._row_to_node(dict(row)) for row in rows]

    def block_ids_for_node(self, node_id: str) -> list[str]:
        """Return ids of every block this node belongs to."""
        rows = self._fetchall("SELECT block_id FROM block_nodes WHERE node_id = ? ORDER BY added_at", (node_id,))
        return [str(row[0]) for row in rows]

    def edges_in_block(self, block_id: str) -> list[MemoryEdge]:
        """Return all edges scoped to a block."""
        rows = self._fetchall("SELECT * FROM edges WHERE block_id = ? ORDER BY created_at", (block_id,))
        return [self._row_to_edge(dict(row)) for row in rows]

    def store_block_link(self, link: BlockLink) -> str:
        """Insert or replace a directed link between two blocks."""
        with self._writing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO block_links
                    (id, source_block_id, target_block_id, relation, weight,
                     agent_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.source_block_id,
                    link.target_block_id,
                    link.relation,
                    link.weight,
                    link.agent_id,
                    json.dumps(link.metadata),
                    link.created_at.isoformat(),
                ),
            )
        return link.id

    def links_from_block(self, block_id: str) -> list[BlockLink]:
        """Return outgoing (forward) links from a block."""
        rows = self._fetchall(
            "SELECT * FROM block_links WHERE source_block_id = ? ORDER BY created_at", (block_id,)
        )
        return [self._row_to_block_link(dict(row)) for row in rows]

    def links_to_block(self, block_id: str) -> list[BlockLink]:
        """Return incoming (backward) links to a block."""
        rows = self._fetchall(
            "SELECT * FROM block_links WHERE target_block_id = ? ORDER BY created_at", (block_id,)
        )
        return [self._row_to_block_link(dict(row)) for row in rows]

    def delete_block_link(self, link_id: str) -> bool:
        """Delete a block link by id. Returns True if a row was deleted."""
        with self._writing() as conn:
            cursor = conn.execute("DELETE FROM block_links WHERE id = ?", (link_id,))
        return cursor.rowcount > 0

    def list_blocks_by_status(self, agent_id: str, status: BlockStatus) -> list[MemoryBlock]:
        """Return an agent's blocks filtered by lifecycle status."""
        rows = self._fetchall(
            "SELECT * FROM blocks WHERE agent_id = ? AND status = ? ORDER BY created_at",
            (agent_id, status.value),
        )
        return [self._row_to_block(dict(row)) for row in rows]

    def latest_sealed_block(self, agent_id: str, *, exclude_id: str | None = None) -> MemoryBlock | None:
        """Return the most recently sealed block in this agent's chain, if any."""
        sql = "SELECT * FROM blocks WHERE agent_id = ? AND status = ? AND sealed_at IS NOT NULL"
        params: list[object] = [agent_id, BlockStatus.SEALED.value]
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        sql += " ORDER BY sealed_at DESC, created_at DESC LIMIT 1"
        row = self._fetchone(sql, params)
        return None if row is None else self._row_to_block(dict(row))

    def sealed_chain(self, agent_id: str) -> list[MemoryBlock]:
        """Return this agent's sealed blocks in chain (seal-time) order."""
        rows = self._fetchall(
            """
            SELECT * FROM blocks
            WHERE agent_id = ? AND sealed_at IS NOT NULL AND status IN (?, ?)
            ORDER BY sealed_at, created_at
            """,
            (agent_id, BlockStatus.SEALED.value, BlockStatus.QUARANTINED.value),
        )
        return [self._row_to_block(dict(row)) for row in rows]

    def store_document(self, doc: MemoryDocument) -> str:
        """Persist a document."""
        with self._writing() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.title,
                    doc.content,
                    doc.source,
                    doc.agent_id,
                    doc.scope.value,
                    json.dumps(doc.fragment_ids),
                    json.dumps(doc.tags),
                    json.dumps(doc.metadata),
                    doc.created_at.isoformat(),
                ),
            )
        return doc.id

    def get_document(self, doc_id: str) -> MemoryDocument | None:
        """Fetch a document by id, or None."""
        row = self._fetchone("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return None if row is None else self._row_to_document(dict(row))

    def store_node(self, node: MemoryNode) -> str:
        """Persist a graph node."""
        with self._writing() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.id,
                    node.node_type.value,
                    node.label,
                    node.agent_id,
                    node.fragment_id,
                    node.document_id,
                    node.weight,
                    json.dumps(node.metadata),
                    node.created_at.isoformat(),
                ),
            )
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Fetch a node by id, or None."""
        row = self._fetchone("SELECT * FROM nodes WHERE id = ?", (node_id,))
        return None if row is None else self._row_to_node(dict(row))

    def get_node_by_label(self, label: str, agent_id: str) -> MemoryNode | None:
        """Find the most-recently-created node with this label for this agent."""
        row = self._fetchone(
            "SELECT * FROM nodes WHERE label = ? AND agent_id = ? ORDER BY created_at DESC LIMIT 1",
            (label, agent_id),
        )
        return None if row is None else self._row_to_node(dict(row))

    def delete_edges_between(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | None = None,
    ) -> int:
        """Delete edges from source_id to target_id. Returns count of deleted rows."""
        sql    = "DELETE FROM edges WHERE source_id = ? AND target_id = ?"
        params: list[object] = [source_id, target_id]
        if relation is not None:
            sql += " AND relation = ?"
            params.append(relation.value)
        with self._writing() as conn:
            cursor = conn.execute(sql, params)
        return cursor.rowcount

    def nodes_for_fragment(self, fragment_id: str) -> list[MemoryNode]:
        """Return nodes attached to a fragment."""
        rows = self._fetchall("SELECT * FROM nodes WHERE fragment_id = ?", (fragment_id,))
        return [self._row_to_node(dict(row)) for row in rows]

    def nodes_for_agent(self, agent_id: str) -> list[MemoryNode]:
        """Return an agent's nodes by weight."""
        rows = self._fetchall("SELECT * FROM nodes WHERE agent_id = ? ORDER BY weight DESC", (agent_id,))
        return [self._row_to_node(dict(row)) for row in rows]

    def store_edge(self, edge: MemoryEdge) -> str:
        """Insert or replace an edge row; block_id scopes the edge to a block."""
        with self._writing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO edges
                    (id, source_id, target_id, relation, weight, agent_id,
                     metadata, created_at, block_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value,
                    edge.weight,
                    edge.agent_id,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                    edge.block_id,
                ),
            )
        return edge.id

    def edges_from_node(
        self,
        node_id: str,
        relation_filter: list[EdgeRelation] | None = None,
    ) -> list[MemoryEdge]:
        """Return outgoing edges from a node."""
        sql = "SELECT * FROM edges WHERE source_id = ?"
        params: list[object] = [node_id]
        if relation_filter:
            placeholders = ",".join("?" for _ in relation_filter)
            sql += f" AND relation IN ({placeholders})"
            params.extend(relation.value for relation in relation_filter)
        rows = self._fetchall(sql, params)
        return [self._row_to_edge(dict(row)) for row in rows]

    def edges_to_node(self, node_id: str) -> list[MemoryEdge]:
        """Return all edges whose target is node_id (incoming edges)."""
        rows = self._fetchall("SELECT * FROM edges WHERE target_id = ?", (node_id,))
        return [self._row_to_edge(dict(row)) for row in rows]

    def contradiction_edges(self, node_id: str) -> list[MemoryEdge]:
        """Return CONTRADICTS edges touching a node."""
        rows = self._fetchall(
            "SELECT * FROM edges WHERE relation = ? AND (source_id = ? OR target_id = ?)",
            (EdgeRelation.CONTRADICTS.value, node_id, node_id),
        )
        return [self._row_to_edge(dict(row)) for row in rows]

    # ── connections ───────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        if self.db_path == ":memory:":
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        local = self._local
        conn = getattr(local, "conn", None)
        if conn is None or getattr(local, "generation", -1) != self._generation:
            conn = self._open(self.db_path)
            with self._lock:
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
                    raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
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
            try:
                yield conn
                if structural:
                    conn.execute(
                        "UPDATE schema_info SET value = CAST(value AS INTEGER) + 1"
                        " WHERE key = 'write_version'"
                    )
            except BaseException:
                conn.rollback()
                raise
            conn.commit()

    def _fetchall(self, sql: str, params: Iterable[object] = ()) -> list[sqlite3.Row]:
        with self._reading() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def _fetchone(self, sql: str, params: Iterable[object] = ()) -> sqlite3.Row | None:
        with self._reading() as conn:
            return conn.execute(sql, tuple(params)).fetchone()

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-32000")
        return conn

    # ── plasticity ────────────────────────────────────────────────────────────

    def update_importance(self, fragment_id: str, importance: float) -> None:
        """Update the importance of a fragment, clamped to [0, 1]."""
        self.update_importance_many({fragment_id: importance})

    def update_importance_many(self, updates: dict[str, float]) -> None:
        """Update many importances (each clamped to [0, 1]) in one transaction."""
        if not updates:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            conn.executemany(
                "UPDATE fragments SET importance = ?, updated_at = ? WHERE id = ?",
                [
                    (max(0.0, min(1.0, float(importance))), now, fragment_id)
                    for fragment_id, importance in updates.items()
                ],
            )

    def touch(self, fragment_id: str) -> None:
        """Record an access: update last_accessed_at and increment access_count in metadata."""
        self.touch_many([fragment_id])

    def touch_many(self, ids: Iterable[str]) -> None:
        """Record accesses for many fragments in one transaction (same semantics as touch)."""
        counts = Counter(ids)
        if not counts:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            updates: list[tuple[str, str, str, str]] = []
            for chunk in _chunks(list(counts), _SQL_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT id, metadata FROM fragments WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for row in rows:
                    fragment_id = str(row[0])
                    meta = json.loads(str(row[1]))
                    meta["access_count"] = int(meta.get("access_count", 0)) + counts[fragment_id]
                    updates.append((now, json.dumps(meta), now, fragment_id))
            conn.executemany(
                "UPDATE fragments SET last_accessed_at = ?, metadata = ?, updated_at = ? WHERE id = ?",
                updates,
            )

    def list_edges(self, agent_id: str) -> list[MemoryEdge]:
        """Return all edges owned by an agent, ordered by weight descending."""
        rows = self._fetchall("SELECT * FROM edges WHERE agent_id = ? ORDER BY weight DESC", (agent_id,))
        return [self._row_to_edge(dict(row)) for row in rows]

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by id. Returns True if a row was deleted."""
        with self._writing() as conn:
            cursor = conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        return cursor.rowcount > 0

    def update_edge_weight(self, edge_id: str, weight: float) -> None:
        """Update the weight of an edge, clamped to [0, 1]."""
        with self._writing(structural=False) as conn:
            conn.execute(
                "UPDATE edges SET weight = ? WHERE id = ?",
                (max(0.0, min(1.0, weight)), edge_id),
            )

    def integrity_check(self) -> list[str]:
        """Run PRAGMA integrity_check. Returns [] on a healthy database."""
        rows = self._fetchall("PRAGMA integrity_check")
        results = [str(row[0]) for row in rows]
        return [] if results == ["ok"] else results

    def export(self, dest_path: str | Path) -> None:
        """Backup the database to dest_path using the SQLite backup API."""
        if self.db_path == ":memory:":
            raise ValueError("Cannot export an in-memory database")
        dest = str(Path(dest_path).expanduser().resolve())
        dest_conn = sqlite3.connect(dest)
        try:
            with self._reading() as conn:
                conn.backup(dest_conn)
        finally:
            dest_conn.close()

    def schema_version(self) -> int:
        """Return the current schema version recorded in schema_info."""
        value = self.get_meta("schema_version")
        return int(value) if value is not None else 0

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Bring the schema up to CURRENT_SCHEMA_VERSION.

        Column additions are decided by inspecting PRAGMA table_info for every
        version (not just those above ``from_version``), so pre-versioned and
        partially-migrated databases converge regardless of their stamp.
        Statement migrations are idempotent and run for versions above
        ``from_version``.
        """
        columns: dict[str, set[str]] = {}
        for v in range(1, self.CURRENT_SCHEMA_VERSION + 1):
            for table, column, decl in self._COLUMN_MIGRATIONS.get(v, []):
                if table not in columns:
                    columns[table] = _table_columns(conn, table)
                if column not in columns[table]:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                    columns[table].add(column)
        for v in range(from_version + 1, self.CURRENT_SCHEMA_VERSION + 1):
            for sql in self._MIGRATIONS.get(v, []):
                conn.execute(sql)

    # ── facts (bi-temporal) ───────────────────────────────────────────────────

    def fact_set(
        self,
        agent_id: str,
        subject: str,
        attribute: str,
        value: str,
        *,
        valid_from: datetime,
        source_fragment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record that ``subject.attribute`` had ``value`` from ``valid_from`` on.

        Facts are never overwritten. The timeline for (subject, attribute) is
        re-derived on every write, so out-of-order arrivals slot into history
        instead of becoming "current". Re-asserting the same value at the same
        time is a no-op; a different value at the same time retracts the old
        row (kept for audit) and records the correction.
        """
        subject_key, attribute_key = _fact_key(subject), _fact_key(attribute)
        if not subject_key or not attribute_key:
            raise ValueError("fact subject and attribute must contain letters or digits")
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=timezone.utc)
        valid_text = valid_from.astimezone(timezone.utc).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        value = str(value)
        with self._writing(structural=False) as conn:
            rows = conn.execute(
                "SELECT id, value, valid_from FROM facts WHERE agent_id = ? AND subject_key = ?"
                " AND attribute_key = ? AND retracted_at IS NULL",
                (agent_id, subject_key, attribute_key),
            ).fetchall()
            for row in rows:
                if str(row["valid_from"]) == valid_text:
                    if self.encryption.decrypt_str(row["value"]) == value:
                        return self._fact_by_id(conn, str(row["id"]))
                    conn.execute("UPDATE facts SET retracted_at = ? WHERE id = ?", (now, row["id"]))
            fact_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO facts (id, agent_id, subject, subject_key, attribute, attribute_key, value,"
                " valid_from, valid_to, recorded_at, retracted_at, source_fragment_id, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)",
                (fact_id, agent_id, subject.strip(), subject_key, attribute.strip(), attribute_key,
                 self.encryption.encrypt_str(value), valid_text, now, source_fragment_id,
                 json.dumps(metadata or {})),
            )
            self._rechain_facts(conn, agent_id, subject_key, attribute_key)
            return self._fact_by_id(conn, fact_id)

    def fact_history(self, agent_id: str, subject: str, attribute: str) -> list[dict[str, Any]]:
        """Every value (subject, attribute) has had, oldest first; repeats collapsed."""
        rows = self._fetchall(
            "SELECT * FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
            " AND retracted_at IS NULL ORDER BY valid_from, recorded_at",
            (agent_id, _fact_key(subject), _fact_key(attribute)),
        )
        history: list[dict[str, Any]] = []
        for row in rows:
            fact = self._row_to_fact(row)
            if history and history[-1]["value"] == fact["value"]:
                history[-1]["valid_to"] = fact["valid_to"]
                continue
            history.append(fact)
        return history

    def fact_current(
        self, agent_id: str, subject: str | None = None, attribute: str | None = None,
    ) -> list[dict[str, Any]]:
        """Facts valid now (no later value recorded), optionally for one subject/attribute."""
        sql = "SELECT * FROM facts WHERE agent_id = ? AND valid_to IS NULL AND retracted_at IS NULL"
        params: list[object] = [agent_id]
        if subject is not None:
            sql += " AND subject_key = ?"
            params.append(_fact_key(subject))
        if attribute is not None:
            sql += " AND attribute_key = ?"
            params.append(_fact_key(attribute))
        sql += " ORDER BY subject_key, attribute_key"
        return [self._row_to_fact(row) for row in self._fetchall(sql, params)]

    def fact_as_of(self, agent_id: str, subject: str, attribute: str, when: datetime) -> dict[str, Any] | None:
        """The value (subject, attribute) had at time ``when``, or None."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        stamp = when.astimezone(timezone.utc).isoformat()
        row = self._fetchone(
            "SELECT * FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
            " AND retracted_at IS NULL AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
            " ORDER BY valid_from DESC LIMIT 1",
            (agent_id, _fact_key(subject), _fact_key(attribute), stamp, stamp),
        )
        return None if row is None else self._row_to_fact(row)

    def fact_subjects(self, agent_id: str) -> list[str]:
        """Normalised subject keys that have at least one fact."""
        rows = self._fetchall(
            "SELECT DISTINCT subject_key FROM facts WHERE agent_id = ? AND retracted_at IS NULL", (agent_id,)
        )
        return [str(row[0]) for row in rows]

    def _rechain_facts(self, conn: sqlite3.Connection, agent_id: str, subject_key: str, attribute_key: str) -> None:
        rows = conn.execute(
            "SELECT id, valid_from FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
            " AND retracted_at IS NULL ORDER BY valid_from, recorded_at",
            (agent_id, subject_key, attribute_key),
        ).fetchall()
        updates = [
            (str(rows[i + 1]["valid_from"]) if i + 1 < len(rows) else None, str(row["id"]))
            for i, row in enumerate(rows)
        ]
        conn.executemany("UPDATE facts SET valid_to = ? WHERE id = ?", updates)

    def _fact_by_id(self, conn: sqlite3.Connection, fact_id: str) -> dict[str, Any]:
        return self._row_to_fact(conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone())

    def _row_to_fact(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "subject": str(row["subject"]),
            "attribute": str(row["attribute"]),
            "value": self.encryption.decrypt_str(row["value"]),
            "valid_from": _parse_dt(row["valid_from"]),
            "valid_to": _parse_optional_dt(row["valid_to"]),
            "recorded_at": _parse_dt(row["recorded_at"]),
            "source_fragment_id": _optional_str(row["source_fragment_id"]),
            "metadata": json.loads(str(row["metadata"] or "{}")),
        }

    def log_event(
        self,
        agent_id: str,
        event_type: str,
        *,
        session_id: str | None = None,
        fragment_id: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a session event. Returns the new event id."""
        event_id = str(uuid.uuid4())
        now      = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            conn.execute(
                "INSERT INTO session_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, agent_id, session_id, event_type, fragment_id, query,
                 json.dumps(metadata or {}), now),
            )
        return event_id

    def list_events(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return session events newest-first."""
        sql    = "SELECT * FROM session_events WHERE agent_id = ?"
        params: list[object] = [agent_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [
            {
                "id":          str(row["id"]),
                "agent_id":    str(row["agent_id"]),
                "session_id":  _optional_str(row["session_id"]),
                "event_type":  str(row["event_type"]),
                "fragment_id": _optional_str(row["fragment_id"]),
                "query":       _optional_str(row["query"]),
                "metadata":    json.loads(str(row["metadata"])),
                "created_at":  str(row["created_at"]),
            }
            for row in rows
        ]

    # ── schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._writing() as conn:
            # Take the write lock before reading the schema stamp: two processes
            # creating the same new database must not both run migrations.
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            for ddl in [self._DDL_SCHEMA_INFO, self._DDL_BLOCKS, self._DDL_FRAGMENTS,
                        self._DDL_DOCUMENTS, self._DDL_NODES, self._DDL_EDGES,
                        self._DDL_SESSION_EVENTS, self._DDL_BLOCK_NODES, self._DDL_BLOCK_LINKS]:
                conn.execute(ddl)
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
            # Pre-versioned databases (no stamp) are treated as version 0. Which
            # columns to add is decided from the actual table shape, not data.
            stored = 0 if row is None else int(row[0])
            self._run_migrations(conn, stored)
            if row is None:
                conn.execute(
                    "INSERT OR IGNORE INTO schema_info VALUES ('schema_version', ?)",
                    (str(self.CURRENT_SCHEMA_VERSION),),
                )
            elif stored < self.CURRENT_SCHEMA_VERSION:
                conn.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    (str(self.CURRENT_SCHEMA_VERSION),),
                )
            # Indexes are created after migrations so indexes over migrated
            # columns (e.g. edges.block_id) can be built on upgraded databases.
            for index in self._INDEXES:
                conn.execute(index)
            conn.execute("INSERT OR IGNORE INTO schema_info VALUES ('write_version', '0')")
            self._init_fts(conn)

    def _init_fts(self, conn: sqlite3.Connection) -> None:
        self._fts_supported = _probe_fts5(conn)
        self._fts_enabled = self._fts_supported and isinstance(self.encryption, NoOpEncryptionProvider)
        if not self._fts_supported:
            return
        if not self._fts_enabled:
            # Encrypted store: never keep a plaintext lexical index around.
            conn.execute(f"DROP TABLE IF EXISTS {_FTS_TABLE}")
            conn.execute("DELETE FROM schema_info WHERE key = ?", (_FTS_META_KEY,))
            return
        conn.execute(self._DDL_FTS)
        done = conn.execute("SELECT value FROM schema_info WHERE key = ?", (_FTS_META_KEY,)).fetchone()
        if done is not None and str(done[0]) == "1":
            return
        conn.execute(f"DELETE FROM {_FTS_TABLE}")
        rows = conn.execute("SELECT rowid, id, content, tags, agent_id FROM fragments").fetchall()
        for row in rows:
            try:
                content = self.encryption.decrypt_str(_as_bytes(row["content"]))
                tags = json.loads(str(row["tags"] or "[]"))
            except Exception:  # noqa: BLE001 - skip undecodable legacy rows
                logger.warning("Skipping FTS backfill for undecodable fragment %s", row["id"])
                continue
            self._fts_upsert(conn, int(row["rowid"]), str(row["id"]), str(row["agent_id"]), content, tags)
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES (?, '1')"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_FTS_META_KEY,),
        )

    @staticmethod
    def _fts_upsert(
        conn: sqlite3.Connection,
        rowid: int,
        fragment_id: str,
        agent_id: str,
        content: str,
        tags: Iterable[str],
    ) -> None:
        conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE rowid = ?", (rowid,))
        conn.execute(
            f"INSERT INTO {_FTS_TABLE} (rowid, content, tags, fragment_id, agent_id) VALUES (?, ?, ?, ?, ?)",
            (rowid, content, " ".join(str(tag) for tag in tags), fragment_id, agent_id),
        )

    # ── decoding ──────────────────────────────────────────────────────────────

    def _fragment_filters(
        self,
        agent_id: str,
        *,
        scope: MemoryScope | None,
        states: Iterable[MemoryState] | None,
        tier: MemoryTier | None,
        session_id: str | None,
        session_prefix: str | None,
        tags: list[str] | None,
        block_id: str | None,
        block_ids: list[str] | None,
    ) -> tuple[str, list[object]]:
        sql = "agent_id = ?"
        params: list[object] = [agent_id]
        # Block-level quarantine: a quarantined block's members never surface.
        sql += (
            " AND (block_id IS NULL OR block_id NOT IN"
            " (SELECT id FROM blocks WHERE status = ?))"
        )
        params.append(BlockStatus.QUARANTINED.value)
        if block_id is not None:
            sql += " AND block_id = ?"
            params.append(block_id)
        elif block_ids:
            placeholders = ",".join("?" for _ in block_ids)
            sql += f" AND block_id IN ({placeholders})"
            params.extend(block_ids)
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope.value)
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier.value)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        elif session_prefix is not None:
            sql += " AND session_id LIKE ? ESCAPE '\\'"
            params.append(f"{_escape_like(session_prefix)}%")
        if tags:
            for tag in tags:
                sql += " AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
                params.append(tag)
        if states:
            state_list = list(states)
            placeholders = ",".join("?" for _ in state_list)
            sql += f" AND state IN ({placeholders})"
            params.extend(state.value for state in state_list)
        return sql, params

    def _persist_tampered(self, fragment_ids: list[str]) -> None:
        """Persist CONTRADICTED for fragments whose content hash check failed."""
        now = datetime.now(timezone.utc).isoformat()
        changed: list[str] = []
        with self._writing() as conn:
            for fragment_id in fragment_ids:
                cursor = conn.execute(
                    "UPDATE fragments SET state = ?, updated_at = ? WHERE id = ? AND state != ?",
                    (MemoryState.CONTRADICTED.value, now, fragment_id, MemoryState.CONTRADICTED.value),
                )
                if cursor.rowcount > 0:
                    changed.append(fragment_id)
        if changed:
            logger.warning("Content hash mismatch; marked CONTRADICTED: %s", ", ".join(changed))
            self._emit("unretrievable", changed)

    def _decode_fragment(self, row: dict[str, object]) -> tuple[MemoryFragment, bool]:
        """Decode a row; returns (fragment, tampered). Tampered rows are relabelled CONTRADICTED."""
        fragment = MemoryFragment(
            id=str(row["id"]),
            content=self.encryption.decrypt_str(_as_bytes(row["content"])),
            block_id=_optional_str(row["block_id"]),
            scope=MemoryScope(str(row["scope"])),
            tier=MemoryTier(str(row["tier"])),
            agent_id=str(row["agent_id"]),
            session_id=_optional_str(row["session_id"]),
            importance=float(row["importance"]),
            state=MemoryState(str(row["state"])),
            confidence=float(row["confidence"]),
            ttl_seconds=None if row["ttl_seconds"] is None else int(row["ttl_seconds"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            last_accessed_at=_parse_optional_dt(row["last_accessed_at"]),
            tags=json.loads(str(row["tags"])),
            source=_optional_str(row["source"]),
            metadata=json.loads(str(row["metadata"])),
            embedding=_decode_embedding(row.get("embedding_blob"), row["embedding"]),
            contradiction_ids=json.loads(str(row["contradiction_ids"])),
        )
        if not ContentHasher.check(fragment, self.encryption):
            already = fragment.state == MemoryState.CONTRADICTED
            return fragment.model_copy(update={"state": MemoryState.CONTRADICTED}), not already
        return fragment, False

    def _row_to_fragment(self, row: dict[str, object]) -> MemoryFragment:
        return self._decode_fragment(row)[0]

    @staticmethod
    def _row_to_block(row: dict[str, object]) -> MemoryBlock:
        return MemoryBlock(
            id=str(row["id"]),
            name=str(row["name"]),
            agent_id=str(row["agent_id"]),
            scope=MemoryScope(str(row["scope"])),
            block_type=str(row["block_type"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
            session_id=_optional_str(row.get("session_id")),
            status=BlockStatus(str(row.get("status") or "open")),
            content_hash=_optional_str(row.get("content_hash")),
            prev_block_id=_optional_str(row.get("prev_block_id")),
            prev_block_hash=_optional_str(row.get("prev_block_hash")),
            sealed_at=_parse_optional_dt(row.get("sealed_at")),
        )

    @staticmethod
    def _row_to_document(row: dict[str, object]) -> MemoryDocument:
        return MemoryDocument(
            id=str(row["id"]),
            title=str(row["title"]),
            content=str(row["content"]),
            source=_optional_str(row["source"]),
            agent_id=str(row["agent_id"]),
            scope=MemoryScope(str(row["scope"])),
            fragment_ids=json.loads(str(row["fragment_ids"])),
            tags=json.loads(str(row["tags"])),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_node(row: dict[str, object]) -> MemoryNode:
        return MemoryNode(
            id=str(row["id"]),
            node_type=NodeType(str(row["node_type"])),
            label=str(row["label"]),
            agent_id=_optional_str(row["agent_id"]),
            fragment_id=_optional_str(row["fragment_id"]),
            document_id=_optional_str(row["document_id"]),
            weight=float(row["weight"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_block_link(row: dict[str, object]) -> BlockLink:
        return BlockLink(
            id=str(row["id"]),
            source_block_id=str(row["source_block_id"]),
            target_block_id=str(row["target_block_id"]),
            relation=str(row["relation"]),
            weight=float(row["weight"]),
            agent_id=str(row["agent_id"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_edge(row: dict[str, object]) -> MemoryEdge:
        return MemoryEdge(
            id=str(row["id"]),
            source_id=str(row["source_id"]),
            target_id=str(row["target_id"]),
            relation=EdgeRelation(str(row["relation"])),
            weight=float(row["weight"]),
            agent_id=str(row["agent_id"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
            block_id=_optional_str(row.get("block_id")),
        )


def _pack_embedding(embedding: list[float] | None) -> bytes | None:
    """Encode an embedding as float32 little-endian bytes (no numpy needed)."""
    if embedding is None:
        return None
    packed = array("f", (float(value) for value in embedding))
    if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
        packed.byteswap()
    return packed.tobytes()


def _unpack_embedding(blob: bytes) -> list[float]:
    unpacked = array("f")
    unpacked.frombytes(bytes(blob))
    if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
        unpacked.byteswap()
    return unpacked.tolist()


def _decode_embedding(blob: object, legacy_json: object) -> list[float] | None:
    """Prefer the float32 blob; fall back to the legacy JSON column."""
    if blob is not None:
        return _unpack_embedding(blob)  # type: ignore[arg-type]
    if legacy_json:
        return [float(value) for value in json.loads(str(legacy_json))]
    return None


def _fts_match_expression(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression: quoted tokens OR-joined."""
    tokens = [
        token
        for token in re.findall(r"\w+", query.lower())
        if len(token) >= 2 or token.isdigit()
    ]
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in dict.fromkeys(tokens))


def _probe_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._mark_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS temp._mark_fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _as_bytes(value: object) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _dt_to_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_optional_dt(value: object) -> datetime | None:
    if value is None:
        return None
    return _parse_dt(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _fact_key(text: str) -> str:
    """Normalise a fact subject/attribute key.

    Case, whitespace, hyphens and underscores are folded ('Atlas-Core' ==
    'atlas core' == 'atlas_core'); every other symbol is kept, so 'C++',
    'C#' and 'C' remain distinct subjects.
    """
    folded = re.sub(r"[\s_\-]+", " ", str(text).lower().replace("’", "'"))
    return folded.strip(" \"'()[]{}").rstrip(" .,;:!?")
