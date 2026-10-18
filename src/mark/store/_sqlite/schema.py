"""Schema lifecycle, FTS5 setup, meta key/value access and maintenance.

``_init_schema`` runs once per store, inside a single write transaction:
create tables, migrate, stamp the schema version, build indexes, seed the
``write_version`` counter, check the encryption provider against the store
(see :mod:`mark.store._sqlite.encryption_guard`) and set up (or tear down)
the FTS5 lexical index.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable

from mark.security import NoOpEncryptionProvider
from mark.store._sqlite.codec import (
    _FTS_META_KEY,
    _FTS_ML_META_KEY,
    _FTS_ML_TABLE,
    _FTS_TABLE,
    _as_bytes,
    _probe_fts5,
    _table_columns,
)
from mark.store._sqlite.fts_text import multilingual_index_text
from mark.store._sqlite.ddl import SchemaDDL
from mark.store._sqlite.scope_delta import WRITE_LOG_DDL

logger = logging.getLogger("mark.store.sqlite")


class SchemaMixin(SchemaDDL):
    """Creates and migrates the schema; owns the FTS index and schema_info."""

    _fts_supported: bool
    _fts_enabled: bool

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

    def schema_version(self) -> int:
        """Return the current schema version recorded in schema_info."""
        value = self.get_meta("schema_version")
        return int(value) if value is not None else 0

    # ── maintenance ───────────────────────────────────────────────────────────

    def integrity_check(self) -> list[str]:
        """Run PRAGMA integrity_check (and list unreadable encrypted rows). Returns [] when healthy."""
        rows = self._fetchall("PRAGMA integrity_check")
        results = [str(row[0]) for row in rows]
        # Plus rows an encrypted store could not decrypt (quarantined at open).
        return ([] if results == ["ok"] else results) + self.unreadable_report()

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

    # ── schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create/migrate the schema; raise StoreEncryptionError on a key mismatch.

        A store stamped with a newer schema than this build knows is refused
        before anything is written: older code must never "migrate" or write
        into a layout it does not understand.
        """
        self._refuse_newer_schema()
        self._refuse_migration_under_server()
        with self._writing() as conn:
            # Take the write lock before reading the schema stamp: two processes
            # creating the same new database must not both run migrations.
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            for ddl in [self._DDL_SCHEMA_INFO, self._DDL_BLOCKS, self._DDL_FRAGMENTS,
                        self._DDL_DOCUMENTS, self._DDL_NODES, self._DDL_EDGES,
                        self._DDL_SESSION_EVENTS, self._DDL_BLOCK_NODES, self._DDL_BLOCK_LINKS, WRITE_LOG_DDL]:
                conn.execute(ddl)
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
            # Pre-versioned databases (no stamp) are treated as version 0. Which
            # columns to add is decided from the actual table shape, not data.
            stored = 0 if row is None else int(row[0])
            if row is not None and stored == self.CURRENT_SCHEMA_VERSION:
                # A plain reopen changes no fragment: log an empty change set, so
                # id caches (and snapshot scopes) in any process stay valid.
                self._note_changed(())
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
            # Seed the cache-validation counter bumped by structural writes.
            conn.execute("INSERT OR IGNORE INTO schema_info VALUES ('write_version', '0')")
            # Must run before FTS setup: a keyless open of an encrypted store
            # would otherwise "backfill" ciphertext into the plaintext index.
            error = self._init_encryption(conn)
            if error is None and self._encrypting:
                self._quarantine_unreadable(conn)  # plaintext written by a keyless a8 (see unreadable.py)
            if error is None:
                self._init_fts(conn)
        if error is not None:
            # Markers decided above are committed; then refuse the open.
            self.close()
            raise error

    def _refuse_newer_schema(self) -> None:
        """Raise StoreVersionError if the store was written by a newer MARK."""
        from mark.errors import StoreVersionError

        with self._reading() as conn:
            try:
                row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
            except sqlite3.OperationalError:  # new database: no schema_info yet
                return
        if row is not None and int(row[0]) > self.CURRENT_SCHEMA_VERSION:
            self.close()
            raise StoreVersionError(str(self.db_path), int(row[0]), self.CURRENT_SCHEMA_VERSION)

    def _refuse_migration_under_server(self) -> None:
        """Raise StoreInUseError if a migration is due while another process serves the store.

        See :mod:`mark.store._sqlite.server_guard`: an older ``mark serve``
        would keep writing into the migrated layout with its old code.
        """
        from mark.store._sqlite.server_guard import StoreInUseError, foreign_server

        if self.db_path == ":memory:":
            return
        try:
            row = self._fetchone("SELECT value FROM schema_info WHERE key = 'schema_version'")
        except sqlite3.OperationalError:  # new database: nothing to migrate
            return
        if row is None or int(row[0]) >= self.CURRENT_SCHEMA_VERSION:
            return
        held, pid = foreign_server(self.db_path)
        if held:
            self.close()
            raise StoreInUseError(self.db_path, pid)

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

    # ── FTS5 lexical index ────────────────────────────────────────────────────

    def _init_fts(self, conn: sqlite3.Connection) -> None:
        """Enable, backfill, or drop the FTS5 indexes for this store.

        The indexes hold plaintext, so they exist only for unencrypted stores;
        an encrypted store drops any index left by an earlier unencrypted
        open. One-time backfills (guarded by meta keys) index fragments
        written before a table existed: both tables for a new index, only the
        multilingual one when upgrading a schema-5 store.
        """
        self._fts_supported = _probe_fts5(conn)
        self._fts_enabled = self._fts_supported and isinstance(self.encryption, NoOpEncryptionProvider)
        self._fts_ml_enabled = False
        if not self._fts_supported:
            return
        if not self._fts_enabled:
            # Encrypted store: never keep a plaintext lexical index around.
            conn.execute(f"DROP TRIGGER IF EXISTS {_FTS_ML_TABLE}_delete")
            for table, key in ((_FTS_TABLE, _FTS_META_KEY), (_FTS_ML_TABLE, _FTS_ML_META_KEY)):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute("DELETE FROM schema_info WHERE key = ?", (key,))
            return
        conn.execute(self._DDL_FTS)
        try:
            conn.execute(self._DDL_FTS_ML)
        except sqlite3.OperationalError:  # SQLite without the categories option
            conn.execute(self._DDL_FTS_ML_FALLBACK)
        conn.execute(self._DDL_FTS_ML_TRIGGER)
        self._fts_ml_enabled = True
        marker = "SELECT value FROM schema_info WHERE key = ?"
        porter_done = (conn.execute(marker, (_FTS_META_KEY,)).fetchone() or ["0"])[0] == "1"
        ml_done = (conn.execute(marker, (_FTS_ML_META_KEY,)).fetchone() or ["0"])[0] == "1"
        if porter_done and ml_done:
            return
        conn.execute(f"DELETE FROM {_FTS_ML_TABLE}")
        if not porter_done:
            conn.execute(f"DELETE FROM {_FTS_TABLE}")
        # Streamed, not fetchall(): a million-row store must not be held in memory.
        for row in conn.execute("SELECT rowid, id, content, tags, agent_id FROM fragments"):
            try:
                content = self.encryption.decrypt_str(_as_bytes(row["content"]))
                tags = json.loads(str(row["tags"] or "[]"))
            except Exception:  # noqa: BLE001 - skip undecodable legacy rows
                logger.warning("Skipping FTS backfill for undecodable fragment %s", row["id"])
                continue
            if not porter_done:
                self._fts_upsert(conn, int(row["rowid"]), str(row["id"]), str(row["agent_id"]), content, tags)
            else:
                self._fts_ml_insert(conn, int(row["rowid"]), str(row["id"]), str(row["agent_id"]), content)
        for key in (_FTS_META_KEY, _FTS_ML_META_KEY):
            conn.execute(
                "INSERT INTO schema_info (key, value) VALUES (?, '1')"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key,),
            )

    def _fts_upsert(
        self,
        conn: sqlite3.Connection,
        rowid: int,
        fragment_id: str,
        agent_id: str,
        content: str,
        tags: Iterable[str],
    ) -> None:
        """Replace the FTS rows for a fragment (FTS rowid == fragments.rowid).

        Synced from Python rather than triggers because fragments.content may
        be ciphertext; callers pass the plaintext. Must run inside the same
        write transaction as the fragment change.
        """
        conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE rowid = ?", (rowid,))
        conn.execute(
            f"INSERT INTO {_FTS_TABLE} (rowid, content, tags, fragment_id, agent_id) VALUES (?, ?, ?, ?, ?)",
            (rowid, content, " ".join(str(tag) for tag in tags), fragment_id, agent_id),
        )
        self._mark_vector_dirty(conn, agent_id, fragment_id)  # no-op without a vector snapshot
        if getattr(self, "_fts_ml_enabled", False):
            conn.execute(f"DELETE FROM {_FTS_ML_TABLE} WHERE rowid = ?", (rowid,))
            self._fts_ml_insert(conn, rowid, fragment_id, agent_id, content)

    @staticmethod
    def _fts_ml_insert(conn: sqlite3.Connection, rowid: int, fragment_id: str, agent_id: str, content: str) -> None:
        """Index a fragment in the multilingual table when it has non-ASCII text."""
        prepared = multilingual_index_text(content)
        if prepared is not None:
            conn.execute(
                f"INSERT INTO {_FTS_ML_TABLE} (rowid, content, fragment_id, agent_id) VALUES (?, ?, ?, ?)",
                (rowid, prepared, fragment_id, agent_id),
            )
