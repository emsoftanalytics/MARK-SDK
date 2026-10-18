"""Vector-index snapshots: reopen a large store without re-reading every row.

Embeddings live one per fragment row, next to the content, so rebuilding
the in-memory index means reading the whole database through Python (about
10 s of CPU at a million memories). A snapshot stores the same normalised
float32 vectors as a few large blobs (``PART_ROWS`` vectors each) plus their
ids, which load in well under a second.

Consistency:
  - a snapshot belongs to one agent and one embedder fingerprint; another
    fingerprint never loads it;
  - once an agent has a snapshot, every store of one of its fragments (the
    FTS write hook, ``_fts_upsert``) and every pipeline re-embedding records
    the id in ``vector_dirty``; loading skips dirty ids (the caller reads
    them from their rows), so a snapshot never serves a stale vector. New,
    deleted or no longer retrievable fragments need no record: the caller
    compares the snapshot with the retrievable-id set. (SQLite triggers
    would do this too, but any UPDATE trigger on ``fragments`` was measured
    to slow every upsert by ~40%.) Stores without that hook (encrypted, or
    no FTS5) never use snapshots;
  - a snapshot is written only if no structural write happened since the
    rows it was built from were read (``write_version`` check), inside one
    transaction that also clears the agent's dirty set.

Saving a snapshot also records the agent's retrievable scope when the
saved ids are exactly that scope (see :mod:`.scope_snapshot`).

Cost: about 1 KB per memory at dimension 256 (the database grows by roughly
half). Only agents with at least ``MIN_ROWS`` memories get one; below that a
bulk load from the rows already takes well under a second.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Iterator

PART_ROWS = 65_536
MIN_ROWS = 50_000

SNAPSHOT_DDL = [
    """CREATE TABLE IF NOT EXISTS vector_snapshot_meta (
        agent_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, dim INTEGER NOT NULL, count INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS vector_snapshots (
        agent_id TEXT NOT NULL, part INTEGER NOT NULL, ids TEXT NOT NULL, vectors BLOB NOT NULL,
        PRIMARY KEY (agent_id, part)
    )""",
    "CREATE TABLE IF NOT EXISTS vector_dirty (agent_id TEXT NOT NULL, id TEXT NOT NULL,"
    " PRIMARY KEY (agent_id, id)) WITHOUT ROWID",
]
class VectorSnapshotMixin:
    """Save and load per-agent vector snapshots (see module docstring)."""

    def mark_vectors_dirty(self, agent_id: str, fragment_ids: Iterable[str]) -> None:
        """Record re-embedded fragments so a snapshot never serves their old vectors."""
        with self._writing(structural=False) as conn:
            for fragment_id in fragment_ids:
                self._mark_vector_dirty(conn, agent_id, fragment_id)

    @staticmethod
    def _mark_vector_dirty(conn: sqlite3.Connection, agent_id: str, fragment_id: str) -> None:
        """One statement, a no-op unless the agent has a snapshot (called on every store)."""
        conn.execute(
            "INSERT OR IGNORE INTO vector_dirty SELECT ?, ?"
            " WHERE EXISTS (SELECT 1 FROM vector_snapshot_meta WHERE agent_id = ?)",
            (agent_id, fragment_id, agent_id),
        )

    def load_vector_snapshot(
        self, agent_id: str, fingerprint: str,
    ) -> tuple[int, int, Iterator[tuple[list[str], bytes]]] | None:
        """(dim, count, parts) of the agent's snapshot, or None; parts are (ids, float32 blob) minus dirty ids.

        Parts are read one at a time, so loading never holds two copies of
        the vectors in memory.
        """
        if not self._fts_enabled:  # no write hook to record changes: never trust a snapshot
            return None
        meta = self._fetchone("SELECT fingerprint, dim, count FROM vector_snapshot_meta WHERE agent_id = ?", (agent_id,))
        if meta is None or str(meta[0]) != fingerprint:
            return None
        dim, count = int(meta[1]), int(meta[2])
        dirty = {str(row[0]) for row in self._fetchall("SELECT id FROM vector_dirty WHERE agent_id = ?", (agent_id,))}

        def parts() -> Iterator[tuple[list[str], bytes]]:
            # One statement streamed part by part (no per-part query, no fetchall):
            # each 64 MB blob is copied once into Python and released after use.
            with self._reading() as conn:
                # Memory-mapped reads: blobs are copied out of the page cache once,
                # not read() into SQLite's cache and copied again (restored below).
                previous = conn.execute("PRAGMA mmap_size").fetchone()[0]
                conn.execute("PRAGMA mmap_size = 2147418112")
                cursor = conn.cursor()
                cursor.row_factory = None
                rows = cursor.execute(
                    "SELECT ids, vectors FROM vector_snapshots WHERE agent_id = ? ORDER BY part", (agent_id,))
                try:
                    yield from self._snapshot_rows(rows, dirty, dim)
                finally:
                    rows.close()
                    conn.execute(f"PRAGMA mmap_size = {int(previous)}")

        return dim, count, parts()

    @staticmethod
    def _snapshot_rows(rows: Iterable[tuple], dirty: set[str], dim: int) -> Iterator[tuple[list[str], bytes]]:
        """(ids, float32 blob) per part, minus the dirty ids."""
        for ids_text, blob in rows:
            ids, data = str(ids_text).split("\n"), blob
            if dirty and not dirty.isdisjoint(ids):
                keep = [i for i, fid in enumerate(ids) if fid not in dirty]
                ids = [ids[i] for i in keep]
                data = b"".join(data[4 * dim * i:4 * dim * (i + 1)] for i in keep)
            yield ids, data

    def save_vector_snapshot(
        self, agent_id: str, fingerprint: str, ids: list[str], vectors: Iterable[bytes], dim: int, write_version: int,
    ) -> bool:
        """Replace the agent's snapshot; False (nothing written) if the store changed since *write_version*."""
        if not self._fts_enabled:
            return False
        with self._writing(structural=False) as conn:
            current = conn.execute("SELECT value FROM schema_info WHERE key = 'write_version'").fetchone()
            if current is None or int(current[0]) != write_version:
                return False
            for table in ("vector_snapshots", "vector_snapshot_meta", "vector_dirty"):
                conn.execute(f"DELETE FROM {table} WHERE agent_id = ?", (agent_id,))
            for part, data in enumerate(vectors):
                part_ids = ids[part * PART_ROWS:(part + 1) * PART_ROWS]
                conn.execute(
                    "INSERT INTO vector_snapshots (agent_id, part, ids, vectors) VALUES (?, ?, ?, ?)",
                    (agent_id, part, "\n".join(part_ids), data),
                )
            conn.execute(
                "INSERT INTO vector_snapshot_meta (agent_id, fingerprint, dim, count) VALUES (?, ?, ?, ?)",
                (agent_id, fingerprint, dim, len(ids)),
            )
            # The same ids are the agent's scope at this version: record it too (cheap).
            self._record_scope_if_exact(conn, agent_id, ids, write_version)
        return True

    def drop_vector_snapshot(self, agent_id: str) -> None:
        """Delete the agent's snapshot (triggers stop tracking it)."""
        with self._writing(structural=False) as conn:
            for table in ("vector_snapshots", "vector_snapshot_meta", "vector_dirty"):
                conn.execute(f"DELETE FROM {table} WHERE agent_id = ?", (agent_id,))
