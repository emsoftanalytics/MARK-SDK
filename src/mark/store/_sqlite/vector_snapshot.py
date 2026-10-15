"""Vector-index snapshots: reopen a large store without re-reading every row.

Embeddings live one per fragment row, next to the content, so rebuilding
the in-memory index means reading the whole database through Python (about
10 s of CPU at a million memories). A snapshot stores the same normalised
float32 vectors as a few large blobs (``PART_ROWS`` vectors each) plus their
ids, which load in well under a second.

Consistency:
  - a snapshot belongs to one agent and one embedder fingerprint; another
    fingerprint never loads it;
  - once an agent has a snapshot, triggers record in ``vector_dirty`` every
    fragment of that agent that is inserted, deleted, or whose embedding,
    state or owner changes. Loading skips dirty ids (the caller fetches them
    from their rows), so a snapshot never serves a stale vector;
  - a snapshot is written only if no structural write happened since the
    rows it was built from were read (``write_version`` check), inside one
    transaction that also clears the agent's dirty set.

Cost: about 1 KB per memory at dimension 256 (the database grows by roughly
half). Only agents with at least ``MIN_ROWS`` memories get one; below that a
bulk load from the rows already takes well under a second.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Iterator

PART_ROWS = 65_536
MIN_ROWS = 50_000

_DDL = [
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
_WATCHED = "EXISTS (SELECT 1 FROM vector_snapshot_meta WHERE agent_id = {row}.agent_id)"
_TRIGGERS = [
    f"""CREATE TRIGGER IF NOT EXISTS vector_dirty_insert AFTER INSERT ON fragments
        WHEN {_WATCHED.format(row='new')}
        BEGIN INSERT OR IGNORE INTO vector_dirty VALUES (new.agent_id, new.id); END""",
    f"""CREATE TRIGGER IF NOT EXISTS vector_dirty_update
        AFTER UPDATE OF embedding_blob, embedding, state, agent_id ON fragments
        WHEN {_WATCHED.format(row='old')} OR {_WATCHED.format(row='new')}
        BEGIN INSERT OR IGNORE INTO vector_dirty VALUES (old.agent_id, old.id); END""",
    f"""CREATE TRIGGER IF NOT EXISTS vector_dirty_delete AFTER DELETE ON fragments
        WHEN {_WATCHED.format(row='old')}
        BEGIN INSERT OR IGNORE INTO vector_dirty VALUES (old.agent_id, old.id); END""",
]


class VectorSnapshotMixin:
    """Save and load per-agent vector snapshots (see module docstring)."""

    def load_vector_snapshot(
        self, agent_id: str, fingerprint: str,
    ) -> tuple[int, int, Iterator[tuple[list[str], bytes]]] | None:
        """(dim, count, parts) of the agent's snapshot, or None; parts are (ids, float32 blob) minus dirty ids.

        Parts are read one at a time, so loading never holds two copies of
        the vectors in memory.
        """
        try:
            meta = self._fetchone(
                "SELECT fingerprint, dim, count FROM vector_snapshot_meta WHERE agent_id = ?", (agent_id,),
            )
        except sqlite3.OperationalError:  # no snapshot tables yet
            return None
        if meta is None or str(meta[0]) != fingerprint:
            return None
        dim, count = int(meta[1]), int(meta[2])
        dirty = {str(row[0]) for row in self._fetchall("SELECT id FROM vector_dirty WHERE agent_id = ?", (agent_id,))}

        def parts() -> Iterator[tuple[list[str], bytes]]:
            for part in range((count + PART_ROWS - 1) // PART_ROWS):
                row = self._fetchone(
                    "SELECT ids, vectors FROM vector_snapshots WHERE agent_id = ? AND part = ?", (agent_id, part),
                )
                if row is None:
                    return
                ids, data = str(row[0]).split("\n"), bytes(row[1])
                if dirty and not dirty.isdisjoint(ids):
                    keep = [i for i, fid in enumerate(ids) if fid not in dirty]
                    ids = [ids[i] for i in keep]
                    data = b"".join(data[4 * dim * i:4 * dim * (i + 1)] for i in keep)
                yield ids, data

        return dim, count, parts()

    def save_vector_snapshot(
        self, agent_id: str, fingerprint: str, ids: list[str], vectors: Iterable[bytes], dim: int, write_version: int,
    ) -> bool:
        """Replace the agent's snapshot; False (nothing written) if the store changed since *write_version*."""
        with self._writing(structural=False) as conn:
            for statement in (*_DDL, *_TRIGGERS):
                conn.execute(statement)
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
        return True

    def drop_vector_snapshot(self, agent_id: str) -> None:
        """Delete the agent's snapshot (triggers stop tracking it)."""
        try:
            with self._writing(structural=False) as conn:
                for table in ("vector_snapshots", "vector_snapshot_meta", "vector_dirty"):
                    conn.execute(f"DELETE FROM {table} WHERE agent_id = ?", (agent_id,))
        except sqlite3.OperationalError:  # no snapshot tables: nothing to drop
            pass
