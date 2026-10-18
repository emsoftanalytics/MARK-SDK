"""Memory lifecycle at the storage layer: cascading forget, fact delete, vacuum.

A forgotten memory must not live on through what was derived from it:

* ``forget_fragment`` deletes the fragment row, its FTS row, its fact-job
  and extraction-ledger rows, and **retracts** every fact whose
  ``source_fragment_id`` is the forgotten memory (retracted facts drop out of
  current values, history and timelines, but stay for audit);
* ``secure=True`` goes further: the content is overwritten before the row is
  deleted, the derived facts are **hard-deleted**, and SQLite's
  ``secure_delete`` pragma zeroes the freed pages. ``vacuum()`` then rewrites
  the file and truncates the WAL so no copy of the text survives on disk.

Also here: listing agent namespaces and the raw row access used by JSONL
export/import; counting/paging fragments lives in
:mod:`mark.store._sqlite.paging` (inherited). Registered on
:class:`mark.store.sqlite.LocalMemoryStore` as ``LifecycleStoreMixin``.
"""
from __future__ import annotations

import sqlite3

import json
from typing import Any, Iterable

from mark.store._sqlite.codec import _FTS_TABLE, _chunks, _fact_key, _SQL_CHUNK
from mark.store._sqlite.paging import PagingStoreMixin


class LifecycleStoreMixin(PagingStoreMixin):
    """Cascading deletes, secure erase and namespace listing (paging from :class:`PagingStoreMixin`)."""

    # ── forget ────────────────────────────────────────────────────────────────

    def forget_fragment(self, fragment_id: str, *, secure: bool = False) -> bool:
        """Delete one fragment and cascade to everything derived from it.

        Returns False when no such fragment exists. See the module docstring
        for what ``secure`` adds.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with self._writing() as conn:
            self._note_changed((fragment_id,))
            row = conn.execute(
                "SELECT rowid, agent_id FROM fragments WHERE id = ?", (fragment_id,)
            ).fetchone()
            if row is None:
                return False
            rowid, agent_id = int(row[0]), str(row[1])
            if secure:
                # Zero freed pages from here on (per connection; cheap to leave on
                # for the rest of the transaction) and overwrite before deleting,
                # so even a page that is not freed no longer holds the text.
                conn.execute("PRAGMA secure_delete = ON")
                conn.execute(
                    "UPDATE fragments SET content = zeroblob(length(content)), metadata = '{}',"
                    " tags = '[]', embedding = NULL, embedding_blob = NULL WHERE id = ?",
                    (fragment_id,),
                )
            if self._fts_enabled:
                conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE rowid = ?", (rowid,))
            conn.execute("DELETE FROM fragments WHERE id = ?", (fragment_id,))
            self._cascade_facts(conn, agent_id, [fragment_id], secure=secure, now=now)
            if secure:  # a saved vector snapshot still holds this memory's vector: drop it
                for table in ("vector_snapshots", "vector_snapshot_meta", "vector_dirty"):
                    try:
                        conn.execute(f"DELETE FROM {table} WHERE agent_id = ?", (agent_id,))
                    except sqlite3.OperationalError:  # no snapshot tables in this store
                        pass
        self._emit("delete", [fragment_id])
        return True

    def _cascade_facts(
        self, conn: Any, agent_id: str, fragment_ids: list[str], *, secure: bool, now: str,
    ) -> None:
        """Retract (or, when ``secure``, delete) facts sourced from *fragment_ids*."""
        for chunk in _chunks(fragment_ids, _SQL_CHUNK):
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id, subject_key, attribute_key FROM facts WHERE agent_id = ?"
                f" AND source_fragment_id IN ({marks})",
                [agent_id, *chunk],
            ).fetchall()
            if secure:
                conn.execute(
                    f"UPDATE facts SET value = zeroblob(length(value)) WHERE agent_id = ?"
                    f" AND source_fragment_id IN ({marks})", [agent_id, *chunk],
                )
                conn.execute(
                    f"DELETE FROM facts WHERE agent_id = ? AND source_fragment_id IN ({marks})",
                    [agent_id, *chunk],
                )
            else:
                conn.execute(
                    f"UPDATE facts SET retracted_at = ? WHERE agent_id = ? AND retracted_at IS NULL"
                    f" AND source_fragment_id IN ({marks})", [now, agent_id, *chunk],
                )
            # Re-derive validity intervals so neighbouring values take over.
            for subject_key, attribute_key in {(str(r[1]), str(r[2])) for r in rows}:
                self._rechain_facts(conn, agent_id, subject_key, attribute_key)
            # Queue + ledger: forgetting then re-remembering must extract again.
            for table in ("fact_jobs", "fact_extractions"):
                if self._has_table(conn, table):
                    conn.execute(f"DELETE FROM {table} WHERE fragment_id IN ({marks})", chunk)

    @staticmethod
    def _has_table(conn: Any, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    # ── facts ─────────────────────────────────────────────────────────────────

    def fact_delete(
        self, agent_id: str, subject: str, attribute: str | None = None, *, value: str | None = None,
    ) -> int:
        """Hard-delete facts (unlike ``fact_retract`` nothing is kept); returns rows deleted.

        Values are overwritten before the rows are removed and freed pages are
        zeroed, so a later ``vacuum()`` leaves no trace of them on disk.
        """
        sql = "SELECT id, attribute_key, value FROM facts WHERE agent_id = ? AND subject_key = ?"
        params: list[object] = [agent_id, _fact_key(subject)]
        if attribute is not None:
            sql += " AND attribute_key = ?"
            params.append(_fact_key(attribute))
        with self._writing(structural=False) as conn:
            conn.execute("PRAGMA secure_delete = ON")
            rows = [
                row for row in conn.execute(sql, params).fetchall()
                if value is None or self.encryption.decrypt_str(row["value"]) == str(value)
            ]
            for row in rows:
                conn.execute("UPDATE facts SET value = zeroblob(length(value)) WHERE id = ?", (row["id"],))
                conn.execute("DELETE FROM facts WHERE id = ?", (row["id"],))
            for attribute_key in {str(row["attribute_key"]) for row in rows}:
                self._rechain_facts(conn, agent_id, _fact_key(subject), attribute_key)
        return len(rows)

    def facts_from_fragments(self, agent_id: str, fragment_ids: Iterable[str], *,
                             after: int | None = None) -> int:
        """How many live facts were extracted from *fragment_ids* (the CLI's "extracted N").

        ``after`` (a :meth:`fact_watermark` taken before a write) counts only
        facts created since, so a re-ingest reports 0 new facts.
        """
        total = 0
        newer = " AND rowid > ?" if after is not None else ""
        for chunk in _chunks(list(fragment_ids), _SQL_CHUNK):
            marks = ",".join("?" for _ in chunk)
            row = self._fetchone(
                f"SELECT COUNT(*) FROM facts WHERE agent_id = ? AND retracted_at IS NULL"
                f" AND source_fragment_id IN ({marks}){newer}",
                [agent_id, *chunk, *([int(after)] if after is not None else [])],
            )
            total += int(row[0]) if row else 0
        return total

    def fact_watermark(self) -> int:
        """The newest fact row id (0 for none): pass to ``facts_from_fragments(after=...)``."""
        row = self._fetchone("SELECT COALESCE(MAX(rowid), 0) FROM facts", ())
        return int(row[0]) if row else 0

    # ── disk hygiene ──────────────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Rewrite the database so deleted content is gone from the file and the WAL.

        Merges the FTS index (dropping deleted terms), checkpoints and
        truncates the WAL, runs ``VACUUM`` and truncates the WAL again.
        """
        if self._shared_conn is not None:  # :memory: — nothing on disk
            return
        with self._lock:
            conn = self._conn()
            conn.commit()
            conn.execute("PRAGMA secure_delete = ON")
            if self._fts_enabled:
                conn.execute(f"INSERT INTO {_FTS_TABLE}({_FTS_TABLE}) VALUES ('optimize')")
                conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ── namespaces and raw rows (export / import) ─────────────────────────────

    def agent_ids(self) -> list[str]:
        """Every agent namespace holding memories or facts, sorted."""
        rows = self._fetchall(
            "SELECT agent_id FROM fragments UNION SELECT agent_id FROM facts ORDER BY 1"
        )
        return [str(row[0]) for row in rows]

    def fact_rows(self, agent_id: str) -> list[dict[str, Any]]:
        """Every non-retracted fact row of an agent (decrypted), oldest first."""
        rows = self._fetchall(
            "SELECT * FROM facts WHERE agent_id = ? AND retracted_at IS NULL"
            " ORDER BY valid_from, recorded_at, rowid", (agent_id,),
        )
        return [self._row_to_fact(row) for row in rows]

    def document_rows(self, agent_id: str) -> list[dict[str, Any]]:
        """Raw ``documents`` rows of an agent (JSON columns decoded)."""
        rows = self._fetchall("SELECT * FROM documents WHERE agent_id = ?", (agent_id,))
        out = []
        for row in rows:
            item = dict(row)
            for key in ("fragment_ids", "tags", "metadata"):
                if isinstance(item.get(key), str):
                    item[key] = json.loads(item[key] or "null")
            out.append(item)
        return out

    def import_document_row(self, row: dict[str, Any]) -> bool:
        """Insert a document row exported by ``document_rows`` unless its id exists."""
        columns = [name for name in row if name.isidentifier()]
        values = [json.dumps(row[c]) if isinstance(row[c], (list, dict)) else row[c] for c in columns]
        with self._writing(structural=False) as conn:
            known = {str(r[1]) for r in conn.execute("PRAGMA table_info(documents)")}
            keep = [i for i, c in enumerate(columns) if c in known]
            sql = (f"INSERT OR IGNORE INTO documents ({', '.join(columns[i] for i in keep)})"
                   f" VALUES ({', '.join('?' for _ in keep)})")
            return conn.execute(sql, [values[i] for i in keep]).rowcount > 0

    @property
    def encrypted(self) -> bool:
        """True when content is encrypted at rest (any provider but the no-op one)."""
        from mark.security import NoOpEncryptionProvider

        return not isinstance(self.encryption, NoOpEncryptionProvider)
