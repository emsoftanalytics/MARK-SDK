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

Also here: listing agent namespaces, counting/paging fragments and the raw
row access used by JSONL export/import. Registered on
:class:`mark.store.sqlite.LocalMemoryStore` as ``LifecycleStoreMixin``.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from mark.store._sqlite.codec import _FTS_TABLE, _chunks, _fact_key, _SQL_CHUNK


class LifecycleStoreMixin:
    """Cascading deletes, secure erase and namespace listing."""

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

    def facts_from_fragments(self, agent_id: str, fragment_ids: Iterable[str]) -> int:
        """How many live facts were extracted from *fragment_ids* (the CLI's "extracted N")."""
        total = 0
        for chunk in _chunks(list(fragment_ids), _SQL_CHUNK):
            marks = ",".join("?" for _ in chunk)
            row = self._fetchone(
                f"SELECT COUNT(*) FROM facts WHERE agent_id = ? AND retracted_at IS NULL"
                f" AND source_fragment_id IN ({marks})", [agent_id, *chunk],
            )
            total += int(row[0]) if row else 0
        return total

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

    # ── paging and counting (no silent caps) ──────────────────────────────────

    def _page_where(
        self, agent_id: str, *, states: Any = None, tags: list[str] | None = None,
        exclude_tags: list[str] | None = None, session_id: str | None = None,
    ) -> tuple[str, list[object]]:
        where, params = self._fragment_filters(
            agent_id, scope=None, states=states, tier=None, session_id=session_id,
            session_prefix=None, tags=tags, block_id=None, block_ids=None,
        )
        for tag in exclude_tags or ():
            where += " AND NOT EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
            params.append(tag)
        return where, params

    def page_fragments(
        self, agent_id: str, *, limit: int | None, offset: int = 0, include_unreadable: bool = False,
        **filters: Any,
    ) -> list[Any]:
        """One page of an agent's fragments in ``list_by_agent`` order (rowid breaks ties).

        Memories an encrypted store could not decrypt (``mark:unreadable``,
        see ``unreadable.py``) are left out unless ``include_unreadable``; then
        their stored text is returned as it is and a warning is logged.
        """
        from mark.store._sqlite.unreadable import UNREADABLE_TAG

        if not include_unreadable:
            filters["exclude_tags"] = [*(filters.get("exclude_tags") or ()), UNREADABLE_TAG]
        where, params = self._page_where(agent_id, **filters)
        sql = f"SELECT * FROM fragments WHERE {where} ORDER BY importance DESC, created_at DESC, rowid DESC"
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params += [-1 if limit is None else int(limit), max(0, int(offset))]
        out = [self._row_to_fragment(dict(row)) for row in self._fetchall(sql, params)]
        return self._raw_unreadable(out) if include_unreadable else out

    def _raw_unreadable(self, fragments: list[Any]) -> list[Any]:
        """Show unreadable memories as stored (they were never encrypted), with a warning."""
        import logging

        from mark.store._sqlite.codec import _as_bytes
        from mark.store._sqlite.unreadable import UNREADABLE_TAG

        bad = {f.id for f in fragments if UNREADABLE_TAG in f.tags}
        if not bad:
            return fragments
        logging.getLogger("mark.store.sqlite").warning(
            "including %d unreadable memories as stored (not encrypted with this store's key)", len(bad))
        raw = {str(r[0]): _as_bytes(r[1]).decode("utf-8", "replace") for chunk in _chunks(list(bad), _SQL_CHUNK)
               for r in self._fetchall(f"SELECT id, content FROM fragments WHERE id IN ({','.join('?' * len(chunk))})", chunk)}
        return [f.model_copy(update={"content": raw[f.id]}) if f.id in raw else f for f in fragments]

    def count_fragments(self, agent_id: str, **filters: Any) -> int:
        """How many fragments ``page_fragments`` would return without a limit."""
        from mark.store._sqlite.unreadable import UNREADABLE_TAG

        if not filters.pop("include_unreadable", False):
            filters["exclude_tags"] = [*(filters.get("exclude_tags") or ()), UNREADABLE_TAG]
        where, params = self._page_where(agent_id, **filters)
        row = self._fetchone(f"SELECT COUNT(*) FROM fragments WHERE {where}", params)
        return int(row[0]) if row else 0

    def ids_with_prefix(self, agent_id: str, prefix: str, limit: int = 2) -> list[str]:
        """Fragment ids of an agent starting with *prefix* (CLI short ids); at most *limit*."""
        rows = self._fetchall(
            "SELECT id FROM fragments WHERE agent_id = ? AND substr(id, 1, ?) = ? LIMIT ?",
            (agent_id, len(prefix), prefix, int(limit)),
        )
        return [str(row[0]) for row in rows]

    def tag_values(self, agent_id: str, prefix: str) -> list[str]:
        """Distinct tags of an agent starting with *prefix* (e.g. every ``block:`` label)."""
        rows = self._fetchall(
            "SELECT DISTINCT j.value FROM fragments AS f, json_each(f.tags) AS j"
            " WHERE f.agent_id = ? AND j.value LIKE ? ESCAPE '\\'",
            (agent_id, prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"),
        )
        return sorted(str(row[0]) for row in rows)

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
