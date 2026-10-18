"""Quarantine rows of an encrypted store that are not ciphertext (MARK 0.2.0a8 writes).

MARK 0.2.0a8 and older know nothing about the encryption marker: opened
without the key, they write plaintext into an encrypted store, and they
cannot be fixed. When this build opens the store *with* the key it checks
the rows added since its last check (``rowid`` above the stored mark, so a
normal open reads nothing):

* a fragment whose content does not decrypt is set to QUARANTINED and
  tagged ``mark:unreadable``: it never reaches recall, reads leave it as
  is, exports skip it unless asked (``page_fragments(include_unreadable=True)``,
  which logs a warning and returns the stored text as it is);
* a fact whose value does not decrypt is retracted and flagged
  ``"mark:unreadable"`` in its metadata, and its timeline is re-derived;
* ``integrity_check()`` lists both.

Also repaired here: fact values stored as the text of a Python bytes repr
(``"b'gAAAA...'"``) are turned back into the ciphertext they spell, when
that ciphertext decrypts with the store's key.

Rows rewritten in place by an old build (same rowid) are only caught on
read, where undecryptable content is already treated as tampered.
"""
from __future__ import annotations

import ast
import json
import logging
import sqlite3
from datetime import datetime, timezone

from mark.store._sqlite.codec import _as_bytes
from mark.types import MemoryState

logger = logging.getLogger("mark.store.sqlite")

UNREADABLE_TAG = "mark:unreadable"
_SCAN_KEY = "unreadable_scan"  # {"fragments": last checked rowid, "facts": last checked rowid}
# Fernet tokens start with "gAAAA" (base64 of version byte 0x80); passphrase mode
# puts a 16-byte salt first. Rows matching either are only decrypted on read.
_NOT_FERNET = ("substr({col}, 1, 5) IS NOT X'6741414141' AND substr({col}, 17, 5) IS NOT X'6741414141'")
_REPORT_LIMIT = 50


def _token_shaped(data: bytes) -> bool:
    """Could *data* be a Fernet token, bare or after a 16-byte passphrase salt?

    The SQL filter compares BLOB prefixes, so tokens stored as TEXT reach the
    scan too; those are decrypted normally.
    """
    return data[:5] == b"gAAAA" or data[16:21] == b"gAAAA"


class UnreadableRowsMixin:
    """Finds, quarantines and reports rows an encrypted store cannot decrypt."""

    def _quarantine_unreadable(self, conn: sqlite3.Connection) -> None:
        """Inside ``_init_schema`` (key already verified): check rows added since the last open."""
        row = conn.execute("SELECT value FROM schema_info WHERE key = ?", (_SCAN_KEY,)).fetchone()
        marks = json.loads(str(row[0])) if row else {}
        fernet = type(self.encryption).__name__ == "FernetEncryptionProvider"
        self._repair_fact_reprs(conn, int(marks.get("facts", 0)))
        bad_fragments = self._scan(conn, "fragments", "content", int(marks.get("fragments", 0)), fernet)
        if bad_fragments:
            self._note_changed(fid for fid, _ in bad_fragments)
            for fid, tags in bad_fragments:
                labels = json.loads(str(tags or "[]"))
                conn.execute(
                    "UPDATE fragments SET state = ?, tags = ? WHERE id = ?",
                    (MemoryState.QUARANTINED.value, json.dumps([*labels, UNREADABLE_TAG]), fid),
                )
            logger.warning("quarantined %d memories that are not encrypted with this store's key "
                           "(written by MARK 0.2.0a8 or older without the key)", len(bad_fragments))
        bad_facts = self._scan(conn, "facts", "value", int(marks.get("facts", 0)), fernet, live_facts=True)
        now = datetime.now(timezone.utc).isoformat()
        for fact_id, metadata in bad_facts:
            meta = {**json.loads(str(metadata or "{}")), UNREADABLE_TAG: True}
            conn.execute("UPDATE facts SET retracted_at = ?, metadata = ? WHERE id = ?", (now, json.dumps(meta), fact_id))
            keys = conn.execute("SELECT agent_id, subject_key, attribute_key FROM facts WHERE id = ?", (fact_id,)).fetchone()
            self._rechain_facts(conn, str(keys[0]), str(keys[1]), str(keys[2]))
        if bad_facts:
            logger.warning("retracted %d fact values that are not encrypted with this store's key", len(bad_facts))
        top = {table: int(conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").fetchone()[0])
               for table in ("fragments", "facts")}
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_SCAN_KEY, json.dumps(top)),
        )

    def _scan(
        self, conn: sqlite3.Connection, table: str, column: str, after: int, fernet: bool, *, live_facts: bool = False,
    ) -> list[tuple[str, object]]:
        """(id, tags-or-metadata) of rows after rowid *after* whose *column* does not decrypt."""
        extra = "tags" if table == "fragments" else "metadata"
        sql = f"SELECT id, {column}, {extra} FROM {table} WHERE rowid > ?"
        if live_facts:
            sql += " AND retracted_at IS NULL"
        if fernet:  # everything that looks like a token is left to the read path
            sql += " AND " + _NOT_FERNET.format(col=column)
        bad = []
        for row_id, data, label in conn.execute(sql, (after,)).fetchall():
            if fernet and not _token_shaped(_as_bytes(data)):
                # Plaintext from a keyless a8: decrypting it would derive a key
                # from a "salt" of plaintext bytes (one PBKDF2 run, seconds,
                # per row: a 14 s first reopen in the a10 re-test).
                bad.append((str(row_id), label))
                continue
            try:
                self.encryption.decrypt(_as_bytes(data))
            except Exception:  # noqa: BLE001 - any provider error: not ciphertext for this key
                bad.append((str(row_id), label))
        return bad

    def _repair_fact_reprs(self, conn: sqlite3.Connection, after: int) -> None:
        """Turn text values like "b'gAAAA...'" back into the token they spell (when it decrypts)."""
        for rowid, text in conn.execute(
            "SELECT rowid, value FROM facts WHERE rowid > ? AND typeof(value) = 'text'"
            " AND (value LIKE 'b''%' OR value LIKE 'b\"%')", (after,),
        ).fetchall():
            try:
                token = ast.literal_eval(str(text))
                if not isinstance(token, bytes):
                    continue
                self.encryption.decrypt(token)
            except Exception:  # noqa: BLE001 - not a repr of our ciphertext: leave it to the scan
                continue
            conn.execute("UPDATE facts SET value = ? WHERE rowid = ?", (token, rowid))

    def unreadable_report(self) -> list[str]:
        """``integrity_check`` lines for quarantined unreadable fragments and retracted unreadable facts."""
        if not getattr(self, "_encrypting", False):
            return []
        lines = []
        fragments = self._fetchall(
            "SELECT id FROM fragments WHERE state = ? AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)",
            (MemoryState.QUARANTINED.value, UNREADABLE_TAG),
        )
        facts = self._fetchall(
            "SELECT id FROM facts WHERE json_extract(metadata, ?) IS NOT NULL", (f'$."{UNREADABLE_TAG}"',),
        )
        for kind, rows in (("memory", fragments), ("fact", facts)):
            for (row_id,) in rows[:_REPORT_LIMIT]:
                lines.append(f"{kind} {row_id}: unreadable (not encrypted with this store's key, e.g. written "
                             f"by MARK 0.2.0a8 without the key); {'quarantined' if kind == 'memory' else 'retracted'}")
            if len(rows) > _REPORT_LIMIT:
                lines.append(f"... and {len(rows) - _REPORT_LIMIT} more unreadable {kind} rows")
        return lines
