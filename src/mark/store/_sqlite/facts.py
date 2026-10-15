"""Bi-temporal facts: (subject, attribute) -> value with validity intervals.

Each fact row carries two time axes:
  - valid time: ``valid_from`` / ``valid_to`` — when the value was true in
    the world. ``valid_to`` is derived, never supplied: it is the next row's
    ``valid_from`` on the same (subject, attribute) timeline, or NULL for
    the current value;
  - transaction time: ``recorded_at`` / ``retracted_at`` — when the store
    learned (or withdrew) the row. Retracted rows are kept for audit and
    ignored by every query.

Subjects and attributes are matched by normalised keys (see ``_fact_key``);
values are typed (see :mod:`.fact_codec`) and encrypted at rest like
fragment content. Fact writes are non-structural: they never change which
fragments are retrievable.

"Current" is judged against the clock: a value whose ``valid_from`` is still
in the future is *scheduled* (see ``fact_scheduled``), and the value it will
replace stays current until then.

Multi-valued attributes (rows flagged ``metadata["multi_valued"]``, written
by ``fact_add`` in :mod:`.facts_multi`) sit outside the single-value chain:
they stay current until removed, or until ``fact_set`` replaces every value
of the attribute.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from mark.store._sqlite.codec import _fact_key, _optional_str, _parse_dt, _parse_optional_dt
from mark.store._sqlite.fact_codec import decode_value, encode_value, value_text

MULTI_KEY = "multi_valued"  # metadata flag of rows written by fact_add


class FactMixin:
    """Record and query bi-temporal facts."""

    def fact_set(
        self,
        agent_id: str,
        subject: str,
        attribute: str,
        value: Any,
        *,
        valid_from: datetime,
        source_fragment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        correction: bool = False,
    ) -> dict[str, Any]:
        """Record that ``subject.attribute`` had ``value`` from ``valid_from`` on.

        Facts are never overwritten. The timeline for (subject, attribute) is
        re-derived on every write, so out-of-order arrivals slot into history
        instead of becoming "current".

        Several values at the same ``valid_from`` (e.g. two changes logged on
        one day) form a sequence ordered by when they were recorded; the last
        one is current and every step stays in history. Re-asserting the
        latest value at that time is a no-op. ``correction=True`` instead
        replaces the values recorded at that time: they are retracted (kept for
        audit) and the new value is recorded. A set value also ends every
        multi-valued value (``fact_add``) of the attribute that began earlier.
        """
        subject_key, attribute_key = _fact_key(subject), _fact_key(attribute)
        if not subject_key or not attribute_key:
            raise ValueError("fact subject and attribute must contain letters or digits")
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=timezone.utc)
        valid_text = valid_from.astimezone(timezone.utc).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        encoded = encode_value(value)
        meta = {key: val for key, val in (metadata or {}).items() if key != MULTI_KEY}
        # Read, dedupe/retract, insert and re-chain in one locked transaction
        # so concurrent writers cannot interleave and corrupt the timeline.
        with self._writing(structural=False) as conn:
            same_time = [row for row in conn.execute(
                "SELECT id, value, metadata FROM facts WHERE agent_id = ? AND subject_key = ?"
                " AND attribute_key = ? AND valid_from = ? AND retracted_at IS NULL"
                " ORDER BY recorded_at, rowid",
                (agent_id, subject_key, attribute_key, valid_text),
            ).fetchall() if not _is_multi(row)]
            if correction:
                # Explicit correction: what was recorded for this time was wrong.
                for row in same_time:
                    conn.execute("UPDATE facts SET retracted_at = ? WHERE id = ?", (now, row["id"]))
            elif same_time and self.encryption.decrypt_str(same_time[-1]["value"]) == encoded:
                # Re-asserting the latest value for this time (e.g. re-ingesting a log).
                return self._fact_by_id(conn, str(same_time[-1]["id"]))
            fact_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO facts (id, agent_id, subject, subject_key, attribute, attribute_key, value,"
                " valid_from, valid_to, recorded_at, retracted_at, source_fragment_id, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)",
                (fact_id, agent_id, subject.strip(), subject_key, attribute.strip(), attribute_key,
                 self.encryption.encrypt_str(encoded), valid_text, now, source_fragment_id,
                 json.dumps(meta)),
            )
            # set() replaces: added (multi-valued) values that began by then end here.
            for row in conn.execute(
                "SELECT id, metadata FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
                " AND retracted_at IS NULL AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
                (agent_id, subject_key, attribute_key, valid_text, valid_text),
            ).fetchall():
                if _is_multi(row):
                    conn.execute("UPDATE facts SET valid_to = ? WHERE id = ?", (valid_text, row["id"]))
            self._rechain_facts(conn, agent_id, subject_key, attribute_key)
            return self._fact_by_id(conn, fact_id)

    def fact_history(self, agent_id: str, subject: str, attribute: str) -> list[dict[str, Any]]:
        """Every value (subject, attribute) has had, oldest first; repeats collapsed."""
        rows = self._fetchall(
            "SELECT * FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
            " AND retracted_at IS NULL ORDER BY valid_from, recorded_at, rowid",
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
        """Facts valid now, optionally for one subject/attribute.

        Valid now: started at or before the current time and not ended by
        then. A scheduled (future ``valid_from``) value is not current yet.
        """
        now = datetime.now(timezone.utc).isoformat()
        sql = ("SELECT * FROM facts WHERE agent_id = ? AND retracted_at IS NULL"
               " AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)")
        params: list[object] = [agent_id, now, now]
        if subject is not None:
            sql += " AND subject_key = ?"
            params.append(_fact_key(subject))
        if attribute is not None:
            sql += " AND attribute_key = ?"
            params.append(_fact_key(attribute))
        sql += " ORDER BY subject_key, attribute_key, valid_from, recorded_at, rowid"
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

    def fact_retract(
        self, agent_id: str, subject: str, attribute: str | None = None, *, value: Any = None,
    ) -> int:
        """Retract facts recorded by mistake; returns how many rows were retracted.

        Rows are kept (``retracted_at`` set) for audit and drop out of current
        values and history. ``attribute``/``value`` narrow what is retracted.
        """
        now = datetime.now(timezone.utc).isoformat()
        sql = ("SELECT id, attribute_key, value FROM facts WHERE agent_id = ? AND subject_key = ?"
               " AND retracted_at IS NULL")
        params: list[object] = [agent_id, _fact_key(subject)]
        if attribute is not None:
            sql += " AND attribute_key = ?"
            params.append(_fact_key(attribute))
        with self._writing(structural=False) as conn:
            rows = [
                row for row in conn.execute(sql, params).fetchall()
                if value is None or _same_value(decode_value(self.encryption.decrypt_str(row["value"])), value)
            ]
            for row in rows:
                conn.execute("UPDATE facts SET retracted_at = ? WHERE id = ?", (now, row["id"]))
            # Neighbouring values take over the retracted intervals.
            for attribute_key in {str(row["attribute_key"]) for row in rows}:
                self._rechain_facts(conn, agent_id, _fact_key(subject), attribute_key)
        return len(rows)

    def fact_subjects(self, agent_id: str) -> list[str]:
        """Normalised subject keys that have at least one fact."""
        rows = self._fetchall(
            "SELECT DISTINCT subject_key FROM facts WHERE agent_id = ? AND retracted_at IS NULL", (agent_id,)
        )
        return [str(row[0]) for row in rows]

    def _rechain_facts(self, conn: sqlite3.Connection, agent_id: str, subject_key: str, attribute_key: str) -> None:
        """Recompute valid_to for one timeline: each row ends where the next begins.

        Re-deriving the whole chain (rather than closing only the previous
        "current" row) lets late, out-of-order facts slot into history.
        """
        rows = [row for row in conn.execute(
            "SELECT id, valid_from, metadata FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
            " AND retracted_at IS NULL ORDER BY valid_from, recorded_at, rowid",
            (agent_id, subject_key, attribute_key),
        ).fetchall() if not _is_multi(row)]  # multi-valued rows keep their own intervals
        updates = [
            (str(rows[i + 1]["valid_from"]) if i + 1 < len(rows) else None, str(row["id"]))
            for i, row in enumerate(rows)
        ]
        conn.executemany("UPDATE facts SET valid_to = ? WHERE id = ?", updates)

    def _fact_by_id(self, conn: sqlite3.Connection, fact_id: str) -> dict[str, Any]:
        return self._row_to_fact(conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone())

    def _row_to_fact(self, row: sqlite3.Row) -> dict[str, Any]:
        """Decode a facts row (decrypting the value) into a plain dict."""
        return {
            "id": str(row["id"]),
            "subject": str(row["subject"]),
            "attribute": str(row["attribute"]),
            "value": decode_value(self.encryption.decrypt_str(row["value"])),
            "valid_from": _parse_dt(row["valid_from"]),
            "valid_to": _parse_optional_dt(row["valid_to"]),
            "recorded_at": _parse_dt(row["recorded_at"]),
            "source_fragment_id": _optional_str(row["source_fragment_id"]),
            "metadata": json.loads(str(row["metadata"] or "{}")),
        }


def _is_multi(row: sqlite3.Row) -> bool:
    """True for a multi-valued row (cheap substring test before parsing the JSON)."""
    text = str(row["metadata"] or "")
    return MULTI_KEY in text and bool(json.loads(text).get(MULTI_KEY))


def _same_value(stored: Any, wanted: Any) -> bool:
    """Retract-by-value match: same typed value, or the same rendering ("4444" matches 4444)."""
    return encode_value(stored) == encode_value(wanted) or value_text(stored) == value_text(wanted)
