"""Multi-valued and scheduled facts (store side).

A *multi-valued* attribute holds several current values at once (phone
numbers, languages, team members). ``fact_add`` records one more value
without ending the others; ``fact_remove`` ends one value at a given time;
``fact_set`` (in :mod:`.facts`) still replaces — it ends every value of the
attribute, added ones included. Added rows carry ``metadata["multi_valued"]``
and keep their own interval: the single-value chain (``_rechain_facts``)
never touches them.

``fact_scheduled`` lists values whose ``valid_from`` is still in the future:
they are not current yet (``fact_current`` ignores them) but are known.

Like every fact write these are non-structural (no retrieval cache churn).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from mark.store._sqlite.codec import _fact_key
from mark.store._sqlite.fact_codec import encode_value
from mark.store._sqlite.facts import MULTI_KEY, _is_multi


class FactMultiMixin:
    """fact_add / fact_remove / fact_scheduled."""

    def fact_add(
        self,
        agent_id: str,
        subject: str,
        attribute: str,
        value: Any,
        *,
        valid_from: datetime,
        source_fragment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add ``value`` to the attribute's current values (others stay current).

        Adding a value that is already current (or already added for a later
        time) is a no-op that returns the existing fact.
        """
        subject_key, attribute_key = _fact_key(subject), _fact_key(attribute)
        if not subject_key or not attribute_key:
            raise ValueError("fact subject and attribute must contain letters or digits")
        valid_text = _utc_text(valid_from)
        encoded = encode_value(value)
        with self._writing(structural=False) as conn:
            for row in self._open_rows(conn, agent_id, subject_key, attribute_key, valid_text):
                if self.encryption.decrypt_str(row["value"]) == encoded:
                    return self._fact_by_id(conn, str(row["id"]))
            fact_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO facts (id, agent_id, subject, subject_key, attribute, attribute_key, value,"
                " valid_from, valid_to, recorded_at, retracted_at, source_fragment_id, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)",
                (fact_id, agent_id, subject.strip(), subject_key, attribute.strip(), attribute_key,
                 self.encryption.encrypt_str(encoded), valid_text, datetime.now(timezone.utc).isoformat(),
                 source_fragment_id, json.dumps({**(metadata or {}), MULTI_KEY: True})),
            )
            return self._fact_by_id(conn, fact_id)

    def fact_remove(self, agent_id: str, subject: str, attribute: str, value: Any, *,
                    valid_to: datetime) -> int:
        """End an added value at ``valid_to`` (history keeps it); returns how many ended.

        Only multi-valued (added) values are ended this way; a value recorded
        with ``fact_set`` changes when the next value is set, or is withdrawn
        with ``fact_retract``.
        """
        subject_key, attribute_key = _fact_key(subject), _fact_key(attribute)
        stamp = _utc_text(valid_to)
        encoded = encode_value(value)
        ended = 0
        with self._writing(structural=False) as conn:
            for row in self._open_rows(conn, agent_id, subject_key, attribute_key, stamp):
                if _is_multi(row) and self.encryption.decrypt_str(row["value"]) == encoded:
                    conn.execute("UPDATE facts SET valid_to = ? WHERE id = ?", (stamp, row["id"]))
                    ended += 1
        return ended

    def fact_scheduled(self, agent_id: str, subject: str | None = None,
                       attribute: str | None = None) -> list[dict[str, Any]]:
        """Values that take effect in the future (``valid_from`` after now), soonest first."""
        sql = "SELECT * FROM facts WHERE agent_id = ? AND retracted_at IS NULL AND valid_from > ?"
        params: list[object] = [agent_id, datetime.now(timezone.utc).isoformat()]
        if subject is not None:
            sql += " AND subject_key = ?"
            params.append(_fact_key(subject))
        if attribute is not None:
            sql += " AND attribute_key = ?"
            params.append(_fact_key(attribute))
        rows = self._fetchall(sql + " ORDER BY valid_from, recorded_at, rowid", params)
        return [self._row_to_fact(row) for row in rows]

    def _open_rows(self, conn: Any, agent_id: str, subject_key: str, attribute_key: str, stamp: str) -> list[Any]:
        """Rows of one timeline still open at ``stamp`` (or starting later)."""
        return conn.execute(
            "SELECT id, value, metadata FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
            " AND retracted_at IS NULL AND (valid_to IS NULL OR valid_to > ?)",
            (agent_id, subject_key, attribute_key, stamp),
        ).fetchall()


def _utc_text(moment: datetime) -> str:
    moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()

