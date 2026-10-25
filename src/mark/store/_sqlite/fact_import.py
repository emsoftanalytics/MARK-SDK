"""Idempotent restore of exported fact rows (``import_jsonl``).

Replaying an export through ``fact_set`` is not idempotent: two values
recorded at the same ``valid_from`` (a status that changed twice in a day)
re-import as 6, 7, 6, 7, because each re-asserted value differs from the
latest one at that time. An exported row carries its ``recorded_at``, so
(subject, attribute, valid_from, recorded_at, value) identifies it exactly:
``import_fact_row`` inserts the row with that ``recorded_at`` unless an
identical live row is already there, then re-derives the timeline. Rows
from exports without ``recorded_at`` are matched on the value at that
``valid_from`` instead.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from mark.store._sqlite.codec import _fact_key
from mark.store._sqlite.fact_codec import encode_value
from mark.store._sqlite.facts import MULTI_KEY


def _utc_text(value: Any) -> str:
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat()


class FactImportMixin:
    """``import_fact_row``: add an exported fact row once, however often it is imported."""

    def import_fact_row(self, agent_id: str, record: dict[str, Any]) -> bool:
        """Insert an exported fact row unless an identical one exists; True when it was added.

        ``record`` is a row as ``export_jsonl`` writes it (``subject``,
        ``attribute``, ``value``, ``valid_from``, optional ``recorded_at``,
        ``source_fragment_id`` and ``metadata``).
        """
        subject, attribute = str(record["subject"]), str(record["attribute"])
        subject_key, attribute_key = _fact_key(subject), _fact_key(attribute)
        if not subject_key or not attribute_key:
            raise ValueError("fact subject and attribute must contain letters or digits")
        valid_text = _utc_text(record["valid_from"])
        recorded = record.get("recorded_at")
        recorded_text = _utc_text(recorded) if recorded else None
        encoded = encode_value(record["value"])
        metadata = dict(record.get("metadata") or {})
        with self._writing(structural=False) as conn:
            for row in conn.execute(
                "SELECT value, recorded_at FROM facts WHERE agent_id = ? AND subject_key = ? AND attribute_key = ?"
                " AND valid_from = ? AND retracted_at IS NULL",
                (agent_id, subject_key, attribute_key, valid_text),
            ).fetchall():
                same_time = recorded_text is None or str(row["recorded_at"]) == recorded_text
                if same_time and self.encryption.decrypt_str(row["value"]) == encoded:
                    return False  # already imported (or recorded here in the first place)
            conn.execute(
                "INSERT INTO facts (id, agent_id, subject, subject_key, attribute, attribute_key, value,"
                " valid_from, valid_to, recorded_at, retracted_at, source_fragment_id, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)",
                (str(uuid.uuid4()), agent_id, subject.strip(), subject_key, attribute.strip(), attribute_key,
                 self.encryption.encrypt_str(encoded), valid_text,
                 recorded_text or datetime.now(timezone.utc).isoformat(),
                 record.get("source_fragment_id"), json.dumps(metadata)),
            )
            if not metadata.get(MULTI_KEY):  # multi-valued rows keep their own interval
                self._rechain_facts(conn, agent_id, subject_key, attribute_key)
        return True
