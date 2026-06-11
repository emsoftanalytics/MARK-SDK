"""Redaction applied to records before any sync payload is built."""
from __future__ import annotations

from mark.memory import MemoryRecord
from mark.security import redact_secrets, redact_sync_value


def redact_records_for_sync(records: list[MemoryRecord]) -> list[dict[str, object]]:
    """Return records as redacted dictionaries for sync."""
    return [
        {
            **redact_sync_value(record.to_dict()),
            "content": redact_secrets(record.content),
        }
        for record in records
    ]
