# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# One-time migration from the legacy JSON store to the SQLite backend.
# Triggered automatically when Mark.local() detects memory.json in .mark/.
"""Migration of legacy JSON stores into the SQLite store."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

_SIMPLE_AGENT = "__mark__"

# Old MemoryState values → new MemoryState values.
# Old RAW → UNVERIFIED: records that were stored (not quarantined/contradicted)
# should be retrievable in the new store.
_STATE_MAP: dict[str, MemoryState] = {
    "raw":          MemoryState.UNVERIFIED,
    "unverified":   MemoryState.UNVERIFIED,
    "verified":     MemoryState.VERIFIED,
    "promoted":     MemoryState.PROMOTED,
    "quarantined":  MemoryState.QUARANTINED,
    "contradicted": MemoryState.CONTRADICTED,
}

log = logging.getLogger("mark.migration")


@dataclass
class MigrationResult:
    """Summary of a JSON-to-SQLite store migration."""
    records_migrated: int = 0
    records_skipped:  int = 0
    source_path:      str = ""
    backup_path:      str = ""


def migrate_json_to_sqlite(json_path: Path, store: Any) -> MigrationResult:
    """
    Read a legacy .mark/memory.json and write each record into the SQLite store.

    On success, renames memory.json → memory.json.bak so the file is preserved
    but will not be picked up again on the next open.

    Idempotent: already-migrated fragments are silently overwritten (INSERT OR
    REPLACE semantics in LocalMemoryStore.store()).
    """
    result = MigrationResult(source_path=str(json_path))

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cannot read %s for migration: %s", json_path, exc)
        return result

    # Build block_id → label map so we can tag fragments with block:<label>
    blocks_by_id: dict[str, str] = {}
    for b in raw.get("blocks", []):
        bid   = str(b.get("id", ""))
        label = str(b.get("label", bid))
        if bid:
            blocks_by_id[bid] = label

    for item in raw.get("records", []):
        try:
            block_id  = str(item.get("block_id", ""))
            label     = blocks_by_id.get(block_id, block_id)
            old_state = str(item.get("state", "raw")).lower()
            new_state = _STATE_MAP.get(old_state, MemoryState.UNVERIFIED)

            meta: dict[str, Any] = dict(item.get("metadata") or {})
            meta["migrated_from"]     = "json"
            meta["original_block_id"] = block_id
            if item.get("feedback_score"):
                meta["feedback_score"] = float(item["feedback_score"])

            frag = MemoryFragment(
                id               = str(item["id"]),
                content          = str(item["content"]),
                agent_id         = _SIMPLE_AGENT,
                scope            = MemoryScope.AGENT,
                tier             = MemoryTier.EPISODIC,
                state            = new_state,
                importance       = float(item.get("importance", 0.5)),
                confidence       = float(item.get("confidence", 0.5)),
                source           = str(item.get("source", "local")),
                tags             = [f"block:{label}"] if label else [],
                metadata         = meta,
                created_at       = _parse_dt(item.get("created_at")),
                updated_at       = _parse_dt(item.get("created_at")),
                last_accessed_at = _parse_optional_dt(item.get("last_accessed_at")),
            )
            store.store(frag)
            result.records_migrated += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping record %s during migration: %s", item.get("id"), exc)
            result.records_skipped += 1

    backup = json_path.with_suffix(".json.bak")
    json_path.rename(backup)
    result.backup_path = str(backup)

    log.info(
        "Migrated %d record(s) from %s → SQLite. Backup: %s",
        result.records_migrated,
        json_path,
        backup,
    )
    return result


def _parse_dt(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_optional_dt(value: Any) -> datetime | None:
    return None if not value else _parse_dt(value)
