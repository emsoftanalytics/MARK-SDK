# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Build stores that look like ones written by older MARK versions (schema 4 / 5).

The store is written with this build and then rolled back to the older
layout: tables, indexes and markers added by later schema versions are
dropped and the version stamp is lowered. End-to-end tests with stores
written by the real a8/a9 releases live in ``test_a10_upgrade_e2e.py``.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryFragment
from mark.types.memory_document import MemoryDocument

AGENT = "a"
TEXTS = [
    "Ana moved to Lisbon in 2024.",
    "上海仓库的库存周五盘点。",
    "Deploy finished 🚀 without errors.",
    "TODO: rotate the staging keys.",
]
KEY_TABLES = ("fragments", "facts", "documents")


def build_store(path: Path, *, version: int = 5, encryption=None) -> dict[str, int]:
    """Write memories (CJK/emoji), a fact with history and a document; return key-table counts."""
    store = LocalMemoryStore(path, encryption=encryption, backup_on_upgrade=False)
    for text in TEXTS:
        store.store(MemoryFragment(content=text, agent_id=AGENT))
    store.fact_set(AGENT, "ana", "city", "Lisbon", valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc))
    store.fact_set(AGENT, "ana", "city", "Porto", valid_from=datetime(2025, 3, 1, tzinfo=timezone.utc))
    store.store_document(MemoryDocument(title="Runbook", content="Restart the API first.", agent_id=AGENT))
    store.close()
    downgrade(path, version)
    return counts(path)


def downgrade(path: Path, version: int) -> None:
    """Remove what schema versions above *version* added and stamp *version*."""
    conn = sqlite3.connect(path)
    try:
        if version <= 5:
            conn.execute("DROP TRIGGER IF EXISTS fragments_fts_ml_delete")
            conn.execute("DROP TABLE IF EXISTS fragments_fts_ml")
            conn.execute("DROP INDEX IF EXISTS idx_frag_ttl")
            for table in ("vector_snapshot_meta", "vector_snapshots", "vector_dirty"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("DELETE FROM schema_info WHERE key = 'fts_ml_backfilled'")
        if version <= 4:
            for table in ("fact_jobs", "fact_extractions"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("UPDATE schema_info SET value = ? WHERE key = 'schema_version'", (str(version),))
        conn.commit()
    finally:
        conn.close()


def counts(path: Path, tables=KEY_TABLES) -> dict[str, int]:
    """Row counts of *tables* (read-only open: never creates -wal/-shm next to a backup)."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        conn.close()


def stamp(path: Path) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()[0])
    finally:
        conn.close()


def dump(path: Path) -> list[str]:
    """Every statement of the database (schema and rows): equal dumps = same content."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def backups(folder: Path) -> list[Path]:
    return sorted(folder.glob("*-backup-*.db"))
