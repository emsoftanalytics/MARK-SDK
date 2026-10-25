# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Hippocampus maintenance: LocalMemoryStore methods it relies on (update_importance,
touch, list_edges, delete_edge, update_edge_weight, integrity_check, export,
schema_version).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.store import LocalMemoryStore
from mark.types import MemoryEdge, MemoryFragment, MemoryNode, MemoryState
from mark.types.graph import EdgeRelation


# ---------------------------------------------------------------------------
# LocalMemoryStore — new methods
# ---------------------------------------------------------------------------

def test_update_importance() -> None:
    store = LocalMemoryStore()
    frag = MemoryFragment(content="Alpha fact.", agent_id="a", state=MemoryState.UNVERIFIED)
    store.store(frag)

    store.update_importance(frag.id, 0.9)
    loaded = store.get(frag.id)
    assert loaded is not None
    assert abs(loaded.importance - 0.9) < 1e-6


def test_update_importance_clamps_to_unit_interval() -> None:
    store = LocalMemoryStore()
    frag = MemoryFragment(content="Beta fact.", agent_id="a", state=MemoryState.UNVERIFIED)
    store.store(frag)

    store.update_importance(frag.id, 1.5)
    assert store.get(frag.id).importance == 1.0

    store.update_importance(frag.id, -0.3)
    assert store.get(frag.id).importance == 0.0


def test_touch_updates_last_accessed_and_increments_access_count() -> None:
    store = LocalMemoryStore()
    frag = MemoryFragment(content="Gamma fact.", agent_id="a", state=MemoryState.UNVERIFIED)
    store.store(frag)

    assert store.get(frag.id).last_accessed_at is None

    store.touch(frag.id)
    loaded = store.get(frag.id)
    assert loaded.last_accessed_at is not None
    assert loaded.metadata.get("access_count") == 1

    store.touch(frag.id)
    assert store.get(frag.id).metadata.get("access_count") == 2


def test_list_edges_and_delete_edge() -> None:
    store = LocalMemoryStore()
    n1 = MemoryNode(label="n1", agent_id="a")
    n2 = MemoryNode(label="n2", agent_id="a")
    store.store_node(n1)
    store.store_node(n2)
    edge = MemoryEdge(source_id=n1.id, target_id=n2.id,
                      relation=EdgeRelation.RELATED_TO, agent_id="a", weight=0.8)
    store.store_edge(edge)

    edges = store.list_edges("a")
    assert any(e.id == edge.id for e in edges)

    deleted = store.delete_edge(edge.id)
    assert deleted is True
    assert not any(e.id == edge.id for e in store.list_edges("a"))

    # Deleting a non-existent edge returns False
    assert store.delete_edge("nonexistent") is False


def test_update_edge_weight() -> None:
    store = LocalMemoryStore()
    n1 = MemoryNode(label="x", agent_id="a")
    n2 = MemoryNode(label="y", agent_id="a")
    store.store_node(n1)
    store.store_node(n2)
    edge = MemoryEdge(source_id=n1.id, target_id=n2.id,
                      relation=EdgeRelation.RELATED_TO, agent_id="a", weight=0.4)
    store.store_edge(edge)

    store.update_edge_weight(edge.id, 0.7)
    reloaded = store.edges_from_node(n1.id)
    assert abs(reloaded[0].weight - 0.7) < 1e-6


def test_integrity_check_passes_on_fresh_store() -> None:
    store = LocalMemoryStore()
    assert store.integrity_check() == []


def test_schema_version_is_recorded() -> None:
    store = LocalMemoryStore()
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION


def test_export_creates_backup(tmp_path: Path) -> None:
    import sqlite3

    store = LocalMemoryStore(tmp_path / "source.db")
    frag = MemoryFragment(content="Exportable fact.", agent_id="a", state=MemoryState.UNVERIFIED)
    store.store(frag)

    dest = tmp_path / "backup.db"
    store.export(dest)

    assert dest.exists()
    with sqlite3.connect(str(dest)) as conn:
        row = conn.execute("SELECT content FROM fragments WHERE id = ?", (frag.id,)).fetchone()
    assert row is not None  # content present (may be encrypted; just check existence)


def test_pre_versioned_db_is_migrated_on_open(tmp_path: Path) -> None:
    """A DB created before schema_info existed is stamped to current version on open."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    # Simulate a pre-versioned DB: create fragments table but no schema_info
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE fragments (
                id TEXT PRIMARY KEY, content BLOB NOT NULL,
                block_id TEXT, scope TEXT NOT NULL, tier TEXT NOT NULL,
                agent_id TEXT NOT NULL, session_id TEXT,
                importance REAL DEFAULT 0.5, state TEXT DEFAULT 'raw',
                confidence REAL DEFAULT 1.0, ttl_seconds INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                last_accessed_at TEXT, tags TEXT DEFAULT '[]',
                source TEXT, metadata TEXT DEFAULT '{}',
                embedding TEXT, contradiction_ids TEXT DEFAULT '[]'
            )
        """)
        conn.execute(
            "INSERT INTO fragments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("legacy-id", "legacy content", None, "agent", "episodic", "legacy-agent",
             None, 0.5, "unverified", 1.0, None,
             "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00",
             None, "[]", None, "{}", None, "[]"),
        )
        conn.commit()

    # Opening the store should detect the legacy DB and stamp current version
    store = LocalMemoryStore(db_path)
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION


def test_export_raises_for_in_memory_store() -> None:
    store = LocalMemoryStore()  # :memory:
    with pytest.raises(ValueError, match="in-memory"):
        store.export("/tmp/nope.db")
