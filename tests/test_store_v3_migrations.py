# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""LocalMemoryStore schema v3: in-place migrations from older schemas, meta helpers,
float32 embedding storage, and batch get/touch/importance APIs.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryState

from _helpers_store_v3 import AGENT, _frag, store


# ── migrations ───────────────────────────────────────────────────────────────

_LEGACY_FRAGMENTS = """
    CREATE TABLE fragments (
        id TEXT PRIMARY KEY, content BLOB NOT NULL, block_id TEXT, scope TEXT NOT NULL,
        tier TEXT NOT NULL, agent_id TEXT NOT NULL, session_id TEXT,
        importance REAL DEFAULT 0.5, state TEXT DEFAULT 'raw', confidence REAL DEFAULT 1.0,
        ttl_seconds INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        last_accessed_at TEXT, tags TEXT DEFAULT '[]', source TEXT,
        metadata TEXT DEFAULT '{}', embedding TEXT, contradiction_ids TEXT DEFAULT '[]'
    )
"""


def test_pre_versioned_db_with_blocks_and_edges_but_no_fragments_migrates(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            _LEGACY_FRAGMENTS
            + """;
            CREATE TABLE blocks (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, agent_id TEXT NOT NULL,
                scope TEXT NOT NULL, block_type TEXT DEFAULT 'ltm',
                metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
            );
            INSERT INTO blocks VALUES ('b1', 'n', 'agent-a', 'agent', 'ltm', '{}', '2025-01-01T00:00:00+00:00');
            CREATE TABLE edges (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
                relation TEXT NOT NULL, weight REAL DEFAULT 1.0, agent_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
            );
            """
        )
    store = LocalMemoryStore(db)
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION
    assert store.get_block("b1").status == BlockStatus.OPEN
    assert store.edges_in_block("b1") == []  # previously: no such column: block_id
    store.close()


def test_v2_db_migrates_to_v3_in_place_and_keeps_legacy_embeddings(tmp_path: Path) -> None:
    db = tmp_path / "v2.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            _LEGACY_FRAGMENTS
            + """;
            CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_info VALUES ('schema_version', '2');
            """
        )
        conn.execute(
            "INSERT INTO fragments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("legacy", "the brass key opens the warehouse", None, "agent", "episodic", AGENT,
             None, 0.5, "unverified", 1.0, None, "2025-01-01T00:00:00+00:00",
             "2025-01-01T00:00:00+00:00", None, '["keys"]', None, "{}", json.dumps([0.6, 0.8]), "[]"),
        )
    store = LocalMemoryStore(db)
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION
    cols = {row[1] for row in store._conn().execute("PRAGMA table_info(fragments)")}
    assert "embedding_blob" in cols
    assert store.get("legacy").embedding == pytest.approx([0.6, 0.8])
    assert store.embeddings_for_agent(AGENT) == [("legacy", pytest.approx([0.6, 0.8]))]
    indexes = {row[1] for row in store._conn().execute("SELECT type, name FROM sqlite_master WHERE type='index'")}
    assert {"idx_frag_block", "idx_frag_agent_state_imp", "idx_node_label_agent"} <= indexes
    if store.fts_available:
        assert store.lexical_search(AGENT, "warehouse")[0][0] == "legacy"
    store.close()
    # Idempotent: reopening does not fail or duplicate FTS rows.
    again = LocalMemoryStore(db)
    if again.fts_available:
        assert len(again.lexical_search(AGENT, "warehouse")) == 1
    again.close()


def test_meta_helpers(store: LocalMemoryStore) -> None:
    assert store.get_meta("embedder") is None
    store.set_meta("embedder", "mark.hash-v2:256")
    store.set_meta("embedder", "mark.hash-v2:512")
    assert store.get_meta("embedder") == "mark.hash-v2:512"
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION


# ── embeddings ───────────────────────────────────────────────────────────────


def test_embeddings_stored_as_float32_blob(store: LocalMemoryStore) -> None:
    frag = _frag("vec", embedding=[0.1, 0.2, 0.3])
    store.store(frag)
    blob, legacy = store._conn().execute(
        "SELECT embedding_blob, embedding FROM fragments WHERE id = ?", (frag.id,)
    ).fetchone()
    assert isinstance(blob, bytes) and len(blob) == 12
    assert legacy is None
    assert store.get(frag.id).embedding == pytest.approx([0.1, 0.2, 0.3], abs=1e-7)

    store.update_embedding(frag.id, [1.0, 0.0])
    assert store.get(frag.id).embedding == [1.0, 0.0]


def test_embeddings_for_agent_is_unlimited_and_state_filtered(store: LocalMemoryStore) -> None:
    for i in range(250):
        store.store(_frag(f"f{i}", embedding=[float(i), 1.0]))
    store.store(_frag("raw", state=MemoryState.RAW, embedding=[1.0, 1.0]))
    store.store(_frag("no embedding"))
    rows = store.embeddings_for_agent(AGENT)
    assert len(rows) == 251
    assert sum(1 for _, vec in rows if vec is None) == 1


# ── batch APIs ───────────────────────────────────────────────────────────────


def test_get_many_chunks_and_matches_get(store: LocalMemoryStore) -> None:
    frags = [_frag(f"fact {i}") for i in range(1000)]
    for f in frags:
        store.store(f)
    got = store.get_many([f.id for f in frags] + ["missing", frags[0].id])
    assert len(got) == 1000
    assert got[frags[5].id] == store.get(frags[5].id)


def test_touch_many_and_update_importance_many(store: LocalMemoryStore) -> None:
    a, b = _frag("a"), _frag("b")
    store.store(a)
    store.store(b)
    store.touch_many([a.id, b.id, a.id, "missing"])
    assert store.get(a.id).metadata["access_count"] == 2
    assert store.get(b.id).metadata["access_count"] == 1
    assert store.get(a.id).last_accessed_at is not None

    store.update_importance_many({a.id: 1.5, b.id: 0.25})
    assert store.get(a.id).importance == 1.0
    assert store.get(b.id).importance == 0.25
