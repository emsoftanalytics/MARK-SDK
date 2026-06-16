# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Tests for JSON → SQLite migration.
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from mark import Mark
from mark.memory.migration import MigrationResult, migrate_json_to_sqlite
from mark.store import LocalMemoryStore


def _make_json_store(mark_dir: Path, records: list[dict], blocks: list[dict] | None = None) -> None:
    mark_dir.mkdir(parents=True, exist_ok=True)
    data = {"blocks": blocks or [], "records": records}
    (mark_dir / "memory.json").write_text(json.dumps(data), encoding="utf-8")


def test_migrate_json_records_into_sqlite(tmp_path: Path) -> None:
    rid = f"rec_{uuid4().hex}"
    bid = f"blk_{uuid4().hex}"
    _make_json_store(
        tmp_path / ".mark",
        blocks=[{"id": bid, "label": "project", "kind": "custom", "description": "",
                 "active": True, "metadata": {}, "created_at": "2025-01-01T00:00:00+00:00",
                 "updated_at": "2025-01-01T00:00:00+00:00"}],
        records=[{"id": rid, "block_id": bid, "content": "FastAPI is used.",
                  "importance": 0.9, "confidence": 0.8, "state": "raw",
                  "metadata": {}, "source": "local",
                  "created_at": "2025-01-01T00:00:00+00:00",
                  "last_accessed_at": None, "feedback_score": 0.0}],
    )

    with Mark.local(project_path=tmp_path) as mark:
        bundle = mark.memory.retrieve("FastAPI", blocks=["project"])
        assert any("FastAPI" in r.content for r in bundle.memories)
        assert not (tmp_path / ".mark" / "memory.json").exists()
        assert (tmp_path / ".mark" / "memory.json.bak").exists()


def test_migration_preserves_record_id(tmp_path: Path) -> None:
    rid = f"rec_{uuid4().hex}"
    store = LocalMemoryStore(tmp_path / "mem.db")
    json_path = tmp_path / "memory.json"
    _make_json_store(
        tmp_path,
        records=[{"id": rid, "block_id": "b1", "content": "Preserved ID content.",
                  "importance": 0.7, "confidence": 0.7, "state": "verified",
                  "metadata": {}, "source": "local",
                  "created_at": "2025-06-01T12:00:00+00:00",
                  "last_accessed_at": None, "feedback_score": 0.0}],
    )
    json_path = tmp_path / "memory.json"

    result = migrate_json_to_sqlite(json_path, store)

    assert result.records_migrated == 1
    assert result.records_skipped == 0
    frag = store.get(rid)
    assert frag is not None
    assert frag.id == rid
    assert frag.content == "Preserved ID content."
    assert frag.metadata.get("migrated_from") == "json"


def test_migration_maps_raw_state_to_unverified(tmp_path: Path) -> None:
    from mark.types import MemoryState

    rid = f"rec_{uuid4().hex}"
    store = LocalMemoryStore(tmp_path / "mem.db")
    _make_json_store(
        tmp_path,
        records=[{"id": rid, "block_id": "b", "content": "Raw record.",
                  "importance": 0.5, "confidence": 0.5, "state": "raw",
                  "metadata": {}, "source": "local",
                  "created_at": "2025-01-01T00:00:00+00:00",
                  "last_accessed_at": None, "feedback_score": 0.0}],
    )

    migrate_json_to_sqlite(tmp_path / "memory.json", store)

    frag = store.get(rid)
    assert frag.state == MemoryState.UNVERIFIED


def test_migration_tags_fragment_with_block_label(tmp_path: Path) -> None:
    rid = f"rec_{uuid4().hex}"
    bid = f"blk_{uuid4().hex}"
    store = LocalMemoryStore(tmp_path / "mem.db")
    _make_json_store(
        tmp_path,
        blocks=[{"id": bid, "label": "api-docs", "kind": "custom", "description": "",
                 "active": True, "metadata": {}, "created_at": "2025-01-01T00:00:00+00:00",
                 "updated_at": "2025-01-01T00:00:00+00:00"}],
        records=[{"id": rid, "block_id": bid, "content": "API docs content.",
                  "importance": 0.6, "confidence": 0.6, "state": "verified",
                  "metadata": {}, "source": "local",
                  "created_at": "2025-01-01T00:00:00+00:00",
                  "last_accessed_at": None, "feedback_score": 0.0}],
    )

    migrate_json_to_sqlite(tmp_path / "memory.json", store)

    frag = store.get(rid)
    assert "block:api-docs" in frag.tags


def test_migration_renames_json_to_bak(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "mem.db")
    _make_json_store(tmp_path, records=[])
    json_path = tmp_path / "memory.json"

    result = migrate_json_to_sqlite(json_path, store)

    assert not json_path.exists()
    assert Path(result.backup_path).exists()
    assert result.backup_path.endswith(".json.bak")


def test_migration_handles_corrupt_json_gracefully(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "mem.db")
    bad_path = tmp_path / "memory.json"
    bad_path.write_text("{not valid json", encoding="utf-8")

    result = migrate_json_to_sqlite(bad_path, store)

    assert result.records_migrated == 0
    assert bad_path.exists()  # not renamed — migration did not complete


def test_second_mark_local_open_skips_migration(tmp_path: Path) -> None:
    """After migration (json renamed to .bak), second open finds no json to migrate."""
    rid = f"rec_{uuid4().hex}"
    _make_json_store(
        tmp_path / ".mark",
        records=[{"id": rid, "block_id": "b", "content": "Once migrated.",
                  "importance": 0.5, "confidence": 0.5, "state": "raw",
                  "metadata": {}, "source": "local",
                  "created_at": "2025-01-01T00:00:00+00:00",
                  "last_accessed_at": None, "feedback_score": 0.0}],
    )

    with Mark.local(project_path=tmp_path):
        pass  # first open triggers migration

    # Second open — no json present, should not raise or re-migrate
    with Mark.local(project_path=tmp_path) as mark:
        bundle = mark.memory.retrieve("migrated")
        assert any("migrated" in r.content for r in bundle.memories)
