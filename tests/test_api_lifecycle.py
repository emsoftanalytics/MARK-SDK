# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""C3 memory lifecycle: cascading forget, secure erase, fact delete, supersede/resolve/update, kinds."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.errors import InvalidInputError, MemoryNotFoundError
from mark.memory.runtime import MarkRuntime


@pytest.fixture()
def mem(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    yield runtime.memory("agent")
    runtime.shutdown()


def _db_bytes(path: Path) -> bytes:
    """Every byte MARK keeps on disk for a database: the file plus its WAL/SHM."""
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*") if p.is_file())


# ── forget cascades ──────────────────────────────────────────────────────────


def test_forget_retracts_facts_extracted_from_the_memory(mem) -> None:
    fid = mem.remember("[2026-03-01] atlas-core runs on port 6430.", extract_facts=True)
    keep = mem.remember("[2026-03-02] billing runs on port 7000.", extract_facts=True)
    assert mem.facts.get("atlas-core", "port").value == "6430"
    assert mem.forget(fid) == 1
    assert mem.get(fid) is None
    assert mem.facts.current("atlas-core") == []            # gone from current values
    assert mem.facts.history("atlas-core", "port") == []    # and from history
    assert len(mem.timeline("atlas-core")) == 0             # and from the timeline
    assert mem.facts.get("billing", "port").value == "7000"  # other memories untouched
    assert mem.get(keep) is not None
    assert all(f.id != fid for f in mem.recall("atlas-core port").fragments)


def test_forget_unknown_id_and_other_agents_memory(mem) -> None:
    other = mem._runtime.memory("other")
    oid = other.remember("Belongs to someone else.")
    assert mem.forget("nope") == 0
    assert mem.forget(oid) == 0          # never forgets across namespaces
    assert other.get(oid) is not None


def test_forget_by_match_is_literal_and_case_insensitive(mem) -> None:
    mem.remember("The VPN password hint is otter.")
    mem.remember("Otter is the team mascot.")
    mem.remember("Unrelated note.")
    assert mem.forget(match="otter") == 2
    assert [f.content for f in mem.list(limit=None)] == ["Unrelated note."]
    with pytest.raises(InvalidInputError):
        mem.forget()
    with pytest.raises(InvalidInputError):
        mem.forget("x", match="y")


def test_secure_forget_and_vacuum_remove_text_from_disk(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    runtime = MarkRuntime.local(store_path=db)
    mem = runtime.memory("a")
    secret = "zqxv-secret-token-91"
    fid = mem.remember(f"[2026-01-01] vault-svc runs on port 5511 with key {secret}.", extract_facts=True)
    mem.remember("A harmless memory that stays.")
    assert secret.encode() in _db_bytes(db)
    assert mem.forget(fid, secure=True) == 1
    mem.vacuum()
    data = _db_bytes(db)
    assert secret.encode() not in data
    assert b"vault-svc" not in data  # extracted fact rows were hard-deleted too
    assert b"harmless memory" in data
    runtime.shutdown()


def test_facts_delete_is_a_hard_delete(mem) -> None:
    mem.facts.set("me", "city", "Lisbon", valid_from="2024-01-01")
    mem.facts.set("me", "city", "Porto", valid_from="2026-01-01")
    store = mem._store
    assert mem.facts.delete("me", "city", value="Porto") == 1
    assert mem.facts.get("me", "city").value == "Lisbon"  # neighbour takes over
    rows = store._fetchall("SELECT COUNT(*) FROM facts WHERE agent_id = 'agent'")
    assert rows[0][0] == 1  # nothing kept for audit, unlike retract


# ── supersede / update / resolve ─────────────────────────────────────────────


def test_supersede_keeps_history_and_retires_the_old_memory(mem) -> None:
    old = mem.remember("Storage backend is JSON files.", kind="decision", tags=["storage"], session_id="s1")
    new = mem.supersede(old, "Storage backend is SQLite.")
    retired, current = mem.get(old), mem.get(new)
    assert "mark:superseded" in retired.tags
    assert retired.metadata["lifecycle"] == "superseded" and retired.metadata["superseded_by"] == new
    assert current.metadata["supersedes"] == old and current.metadata["history"] == [old]
    assert set(current.tags) == {"storage", "kind:decision"} and current.session_id == "s1"
    newer = mem.update(new, "Storage backend is Postgres.")
    assert mem.get(newer).metadata["history"] == [old, new]
    assert [f.id for f in mem.list(kind="decision", open=True)] == [newer]
    assert mem.count(kind="decision") == 3 and mem.count(kind="decision", open=False) == 2
    with pytest.raises(MemoryNotFoundError):
        mem.supersede("missing", "x")


def test_resolve_closes_todos(mem) -> None:
    first = mem.remember("TODO: add retries to the uploader", kind="todo")
    second = mem.remember("TODO: write the changelog", kind="todos")  # plural forgiven
    mem.remember("Decision: keep the CLI stdlib-only", kind="decision")
    assert mem.resolve(first, note="done in #42") == first
    resolved = mem.get(first)
    assert {"mark:resolved", "mark:superseded"} <= set(resolved.tags)
    assert resolved.metadata["resolution"] == "done in #42" and "resolved_at" in resolved.metadata
    assert [f.id for f in mem.list(kind="todo", open=True)] == [second]
    assert mem.get(first) is not None  # still stored, still findable
    with pytest.raises(MemoryNotFoundError):
        mem.resolve("missing")


def test_kinds_are_tags_and_are_validated(mem) -> None:
    fid = mem.remember("Use ruff for linting.", kind="Preference")
    assert "kind:preference" in mem.get(fid).tags
    assert [f.id for f in mem.recall("linting", kind="preference").fragments] == [fid]
    assert mem.recall("linting", kind="bug").fragments == []
    with pytest.raises(InvalidInputError, match="decision, todo"):
        mem.remember("x", kind="banana")
