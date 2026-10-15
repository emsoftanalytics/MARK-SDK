# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Incremental retrievable-id scopes: writes are applied as deltas, not rescans.

Each test spies on ``_full_scope`` (the O(N) scan) and checks the cached set
against a fresh, uncached computation.
"""
from __future__ import annotations

import pickle
import random
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from mark import Mark
from mark.store import LocalMemoryStore
from mark.store._sqlite import scope_delta
from mark.store._sqlite.scope_ids import ScopeIds
from mark.types import BlockStatus, MemoryBlock, MemoryFragment, MemoryState

AGENT = "delta"


def _frag(text: str, **kw) -> MemoryFragment:
    kw.setdefault("agent_id", AGENT)
    kw.setdefault("state", MemoryState.UNVERIFIED)
    return MemoryFragment(content=text, **kw)


def _spy_full(store: LocalMemoryStore) -> list[int]:
    calls: list[int] = []
    original = store._full_scope
    store._full_scope = lambda *a: calls.append(1) or original(*a)  # type: ignore[method-assign]
    return calls


def _fresh(path, **filters) -> set[str]:
    ref = LocalMemoryStore(path)
    try:
        return ref.retrievable_ids(AGENT, **filters)
    finally:
        ref.close()


def test_write_then_read_applies_a_delta_without_rescanning(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "m.db")
    first = store.store_many([_frag(f"note {i}") for i in range(20)])
    before = store.retrievable_id_set(AGENT)
    calls = _spy_full(store)
    new = store.store(_frag("a fresh note"))
    after = store.retrievable_id_set(AGENT)
    assert calls == [] and after == set(first) | {new}
    assert after.parent == before.token and after.added == {new} and not after.removed
    store.update_state(first[0], MemoryState.CONTRADICTED)
    store.delete(first[1])
    final = store.retrievable_id_set(AGENT)
    assert calls == [] and final.removed == {first[0], first[1]}
    store.touch_many([new])  # non-structural: same object
    assert store.retrievable_id_set(AGENT) is final
    store.close()


def test_writes_without_fragment_changes_keep_the_same_set(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "m.db")
    store.store_many([_frag("one"), _frag("two")])
    before = store.retrievable_id_set(AGENT)
    calls = _spy_full(store)
    store.add_node_to_block("b1", "n1")
    assert store.retrievable_id_set(AGENT) is before and calls == []
    store.close()


def test_other_process_writes_are_applied_as_deltas(tmp_path) -> None:
    path = tmp_path / "shared.db"
    reader, writer = LocalMemoryStore(path), LocalMemoryStore(path)
    reader.store(_frag("seed"))
    reader.retrievable_id_set(AGENT)
    calls = _spy_full(reader)
    fid = writer.store(_frag("written elsewhere"))
    assert fid in reader.retrievable_id_set(AGENT) and calls == []
    writer.delete(fid)
    assert fid not in reader.retrievable_id_set(AGENT) and calls == []
    reader.close()
    writer.close()


def test_unknown_or_large_change_sets_rescan(tmp_path, monkeypatch) -> None:
    path = tmp_path / "m.db"
    store = LocalMemoryStore(path)
    block = MemoryBlock(name="b", agent_id=AGENT)
    store.store_block(block)
    inside = store.store(_frag("in a block", block_id=block.id))
    outside = store.store(_frag("loose"))
    assert store.retrievable_ids(AGENT) == {inside, outside}
    calls = _spy_full(store)
    store.store_block(block.model_copy(update={"status": BlockStatus.QUARANTINED}))  # unlogged ids
    assert store.retrievable_ids(AGENT) == {outside} and len(calls) == 1
    # Code that does not log (an older process) bumps the version only.
    raw = sqlite3.connect(path)
    raw.execute("UPDATE schema_info SET value = CAST(value AS INTEGER) + 1 WHERE key = 'write_version'")
    raw.commit()
    raw.close()
    assert store.retrievable_ids(AGENT) == {outside} and len(calls) == 2
    monkeypatch.setattr(scope_delta, "MAX_DELTA_IDS", 3)
    many = store.store_many([_frag(f"bulk {i}") for i in range(5)])
    assert store.retrievable_ids(AGENT) == {outside, *many} and len(calls) == 3
    store.close()


def test_ttl_expiry_removes_members_without_rescanning(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "ttl.db")
    keep = store.store(_frag("stays"))
    store.retrievable_id_set(AGENT)
    calls = _spy_full(store)
    created = datetime.now(timezone.utc) - timedelta(seconds=9)
    short = store.store(_frag("short lived", ttl_seconds=10, created_at=created))
    assert store.retrievable_ids(AGENT) == {keep, short}
    time.sleep(1.3)
    assert store.retrievable_ids(AGENT) == {keep} and calls == []
    store.close()


def test_random_operations_match_a_fresh_scan(tmp_path) -> None:
    path = tmp_path / "rand.db"
    store, other = LocalMemoryStore(path), LocalMemoryStore(path)
    rnd = random.Random(11)
    filters = [{}, {"session_id": "s1"}, {"tags": ["t"]}]
    ids: list[str] = []
    for step in range(120):
        op = rnd.random()
        writer = other if rnd.random() < 0.3 else store
        if op < 0.45 or not ids:
            frag = _frag(f"m{step}", session_id=rnd.choice(["s1", "s2"]), tags=rnd.choice([[], ["t"]]))
            ids.append(writer.store(frag))
        elif op < 0.6:
            writer.update_state(rnd.choice(ids), rnd.choice(list(MemoryState)))
        elif op < 0.7:
            writer.delete(rnd.choice(ids))
        elif op < 0.8:
            fid = rnd.choice(ids)
            frag = writer.get(fid)
            if frag is not None:  # re-store with other tags / session
                writer.store(frag.model_copy(update={"tags": rnd.choice([[], ["t"]]), "session_id": "s1"}))
        else:
            writer.forget_fragment(rnd.choice(ids))
        for kw in filters:
            assert store.retrievable_ids(AGENT, **kw) == _fresh(path, **kw), (step, kw)
    store.close()
    other.close()


def test_scope_ids_are_immutable_and_compare_by_identity() -> None:
    base = ScopeIds({"a", "b"})
    with pytest.raises(TypeError):
        base.add("c")
    alias = base
    alias |= {"c"}
    assert base == {"a", "b"} and alias == {"a", "b", "c"} and isinstance(alias, frozenset)
    derived = ScopeIds.derive(base, {"c", "a"}, {"b", "z"})
    assert derived == {"a", "c"} and derived.added == {"c"} and derived.removed == {"b"}
    assert derived.parent == base.token and hash(base) == hash(frozenset({"a", "b"}))
    assert pickle.loads(pickle.dumps(derived)) == frozenset({"a", "c"})


def test_recall_after_remember_syncs_the_index_by_delta(tmp_path) -> None:
    with Mark.local(tmp_path) as mark:
        runtime = mark.runtime
        memory = runtime.memory(AGENT)
        ids = memory.store_many_sync([f"svc-{i:03d} was patched" for i in range(60)])
        memory.retrieve_sync("svc-001 patched")
        sizes: list[int] = []
        original = runtime.index.missing
        runtime.index.missing = lambda wanted, *a: sizes.append(len(wanted)) or original(wanted, *a)
        new = memory.store_sync("zebra-42 handled the outage")
        assert memory.retrieve_sync("zebra-42 outage").fragments[0].id == new
        runtime.store.delete(ids[0])
        hits = {f.id for f in memory.retrieve_sync("svc-000 patched").fragments}
        assert ids[0] not in hits and runtime.index.count(AGENT) == 60
        assert sizes and max(sizes) <= 1  # only the delta was checked, never the whole scope
