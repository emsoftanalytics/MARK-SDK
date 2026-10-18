"""Cold start from a vector snapshot: scope from the snapshot + write-log delta, cheap load.

At a million memories the cold start used to scan every row for the
retrievable-id set (4.4 s) and copy + renormalise the snapshot vectors.
These tests pin the faster path and, above all, that it answers exactly
like a full scan.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from mark import Mark
from mark.index import vector as vector_module
from mark.store._sqlite import scope_delta, scope_snapshot, vector_snapshot
from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryFragment, MemoryState

np = pytest.importorskip("numpy")
AGENT = "scale"


def _texts(n: int, start: int = 0) -> list[str]:
    return [f"svc-{i % 40:03d} was patched after a spike (ticket T-{i})." for i in range(start, start + n)]


@pytest.fixture
def snapshot_store(tmp_path, monkeypatch):
    """A 400-memory store whose first cold start has saved a snapshot (with its scope)."""
    monkeypatch.setattr(vector_module, "AUTO_NUMPY_THRESHOLD", 100)
    monkeypatch.setattr(vector_snapshot, "MIN_ROWS", 200)
    monkeypatch.setattr(vector_snapshot, "PART_ROWS", 128)
    with Mark.local(tmp_path) as mark:
        ids = mark.runtime.memory(AGENT).store_many_sync(_texts(400))
    with Mark.local(tmp_path) as mark:  # first cold start reads rows, then saves the snapshot
        mark.runtime.memory(AGENT).retrieve_sync("svc-001 patched")
        assert mark.runtime.store._fetchall("SELECT count FROM vector_snapshot_meta")[0][0] == len(ids)
    return tmp_path, ids


def _no_full_scan(monkeypatch) -> list[str]:
    scans: list[str] = []
    original = scope_delta.ScopeCacheMixin._full_scope

    def counting(self, version, where, params, agent_id, now):
        scans.append(where)
        return original(self, version, where, params, agent_id, now)

    monkeypatch.setattr(scope_delta.ScopeCacheMixin, "_full_scope", counting)
    return scans


def _fresh_ids(path) -> set[str]:
    store = LocalMemoryStore(path / ".mark" / "memory.db")
    try:
        return set(store.retrievable_ids(AGENT))
    finally:
        store.close()


def test_scope_comes_from_the_snapshot_plus_the_logged_delta(snapshot_store, monkeypatch) -> None:
    path, ids = snapshot_store
    with Mark.local(path) as mark:  # writes after the snapshot: logged in write_log
        memory = mark.runtime.memory(AGENT)
        new = memory.store_sync("zebra-42 handled the outage")
        mark.runtime.store.delete(ids[3])
        mark.runtime.store.forget_fragment(ids[4]) if hasattr(mark.runtime.store, "forget_fragment") else None
    expected = _fresh_ids(path)
    scans = _no_full_scan(monkeypatch)
    with Mark.local(path) as mark:
        memory = mark.runtime.memory(AGENT)
        assert set(mark.runtime.store.retrievable_id_set(AGENT)) == expected
        assert memory.retrieve_sync("zebra-42 outage").fragments[0].id == new
        assert mark.runtime.index.count(AGENT) == len(expected)
    assert scans == []  # no O(N) scan of the fragments table


def test_unknown_or_tampered_snapshot_scope_falls_back_to_a_full_scan(snapshot_store, monkeypatch) -> None:
    path, _ = snapshot_store
    store = LocalMemoryStore(path / ".mark" / "memory.db")
    key = f"{scope_snapshot.SCOPE_META_PREFIX}{AGENT}"
    info = json.loads(store.get_meta(key))
    store.set_meta(key, json.dumps({**info, "digest": "0" * len(info["digest"])}))
    store.close()
    scans = _no_full_scan(monkeypatch)
    expected = _fresh_ids(path)
    with Mark.local(path) as mark:
        assert set(mark.runtime.store.retrievable_id_set(AGENT)) == expected
    assert scans  # digest mismatch: the snapshot ids were not trusted


def test_expired_ttl_memories_never_come_back_from_a_snapshot_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vector_module, "AUTO_NUMPY_THRESHOLD", 100)
    monkeypatch.setattr(vector_snapshot, "MIN_ROWS", 200)
    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory(AGENT)
        memory.store_many_sync(_texts(300))
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        short = MemoryFragment(content="short lived note", agent_id=AGENT, ttl_seconds=1800, state=MemoryState.UNVERIFIED,
                               created_at=old + timedelta(minutes=59, seconds=59))
        mark.runtime.store.store(short)
    with Mark.local(tmp_path) as mark:  # snapshot saved while the note is still live
        mark.runtime.memory(AGENT).retrieve_sync("short lived note")
    store = LocalMemoryStore(tmp_path / ".mark" / "memory.db")
    with store._writing(structural=False) as conn:  # age it past its TTL without a logged write
        conn.execute("UPDATE fragments SET created_at = ? WHERE id = ?", (old.isoformat(), short.id))
    store.close()
    with Mark.local(tmp_path) as mark:
        assert short.id not in mark.runtime.store.retrievable_id_set(AGENT)


def test_snapshot_load_keeps_headroom_and_skips_renormalising(snapshot_store, monkeypatch) -> None:
    path, ids = snapshot_store
    norms: list[int] = []
    original = np.linalg.norm
    monkeypatch.setattr(np.linalg, "norm", lambda *a, **k: norms.append(1) or original(*a, **k))
    with Mark.local(path) as mark:
        mark.runtime.memory(AGENT).retrieve_sync("svc-001 patched")
        partition = mark.runtime.index._partitions[AGENT]
        assert partition._matrix.shape[0] >= len(ids) + len(ids) // 8  # room for the next writes
        grown = partition._matrix
        mark.runtime.memory(AGENT).store_sync("one more memory after reopen")
        mark.runtime.memory(AGENT).retrieve_sync("one more memory")
        assert partition._matrix is grown  # the first write after reopen did not copy the matrix
    assert norms == []  # snapshot vectors are stored normalised


def test_serve_saves_a_fresh_snapshot_on_graceful_stop(snapshot_store, monkeypatch) -> None:
    from mark._cli.server import serve

    path, ids = snapshot_store
    db = path / ".mark" / "memory.db"
    stop = threading.Event()
    ready = threading.Event()
    monkeypatch.setattr("mark._cli.server._warm", lambda engine: (engine.runtime.warm_up(), ready.set()))
    thread = threading.Thread(target=serve, args=(db,), kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    assert ready.wait(60)
    writer = LocalMemoryStore(db)  # another process writes while the server runs
    writer.store(MemoryFragment(content="written beside the server", agent_id=AGENT, state=MemoryState.UNVERIFIED))
    version = writer._write_version()
    writer.close()
    stop.set()
    thread.join(60)
    store = LocalMemoryStore(db)
    info = json.loads(store.get_meta(f"{scope_snapshot.SCOPE_META_PREFIX}{AGENT}"))
    assert info["version"] == version and info["count"] == len(ids) + 1
    store.close()


def test_a_cold_start_that_had_to_scan_saves_the_scope_for_the_next_one(snapshot_store, monkeypatch) -> None:
    path, ids = snapshot_store
    store = LocalMemoryStore(path / ".mark" / "memory.db")
    with store._writing(structural=False) as conn:  # as before this build, or after an unknown write
        conn.execute("DELETE FROM schema_info WHERE key = ?", (f"{scope_snapshot.SCOPE_META_PREFIX}{AGENT}",))
        conn.execute("DELETE FROM scope_snapshots")
    store.close()
    scans = _no_full_scan(monkeypatch)
    with Mark.local(path) as mark:  # scans once, and keeps the result
        assert len(mark.runtime.store.retrievable_id_set(AGENT)) == len(ids)
    with Mark.local(path) as mark:
        assert len(mark.runtime.store.retrievable_id_set(AGENT)) == len(ids)
    assert len(scans) == 1


def test_a_few_rows_since_the_snapshot_are_read_by_id(snapshot_store, monkeypatch) -> None:
    path, _ = snapshot_store
    with Mark.local(path) as mark:
        mark.runtime.memory(AGENT).store_many_sync(_texts(300, start=1000))  # > 256: the bulk path
    from mark.intelligence import index_bulk_load

    monkeypatch.setattr(index_bulk_load, "BY_ID_FRACTION", 2)  # 300 of 700 counts as "a few" here
    pages: list[str] = []
    monkeypatch.setattr(LocalMemoryStore, "embedding_pages", lambda self, agent_id, page_size=20_000: pages.append(agent_id) or iter(()))
    with Mark.local(path) as mark:
        memory = mark.runtime.memory(AGENT)
        assert memory.retrieve_sync("ticket T-1299").fragments
        assert mark.runtime.index.count(AGENT) == 700
    assert pages == []  # never streamed the whole agent for 300 new rows
