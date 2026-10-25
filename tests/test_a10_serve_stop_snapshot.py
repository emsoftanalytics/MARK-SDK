"""``mark serve`` stop rewrites a vector snapshot only for a large delta.

Release scale check: at a million memories the stop re-synced the index and
rewrote the ~1 GB snapshot after any write (38-60 s, past service-manager
stop timeouts). A small delta is left in ``vector_dirty`` for the next cold
start, which reads exactly those rows.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mark import Mark
from mark._cli import server_snapshots
from mark._cli.server_snapshots import _save_snapshots, snapshots_need_refresh
from mark.index import vector as vector_module
from mark.store._sqlite import vector_snapshot

pytest.importorskip("numpy")
AGENT = "scale"


@pytest.fixture
def snapshotted(tmp_path, monkeypatch):
    monkeypatch.setenv("MARK_EMBEDDER", "hash")
    monkeypatch.setattr(vector_module, "AUTO_NUMPY_THRESHOLD", 100)
    monkeypatch.setattr(vector_snapshot, "MIN_ROWS", 200)
    monkeypatch.setattr(vector_snapshot, "PART_ROWS", 128)
    with Mark.local(tmp_path) as mark:
        mark.runtime.memory(AGENT).store_many_sync(
            [f"svc-{i % 40:03d} was patched after a spike (ticket T-{i})." for i in range(400)])
    with Mark.local(tmp_path) as mark:   # the first cold start saves the snapshot
        mark.runtime.memory(AGENT).retrieve_sync("svc-001 patched")
    return tmp_path


def _spy(runtime) -> list[int]:
    calls: list[int] = []
    original = runtime.save_vector_snapshots
    runtime.save_vector_snapshots = lambda: calls.append(1) or original()
    return calls


def test_a_small_delta_is_left_for_the_next_cold_start(snapshotted) -> None:
    with Mark.local(snapshotted) as mark:
        runtime = mark.runtime
        memory = runtime.memory(AGENT)
        memory.retrieve_sync("svc-002")                       # index loaded, as in a server
        assert snapshots_need_refresh(runtime) is False      # nothing written yet
        new = memory.store_sync("zebra-42 handled the outage")
        assert runtime.store.vector_snapshot_delta(AGENT, runtime.embedder.fingerprint) >= 1
        calls = _spy(runtime)
        _save_snapshots(SimpleNamespace(runtime=runtime))
        assert calls == []                                    # stop: no 1 GB rewrite
    with Mark.local(snapshotted) as mark:                     # cold start: snapshot + delta
        memory = mark.runtime.memory(AGENT)
        assert memory.retrieve_sync("zebra-42 outage").fragments[0].id == new
        assert mark.runtime.index.count(AGENT) == 401


def test_a_large_delta_or_no_snapshot_is_refreshed(snapshotted, monkeypatch) -> None:
    monkeypatch.setattr(server_snapshots, "REFRESH_MIN_ROWS", 2)
    monkeypatch.setattr(server_snapshots, "REFRESH_FRACTION", 1000)
    with Mark.local(snapshotted) as mark:
        runtime = mark.runtime
        memory = runtime.memory(AGENT)
        memory.retrieve_sync("svc-002")
        memory.store_many_sync(["first new memo", "second new memo", "third new memo"])
        assert snapshots_need_refresh(runtime) is True
        calls = _spy(runtime)
        _save_snapshots(SimpleNamespace(runtime=runtime))
        assert calls == [1]
        assert runtime.store.vector_snapshot_delta(AGENT, runtime.embedder.fingerprint) == 0
        runtime.store.drop_vector_snapshot(AGENT)
        assert snapshots_need_refresh(runtime) is True        # large agent without a snapshot


def test_unknown_hosts_keep_saving() -> None:
    assert snapshots_need_refresh(SimpleNamespace(store=object(), index=None, embedder=None)) is True
