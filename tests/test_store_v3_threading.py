# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""LocalMemoryStore schema v3: thread safety, connection lifecycle, pragmas, and
agent/session/block filters (including retrievable_ids semantics).
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryBlock, MemoryScope, MemoryState, MemoryTier

from _helpers_store_v3 import AGENT, _frag, store


# ── thread safety ────────────────────────────────────────────────────────────


def test_concurrent_reads_and_writes_do_not_corrupt_rows(store: LocalMemoryStore) -> None:
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(40):
                frag = _frag(f"worker {n} fact {i}", tags=[f"w{n}"])
                store.store(frag)
                got = store.get(frag.id)
                assert got is not None and got.content == frag.content
                store.list_by_agent(AGENT, tags=[f"w{n}"], limit=None)
                store.retrievable_ids(AGENT)
                store.touch(frag.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count(AGENT) == 240
    assert all(f.state == MemoryState.UNVERIFIED for f in store.list_by_agent(AGENT, limit=None))


def test_close_closes_connections_from_every_thread(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "mem.db")
    opened: list[sqlite3.Connection] = []

    def worker() -> None:
        store.count(AGENT)
        opened.append(store._conn())

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    opened.append(store._conn())
    store.close()
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_export_closes_destination(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "src.db")
    frag = _frag("exported")
    store.store(frag)
    dest = tmp_path / "dest.db"
    store.export(dest)
    # Destination must not be held open: it can be removed and reopened cleanly.
    reopened = LocalMemoryStore(dest)
    assert reopened.get(frag.id).content == "exported"
    reopened.close()
    store.close()


def test_pragmas_applied(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "mem.db")
    conn = store._conn()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2  # MEMORY
    assert conn.execute("PRAGMA cache_size").fetchone()[0] == -32000
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    store.close()


# ── filters ──────────────────────────────────────────────────────────────────


def test_session_prefix_escapes_like_wildcards(store: LocalMemoryStore) -> None:
    store.store(_frag("literal", session_id="ep_1%x"))
    store.store(_frag("wildcard victim", session_id="epA1zzz"))
    store.store(_frag("underscore victim", session_id="ep11%x"))
    hits = store.list_by_agent(AGENT, session_prefix="ep_1%")
    assert [f.content for f in hits] == ["literal"]
    assert len(store.retrievable_ids(AGENT, session_prefix="ep_1%")) == 1


def test_list_by_agent_limit_none_returns_everything(store: LocalMemoryStore) -> None:
    for i in range(250):
        store.store(_frag(f"fact {i}"))
    assert len(store.list_by_agent(AGENT)) == 200
    assert len(store.list_by_agent(AGENT, limit=None)) == 250


def test_retrievable_ids_semantics(store: LocalMemoryStore) -> None:
    keep = _frag("keep", tags=["a", "b"])
    one_tag = _frag("one tag", tags=["a"])
    raw = _frag("raw", state=MemoryState.RAW, tags=["a", "b"])
    quarantined_state = _frag("q", state=MemoryState.QUARANTINED, tags=["a", "b"])
    expired = _frag(
        "expired",
        tags=["a", "b"],
        ttl_seconds=10,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    live_ttl = _frag("live ttl", tags=["a", "b"], ttl_seconds=3600)
    block = MemoryBlock(name="bad", agent_id=AGENT, scope=MemoryScope.AGENT, status=BlockStatus.QUARANTINED)
    store.store_block(block)
    in_quarantine = _frag("in quarantined block", tags=["a", "b"], block_id=block.id)
    other_agent = _frag("other", agent_id="agent-b", tags=["a", "b"])
    for f in [keep, one_tag, raw, quarantined_state, expired, live_ttl, in_quarantine, other_agent]:
        store.store(f)

    assert store.retrievable_ids(AGENT, tags=["a", "b"]) == {keep.id, live_ttl.id}
    assert store.retrievable_ids(AGENT) == {keep.id, one_tag.id, live_ttl.id}
    working = _frag("working", tier=MemoryTier.WORKING)
    store.store(working)
    assert store.retrievable_ids(AGENT, tier=MemoryTier.WORKING) == {working.id}


def test_retrievable_ids_has_no_row_limit(store: LocalMemoryStore) -> None:
    for i in range(1200):
        store.store(_frag(f"fact {i}"))
    assert len(store.retrievable_ids(AGENT)) == 1200


def test_retrievable_ids_block_filters(store: LocalMemoryStore) -> None:
    b1 = MemoryBlock(name="b1", agent_id=AGENT, scope=MemoryScope.AGENT)
    b2 = MemoryBlock(name="b2", agent_id=AGENT, scope=MemoryScope.AGENT)
    store.store_block(b1)
    store.store_block(b2)
    f1 = _frag("one", block_id=b1.id)
    f2 = _frag("two", block_id=b2.id)
    store.store(f1)
    store.store(f2)
    assert store.retrievable_ids(AGENT, block_id=b1.id) == {f1.id}
    assert store.retrievable_ids(AGENT, block_ids=[b1.id, b2.id]) == {f1.id, f2.id}
