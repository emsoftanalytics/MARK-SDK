# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""A secure forget must not leave the memory's vector behind in a saved vector snapshot."""
from __future__ import annotations

from pathlib import Path

from mark.memory.runtime import MarkRuntime


def test_secure_forget_drops_the_agents_vector_snapshot(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "s.db")
    store = runtime.store
    fid = runtime.memory("a").remember("The launch codename is qzx-hidden-77.")
    keep = runtime.memory("b").remember("Another agent's note.")
    for agent, ids in (("a", [fid]), ("b", [keep])):
        version = int(store._fetchall("SELECT value FROM schema_info WHERE key = 'write_version'")[0][0])
        assert store.save_vector_snapshot(agent, "fp", ids, [b"\0" * 16], 4, version)
    count = "SELECT count(*) FROM vector_snapshots WHERE agent_id = ?"
    assert runtime.memory("a").forget(fid, secure=True) == 1
    assert store._fetchall(count, ("a",))[0][0] == 0      # forgotten agent's snapshot is gone
    assert store._fetchall(count, ("b",))[0][0] == 1      # other agents keep theirs
    runtime.shutdown()


def test_plain_forget_keeps_the_snapshot(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "p.db")
    store = runtime.store
    fid = runtime.memory("a").remember("ordinary note")
    version = int(store._fetchall("SELECT value FROM schema_info WHERE key = 'write_version'")[0][0])
    assert store.save_vector_snapshot("a", "fp", [fid], [b"\0" * 16], 4, version)
    runtime.memory("a").forget(fid)
    assert store._fetchall("SELECT count(*) FROM vector_snapshots WHERE agent_id = 'a'")[0][0] == 1
    runtime.shutdown()
