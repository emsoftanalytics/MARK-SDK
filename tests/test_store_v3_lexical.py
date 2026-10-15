# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""LocalMemoryStore schema v3: FTS5 lexical channel (BM25, safety, index upkeep,
encryption) and integrity guards / store listeners.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.memory.runtime import MarkRuntime
from mark.security import EncryptionProvider
from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryBlock, MemoryScope, MemoryState

from _helpers_store_v3 import AGENT, _frag, store


class _XorEncryption(EncryptionProvider):
    def encrypt(self, data: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in data)

    def decrypt(self, data: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in data)


# ── FTS5 lexical channel ─────────────────────────────────────────────────────


def test_lexical_search_bm25(store: LocalMemoryStore) -> None:
    if not store.fts_available:
        pytest.skip("SQLite built without FTS5")
    k = _frag("Elena carries a brass key into the warehouse", tags=["props"])
    s = _frag("Elena wears a red scarf")
    other = _frag("brass key warehouse", agent_id="agent-b")
    for f in [k, s, other]:
        store.store(f)
    hits = store.lexical_search(AGENT, "Where is the brass KEY?")
    assert hits[0][0] == k.id
    assert all(score > 0 for _, score in hits)
    assert other.id not in {fid for fid, _ in hits}
    # Porter stemming: "carrying" matches "carries".
    assert store.lexical_search(AGENT, "carrying")[0][0] == k.id
    # Tags are indexed.
    assert store.lexical_search(AGENT, "props")[0][0] == k.id
    # include_ids restricts results (SQL path and Python path).
    assert [fid for fid, _ in store.lexical_search(AGENT, "elena", include_ids={s.id})] == [s.id]
    big = {s.id} | {f"pad-{i}" for i in range(1000)}
    assert [fid for fid, _ in store.lexical_search(AGENT, "elena", include_ids=big)] == [s.id]
    assert store.lexical_search(AGENT, "elena", include_ids=set()) == []


@pytest.mark.parametrize(
    "query",
    ['"', "AND OR NOT", "foo*", "NEAR(a b)", "col:value", "a", "", "   ", "(((", "^x", "' OR 1=1 --"],
)
def test_lexical_search_never_raises_on_user_text(store: LocalMemoryStore, query: str) -> None:
    store.store(_frag("ordinary content about value and things"))
    result = store.lexical_search(AGENT, query)
    assert isinstance(result, list)


def test_lexical_index_tracks_updates_deletes_and_clear(store: LocalMemoryStore) -> None:
    if not store.fts_available:
        pytest.skip("SQLite built without FTS5")
    f = _frag("alpha content")
    store.store(f)
    assert [fid for fid, _ in store.lexical_search(AGENT, "alpha")] == [f.id]
    store.store(f.model_copy(update={"content": "beta content"}))
    assert store.lexical_search(AGENT, "alpha") == []
    assert [fid for fid, _ in store.lexical_search(AGENT, "beta")] == [f.id]
    store.delete(f.id)
    assert store.lexical_search(AGENT, "beta") == []
    g = _frag("gamma")
    h = _frag("gamma", agent_id="agent-b")
    store.store(g)
    store.store(h)
    store.clear(AGENT)
    assert store.lexical_search(AGENT, "gamma") == []
    assert [fid for fid, _ in store.lexical_search("agent-b", "gamma")] == [h.id]
    store.clear()
    assert store.lexical_search("agent-b", "gamma") == []


def test_encrypted_store_never_indexes_plaintext(tmp_path: Path) -> None:
    db = tmp_path / "enc.db"
    store = LocalMemoryStore(db, encryption=_XorEncryption())
    assert store.fts_available is False
    store.store(_frag("top secret launch codes"))
    assert store.lexical_search(AGENT, "secret") == []
    tables = {row[0] for row in store._conn().execute("SELECT name FROM sqlite_master")}
    assert "fragments_fts" not in tables
    store.close()


# ── integrity guards & listeners ─────────────────────────────────────────────


def test_store_refuses_cross_agent_overwrite(store: LocalMemoryStore) -> None:
    f = _frag("mine")
    store.store(f)
    with pytest.raises(ValueError):
        store.store(f.model_copy(update={"agent_id": "agent-b", "content": "stolen"}))
    assert store.get(f.id).content == "mine"
    assert store.get(f.id).agent_id == AGENT
    # Same agent may update.
    store.store(f.model_copy(update={"content": "mine v2"}))
    assert store.get(f.id).content == "mine v2"


def test_cannot_move_fragment_out_of_sealed_block(store: LocalMemoryStore) -> None:
    sealed = MemoryBlock(name="sealed", agent_id=AGENT, scope=MemoryScope.AGENT, status=BlockStatus.SEALED)
    open_block = MemoryBlock(name="open", agent_id=AGENT, scope=MemoryScope.AGENT)
    store.store_block(sealed)
    store.store_block(open_block)
    f = _frag("sealed member", block_id=sealed.id)
    store.store(f)
    with pytest.raises(ValueError, match="sealed"):
        store.assign_fragment_block(f.id, open_block.id)
    with pytest.raises(ValueError, match="sealed"):
        store.assign_fragment_block(f.id, None)
    assert store.get(f.id).block_id == sealed.id
    store.assign_fragment_block(f.id, sealed.id)  # no-op move is allowed
    loose = _frag("loose")
    store.store(loose)
    store.assign_fragment_block(loose.id, open_block.id)
    assert store.get(loose.id).block_id == open_block.id


def test_listeners_fire_on_delete_clear_and_unretrievable(store: LocalMemoryStore) -> None:
    events: list[tuple[str, str]] = []
    store.add_listener(lambda event, fid: events.append((event, fid)))

    def broken(event: str, fid: str) -> None:
        raise RuntimeError("listener bug")

    store.add_listener(broken)
    a, b, c = _frag("a"), _frag("b"), _frag("c")
    for f in (a, b, c):
        store.store(f)
    store.update_state(a.id, MemoryState.VERIFIED)
    store.update_state(a.id, MemoryState.QUARANTINED)
    assert store.delete(b.id) is True
    assert store.delete(b.id) is False
    store.clear(AGENT)
    assert events[:2] == [("unretrievable", a.id), ("delete", b.id)]
    assert sorted(events[2:]) == sorted([("delete", a.id), ("delete", c.id)])


def test_tampered_fragment_is_persisted_contradicted(store: LocalMemoryStore) -> None:
    events: list[tuple[str, str]] = []
    store.add_listener(lambda event, fid: events.append((event, fid)))
    f = _frag("original truth")
    store.store(f)
    store._conn().execute(
        "UPDATE fragments SET content = ? WHERE id = ?", (store.encryption.encrypt_str("forged"), f.id)
    )
    store._conn().commit()
    assert store.get(f.id).state == MemoryState.CONTRADICTED
    assert events == [("unretrievable", f.id)]
    assert f.id not in store.retrievable_ids(AGENT)
    # Already persisted: a second read does not fire again.
    assert store.get_many([f.id])[f.id].state == MemoryState.CONTRADICTED
    assert events == [("unretrievable", f.id)]


def test_runtime_index_is_pipeline_index_and_follows_store_events() -> None:
    from mark.index import VectorIndex

    shared = VectorIndex()
    runtime = MarkRuntime.local()
    try:
        assert runtime.pipeline._index is runtime.index
        rt2 = MarkRuntime(store=LocalMemoryStore(), embedder=runtime.embedder, plugins=runtime.plugins, index=shared)
        assert rt2.index is shared
        mem = runtime.memory(AGENT)
        frag = mem.store_sync("Elena carries a brass key.", importance=0.9)
        fid = frag.id if hasattr(frag, "id") else frag
        runtime.pipeline.retrieve("brass key", AGENT)
        assert fid in runtime.index
        runtime.store.update_state(fid, MemoryState.QUARANTINED)
        assert fid not in runtime.index
        runtime.index.add(fid, [1.0, 0.0] + [0.0] * (runtime.embedder.dim - 2))
        runtime.store.delete(fid)
        assert fid not in runtime.index
        rt2.shutdown()
    finally:
        runtime.shutdown()
