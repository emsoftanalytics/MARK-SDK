# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# LocalMemoryStore schema v3: thread safety, migrations, float32 embeddings,
# FTS5 lexical channel, batch APIs, listeners, and integrity guards.
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mark.memory.runtime import MarkRuntime
from mark.security import EncryptionProvider
from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryBlock, MemoryFragment, MemoryScope, MemoryState, MemoryTier

AGENT = "agent-a"


def _frag(content: str, **kwargs) -> MemoryFragment:
    kwargs.setdefault("agent_id", AGENT)
    kwargs.setdefault("state", MemoryState.UNVERIFIED)
    return MemoryFragment(content=content, **kwargs)


class _XorEncryption(EncryptionProvider):
    def encrypt(self, data: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in data)

    def decrypt(self, data: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in data)


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path: Path):
    path = ":memory:" if request.param == "memory" else tmp_path / "mem.db"
    s = LocalMemoryStore(path)
    yield s
    s.close()


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
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION == 3
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
    assert store.schema_version() == 3
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
    assert store.schema_version() == 3


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
