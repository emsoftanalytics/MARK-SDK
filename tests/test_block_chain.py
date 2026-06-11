# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Block sealing, hash-chain verification, quarantine isolation, and migration."""

import sqlite3

import pytest

from mark.memory.block_chain import BlockChain
from mark.memory.block_graph import BlockGraph
from mark.runtime import Mark
from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryFragment, MemoryNode
from mark.types.graph import NodeType

AGENT = "chain-test-agent"


@pytest.fixture()
def store():
    s = LocalMemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def graph(store):
    return BlockGraph(store, AGENT)


@pytest.fixture()
def chain(store):
    return BlockChain(store, AGENT)


def _populated_block(graph, store, name="scene-1"):
    block = graph.create_block(name)
    fragment = MemoryFragment(content=f"{name}: Elena enters the warehouse.", agent_id=AGENT)
    store.store(fragment)
    graph.add_fragment(block.id, fragment.id)
    node = MemoryNode(label=f"Elena-{name}", node_type=NodeType.CHARACTER, agent_id=AGENT)
    store.store_node(node)
    graph.add_node(block.id, node.id)
    return block, fragment, node


def test_seal_sets_hash_and_status(graph, store, chain):
    block, _, _ = _populated_block(graph, store)
    sealed = chain.seal(block.id)
    assert sealed.status == BlockStatus.SEALED
    assert sealed.content_hash is not None
    assert sealed.sealed_at is not None
    assert sealed.prev_block_id is None  # first block in the chain


def test_seal_links_to_previous_block(graph, store, chain):
    b1, _, _ = _populated_block(graph, store, "scene-1")
    s1 = chain.seal(b1.id)
    b2, _, _ = _populated_block(graph, store, "scene-2")
    s2 = chain.seal(b2.id)
    assert s2.prev_block_id == s1.id
    assert s2.prev_block_hash == s1.content_hash


def test_seal_rejects_non_open_blocks(graph, store, chain):
    block, _, _ = _populated_block(graph, store)
    chain.seal(block.id)
    with pytest.raises(ValueError):
        chain.seal(block.id)
    with pytest.raises(KeyError):
        chain.seal("missing")


def test_verify_clean_block(graph, store, chain):
    block, _, _ = _populated_block(graph, store)
    chain.seal(block.id)
    report = chain.verify(block.id)
    assert report.valid is True
    assert report.reason == "ok"


def test_verify_unsealed_block(graph, chain):
    block = graph.create_block("never-sealed")
    report = chain.verify(block.id)
    assert report.valid is False
    assert "never been sealed" in report.reason


def test_verify_pinpoints_corrupted_fragment(graph, store, chain):
    block, fragment, _ = _populated_block(graph, store)
    chain.seal(block.id)

    # Tamper with the fragment content directly in SQLite, bypassing the API.
    conn = store._conn()
    conn.execute(
        "UPDATE fragments SET content = ? WHERE id = ?",
        (store.encryption.encrypt_str("Elena never existed."), fragment.id),
    )
    conn.commit()

    report = chain.verify(block.id)
    assert report.valid is False
    assert report.corrupted_member_ids == [fragment.id]


def test_verify_detects_deleted_and_added_members(graph, store, chain):
    block, fragment, _ = _populated_block(graph, store)
    chain.seal(block.id)

    store.delete(fragment.id)
    intruder = MemoryFragment(content="Injected after seal.", agent_id=AGENT, block_id=block.id)
    store.store(intruder)

    report = chain.verify(block.id)
    assert report.valid is False
    assert fragment.id in report.missing_member_ids
    assert intruder.id in report.added_member_ids


def test_plasticity_updates_do_not_break_seal(graph, store, chain):
    block, fragment, node = _populated_block(graph, store)
    chain.seal(block.id)

    # Mutable plasticity fields are excluded from hashes by design.
    store.update_importance(fragment.id, 0.99)
    store.touch(fragment.id)

    assert chain.verify(block.id).valid is True


def test_verify_chain_healthy_and_broken(graph, store, chain):
    b1, f1, _ = _populated_block(graph, store, "scene-1")
    chain.seal(b1.id)
    b2, _, _ = _populated_block(graph, store, "scene-2")
    chain.seal(b2.id)

    healthy = chain.verify_chain()
    assert healthy.valid is True
    assert healthy.checked == 2

    conn = store._conn()
    conn.execute(
        "UPDATE fragments SET content = ? WHERE id = ?",
        (store.encryption.encrypt_str("history rewritten"), f1.id),
    )
    conn.commit()

    broken = chain.verify_chain()
    assert broken.valid is False
    assert broken.broken_block_ids == [b1.id]


def test_verify_chain_detects_broken_linkage(graph, store, chain):
    b1, _, _ = _populated_block(graph, store, "scene-1")
    chain.seal(b1.id)
    b2, _, _ = _populated_block(graph, store, "scene-2")
    chain.seal(b2.id)

    # Tamper with the recorded chain link.
    conn = store._conn()
    conn.execute("UPDATE blocks SET prev_block_hash = 'forged' WHERE id = ?", (b2.id,))
    conn.commit()

    report = chain.verify_chain()
    assert report.valid is False
    assert any(block_id == b2.id for block_id, _ in report.broken_links)


def test_quarantine_isolates_block_from_retrieval(tmp_path):
    with Mark.local(project_path=str(tmp_path)) as mark:
        memory = mark.runtime.memory(AGENT)
        graph = memory.blocks()
        chain = memory.chain()

        good = graph.create_block("good-topic")
        bad = graph.create_block("bad-topic")
        good_id = memory.store_sync("The deploy password rotation works fine.")
        bad_id = memory.store_sync("The deploy password rotation is corrupted data.")
        graph.add_fragment(good.id, good_id)
        graph.add_fragment(bad.id, bad_id)

        before = memory.retrieve_sync("deploy password rotation")
        assert {f.id for f in before.fragments} >= {good_id, bad_id}

        chain.quarantine(bad.id)
        after = memory.retrieve_sync("deploy password rotation")
        ids = {f.id for f in after.fragments}
        assert good_id in ids
        assert bad_id not in ids  # quarantined block is isolated

        chain.release(bad.id)
        restored = memory.retrieve_sync("deploy password rotation")
        assert {f.id for f in restored.fragments} >= {good_id, bad_id}


def test_block_scoped_retrieval_filter(tmp_path):
    with Mark.local(project_path=str(tmp_path)) as mark:
        memory = mark.runtime.memory(AGENT)
        graph = memory.blocks()

        b1 = graph.create_block("topic-a")
        b2 = graph.create_block("topic-b")
        f1 = memory.store_sync("Alpha fact about the warehouse lighting.")
        f2 = memory.store_sync("Beta fact about the warehouse lighting.")
        graph.add_fragment(b1.id, f1)
        graph.add_fragment(b2.id, f2)

        scoped = memory.retrieve_sync("warehouse lighting", block_id=b1.id)
        assert {f.id for f in scoped.fragments} == {f1}

        multi = memory.retrieve_sync("warehouse lighting", block_ids=[b1.id, b2.id])
        assert {f.id for f in multi.fragments} == {f1, f2}


def test_release_unsealed_block_returns_to_open(graph, chain):
    block = graph.create_block("open-then-bad")
    chain.quarantine(block.id)
    released = chain.release(block.id)
    assert released.status == BlockStatus.OPEN


def test_v1_database_migrates_to_v2(tmp_path):
    """A pre-block (schema v1) database upgrades in place without data loss."""
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_info VALUES ('schema_version', '1');
        CREATE TABLE blocks (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, agent_id TEXT NOT NULL,
            scope TEXT NOT NULL, block_type TEXT DEFAULT 'ltm',
            metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
        );
        INSERT INTO blocks VALUES ('blk1', 'legacy-block', 'legacy-agent', 'agent',
                                   'ltm', '{}', '2025-01-01T00:00:00+00:00');
        CREATE TABLE fragments (
            id TEXT PRIMARY KEY, content BLOB NOT NULL, block_id TEXT, scope TEXT NOT NULL,
            tier TEXT NOT NULL, agent_id TEXT NOT NULL, session_id TEXT,
            importance REAL DEFAULT 0.5, state TEXT DEFAULT 'raw', confidence REAL DEFAULT 1.0,
            ttl_seconds INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            last_accessed_at TEXT, tags TEXT DEFAULT '[]', source TEXT,
            metadata TEXT DEFAULT '{}', embedding TEXT, contradiction_ids TEXT DEFAULT '[]'
        );
        INSERT INTO fragments VALUES ('frag1', 'legacy fact', 'blk1', 'agent', 'episodic',
            'legacy-agent', NULL, 0.5, 'unverified', 1.0, NULL,
            '2025-01-01T00:00:00+00:00', '2025-01-01T00:00:00+00:00', NULL,
            '[]', NULL, '{}', NULL, '[]');
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, source TEXT,
            agent_id TEXT NOT NULL, scope TEXT NOT NULL, fragment_ids TEXT DEFAULT '[]',
            tags TEXT DEFAULT '[]', metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL,
            agent_id TEXT, fragment_id TEXT, document_id TEXT, weight REAL DEFAULT 1.0,
            metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE TABLE edges (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
            relation TEXT NOT NULL, weight REAL DEFAULT 1.0, agent_id TEXT NOT NULL,
            metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE TABLE session_events (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, session_id TEXT,
            event_type TEXT NOT NULL, fragment_id TEXT, query TEXT,
            metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    store = LocalMemoryStore(db_path)
    try:
        assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION
        # Legacy data survives and reads through the new columns with defaults.
        block = store.get_block("blk1")
        assert block is not None
        assert block.status == BlockStatus.OPEN
        assert block.content_hash is None
        fragments = store.fragments_for_block("blk1")
        assert [f.id for f in fragments] == ["frag1"]
        # New v2 surfaces work on the migrated database.
        graph = BlockGraph(store, "legacy-agent")
        chain = BlockChain(store, "legacy-agent")
        sealed = chain.seal("blk1")
        assert sealed.status == BlockStatus.SEALED
        assert chain.verify("blk1").valid is True
        assert graph.list(status=BlockStatus.SEALED)[0].id == "blk1"
    finally:
        store.close()
