"""Recall at scale: bounded keyword queries, bulk index loading, O(1) steady-state sync.

The large-index code paths are exercised on small stores by lowering the
size thresholds, and checked against the exact (small-index) behaviour.
"""
from __future__ import annotations

import random

import pytest

from mark.index import VectorIndex
from mark.index import vector as vector_module
from mark.intelligence.index_bulk_load import ScopeIds
from mark import Mark
from mark.store._sqlite import fts_query
from mark.store._sqlite.fts_text import ascii_query_terms
from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryFragment

AGENT = "scale"
SUBJ = [f"svc-{i:03d}" for i in range(40)]
VERB = ["restarted", "deployed", "patched", "audited"]


def _corpus(n: int) -> list[str]:
    rnd = random.Random(7)
    return [f"{rnd.choice(SUBJ)} was {rnd.choice(VERB)} after a spike (ticket T-{i})." for i in range(n)]


@pytest.fixture
def big_store(tmp_path, monkeypatch):
    """A store whose FTS index counts as "large" (planner path), with 600 memories."""
    monkeypatch.setattr(fts_query, "SMALL_INDEX", 50)
    monkeypatch.setattr(fts_query, "RARE_BUDGET", 40)
    store = LocalMemoryStore(tmp_path / "m.db")
    frags = [MemoryFragment(content=text, agent_id=AGENT) for text in _corpus(600)]
    frags.append(MemoryFragment(content="NEEDLE: the vault code for site 9 is 7009.", agent_id=AGENT))
    frags.append(MemoryFragment(content="svc-007 was restarted by another agent", agent_id="other"))
    store.store_many(frags)
    yield store, frags
    store.close()


def test_query_terms_put_stopwords_last_and_keep_identifiers() -> None:
    assert ascii_query_terms("Why was svc-0042 restarted after the outage?") == [
        "svc", "0042", "restarted", "after", "outage", "why", "was", "the",
    ]
    assert ascii_query_terms("who is it") == ["who", "is", "it"]
    assert len(ascii_query_terms(" ".join(f"w{i}" for i in range(30)) + " the")) == 16
    assert ascii_query_terms("Where is the café?")[0] == "cafe"  # folded for the porter table


def test_planned_search_finds_rare_terms_and_ranks_by_the_rest(big_store) -> None:
    store, frags = big_store
    hits = store.lexical_search(AGENT, "vault code for site 9")
    assert hits[0][0] == frags[600].id
    # "svc-007" is rare, "restarted" only ranks: every hit has svc-007, and
    # those that were also restarted come first.
    by_id = {f.id: f.content for f in frags}
    ranked = [by_id[fid] for fid, _ in store.lexical_search(AGENT, "why was svc-007 restarted?", limit=100)]
    assert ranked and all("svc-007" in text for text in ranked)
    restarted = [("restarted" in text) for text in ranked]
    assert restarted == sorted(restarted, reverse=True) and restarted[0]
    # Other agents never leak in.
    assert all(fid in by_id for fid, _ in store.lexical_search(AGENT, "svc-007 restarted another agent"))


def test_planned_search_with_only_common_terms_still_answers(big_store) -> None:
    store, frags = big_store
    hits = store.lexical_search(AGENT, "restarted", limit=10)
    assert len(hits) == 10
    contents = {f.id: f.content for f in frags}
    assert all("restarted" in contents[fid] for fid, _ in hits)


def test_planned_search_scope_filter_is_exact(big_store) -> None:
    store, frags = big_store
    targets = [f for f in frags if "svc-011" in f.content][:3]
    scope = {f.id for f in targets} | {f.id for f in frags[:3000]}  # large scope: filtered in Python
    hits = store.lexical_search(AGENT, "svc-011", limit=50, include_ids={f.id for f in targets})
    assert {fid for fid, _ in hits} == {f.id for f in targets}
    assert {fid for fid, _ in store.lexical_search(AGENT, "svc-011", include_ids=scope)} >= {f.id for f in targets}


def test_small_index_keeps_exact_or_ranking(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "s.db")
    a = MemoryFragment(content="alpha beta gamma", agent_id=AGENT)
    b = MemoryFragment(content="alpha only", agent_id=AGENT)
    store.store_many([a, b])
    assert [fid for fid, _ in store.lexical_search(AGENT, "alpha gamma")] == [a.id, b.id]
    store.close()


def test_doc_frequencies_are_cached_until_the_index_grows(big_store) -> None:
    store, _ = big_store
    cache = fts_query.DocFreqCache()
    calls: list[str] = []

    def fetch(sql: str, params: list) -> list:
        calls.append(sql)
        return store._fetchall(sql, params)

    assert cache.get(fetch, "fragments_fts", "restarted", 600) > 0
    cache.get(fetch, "fragments_fts", "restarted", 610)
    assert len(calls) == 1
    cache.get(fetch, "fragments_fts", "restarted", 1000)
    assert len(calls) == 2
    assert fts_query.total_rows(store._fetchall, "fragments_fts") == 602


# ── vector index ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_add_many_matches_add(backend: str) -> None:
    pytest.importorskip("numpy")
    rng = random.Random(3)
    vectors = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(300)]
    one, many = VectorIndex(backend=backend), VectorIndex(backend=backend)
    for i, vec in enumerate(vectors):
        one.add(f"f{i}", vec, agent_id="a")
    many.add_many([f"f{i}" for i in range(300)], vectors, agent_id="a")
    assert many.count("a") == 300 and many.ids("a") == one.ids("a")
    assert many.missing({"f1", "f2", "zz"}, "a") == {"zz"} and many.missing({"f1"}, "b") == {"f1"}
    for _ in range(5):
        query = [rng.uniform(-1, 1) for _ in range(16)]
        assert [h.fragment_id for h in many.search(query, top_k=7)] == [h.fragment_id for h in one.search(query, top_k=7)]


def test_numpy_removals_compact_and_keep_parity() -> None:
    pytest.importorskip("numpy")
    rng = random.Random(5)
    py, np_ = VectorIndex(backend="python"), VectorIndex(backend="numpy")
    for i in range(400):
        vec = [rng.uniform(-1, 1) for _ in range(8)]
        py.add(f"f{i}", vec)
        np_.add(f"f{i}", vec)
    revision = np_.revision
    for i in range(0, 400, 3):  # enough removals to trigger compaction
        py.remove(f"f{i}")
        np_.remove(f"f{i}")
    assert np_.revision > revision and len(np_) == len(py)
    np_.add("f1", [1.0] * 8)  # re-add in place keeps the insertion position
    py.add("f1", [1.0] * 8)
    for _ in range(10):
        query = [rng.uniform(-1, 1) for _ in range(8)]
        assert [h.fragment_id for h in np_.search(query, top_k=20)] == [h.fragment_id for h in py.search(query, top_k=20)]


def test_scope_ids_compare_by_identity_first() -> None:
    ids = ScopeIds({"a", "b"})
    assert ids == ids and ids == {"a", "b"} and ids != {"a"}
    assert hash(ids) == hash(frozenset({"a", "b"}))


# ── pipeline sync ─────────────────────────────────────────────────────────────

def test_cold_start_bulk_loads_the_index_and_answers_like_a_warm_one(tmp_path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    monkeypatch.setattr(vector_module, "AUTO_NUMPY_THRESHOLD", 100)
    texts = _corpus(500) + ["NEEDLE: the vault code for site 9 is 7009."]
    with Mark.local(tmp_path) as warm:
        memory = warm.runtime.memory(AGENT)
        ids = memory.store_many_sync(texts)
        warm_top = [f.id for f in memory.retrieve_sync("vault code for site 9").fragments]
    with Mark.local(tmp_path) as cold:
        memory = cold.runtime.memory(AGENT)
        cold_top = [f.id for f in memory.retrieve_sync("vault code for site 9").fragments]
        assert cold.runtime.index.count(AGENT) == len(ids) and cold.runtime.index.backend == "numpy"
    assert cold_top[0] == ids[-1] and cold_top == warm_top


def test_unchanged_scope_skips_index_sync(tmp_path) -> None:
    with Mark.local(tmp_path) as mark:
        runtime = mark.runtime
        memory = runtime.memory(AGENT)
        memory.store_many_sync(_corpus(50))
        memory.retrieve_sync("svc-001 restarted")
        pipeline = runtime.pipeline
        calls = []
        original = pipeline._load_missing
        pipeline._load_missing = lambda *a: calls.append(a) or original(*a)  # type: ignore[method-assign]
        memory.retrieve_sync("svc-002 patched")
        assert calls == []  # nothing changed: no O(store) work
        new = memory.store_sync("svc-003 was audited")
        # A write since: the index follows the scope's delta, still no O(store) work.
        assert memory.retrieve_sync("svc-003 audited").fragments[0].id == new
        assert calls == []


# ── vector snapshots ──────────────────────────────────────────────────────────

def test_vector_snapshot_round_trip_with_dirty_rows(tmp_path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    from mark.store._sqlite import vector_snapshot

    monkeypatch.setattr(vector_module, "AUTO_NUMPY_THRESHOLD", 100)
    monkeypatch.setattr(vector_snapshot, "MIN_ROWS", 200)
    monkeypatch.setattr(vector_snapshot, "PART_ROWS", 128)  # several parts
    texts = _corpus(400) + ["NEEDLE: the vault code for site 9 is 7009."]
    with Mark.local(tmp_path) as mark:
        ids = mark.runtime.memory(AGENT).store_many_sync(texts)
    reads: list[str] = []
    original = LocalMemoryStore.embedding_pages

    def counting(self, agent_id, page_size=20_000):
        reads.append(agent_id)
        return original(self, agent_id, page_size)

    monkeypatch.setattr(LocalMemoryStore, "embedding_pages", counting)
    with Mark.local(tmp_path) as mark:  # first cold start: rows, then a snapshot is saved
        memory = mark.runtime.memory(AGENT)
        first = [f.id for f in memory.retrieve_sync("vault code for site 9").fragments]
        assert reads == [AGENT]
        assert mark.runtime.store._fetchall("SELECT count FROM vector_snapshot_meta")[0][0] == len(ids)
        mark.runtime.store.delete(ids[1])
        edited = mark.runtime.store.get(ids[2])  # re-stored with a new vector: dirty, re-read from its row
        mark.runtime.store.store(edited.model_copy(update={"embedding": [1.0] + [0.0] * 255}))
        memory.store_sync("svc-999 was restarted for the needle test")
    with Mark.local(tmp_path) as mark:  # second: snapshot, dirty rows read individually
        memory = mark.runtime.memory(AGENT)
        assert [f.id for f in memory.retrieve_sync("vault code for site 9").fragments] == first
        assert reads == [AGENT]
        index = mark.runtime.index
        assert ids[1] not in index and ids[-1] in index
        assert index.get(ids[2])[0] == pytest.approx(1.0)
        assert index.count(AGENT) == len(mark.runtime.store.retrievable_ids(AGENT))
        assert memory.retrieve_sync("svc-999 restarted needle test").fragments[0].content.startswith("svc-999")


def test_snapshots_need_the_fts_write_hook(tmp_path) -> None:
    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    store = LocalMemoryStore(tmp_path / "e.db", encryption=FernetEncryptionProvider(ephemeral=True))
    assert store.save_vector_snapshot(AGENT, "fp", ["a"], [b"\0" * 4], 1, store._write_version()) is False
    assert store.load_vector_snapshot(AGENT, "fp") is None
    store.close()


def test_update_embedding_invalidates_the_snapshot_entry(tmp_path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    from mark.store._sqlite import vector_snapshot

    monkeypatch.setattr(vector_module, "AUTO_NUMPY_THRESHOLD", 100)
    monkeypatch.setattr(vector_snapshot, "MIN_ROWS", 200)
    with Mark.local(tmp_path) as mark:
        ids = mark.runtime.memory(AGENT).store_many_sync(_corpus(300))
    with Mark.local(tmp_path) as mark:  # cold start saves a snapshot
        mark.runtime.memory(AGENT).retrieve_sync("svc-001 restarted")
        store = mark.runtime.store
        assert store._fetchall("SELECT count FROM vector_snapshot_meta")[0][0] == len(ids)
        store.update_embedding(ids[5], [0.0, 1.0] + [0.0] * 254)  # direct call, no pipeline
        assert store._fetchall("SELECT id FROM vector_dirty")[0][0] == ids[5]
    with Mark.local(tmp_path) as mark:  # the snapshot must not serve the old vector
        mark.runtime.memory(AGENT).retrieve_sync("svc-001 restarted")
        assert mark.runtime.index.get(ids[5])[1] == pytest.approx(1.0)
