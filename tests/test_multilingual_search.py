"""Multilingual keyword search: CJK segmentation, no English stemming of other
languages, emoji/symbol tokens, schema-6 migration of existing stores."""
from __future__ import annotations

import sqlite3

import pytest

from mark import Mark
from mark.embeddings import HashEmbeddingProvider
from mark.embeddings.segments import segment_words
from mark.store._sqlite.fts_text import multilingual_index_text, multilingual_query_chunks
from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryFragment, MemoryState

AGENT = "ml"
TEXTS = {
    "zh": "我们的上海仓库下个月将搬到浦东新区。",
    "ja": "東京オフィスの会議室は毎週金曜日に予約が必要です。",
    "hi": "दिल्ली कार्यालय में नई कैंटीन सोमवार से खुलेगी।",
    "yo": "Ìpàdé àwọn àgbẹ̀ yóò wáyé ní Ìbàdàn ní ọjọ́ Ẹtì.",
    "pt": "A nação inteira trabalha numa padaria perto da Ribeira.",
    "emoji": "Team mascot is the 🦉 owl named Hooty 🎉🎉",
    "en": "The cafe on the corner deploys new menus every week.",
}


@pytest.fixture
def store(tmp_path):
    store = LocalMemoryStore(tmp_path / "ml.db")
    frags = {
        key: MemoryFragment(content=text, agent_id=AGENT, state=MemoryState.UNVERIFIED)
        for key, text in TEXTS.items()
    }
    store.store_many(list(frags.values()))
    yield store, {key: f.id for key, f in frags.items()}
    store.close()


def _top(store, query):
    hits = store.lexical_search(AGENT, query)
    return hits[0][0] if hits else None


@pytest.mark.parametrize("query,key", [
    ("上海仓库", "zh"),            # CJK word inside an unsegmented sentence
    ("仓库什么时候搬家？", "zh"),    # paraphrase sharing one two-character word
    ("会議室を予約", "ja"),
    ("कार्यालय", "hi"),            # combining marks kept inside the word
    ("Ìbàdàn", "yo"),
    ("ibadan", "yo"),             # folded ASCII form still reaches the porter table
    ("nação", "pt"),
    ("🦉", "emoji"),
    ("café", "en"),               # accented query, unaccented memory
    ("deployed", "en"),           # English stemming unchanged
])
def test_keyword_search_across_scripts(store, query, key) -> None:
    db, ids = store
    assert _top(db, query) == ids[key]


def test_non_english_words_are_not_porter_stemmed_in_the_multilingual_table() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(LocalMemoryStore._DDL_FTS_ML)
    conn.execute("CREATE VIRTUAL TABLE v USING fts5vocab(fragments_fts_ml, row)")
    conn.execute("INSERT INTO fragments_fts_ml (content) VALUES (?)", (multilingual_index_text("nações Ìpàdé"),))
    assert {row[0] for row in conn.execute("SELECT term FROM v")} == {"nacoes", "ipade"}


def test_phrase_exists_and_timeline_candidates_for_cjk(store) -> None:
    db, ids = store
    assert db.phrase_exists(AGENT, "上海仓库") is True
    assert db.phrase_exists(AGENT, "北京仓库") is False
    pages = list(db.mention_candidates(AGENT, "浦东新区"))
    assert [f.id for page in pages for f in page] == [ids["zh"]]


def test_query_chunks_and_index_text() -> None:
    assert multilingual_index_text("plain ascii") is None
    assert multilingual_query_chunks("plain ascii") == []
    assert multilingual_query_chunks("上海仓库") == ["上海", "海仓", "仓库"]
    assert "🦉" in multilingual_index_text("a 🦉 owl").split()


def test_hash_embedder_tokens_cover_cjk_and_emoji() -> None:
    assert segment_words("上海仓库 🦉 owl") == ["上海", "海仓", "仓库", "🦉", "owl"]
    assert segment_words("plain words, only") == ["plain", "words", "only"]
    embed = HashEmbeddingProvider()
    owl = embed.embed(TEXTS["emoji"])
    assert sum(a * b for a, b in zip(embed.embed("🦉"), owl)) > 0.2
    shanghai = embed.embed(TEXTS["zh"])
    assert sum(a * b for a, b in zip(embed.embed("上海仓库"), shanghai)) > 0.2


def test_hash_v2_to_v3_upgrade_re_embeds_only_changed_texts(tmp_path) -> None:
    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory(AGENT)
        ids = dict(zip(TEXTS, memory.store_many_sync(list(TEXTS.values()))))
        memory.retrieve_sync("warm up")
        store = mark.runtime.store
        store.set_meta(f"embedder_fingerprint:{AGENT}", "mark.hash-v2:256")
        for fragment_id in ids.values():  # pretend every vector predates v3
            store.update_embedding(fragment_id, [1.0] + [0.0] * 255)
    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory(AGENT)
        assert memory.retrieve_sync("🦉").fragments[0].id == ids["emoji"]
        store = mark.runtime.store
        assert store.get_meta(f"embedder_fingerprint:{AGENT}") == "mark.hash-v3:256"
        vectors = dict(store.embeddings_for_agent(AGENT))
        # Plain-text memories embed identically under v2 and v3: left alone.
        assert vectors[ids["en"]][0] == pytest.approx(1.0)
        assert vectors[ids["pt"]][0] == pytest.approx(1.0)
        assert vectors[ids["zh"]][0] != pytest.approx(1.0)
        assert vectors[ids["emoji"]][0] != pytest.approx(1.0)


def test_schema_5_store_migrates_and_backfills_the_multilingual_table(tmp_path) -> None:
    path = tmp_path / "old.db"
    store = LocalMemoryStore(path)
    fragment = MemoryFragment(content=TEXTS["zh"], agent_id=AGENT)
    store.store(fragment)
    store.close()
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER fragments_fts_ml_delete")
    conn.execute("DROP TABLE fragments_fts_ml")
    conn.execute("DELETE FROM schema_info WHERE key = 'fts_ml_backfilled'")
    conn.execute("UPDATE schema_info SET value = '5' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    store = LocalMemoryStore(path)
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION == 6
    assert _top(store, "上海仓库") == fragment.id
    store.delete(fragment.id)  # the trigger keeps the multilingual table in sync
    assert store._fetchall("SELECT count(*) FROM fragments_fts_ml")[0][0] == 0
    store.close()


def test_encrypted_store_keeps_no_multilingual_index(tmp_path) -> None:
    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    store = LocalMemoryStore(tmp_path / "enc.db", encryption=FernetEncryptionProvider(ephemeral=True))
    store.store(MemoryFragment(content=TEXTS["zh"], agent_id=AGENT))
    tables = {row[0] for row in store._fetchall("SELECT name FROM sqlite_master")}
    assert "fragments_fts_ml" not in tables and "fragments_fts" not in tables
    assert store.lexical_search(AGENT, "上海仓库") == []
    store.close()
