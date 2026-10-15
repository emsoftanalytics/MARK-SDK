# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# v0.2.0a8: temporal ordering, bi-temporal facts, anchor-aware gaps, batch
# store/recall, and the traps found while building them.
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mark import Mark
from mark.intelligence.retrieval import (
    TemporalIntent,
    _stem,
    detect_temporal_intent,
    query_anchors,
)
from mark.memory.runtime import MarkRuntime
from mark.middlewares.governance import GovernanceMiddleware
from mark.store import LocalMemoryStore
from mark.types import GapSeverity

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def runtime(tmp_path: Path):
    rt = MarkRuntime.local(store_path=tmp_path / "mem.db")
    yield rt
    rt.shutdown()


def _port_history(mem) -> None:
    mem.store_sync("[2026-03-01] INVENTORY: atlas-core is owned by Liam, listens on port 6430, uses MongoDB 7.")
    mem.store_sync("[2026-03-30] CHANGE: atlas-core moved from port 9630 to port 4444.")
    mem.store_sync("[2026-03-17] CHANGE: atlas-core moved from port 6430 to port 9239.")
    mem.store_sync("[2026-03-31] DATABASE: atlas-core switched its datastore to PostgreSQL 15.")
    mem.store_sync("[2026-03-29] CHANGE: atlas-gate moved from port 7001 to port 7002.")


# ── query understanding ───────────────────────────────────────────────────────

@pytest.mark.parametrize("query,intent", [
    ("Which port does atlas-core listen on now?", TemporalIntent.CURRENT),
    ("Who currently owns mosaic-gate?", TemporalIntent.CURRENT),
    ("What is the latest release?", TemporalIntent.CURRENT),
    ("What was the original port of quartz-gateway?", TemporalIntent.ORIGINAL),
    ("Which region did it run in at first?", TemporalIntent.ORIGINAL),
    ("Which port does atlas-core listen on?", TemporalIntent.NONE),
])
def test_temporal_intent(query: str, intent: TemporalIntent) -> None:
    assert detect_temporal_intent(query) == intent


def test_query_anchors() -> None:
    assert query_anchors("Which port does atlas-core listen on now?") == ["atlas core"]
    assert query_anchors("What is the status of ticket HAL-0042?") == ["hal 0042"]
    assert query_anchors("Which team is Marcus Webb on?") == ["marcus webb"]
    assert query_anchors("Who owns the service in eu-west-1?") == ["eu west 1"]
    assert query_anchors("How is authentication implemented?") == []


@pytest.mark.parametrize("left,right", [
    ("owned", "owns"), ("owner", "own"), ("listens", "listen"), ("ports", "port"), ("gates", "gate"),
    ("boxes", "box"), ("policies", "policy"), ("running", "run"), ("database", "databases"),
    ("filter", "filtered"), ("order", "ordered"), ("register", "registered"), ("rate", "rating"),
    ("cache", "caching"), ("speed", "speeding"),
])
def test_stemmer_merges_word_forms(left: str, right: str) -> None:
    assert _stem(left) == _stem(right)


@pytest.mark.parametrize("left,right", [("titan", "titanium"), ("status", "state"), ("class", "clas")])
def test_stemmer_keeps_distinct_words_apart(left: str, right: str) -> None:
    assert _stem(left) != _stem(right)


# ── temporal ordering ─────────────────────────────────────────────────────────

def test_current_query_puts_latest_value_first(runtime) -> None:
    mem = runtime.memory("t")
    _port_history(mem)

    result = mem.retrieve_sync("Which port does atlas-core listen on now?")

    assert result.temporal_intent == "current"
    assert "port 4444" in result.fragments[0].content
    # A newer record about a different attribute (datastore) must not jump ahead,
    # and a different service (atlas-gate) is not part of the subject band.
    assert "DATABASE" not in result.fragments[0].content
    assert result.scores == sorted(result.scores, reverse=True)


def test_original_query_puts_first_value_first(runtime) -> None:
    mem = runtime.memory("t")
    _port_history(mem)

    result = mem.retrieve_sync("What was the original port of atlas-core?")

    assert result.temporal_intent == "original"
    assert "port 6430" in result.fragments[0].content and "INVENTORY" in result.fragments[0].content


def test_dates_inside_text_are_not_event_times(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("[2026-03-05] PLAN: the drill for orion-core is scheduled for 2026-04-30.")
    mem.store_sync("[2026-03-20] PLAN: the drill for orion-core was moved to 2026-04-10.")

    result = mem.retrieve_sync("When is the drill for orion-core scheduled now?")

    assert "2026-04-10" in result.fragments[0].content


def test_two_clocks_are_never_compared(runtime) -> None:
    """Dated records are ordered by declared time; an undated record leads only
    when it was stored after every dated record (a live update after a log import)."""
    mem = runtime.memory("t")
    mem.store_sync("Note: mosaic-gate owner is Kavya.")  # undated, stored before the log
    mem.store_sync("[2026-03-01] OWNERSHIP: mosaic-gate is now owned by Elif.")
    mem.store_sync("[2026-03-16] OWNERSHIP: mosaic-gate is now owned by Wanjiru.")

    ranked = [f.content for f in mem.retrieve_sync("Who currently owns mosaic-gate?").fragments]
    assert "Wanjiru" in ranked[0] and "Kavya" in ranked[2]

    mem.store_sync("Live update: mosaic-gate owner is now Zoe.")  # undated, stored last
    assert "Zoe" in mem.retrieve_sync("Who currently owns mosaic-gate?").fragments[0].content


def test_occurred_at_is_the_declared_event_time(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("vega-core runs in eu-west-1", occurred_at="2026-01-01")
    mem.store_sync("vega-core runs in us-east-1", occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc))

    assert "us-east-1" in mem.retrieve_sync("Which region does vega-core run in now?").fragments[0].content
    assert "eu-west-1" in mem.retrieve_sync("Original region of vega-core?").fragments[0].content
    with pytest.raises(ValueError):
        mem.store_sync("bad", occurred_at="last tuesday")


# ── anchor-aware gaps ─────────────────────────────────────────────────────────

def test_unknown_names_and_ids_report_a_gap(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("[2026-03-03] INVENTORY: billing-gate listens on port 9066.")
    mem.store_sync("[2026-03-04] HR: Samir Khan transferred to the Messaging team.")
    mem.store_sync("[2026-03-05] TICKET HAL-8557 status changed to in progress.")

    for query, missing in [
        ("Which port does the titan-billing service listen on?", "titan billing"),
        ("Which team is Marcus Webb on?", "marcus webb"),
        ("What is the status of ticket HAL-0042?", "hal 0042"),
    ]:
        result = mem.retrieve_sync(query)
        assert result.gap_report.severity == GapSeverity.HIGH, query
        assert result.missing_anchors == [missing]


def test_known_anchor_outside_top_k_is_not_a_gap(runtime) -> None:
    mem = runtime.memory("t")
    for i in range(40):
        mem.store_sync(f"Routine note {i} about the payments service and its port.")
    mem.store_sync("The sable-core service listens on port 7777 and handles payments.")

    result = mem.retrieve_sync("payments service port sable-core", policy="fast")

    assert result.missing_anchors == []


def test_titan_never_matches_titanium(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("titanium-core listens on port 5000.")

    result = mem.retrieve_sync("Which port does titan-core listen on?")

    assert result.missing_anchors == ["titan core"]


# ── facts ─────────────────────────────────────────────────────────────────────

def test_facts_current_history_and_out_of_order_writes(runtime) -> None:
    facts = runtime.memory("f").facts
    facts.set("atlas-core", "port", 6430, valid_from="2026-03-01")
    facts.set("atlas-core", "port", 4444, valid_from="2026-03-30")
    facts.set("Atlas Core", "Port", 9239, valid_from="2026-03-17")  # arrives late

    assert facts.get("atlas-core", "port").value == "4444"
    assert [f.value for f in facts.history("atlas-core", "port")] == ["6430", "9239", "4444"]
    assert facts.as_of("atlas-core", "port", "2026-03-20").value == "9239"
    assert facts.as_of("atlas-core", "port", "2026-02-01") is None


def test_facts_idempotent_and_corrections_retract(runtime) -> None:
    facts = runtime.memory("f").facts
    first = facts.set("vega-core", "region", "eu-west-1", valid_from="2026-03-01")
    again = facts.set("vega-core", "region", "eu-west-1", valid_from="2026-03-01")
    assert again.id == first.id
    facts.set("vega-core", "region", "us-east-1", valid_from="2026-03-01")  # correction
    assert [f.value for f in facts.history("vega-core", "region")] == ["us-east-1"]


def test_facts_collapse_repeated_values(runtime) -> None:
    facts = runtime.memory("f").facts
    for day, value in [(1, "open"), (2, "open"), (3, "closed")]:
        facts.set("HAL-1", "status", value, valid_from=f"2026-03-0{day}")
    assert [f.value for f in facts.history("HAL-1", "status")] == ["open", "closed"]


def test_facts_are_agent_scoped_and_surface_in_retrieval(runtime) -> None:
    runtime.memory("other").facts.set("atlas-core", "port", "1111")
    mem = runtime.memory("f")
    mem.facts.set("atlas-core", "port", "4444", valid_from="2026-03-30")
    mem.facts.set("atlas-core", "owner", "Liam", valid_from="2026-03-01")

    result = mem.retrieve_sync("Which port does atlas-core listen on now?")

    assert [f.value for f in result.facts] == ["4444"]
    assert result.missing_anchors == []
    assert "atlas-core port = 4444" in result.as_context()


def test_fact_values_are_encrypted_at_rest(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    import sqlite3

    from mark.security.encryption import FernetEncryptionProvider

    path = tmp_path / "enc.db"
    store = LocalMemoryStore(path, encryption=FernetEncryptionProvider(passphrase="pw"))
    store.fact_set("a", "vault", "pin", "4821", valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert store.fact_current("a", "vault", "pin")[0]["value"] == "4821"
    store.close()
    raw = sqlite3.connect(path).execute("SELECT value FROM facts").fetchone()[0]
    assert b"4821" not in bytes(raw)


# ── batch store / recall ──────────────────────────────────────────────────────

def test_store_many_and_retrieve_many_match_single_calls(runtime) -> None:
    mem = runtime.memory("b")
    texts = [f"Service s{i}-core listens on port {5000 + i}." for i in range(30)]
    ids = mem.store_many_sync(texts, tags=["inventory"])

    assert len(ids) == 30 and len(set(ids)) == 30
    assert [runtime.store.get(fid).content for fid in ids] == texts
    queries = ["Which port does s3-core listen on?", "Which port does s17-core listen on?"]
    many = mem.retrieve_many_sync(queries)
    single = [mem.retrieve_sync(q) for q in queries]
    assert [r.fragments[0].id for r in many] == [r.fragments[0].id for r in single]
    assert "5003" in many[0].fragments[0].content


def test_store_many_with_middleware_still_redacts_each_item(tmp_path: Path) -> None:
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware()]) as mark:
        mem = mark.runtime.memory("gov")
        ids = mem.store_many_sync(["contact bob@example.com", "api_key=abcd1234efgh5678"])
        contents = [mark.runtime.store.get(fid).content for fid in ids]
    assert "bob@example.com" not in contents[0]
    assert "abcd1234efgh5678" not in contents[1]


def test_store_many_rejects_mismatched_occurred_at(runtime) -> None:
    with pytest.raises(ValueError):
        runtime.memory("b").store_many_sync(["a", "b"], occurred_at=["2026-01-01"])


def test_store_many_is_atomic_on_foreign_ids(runtime) -> None:
    from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

    store = runtime.store

    def frag(fid: str, agent: str) -> MemoryFragment:
        return MemoryFragment(id=fid, content=fid, scope=MemoryScope.AGENT, tier=MemoryTier.SEMANTIC,
                              agent_id=agent, state=MemoryState.UNVERIFIED)

    store.store(frag("taken", "alice"))
    with pytest.raises(ValueError):
        store.store_many([frag("fresh", "bob"), frag("taken", "bob")])
    assert store.get("fresh") is None


def test_lexical_filter_is_exact_for_large_scopes(runtime) -> None:
    mem = runtime.memory("lex")
    noise = mem.store_many_sync([f"zebra noise line {i}" for i in range(1200)], session_id="noise")
    target = mem.store_sync("zebra target in scope", session_id="scope")
    scope = set(noise[:950]) | {target}  # > 900 ids, target ranks below many out-of-scope hits

    hits = runtime.store.lexical_search("lex", "zebra target", limit=5, include_ids=scope)

    assert hits and hits[0][0] == target
    assert all(fid in scope for fid, _ in hits)


# ── upgrade from v0.2.0a7 ─────────────────────────────────────────────────────

def test_v0_2_0a7_store_upgrades_in_place(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    shutil.copy(FIXTURES / "store_v0.2.0a7.db", path)

    runtime = MarkRuntime.local(store_path=path)
    try:
        assert runtime.store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION
        mem = runtime.memory("legacy")
        result = mem.retrieve_sync("Which port does atlas-core listen on now?")
        assert "port 4444" in result.fragments[0].content
        mem.facts.set("atlas-core", "port", "4444", valid_from="2026-03-30")
        assert mem.facts.get("atlas-core", "port").value == "4444"
    finally:
        runtime.shutdown()
