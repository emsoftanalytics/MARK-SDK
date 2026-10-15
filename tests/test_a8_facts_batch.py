# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""v0.2.0a8: bi-temporal facts (history, corrections, scoping, encryption), batch
store/recall, and in-place upgrade from a v0.2.0a7 store.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mark import Mark
from mark.memory.runtime import MarkRuntime
from mark.middlewares.governance import GovernanceMiddleware
from mark.store import LocalMemoryStore

from _helpers_a8_temporal import runtime


FIXTURES = Path(__file__).parent / "fixtures"


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
    facts.set("vega-core", "region", "us-east-1", valid_from="2026-03-01", correction=True)
    assert [f.value for f in facts.history("vega-core", "region")] == ["us-east-1"]


def test_same_time_values_form_a_sequence(runtime) -> None:
    """Two changes logged on one day keep both steps; the last one is current."""
    facts = runtime.memory("f").facts
    facts.set("umbra-gateway", "db", "MongoDB 7", valid_from="2026-03-01")
    facts.set("umbra-gateway", "db", "CockroachDB 23", valid_from="2026-03-08")
    facts.set("umbra-gateway", "db", "MongoDB 7", valid_from="2026-03-08")
    again = facts.set("umbra-gateway", "db", "MongoDB 7", valid_from="2026-03-08")  # re-ingest: no-op

    history = facts.history("umbra-gateway", "db")
    assert [f.value for f in history] == ["MongoDB 7", "CockroachDB 23", "MongoDB 7"]
    assert facts.get("umbra-gateway", "db").id == again.id
    assert facts.as_of("umbra-gateway", "db", "2026-03-08T12:00:00").value == "MongoDB 7"


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
