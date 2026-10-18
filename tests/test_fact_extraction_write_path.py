# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Automatic fact extraction on the write path: store_sync / store_many_sync.

Covers: opt-in and runtime defaults, event-time valid_from, source links,
idempotent re-ingest (including same-day sequences), quarantined/redacted
content, and the guarantee that extraction never fails a write.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mark import Mark
from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.fact_extraction import FactCandidate, PatternFactExtractor
from mark.memory.runtime import MarkRuntime
from mark.middlewares import GovernanceMiddleware
from mark.types import MemoryState

LOG = [
    "[2026-03-01] INVENTORY: service atlas-core is owned by Hana Sato, runs in eu-west-1.",
    "[2026-03-10] CHANGE: atlas-core moved from port 9239 to port 4444.",
    "[2026-03-10] CHANGE: atlas-core moved from port 4444 to port 5100.",
    "[2026-03-12] OWNERSHIP: atlas-core is now owned by Samir Khan.",
    "[2026-03-12] SOCIAL: cake in the kitchen at 3pm.",
]


def _runtime(tmp_path: Path | None = None, **kw) -> MarkRuntime:
    path = ":memory:" if tmp_path is None else tmp_path / "m.db"
    return MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32), **kw)


def _history(memory, subject: str, attribute: str) -> list[tuple[str, str]]:
    return [(f.value, f.valid_from.date().isoformat()) for f in memory.facts.history(subject, attribute)]


def test_extraction_is_off_by_default_and_opt_in_per_call() -> None:
    runtime = _runtime()
    memory = runtime.memory("a")
    memory.store_sync(LOG[1])
    assert memory.facts.current() == []
    memory.store_sync(LOG[1], extract_facts=True)
    assert memory.facts.get("atlas-core", "port").value == "4444"
    runtime.shutdown()


def test_runtime_default_extractor_and_per_call_opt_out() -> None:
    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    memory.store_sync("[2026-03-01] web-ui moved to port 80.", extract_facts=False)
    assert memory.facts.current() == []
    memory.store_many_sync(LOG)
    assert memory.facts.get("atlas-core", "port").value == "5100"
    assert memory.facts.get("atlas-core", "owner").value == "Samir Khan"
    runtime.shutdown()


def test_valid_from_is_event_time_and_facts_link_their_source() -> None:
    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    stamped = memory.store_sync("[2026-03-10] atlas-core moved to port 4444.")
    explicit = memory.store_sync("billing moved to port 81.", occurred_at="2026-02-02T09:30:00Z")
    before = datetime.now(timezone.utc)
    undated = memory.store_sync("search moved to port 82.")
    port = memory.facts.get("atlas-core", "port")
    assert port.valid_from == datetime(2026, 3, 10, tzinfo=timezone.utc)
    assert port.source_fragment_id == stamped
    assert port.metadata["extracted_by"] == "patterns" and port.metadata["rule"] == "changes"
    billing = memory.facts.get("billing", "port")
    assert billing.valid_from == datetime(2026, 2, 2, 9, 30, tzinfo=timezone.utc)
    assert billing.source_fragment_id == explicit
    search = memory.facts.get("search", "port")  # no event time: when it was stored
    assert search.source_fragment_id == undated
    assert abs((search.valid_from - before).total_seconds()) < 5
    runtime.shutdown()


def test_same_day_sequence_keeps_every_step_in_order() -> None:
    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    memory.store_many_sync(LOG)
    # "moved from port 9239" also records 9239 as the value held before the move.
    assert _history(memory, "atlas-core", "port") == [("9239", "2026-03-10"), ("4444", "2026-03-10"),
                                                      ("5100", "2026-03-10")]
    assert _history(memory, "atlas-core", "owner") == [("Hana Sato", "2026-03-01"), ("Samir Khan", "2026-03-12")]
    assert memory.facts.as_of("atlas-core", "owner", "2026-03-05").value == "Hana Sato"
    runtime.shutdown()


def test_reingesting_the_same_log_is_idempotent() -> None:
    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    memory.store_many_sync(LOG)
    first = [(f.subject, f.attribute, f.value, f.valid_from) for f in memory.facts.current()]
    memory.store_many_sync(LOG)  # whole log again (new fragments, same text)
    memory.store_many_sync(LOG[1:3])  # and a partial overlap
    for line in LOG:
        memory.store_sync(line)  # and line by line
    assert [(f.subject, f.attribute, f.value, f.valid_from) for f in memory.facts.current()] == first
    # "moved from port 9239" also records 9239 as the value held before the move.
    assert _history(memory, "atlas-core", "port") == [("9239", "2026-03-10"), ("4444", "2026-03-10"),
                                                      ("5100", "2026-03-10")]
    rows = runtime.store._fetchall("SELECT COUNT(*) FROM facts WHERE retracted_at IS NULL")
    assert rows[0][0] == 6  # owner x2, port x3 (9239 from "moved from"), region x1 — no duplicates recorded
    runtime.shutdown()


def test_identical_lines_within_one_batch_are_distinct_events() -> None:
    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    day = ["[2026-04-01] TICKET T-1 status changed to in progress.",
           "[2026-04-01] TICKET T-1 status changed to blocked.",
           "[2026-04-01] TICKET T-1 status changed to in progress."]
    memory.store_many_sync(day)
    memory.store_many_sync(day)  # re-ingest: still idempotent
    assert [v for v, _ in _history(memory, "T-1", "status")] == ["in progress", "blocked", "in progress"]
    assert memory.facts.get("T-1", "status").value == "in progress"
    runtime.shutdown()


def test_quarantined_and_gate_rejected_content_produces_no_facts() -> None:
    class RejectAll:
        def check(self, content: str, *, confidence: float = 1.0):
            return type("R", (), {"passed": False, "reason": "test", "content": content, "truncated": False})()

    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    memory.store_sync("svc-q moved to port 1.", state=MemoryState.QUARANTINED)
    memory.store_many_sync(["svc-q moved to port 2."], state=MemoryState.QUARANTINED)
    memory.quarantine_sync("svc-q moved to port 3.", reason="suspicious")
    memory._consolidation_gate = RejectAll()
    memory.store_sync("svc-q moved to port 4.", extract_facts=True)
    assert len(memory.quarantined()) == 4
    assert memory.facts.current() == []
    runtime.shutdown()


def test_extraction_reads_what_governance_stored() -> None:
    runtime = _runtime(fact_extractor="patterns", middleware=[GovernanceMiddleware()])
    memory = runtime.memory("a")
    memory.store_sync("billing-api owner: ops@example.com")
    memory.store_many_sync(["mailer owner: help@example.com", "mailer moved to port 2525."])
    assert memory.facts.get("billing-api", "owner") is None  # value was redacted before extraction
    assert memory.facts.get("mailer", "owner") is None
    assert memory.facts.get("mailer", "port").value == "2525"  # batch path goes through middleware
    assert "example.com" not in str([f.value for f in memory.facts.current()])
    runtime.shutdown()


def test_failing_extractor_never_fails_the_write(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        name = "broken"

        def extract(self, text: str, *, occurred_at=None):
            raise RuntimeError("extractor exploded")

    runtime = _runtime(fact_extractor=Broken())
    memory = runtime.memory("a")
    fid = memory.store_sync("svc moved to port 1.")
    ids = memory.store_many_sync(["a moved to port 2.", "b moved to port 3."])
    assert runtime.store.get(fid) is not None and len(ids) == 2
    assert memory.facts.current() == []
    assert "fact extraction failed" in caplog.text
    runtime.shutdown()


def test_custom_extractor_candidates_are_validated_and_dated() -> None:
    class Fixed:
        name = "fixed"

        def extract(self, text: str, *, occurred_at=None):
            return [FactCandidate("robot-7", "battery", "low", confidence=0.6),
                    FactCandidate("!!!", "x", "y"),  # no letters: rejected by fact_set
                    FactCandidate("robot-7", "mode", "idle", valid_from=datetime(2026, 1, 2))]

    runtime = _runtime(fact_extractor=Fixed())
    memory = runtime.memory("a")
    memory.store_sync("[2026-01-01] robot telemetry", extract_facts=True)
    assert memory.facts.get("robot-7", "battery").valid_from.date().isoformat() == "2026-01-01"
    assert memory.facts.get("robot-7", "mode").valid_from.date().isoformat() == "2026-01-02"
    assert len(memory.facts.current()) == 2
    runtime.shutdown()


def test_mark_local_and_invalid_extractor_specs(tmp_path: Path) -> None:
    with Mark.local(tmp_path, fact_extractor=PatternFactExtractor(packs=["ownership"])) as mark:
        memory = mark.runtime.memory("agent")
        memory.store_sync("[2026-05-01] docs-site is now owned by Omar.")
        memory.store_sync("[2026-05-01] docs-site moved to port 9.")  # pack not selected
        assert [(f.attribute, f.value) for f in memory.facts.current("docs-site")] == [("owner", "Omar")]
    with pytest.raises(ValueError, match="unknown fact extractor"):
        _runtime(fact_extractor="regexes")
    with pytest.raises(TypeError):
        _runtime(fact_extractor=object())


def test_async_store_extracts_too() -> None:
    import asyncio

    runtime = _runtime(fact_extractor="patterns")
    memory = runtime.memory("a")
    asyncio.run(memory.store("[2026-06-01] api-x moved to port 7."))
    assert memory.facts.get("api-x", "port").value == "7"
    runtime.shutdown()
