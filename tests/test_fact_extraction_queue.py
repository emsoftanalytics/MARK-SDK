# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Durable background extraction: the fact_jobs queue and its worker.

Slow extractors only enqueue a job per fragment on write. Jobs are
idempotent by fragment id, survive reopening the store, are retried with
their errors recorded, and are drained by a worker thread or synchronously
by ``memory.facts.extract_pending()``.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.fact_extraction import FactExtractionService, PatternFactExtractor
from mark.memory.runtime import MarkRuntime
from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryState


class SlowPatterns(PatternFactExtractor):
    """Pattern extractor that pretends to be slow (queued), and can fail on demand."""

    name = "slow-patterns"
    background = True

    def __init__(self, failures: int = 0) -> None:
        super().__init__()
        self.failures = failures
        self.calls = 0

    def extract(self, text: str, *, occurred_at=None):
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("model server unavailable")
        return super().extract(text, occurred_at=occurred_at)


def _runtime(path: Path, extractor, **kw) -> MarkRuntime:
    runtime = MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32), **kw)
    # Swap in a worker-less service so tests drive the queue deterministically.
    runtime.fact_extraction = FactExtractionService(runtime.store, extractor, start_worker=False)
    return runtime


def test_writes_enqueue_and_extract_pending_drains(tmp_path: Path) -> None:
    extractor = SlowPatterns()
    runtime = _runtime(tmp_path / "q.db", extractor)
    memory = runtime.memory("a")
    memory.store_many_sync(["[2026-03-10] api moved to port 1.", "[2026-03-11] api moved to port 2."])
    memory.store_sync("[2026-03-12] api is now owned by Ana.")
    assert extractor.calls == 0 and memory.facts.current() == []  # nothing ran on the write path
    assert runtime.store.fact_job_counts("a")["pending"] == 3
    assert memory.facts.extract_pending() == 3
    assert memory.facts.get("api", "port").value == "2"
    assert memory.facts.get("api", "owner").valid_from.date().isoformat() == "2026-03-12"
    assert runtime.store.fact_job_counts("a") == {"pending": 0, "running": 0, "done": 3, "failed": 0}
    assert memory.facts.extract_pending() == 0
    runtime.shutdown()


def test_jobs_are_idempotent_by_fragment_and_scoped_by_agent(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "q.db", SlowPatterns())
    a, b = runtime.memory("a"), runtime.memory("b")
    fid = a.store_sync("svc moved to port 5.")
    b.store_sync("svc moved to port 6.")
    assert runtime.store.fact_job_enqueue("a", [(fid, 0)], "slow-patterns") == 0  # already queued
    assert a.facts.extract_pending() == 1
    assert runtime.store.fact_job_enqueue("a", [(fid, 0)], "slow-patterns") == 0  # done: not re-queued
    assert b.facts.current() == [] and b.facts.extract_pending() == 1
    assert [f.value for f in b.facts.current()] == ["6"]
    runtime.shutdown()


def test_queue_survives_reopening_the_store(tmp_path: Path) -> None:
    path = tmp_path / "durable.db"
    runtime = _runtime(path, SlowPatterns())
    runtime.memory("a").store_sync("[2026-01-05] queue-svc moved to port 9000.")
    runtime.shutdown()  # process ends with the job still pending

    reopened = _runtime(path, SlowPatterns())
    memory = reopened.memory("a")
    assert reopened.store.fact_job_counts()["pending"] == 1
    assert memory.facts.extract_pending() == 1
    assert memory.facts.get("queue-svc", "port").value == "9000"
    reopened.shutdown()


def test_failures_are_recorded_retried_and_eventually_failed(tmp_path: Path) -> None:
    flaky = SlowPatterns(failures=1)
    runtime = _runtime(tmp_path / "q.db", flaky)
    memory = runtime.memory("a")
    memory.store_sync("svc moved to port 7.")
    assert memory.facts.extract_pending() == 1  # one attempt per pass, no tight retry loop
    [job] = runtime.store.fact_job_list(agent_id="a")
    assert job["status"] == "pending" and job["attempts"] == 1 and "model server unavailable" in job["error"]
    assert memory.facts.extract_pending() == 1
    [job] = runtime.store.fact_job_list(agent_id="a")
    assert job["status"] == "done" and job["error"] is None
    assert memory.facts.get("svc", "port").value == "7"

    broken = SlowPatterns(failures=99)
    runtime.fact_extraction = FactExtractionService(runtime.store, broken, start_worker=False, max_attempts=2)
    memory = runtime.memory("b")
    memory.store_sync("svc moved to port 8.")
    for _ in range(4):
        memory.facts.extract_pending()
    [job] = runtime.store.fact_job_list(agent_id="b")
    assert job["status"] == "failed" and job["attempts"] == 2 and "ConnectionError" in job["error"]
    assert broken.calls == 2
    assert runtime.store.fact_job_list(status="failed")[0]["fragment_id"] == job["fragment_id"]
    runtime.shutdown()


def test_stale_running_jobs_are_reclaimed_after_their_lease(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "q.db", SlowPatterns())
    memory = runtime.memory("a")
    memory.store_sync("svc moved to port 3.")
    claimed = runtime.store.fact_job_claim("slow-patterns")  # a worker that then dies
    assert len(claimed) == 1 and runtime.store.fact_job_claim("slow-patterns") == []
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with runtime.store._writing(structural=False) as conn:
        conn.execute("UPDATE fact_jobs SET updated_at = ?", (old,))
    assert memory.facts.extract_pending() == 1
    assert memory.facts.get("svc", "port").value == "3"
    runtime.shutdown()


def test_deleted_or_quarantined_fragments_complete_without_facts(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "q.db", SlowPatterns())
    memory = runtime.memory("a")
    gone = memory.store_sync("svc-a moved to port 1.")
    flagged = memory.store_sync("svc-b moved to port 2.")
    runtime.store.delete(gone)
    runtime.store.update_state(flagged, MemoryState.QUARANTINED)
    assert memory.facts.extract_pending() == 2
    assert memory.facts.current() == []
    assert runtime.store.fact_job_counts()["done"] == 2
    runtime.shutdown()


def test_background_worker_drains_the_queue(tmp_path: Path) -> None:
    extractor = SlowPatterns()
    runtime = MarkRuntime.local(store_path=tmp_path / "w.db", embedder=HashEmbeddingProvider(dim=32),
                                fact_extractor=extractor)
    memory = runtime.memory("a")
    memory.store_many_sync([f"[2026-02-0{d}] worker-svc moved to port {8000 + d}." for d in range(1, 6)])
    assert runtime.fact_extraction.wait_idle(timeout=10)
    assert memory.facts.get("worker-svc", "port").value == "8005"
    assert len(memory.facts.history("worker-svc", "port")) == 5
    runtime.shutdown()


def test_worker_picks_up_jobs_left_by_an_earlier_process(tmp_path: Path) -> None:
    path = tmp_path / "left.db"
    first = _runtime(path, SlowPatterns())
    first.memory("a").store_sync("leftover moved to port 4.")
    first.shutdown()
    second = MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32), fact_extractor=SlowPatterns())
    assert second.fact_extraction.wait_idle(timeout=10)
    assert second.memory("a").facts.get("leftover", "port").value == "4"
    second.shutdown()


def test_inline_extractors_have_no_queue(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "i.db", fact_extractor="patterns")
    memory = runtime.memory("a")
    memory.store_sync("inline-svc moved to port 1.")
    assert memory.facts.extract_pending() == 0
    assert runtime.store.fact_job_counts() == {"pending": 0, "running": 0, "done": 0, "failed": 0}
    assert memory.facts.get("inline-svc", "port").value == "1"
    runtime.shutdown()


def test_v4_database_migrates_to_v5_with_the_queue_tables(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    LocalMemoryStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE fact_jobs")
    conn.execute("DROP TABLE fact_extractions")
    conn.execute("UPDATE schema_info SET value = '4' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    store = LocalMemoryStore(path)
    assert store.schema_version() == LocalMemoryStore.CURRENT_SCHEMA_VERSION >= 5
    assert store.fact_job_counts() == {"pending": 0, "running": 0, "done": 0, "failed": 0}
    store.close()


class GatedPatterns(SlowPatterns):
    """Background extractor that blocks inside extract() until released."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def extract(self, text: str, *, occurred_at=None):
        self.started.set()
        self.release.wait(10)
        return super().extract(text, occurred_at=occurred_at)


def _held_by_worker(tmp_path: Path):
    extractor = GatedPatterns()
    runtime = MarkRuntime.local(store_path=tmp_path / "held.db", embedder=HashEmbeddingProvider(dim=32),
                                fact_extractor=extractor)
    memory = runtime.memory("a")
    memory.store_sync("[2026-03-10] held-svc moved to port 7.")
    assert extractor.started.wait(10)  # the worker thread now holds the job
    return runtime, memory, extractor


def test_extract_pending_waits_for_jobs_the_worker_holds(tmp_path: Path) -> None:
    runtime, memory, extractor = _held_by_worker(tmp_path)
    threading.Timer(0.2, extractor.release.set).start()
    assert memory.facts.extract_pending() == 1  # default: block until the held job finishes
    assert memory.facts.get("held-svc", "port").value == "7"
    runtime.shutdown()


def test_extract_pending_without_waiting_or_with_a_short_timeout(tmp_path: Path) -> None:
    runtime, memory, extractor = _held_by_worker(tmp_path)
    assert memory.facts.extract_pending(wait=False) == 0  # the job is the worker's, not ours
    started = time.monotonic()
    assert memory.facts.extract_pending(wait=0.1) == 0  # timed out while still running
    assert time.monotonic() - started < 5
    threading.Timer(0.2, extractor.release.set).start()
    assert memory.facts.extract_pending(wait=5.0) == 1
    assert memory.facts.get("held-svc", "port").value == "7"
    runtime.shutdown()
