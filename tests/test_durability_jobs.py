"""Fact-job lifecycle and shutdown (backlog A1, A2).

Jobs of a crashed process are reclaimed at once (not after the lease), claims
are small, the lease is configurable, shutdown is bounded and hands unfinished
jobs back, and a closed store refuses further use.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mark.adapters.backend import run_off_loop
from mark.embeddings import HashEmbeddingProvider
from mark.errors import MarkError, StoreClosedError
from mark.intelligence.fact_extraction import FactExtractionService, PatternFactExtractor
from mark.intelligence.fact_extraction import service_queue
from mark.memory.runtime import MarkRuntime
from mark.store._sqlite.job_owner import current_owner, owner_is_dead
from mark.store.sqlite import LocalMemoryStore


class SlowPatterns(PatternFactExtractor):
    """Background pattern extractor that sleeps ``delay`` seconds per memory."""

    name = "slow-patterns"
    background = True

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay
        self.started = threading.Event()

    def extract(self, text: str, *, occurred_at=None):
        self.started.set()
        time.sleep(self.delay)
        return super().extract(text, occurred_at=occurred_at)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _owner_for(pid: int) -> str:
    parts = current_owner().split("|")
    parts[3], parts[4] = str(pid), ""
    return "|".join(parts)


def _queued(path: Path, n: int) -> None:
    runtime = MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32))
    runtime.fact_extraction = FactExtractionService(runtime.store, SlowPatterns(), start_worker=False)
    runtime.memory("a").store_many_sync([f"svc{i} moved to port {8000 + i}." for i in range(n)])
    runtime.shutdown()


def test_owner_tokens_identify_dead_processes() -> None:
    assert not owner_is_dead(current_owner())
    assert owner_is_dead(_owner_for(_dead_pid()))
    assert not owner_is_dead(_owner_for(os.getppid()))  # alive
    assert not owner_is_dead("v1|some-other-host||1|")  # other hosts: never judged
    assert not owner_is_dead(None) and not owner_is_dead("garbage")


def test_jobs_of_a_dead_process_are_reclaimed_on_open(tmp_path: Path) -> None:
    path = tmp_path / "j.db"
    _queued(path, 3)
    store = LocalMemoryStore(path)
    assert len(store.fact_job_claim("slow-patterns", owner=_owner_for(_dead_pid()))) == 3
    assert store.fact_job_counts()["running"] == 3
    # A new service (as on open) requeues them at once; the 300 s lease is not involved.
    FactExtractionService(store, SlowPatterns(), start_worker=False)
    assert store.fact_job_counts()["pending"] == 3
    store.close()


def test_claim_reclaims_dead_owner_jobs_but_not_live_ones(tmp_path: Path) -> None:
    path = tmp_path / "j.db"
    _queued(path, 2)
    store = LocalMemoryStore(path)
    store.fact_job_claim("slow-patterns", limit=1, owner=_owner_for(_dead_pid()))
    store.fact_job_claim("slow-patterns", limit=1, owner=_owner_for(os.getppid()))
    again = store.fact_job_claim("slow-patterns")
    assert len(again) == 1  # only the dead owner's job; the live one waits for its lease
    store.close()


def test_claims_are_small_batches(tmp_path: Path) -> None:
    path = tmp_path / "j.db"
    _queued(path, 10)
    store = LocalMemoryStore(path)
    assert len(store.fact_job_claim("slow-patterns")) == 4
    store.close()


def test_job_lease_is_configurable(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "l.db", fact_extractor=SlowPatterns(), fact_job_lease=42)
    assert runtime.fact_extraction.lease_seconds == 42
    runtime.shutdown()
    with pytest.raises(ValueError):
        FactExtractionService(LocalMemoryStore(), SlowPatterns(), start_worker=False, lease_seconds=0)


def test_lease_is_renewed_when_a_job_starts(tmp_path: Path) -> None:
    path = tmp_path / "j.db"
    _queued(path, 1)
    store = LocalMemoryStore(path)
    job = store.fact_job_claim("slow-patterns")[0]
    with store._writing(structural=False) as conn:
        conn.execute("UPDATE fact_jobs SET updated_at = '2000-01-01T00:00:00+00:00'")
    store.fact_job_start(job["fragment_id"], "slow-patterns")
    assert store.fact_job_claim("slow-patterns", lease_seconds=60) == []  # fresh lease
    store.close()


def test_default_wait_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert service_queue.DEFAULT_WAIT_SECONDS == 30.0
    monkeypatch.setattr(service_queue, "DEFAULT_WAIT_SECONDS", 0.3)
    extractor = SlowPatterns(delay=3.0)
    runtime = MarkRuntime.local(store_path=tmp_path / "w.db", embedder=HashEmbeddingProvider(dim=32),
                                fact_extractor=extractor)
    memory = runtime.memory("a")
    memory.store_sync("held moved to port 1.")
    assert extractor.started.wait(5)
    started = time.monotonic()
    memory.facts.extract_pending()  # wait=True -> DEFAULT_WAIT_SECONDS
    assert time.monotonic() - started < 2.0
    runtime.shutdown(timeout=0.1)


def test_shutdown_is_bounded_and_hands_jobs_back(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    extractor = SlowPatterns(delay=3.0)
    runtime = MarkRuntime.local(store_path=path, embedder=HashEmbeddingProvider(dim=32), fact_extractor=extractor)
    runtime.memory("a").store_many_sync([f"svc{i} moved to port {9000 + i}." for i in range(5)])
    assert extractor.started.wait(5)
    started = time.monotonic()
    runtime.shutdown(timeout=0.5)
    assert time.monotonic() - started < 1.5
    runtime.shutdown()  # idempotent
    reopened = LocalMemoryStore(path)
    counts = reopened.fact_job_counts()
    assert counts["running"] == 0 and counts["pending"] == 5
    attempts = sorted(int(job["attempts"]) for job in reopened.fact_job_list())
    assert attempts == [0, 0, 0, 1, 1] or attempts == [0, 0, 0, 0, 1]  # only started jobs keep an attempt
    reopened.close()
    time.sleep(3.2)  # the abandoned job finishes against a closed store: nothing is written
    check = LocalMemoryStore(path)
    assert check.fact_job_counts()["done"] == 0 and check.fact_current("a") == []
    check.close()


def test_closed_store_refuses_use(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "c.db")
    store.count("a")
    store.close()
    store.close()  # idempotent
    with pytest.raises(StoreClosedError) as info:
        store.count("a")
    assert isinstance(info.value, MarkError) and "closed" in str(info.value)
    memory_store = LocalMemoryStore()
    memory_store.close()
    with pytest.raises(StoreClosedError):
        memory_store.count("a")


def test_closed_store_is_refused_from_other_threads(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "c.db")
    store.close()
    errors: list[BaseException] = []
    thread = threading.Thread(target=lambda: _capture(errors, lambda: store.count("a")))
    thread.start()
    thread.join()
    assert len(errors) == 1 and isinstance(errors[0], StoreClosedError)


def _capture(errors: list[BaseException], fn) -> None:
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001
        errors.append(exc)


def test_shutdown_retires_the_offload_pool(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "o.db")
    asyncio.run(run_off_loop(lambda: 1))
    assert any(t.name.startswith("mark-backend") for t in threading.enumerate())
    runtime.shutdown()
    deadline = time.monotonic() + 3
    while any(t.name.startswith("mark-backend") for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(t.name.startswith("mark-backend") for t in threading.enumerate())
    assert asyncio.run(run_off_loop(lambda: 2)) == 2  # a fresh pool on next use


_HYGIENE = """
import sys
from mark.memory.runtime import MarkRuntime
from mark.errors import StoreClosedError
rt = MarkRuntime.local(store_path=sys.argv[1], fact_extractor="patterns")
m = rt.memory("a")
m.store_sync("api moved to port 1.")
m.retrieve_sync("api port")
rt.shutdown()
try:
    m.store_sync("after close")
except StoreClosedError:
    print("refused")
"""


def test_no_resource_warnings_after_shutdown(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    out = subprocess.run([sys.executable, "-X", "dev", "-W", "error", "-c", _HYGIENE, str(tmp_path / "h.db")],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "refused" in out.stdout and "ResourceWarning" not in out.stderr
