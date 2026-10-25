"""Write-path validation and bulk import robustness (backlog A3, A6).

Bad input raises MARK errors (never raw pydantic/JSON/Unicode errors) and
writes nothing; a bulk import skips bad items with reasons, embeds before
writing, commits in one transaction and is idempotent with a key.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark._cli.engine import _Engine
from mark.embeddings import HashEmbeddingProvider
from mark.errors import InvalidMemoryError, MarkError, MemoryTooLargeError
from mark.memory.facade.write_validation import StoreReport
from mark.memory.runtime import MarkRuntime
from mark.middlewares.base import BaseMiddleware


@pytest.fixture()
def runtime(tmp_path: Path):
    rt = MarkRuntime.local(store_path=tmp_path / "w.db", embedder=HashEmbeddingProvider(dim=32))
    yield rt
    rt.shutdown()


@pytest.mark.parametrize("content, reason", [
    (None, "not_text"), ("", "empty"), ("   ", "empty"), (12345, "not_text"), ({"a": 1}, "not_text"),
    (Path("notes.txt"), "not_text"), (b"bytes are not text", "not_text"), ("bad \ud800 text", "invalid_unicode"),
])
def test_store_sync_rejects_bad_content_with_mark_errors(runtime, content, reason) -> None:
    memory = runtime.memory("a")
    with pytest.raises(InvalidMemoryError) as info:
        memory.store_sync(content)
    assert info.value.reason == reason
    assert isinstance(info.value, MarkError) and isinstance(info.value, ValueError)
    assert runtime.store.count("a") == 0


@pytest.mark.parametrize("kwargs, reason", [
    ({"metadata": {"raw": b"\x00"}}, "metadata"), ({"metadata": {"s": {1, 2}}}, "metadata"),
    ({"metadata": ["not", "a", "dict"]}, "metadata"), ({"tags": "single-string"}, "tags"),
    ({"tags": ["ok", 3]}, "tags"), ({"importance": 2.0}, "importance"), ({"confidence": float("nan")}, "confidence"),
    ({"occurred_at": "yesterday-ish"}, "occurred_at"),
])
def test_store_sync_rejects_bad_options_up_front(runtime, kwargs, reason) -> None:
    with pytest.raises(InvalidMemoryError) as info:
        runtime.memory("a").store_sync("fine text", **kwargs)
    assert info.value.reason == reason
    assert runtime.store.count("a") == 0


def test_max_memory_size_is_enforced_and_configurable(tmp_path: Path) -> None:
    rt = MarkRuntime.local(store_path=tmp_path / "s.db", embedder=HashEmbeddingProvider(dim=32))
    with pytest.raises(MemoryTooLargeError) as info:
        rt.memory("a").store_sync("x" * 1_000_001)
    assert "store_document" in str(info.value) and info.value.limit == 1_000_000
    rt.max_memory_bytes = None
    assert rt.memory("a").store_sync("x" * 1_000_001)
    rt.shutdown()
    small = MarkRuntime.local(store_path=tmp_path / "t.db", max_memory_bytes=10)
    with pytest.raises(MemoryTooLargeError):
        small.memory("a").store_sync("é" * 6)  # 12 UTF-8 bytes
    small.shutdown()


def test_bulk_import_skips_bad_items_with_reasons(runtime) -> None:
    memory = runtime.memory("a")
    report = memory.store_many_sync(["fine", "", None, "bad \udc80", b"raw", 7, "also fine"])
    assert isinstance(report, StoreReport) and isinstance(report, list)
    assert report[0] and report[6] and report[1:6] == [None] * 5
    assert report.stored == 2 and report.ids == [report[0], report[6]]
    assert [(s.index, s.reason) for s in report.skipped] == [
        (1, "empty"), (2, "not_text"), (3, "invalid_unicode"), (4, "not_text"), (5, "not_text")]
    assert runtime.store.count("a") == 2


def test_bulk_import_strict_raises_before_writing(runtime) -> None:
    with pytest.raises(InvalidMemoryError, match="item 1"):
        runtime.memory("a").store_many_sync(["fine", None, "fine too"], strict=True)
    assert runtime.store.count("a") == 0


def test_bulk_import_validates_shared_metadata_once(runtime) -> None:
    with pytest.raises(InvalidMemoryError) as info:
        runtime.memory("a").store_many_sync(["a", "b"], metadata={"when": object()})
    assert info.value.reason == "metadata" and runtime.store.count("a") == 0
    with pytest.raises(InvalidMemoryError):
        runtime.memory("a").store_many_sync("one string, not a list")


class FlakyEmbedder(HashEmbeddingProvider):
    """Fails on the n-th embed_batch call (like a network embedder timing out)."""

    def __init__(self, fail_on: int) -> None:
        super().__init__(dim=32)
        self.calls, self.fail_on = 0, fail_on

    def embed_batch(self, texts):
        self.calls += 1
        if self.calls == self.fail_on:
            raise TimeoutError("embedding server timed out")
        return super().embed_batch(texts)


def test_embedder_timeout_mid_batch_stores_the_whole_batch(tmp_path: Path) -> None:
    # A transient embedder failure on one batch must neither lose writes nor
    # half-write the batch: every row is stored, the unembedded ones pending.
    rt = MarkRuntime.local(store_path=tmp_path / "e.db", embedder=FlakyEmbedder(fail_on=2))
    report = rt.memory("a").store_many_sync([f"memory {i}" for i in range(200)])  # 4 embed batches
    assert report.stored == 200 and rt.store.count("a") == 200
    rt.shutdown()


def test_bulk_import_is_one_transaction(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    store = runtime.store
    original, calls = store._store_row, {"n": 0}

    def crash_on_third(conn, fragment):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("disk vanished")
        return original(conn, fragment)

    monkeypatch.setattr(store, "_store_row", crash_on_third)
    with pytest.raises(RuntimeError):
        runtime.memory("a").store_many_sync([f"memory {i}" for i in range(10)])
    monkeypatch.undo()
    assert store.count("a") == 0


def test_idempotency_key_makes_retries_safe(runtime) -> None:
    memory = runtime.memory("a")
    lines = [f"line {i}" for i in range(50)] + ["line 0"]  # a repeated line at another position is kept
    first = memory.store_many_sync(lines, idempotency_key="import-1")
    again = memory.store_many_sync(lines, idempotency_key="import-1")
    assert list(first) == list(again) and again.existing == len(lines) and first.existing == 0
    assert runtime.store.count("a") == 51
    other_agent = runtime.memory("b").store_many_sync(lines[:3], idempotency_key="import-1")
    assert set(other_agent).isdisjoint(first)  # keys are per agent


class PassThrough(BaseMiddleware):
    name = "pass-through"


def test_idempotency_key_works_through_middleware(tmp_path: Path) -> None:
    rt = MarkRuntime.local(store_path=tmp_path / "m.db", embedder=HashEmbeddingProvider(dim=32),
                           middleware=[PassThrough()])
    memory = rt.memory("a")
    first = memory.store_many_sync(["x one", "", "x two"], idempotency_key="k")
    again = memory.store_many_sync(["x one", "", "x two"], idempotency_key="k")
    assert list(first) == list(again) and first[1] is None and rt.store.count("a") == 2
    rt.shutdown()


def test_cli_ingest_retry_never_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    lines = [f"deploy step {i}" for i in range(30)]
    engine = _Engine(db)
    assert engine.run("ingest", {"agent": "cli", "lines": lines})["stored"] == 30
    retry = engine.run("ingest", {"agent": "cli", "lines": lines})        # a10: new vs present
    assert (retry["stored"], retry["existing"]) == (0, 30)
    assert engine.runtime.store.count("cli") == 30
    engine.close()
