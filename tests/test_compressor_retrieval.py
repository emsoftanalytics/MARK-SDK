# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Compression in retrieval: compress=True/False in retrieve_sync, gap recomputation,
RetrievalResult.was_compressed, and configure_compressor() late binding.
"""
from __future__ import annotations

from pathlib import Path

from mark import LLMContextualCompressor, Mark, RetrievalResult, SimpleWindowCompressor
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime

from _helpers_compressor import _FakeLLM


# ---------------------------------------------------------------------------
# compress=True in retrieve_sync — NoopCompressor (default)
# ---------------------------------------------------------------------------

def test_retrieve_compress_false_default_no_change(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=False)
    assert isinstance(result, RetrievalResult)
    assert result.was_compressed is False
    runtime.shutdown()


def test_retrieve_compress_true_noop_compressor_no_effect(tmp_path: Path) -> None:
    """compress=True with NoopCompressor (default) → was_compressed stays False."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=True)
    assert result.was_compressed is False   # NoopCompressor doesn't flip the flag
    runtime.shutdown()


def test_retrieve_compress_true_with_window_compressor(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        compressor=SimpleWindowCompressor(max_chars=5),
    )
    mem = runtime.memory("agent")
    mem.store_sync("Elena is a very important character in the story.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=True)
    assert result.was_compressed is True
    assert all(len(f.content) <= 6 for f in result.fragments)
    runtime.shutdown()


def test_retrieve_compress_true_with_llm_compressor(tmp_path: Path) -> None:
    llm = _FakeLLM("Elena is the protagonist.")
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        compressor=LLMContextualCompressor(llm),
    )
    mem = runtime.memory("agent")
    mem.store_sync("Elena is the protagonist of the story and has red hair.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=True)
    assert result.was_compressed is True
    if result.fragments:
        assert result.fragments[0].content == "Elena is the protagonist."
    runtime.shutdown()


def test_retrieve_compress_false_with_real_compressor_skips_it(tmp_path: Path) -> None:
    """Compressor configured but compress=False → no compression applied."""
    llm = _FakeLLM("IRRELEVANT")
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        compressor=LLMContextualCompressor(llm),
    )
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=False)
    assert result.was_compressed is False
    assert len(result.fragments) >= 1    # fragment not dropped because compressor not applied
    runtime.shutdown()


def test_retrieve_recomputes_gap_after_compressor_drops_all(tmp_path: Path) -> None:
    llm = _FakeLLM("IRRELEVANT")
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        compressor=LLMContextualCompressor(llm),
    )
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=True)

    assert result.was_compressed is True
    assert result.fragments == []
    assert result.gap_report.must_not_answer is True
    assert result.gap_report.search_query_hint
    runtime.shutdown()


# ---------------------------------------------------------------------------
# RetrievalResult.was_compressed field
# ---------------------------------------------------------------------------

def test_retrieval_result_was_compressed_default_false(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    result = mem.retrieve_sync("anything")
    assert result.was_compressed is False
    runtime.shutdown()


def test_retrieval_result_as_context_includes_compressed_flag(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        compressor=SimpleWindowCompressor(max_chars=3),
    )
    mem = runtime.memory("agent")
    mem.store_sync("Elena is very important.", importance=0.9)

    result = mem.retrieve_sync("Elena", compress=True)
    ctx = result.as_context()
    if result.was_compressed:
        assert "compressed=true" in ctx
    runtime.shutdown()


# ---------------------------------------------------------------------------
# configure_compressor() — late binding
# ---------------------------------------------------------------------------

def test_configure_compressor_late_binding(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed and has red hair.", importance=0.9)

    # Before configuration: NoopCompressor — was_compressed never True
    r1 = mem.retrieve_sync("Elena", compress=True)
    assert r1.was_compressed is False

    # After configuration with a real compressor
    runtime.configure_compressor(SimpleWindowCompressor(max_chars=5))
    r2 = mem.retrieve_sync("Elena", compress=True)
    assert r2.was_compressed is True
    runtime.shutdown()


def test_mark_configure_compressor(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        mark.memory.block("p").write("Elena is left-handed.", importance=0.9)

        # Before
        r1 = mark.memory.retrieve("Elena", compress=True)

        # Late bind a compressor
        mark.configure_compressor(SimpleWindowCompressor(max_chars=5))
        r2 = mark.memory.retrieve("Elena", compress=True)
        assert r2 is not None   # retrieved without error


def test_mark_local_with_compressor_constructor(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path,
                    compressor=SimpleWindowCompressor(max_chars=10)) as mark:
        mark.memory.block("p").write("Elena is left-handed and has red hair.", importance=0.9)
        result = mark.memory.retrieve("Elena", compress=True)
        assert result is not None
