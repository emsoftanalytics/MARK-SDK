# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Tests for: ContextualCompressor protocol, NoopCompressor, SimpleWindowCompressor,
# LLMContextualCompressor, compress=True in retrieve_sync, configure_compressor,
# and RetrievalResult.was_compressed.
from __future__ import annotations

from pathlib import Path

import pytest

from mark import (
    ContextualCompressor,
    LLMContextualCompressor,
    LLMProvider,
    Mark,
    NoopCompressor,
    RetrievalResult,
    SimpleWindowCompressor,
)
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# Protocol checks
# ---------------------------------------------------------------------------

def test_noop_compressor_satisfies_protocol() -> None:
    assert isinstance(NoopCompressor(), ContextualCompressor)


def test_simple_window_compressor_satisfies_protocol() -> None:
    assert isinstance(SimpleWindowCompressor(), ContextualCompressor)


def test_llm_contextual_compressor_satisfies_protocol() -> None:
    class FakeLLM:
        def complete(self, prompt: str) -> str:
            return "Relevant excerpt."
    assert isinstance(LLMContextualCompressor(FakeLLM()), ContextualCompressor)


def test_noop_returns_fragments_unchanged(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    mem.store_sync("Alpha fact about workflows.")
    mem.store_sync("Beta fact about databases.")

    from mark.types import MemoryFragment
    fake: list[MemoryFragment] = [
        runtime.store.get(f.id)
        for f in runtime.store.list_by_agent("agent")
        if runtime.store.get(f.id) is not None
    ]
    c = NoopCompressor()
    result = c.compress("query", fake)
    assert result is fake          # exact same list object
    runtime.shutdown()


# ---------------------------------------------------------------------------
# SimpleWindowCompressor
# ---------------------------------------------------------------------------

def test_simple_window_trims_long_content(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    long_text = "A" * 1000
    mem.store_sync(long_text)

    frags = [runtime.store.get(f.id) for f in runtime.store.list_by_agent("agent")]
    frags = [f for f in frags if f is not None]

    c = SimpleWindowCompressor(max_chars=100)
    compressed = c.compress("query", frags)

    assert len(compressed) == 1
    assert len(compressed[0].content) <= 101  # 100 chars + possible ellipsis char
    assert compressed[0].content.endswith("…")
    assert compressed[0].id == frags[0].id    # ID preserved
    runtime.shutdown()


def test_simple_window_keeps_short_content_unchanged(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    mem.store_sync("Short fact.")

    frags = [runtime.store.get(f.id) for f in runtime.store.list_by_agent("agent")]
    frags = [f for f in frags if f is not None]

    c = SimpleWindowCompressor(max_chars=500)
    compressed = c.compress("query", frags)

    assert compressed[0].content == "Short fact."
    runtime.shutdown()


def test_simple_window_no_ellipsis_option(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    mem.store_sync("A" * 200)

    frags = [runtime.store.get(f.id) for f in runtime.store.list_by_agent("agent")]
    frags = [f for f in frags if f is not None]

    c = SimpleWindowCompressor(max_chars=50, ellipsis=False)
    compressed = c.compress("query", frags)
    assert not compressed[0].content.endswith("…")
    runtime.shutdown()


# ---------------------------------------------------------------------------
# LLMContextualCompressor
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


def test_llm_compressor_extracts_relevant(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed and is the protagonist.")

    frags = [runtime.store.get(f.id) for f in runtime.store.list_by_agent("agent")]
    frags = [f for f in frags if f is not None]

    llm = _FakeLLM("Elena is left-handed.")
    c = LLMContextualCompressor(llm)
    compressed = c.compress("What hand does Elena use?", frags)

    assert len(compressed) == 1
    assert compressed[0].content == "Elena is left-handed."
    assert compressed[0].id == frags[0].id    # ID preserved
    assert len(llm.prompts) == 1
    assert "Elena" in llm.prompts[0]
    runtime.shutdown()


def test_llm_compressor_drops_irrelevant_fragment(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    mem.store_sync("The weather is nice today.")

    frags = [runtime.store.get(f.id) for f in runtime.store.list_by_agent("agent")]
    frags = [f for f in frags if f is not None]

    llm = _FakeLLM("IRRELEVANT")
    c = LLMContextualCompressor(llm)
    compressed = c.compress("What are Elena's traits?", frags)

    assert compressed == []
    runtime.shutdown()


def test_llm_compressor_fail_open_on_exception(tmp_path: Path) -> None:
    """LLM error → fragment kept (fail-open so context is not lost)."""
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    mem = runtime.memory("agent")
    mem.store_sync("Important memory fragment.")

    frags = [runtime.store.get(f.id) for f in runtime.store.list_by_agent("agent")]
    frags = [f for f in frags if f is not None]

    class BrokenLLM:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

    c = LLMContextualCompressor(BrokenLLM())
    compressed = c.compress("query", frags)

    assert len(compressed) == 1
    assert compressed[0].content == "Important memory fragment."
    runtime.shutdown()


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
