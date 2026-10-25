# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Contextual compressors: protocol conformance, NoopCompressor,
SimpleWindowCompressor, and LLMContextualCompressor behaviour.
"""
from __future__ import annotations

from pathlib import Path

from mark import (
    ContextualCompressor,
    LLMContextualCompressor,
    NoopCompressor,
    SimpleWindowCompressor,
)
from mark.memory.runtime import MarkRuntime

from _helpers_compressor import _FakeLLM


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
