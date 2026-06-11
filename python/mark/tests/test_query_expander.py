# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

from pathlib import Path

import pytest

from mark import (
    HashEmbeddingProvider,
    KeywordQueryExpander,
    LLMQueryExpander,
    Mark,
    NoopQueryExpander,
    QueryExpander,
)
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# Protocol checks
# ---------------------------------------------------------------------------

def test_noop_satisfies_protocol() -> None:
    assert isinstance(NoopQueryExpander(), QueryExpander)


def test_keyword_satisfies_protocol() -> None:
    assert isinstance(KeywordQueryExpander(), QueryExpander)


def test_llm_satisfies_protocol() -> None:
    class FakeLLM:
        def complete(self, p: str) -> str: return "alternate"
    assert isinstance(LLMQueryExpander(FakeLLM()), QueryExpander)


# ---------------------------------------------------------------------------
# NoopQueryExpander
# ---------------------------------------------------------------------------

def test_noop_returns_original_only() -> None:
    q = "What color is Elena's scarf?"
    assert NoopQueryExpander().expand(q) == [q]


def test_noop_first_element_is_original() -> None:
    q = "find the red scarf"
    result = NoopQueryExpander().expand(q)
    assert result[0] == q


# ---------------------------------------------------------------------------
# KeywordQueryExpander
# ---------------------------------------------------------------------------

def test_keyword_strips_question_words() -> None:
    result = KeywordQueryExpander().expand("What color is Elena's scarf?")
    assert len(result) >= 1
    assert result[0].startswith("What")   # original preserved as first


def test_keyword_adds_a_second_query() -> None:
    result = KeywordQueryExpander().expand("What color is Elena's scarf?")
    assert len(result) == 2


def test_keyword_original_is_first() -> None:
    q = "Who is the protagonist?"
    assert KeywordQueryExpander().expand(q)[0] == q


def test_keyword_simple_query_returns_only_original() -> None:
    q = "Elena"
    result = KeywordQueryExpander().expand(q)
    assert result[0] == q


def test_keyword_does_not_produce_empty_alternate() -> None:
    for query in ["Where?", "Who?", "What?"]:
        result = KeywordQueryExpander().expand(query)
        assert all(len(r.strip()) > 0 for r in result)


# ---------------------------------------------------------------------------
# LLMQueryExpander
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response  = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


def test_llm_expander_includes_original() -> None:
    llm    = _FakeLLM("What hand does Elena use?\nElena handedness")
    result = LLMQueryExpander(llm).expand("Which hand is Elena dominant with?")
    assert result[0] == "Which hand is Elena dominant with?"


def test_llm_expander_adds_alternates() -> None:
    llm    = _FakeLLM("Elena left hand\nElena dominant hand")
    result = LLMQueryExpander(llm, n_expansions=2).expand("Which hand?")
    assert len(result) >= 2


def test_llm_expander_calls_llm_once() -> None:
    llm = _FakeLLM("alternate")
    LLMQueryExpander(llm).expand("query")
    assert len(llm.prompts) == 1


def test_llm_expander_prompt_contains_query() -> None:
    llm = _FakeLLM("alternate")
    LLMQueryExpander(llm).expand("Elena scarf location")
    assert "Elena scarf location" in llm.prompts[0]


def test_llm_expander_fail_open_on_exception() -> None:
    class BrokenLLM:
        def complete(self, p: str) -> str:
            raise RuntimeError("LLM down")
    result = LLMQueryExpander(BrokenLLM()).expand("query")
    assert result == ["query"]


def test_llm_expander_strips_duplicate_of_original() -> None:
    q   = "What color is the scarf?"
    llm = _FakeLLM(f"{q}\nScarf color")   # first alternate is same as original
    result = LLMQueryExpander(llm).expand(q)
    # original should appear exactly once
    assert result.count(q) == 1


# ---------------------------------------------------------------------------
# Pipeline integration: expand=True/False
# ---------------------------------------------------------------------------

def test_retrieve_expand_false_no_multi_query(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    result = mem.retrieve_sync("Elena", expand=False)
    assert isinstance(result.fragments, list)
    runtime.shutdown()


def test_retrieve_expand_true_noop_expander_same_as_default(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is the protagonist.", importance=0.9)

    r1 = mem.retrieve_sync("Elena", expand=False)
    r2 = mem.retrieve_sync("Elena", expand=True)   # NoopQueryExpander — same result
    assert {f.id for f in r1.fragments} == {f.id for f in r2.fragments}
    runtime.shutdown()


def test_configure_query_expander_late_binding(tmp_path: Path) -> None:
    llm_expander_called = []

    class _TrackingExpander:
        def expand(self, query: str) -> list[str]:
            llm_expander_called.append(query)
            return [query, query + " elena traits"]

    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    runtime.configure_query_expander(_TrackingExpander())
    mem.retrieve_sync("Elena", expand=True)
    assert len(llm_expander_called) == 1
    runtime.shutdown()


def test_query_expander_via_constructor(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        query_expander=KeywordQueryExpander(),
    )
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    result = mem.retrieve_sync("What are Elena's traits?", expand=True)
    assert isinstance(result.fragments, list)
    runtime.shutdown()


def test_mark_local_accepts_query_expander(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path,
                    query_expander=KeywordQueryExpander()) as mark:
        mark.memory.block("p").write("Elena is left-handed.", importance=0.9)
        result = mark.memory.retrieve("Elena", expand=True)
        assert result is not None
