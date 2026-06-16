# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import json
from pathlib import Path

from mark import HashEmbeddingProvider, RetrievalResult
from mark.memory.runtime import MarkRuntime


def _make_result(tmp_path: Path, content: str = "Elena is left-handed.") -> RetrievalResult:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync(content, importance=0.9)
    result = mem.retrieve_sync("Elena")
    runtime.shutdown()
    return result


# ---------------------------------------------------------------------------
# xml (default)
# ---------------------------------------------------------------------------

def test_default_format_is_xml(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context()
    assert ctx.startswith("<memory_context")
    assert ctx.strip().endswith("</memory_context>")


def test_xml_format_explicit(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="xml")
    assert "<memory_context" in ctx


def test_xml_contains_fragment_content(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "Elena is left-handed.")
    ctx    = result.as_context(format="xml")
    assert "Elena is left-handed." in ctx


def test_xml_compressed_flag_present(tmp_path: Path) -> None:
    from mark.intelligence.retrieval import RetrievalPolicy, RetrievalPolicySpec, GapReport
    from mark.types import GapSeverity
    result = _make_result(tmp_path)
    # Artificially set was_compressed
    compressed = result.__class__(
        query=result.query,
        fragments=result.fragments,
        scores=result.scores,
        policy_used=result.policy_used,
        spec_used=result.spec_used,
        was_compressed=True,
    )
    assert "compressed=true" in compressed.as_context(format="xml")


def test_xml_no_compressed_flag_by_default(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    assert "compressed=true" not in result.as_context(format="xml")


# ---------------------------------------------------------------------------
# numbered
# ---------------------------------------------------------------------------

def test_numbered_format_starts_with_bracket(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="numbered")
    if result.fragments:
        assert "[1]" in ctx


def test_numbered_format_has_footer(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="numbered")
    assert "---" in ctx


def test_numbered_format_contains_score(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="numbered")
    if result.fragments:
        assert "score=" in ctx


def test_numbered_format_no_xml_tags(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="numbered")
    assert "<memory_context" not in ctx


def test_numbered_empty_result(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem    = runtime.memory("agent")
    result = mem.retrieve_sync("nobody here")
    ctx    = result.as_context(format="numbered")
    assert "0 result" in ctx
    runtime.shutdown()


# ---------------------------------------------------------------------------
# json
# ---------------------------------------------------------------------------

def test_json_format_is_valid_json(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="json")
    parsed = json.loads(ctx)
    assert isinstance(parsed, dict)


def test_json_format_has_results_key(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    parsed = json.loads(result.as_context(format="json"))
    assert "results" in parsed


def test_json_format_has_query_key(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    parsed = json.loads(result.as_context(format="json"))
    assert parsed["query"] == "Elena"


def test_json_format_result_has_content(tmp_path: Path) -> None:
    result = _make_result(tmp_path, "Elena is left-handed.")
    parsed = json.loads(result.as_context(format="json"))
    if parsed["results"]:
        assert "Elena is left-handed." in parsed["results"][0]["content"]


def test_json_format_has_score(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    parsed = json.loads(result.as_context(format="json"))
    if parsed["results"]:
        assert "score" in parsed["results"][0]


def test_json_format_has_gap_key(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    parsed = json.loads(result.as_context(format="json"))
    assert "gap" in parsed
    assert "severity" in parsed["gap"]


def test_json_compressed_flag(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    parsed = json.loads(result.as_context(format="json"))
    assert "compressed" in parsed
    assert parsed["compressed"] is False


# ---------------------------------------------------------------------------
# max_tokens budget
# ---------------------------------------------------------------------------

def test_max_tokens_truncates_content(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    # Store a long fragment
    long_text = "Elena is the protagonist. " * 200
    mem.store_sync(long_text, importance=0.9)
    result = mem.retrieve_sync("Elena")
    runtime.shutdown()

    ctx_full    = result.as_context(format="xml")
    ctx_limited = result.as_context(format="xml", max_tokens=10)
    assert len(ctx_limited) <= len(ctx_full)


def test_max_tokens_overrides_max_words(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx1 = result.as_context(max_words=10000)
    ctx2 = result.as_context(max_words=10000, max_tokens=1)
    # With a 1-token budget, should be shorter or equal
    assert len(ctx2) <= len(ctx1)


def test_max_tokens_works_with_numbered(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="numbered", max_tokens=5)
    assert isinstance(ctx, str)


def test_max_tokens_works_with_json(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    ctx    = result.as_context(format="json", max_tokens=5)
    parsed = json.loads(ctx)
    assert isinstance(parsed["results"], list)
