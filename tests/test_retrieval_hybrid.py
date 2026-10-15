# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Retrieval pipeline regressions (v0.2.0a7): uncapped agent-scoped filtering,
authoritative policies, calibrated gap detection, and the default-config quality
regression. Re-exports FACTS/QUERIES for benchmarks/bench_runtime.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.retrieval import BUILTIN_POLICIES, RetrievalPolicy, RetrievalResult
from mark.memory.runtime import MarkRuntime
from mark.plugins import HOOK_SCORING
from mark.types import GapSeverity, MemoryState

from _helpers_retrieval_hybrid import FACTS, QUERIES, runtime


def _rank(result: RetrievalResult, fragment_id: str) -> int | None:
    return next((i for i, f in enumerate(result.fragments, 1) if f.id == fragment_id), None)


# ── Reachability and isolation ────────────────────────────────────────────────

def test_low_importance_memory_beyond_200_is_reachable(runtime) -> None:
    mem = runtime.memory("cap")
    for i in range(250):
        mem.store_sync(f"Routine maintenance note number {i} about the build.", importance=0.6)
    target = mem.store_sync("The vault access code is ZEBRA-7731.", importance=0.5)

    result = mem.retrieve_sync("vault access code ZEBRA-7731")

    assert _rank(result, target) == 1


def test_other_agents_cannot_crowd_out_results(runtime) -> None:
    bob = runtime.memory("bob")
    for i in range(40):
        bob.store_sync(f"Deploy port for service {i} is 8080.")
    alice = runtime.memory("alice")
    mine = alice.store_sync("Deploy port for the gateway is 9090.")

    result = alice.retrieve_sync("deploy port")

    assert result.fragments and result.fragments[0].id == mine
    assert all(f.agent_id == "alice" for f in result.fragments)


def test_runtime_index_is_the_pipeline_index(runtime) -> None:
    assert runtime.index is runtime.pipeline._index


def test_non_retrievable_fragments_never_surface(runtime) -> None:
    mem = runtime.memory("states")
    keep = mem.store_sync("The blue door opens at noon.")
    hidden = mem.store_sync("The blue door opens at midnight.")
    runtime.store.update_state(hidden, MemoryState.CONTRADICTED)

    result = mem.retrieve_sync("blue door opens")

    ids = [f.id for f in result.fragments]
    assert keep in ids and hidden not in ids


# ── Policies ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy", [RetrievalPolicy.FAST, RetrievalPolicy.BALANCED, RetrievalPolicy.DEEP])
def test_explicit_policy_is_authoritative(runtime, policy) -> None:
    mem = runtime.memory("policy")
    mem.store_sync("Short fact about caching.")

    # A short query would make the classifier suggest a different top_k.
    result = mem.retrieve_sync("caching", policy=policy)

    assert result.spec_used == BUILTIN_POLICIES[policy.value]


def test_auto_policy_uses_classifier(runtime) -> None:
    mem = runtime.memory("auto")
    mem.store_sync("Short fact about caching.")

    result = runtime.pipeline.retrieve("caching", "auto", RetrievalPolicy.AUTO)

    assert result.policy_used == RetrievalPolicy.AUTO
    assert result.spec_used.top_k == result.analysis.estimated_top_k


def test_scoring_plugin_output_is_truncated_to_top_k(runtime) -> None:
    mem = runtime.memory("plugin")
    for i in range(20):
        mem.store_sync(f"Scoring plugin fact {i}.")

    def scorer(candidates, **_):
        from mark.intelligence.reranker import RerankResult
        return [RerankResult(c.fragment_id, c.content, 1.0) for c in candidates]

    runtime.plugins._hooks = getattr(runtime.plugins, "_hooks", {})
    original_get = runtime.plugins.get
    runtime.plugins.get = lambda hook: scorer if hook == HOOK_SCORING else original_get(hook)

    result = mem.retrieve_sync("scoring plugin fact", policy=RetrievalPolicy.FAST)

    assert len(result.fragments) <= BUILTIN_POLICIES["fast"].top_k


# ── Gap detection ─────────────────────────────────────────────────────────────

def test_irrelevant_query_reports_a_gap(runtime) -> None:
    mem = runtime.memory("gap")
    for fact in FACTS:
        mem.store_sync(fact, importance=0.9)

    result = mem.retrieve_sync("quantum chromodynamics lattice gauge")

    assert result.gap_report.severity != GapSeverity.NONE


def test_empty_agent_reports_critical_gap(runtime) -> None:
    result = runtime.memory("nobody").retrieve_sync("anything at all")

    assert result.fragments == []
    assert result.gap_report.severity == GapSeverity.CRITICAL


def test_relevant_query_reports_no_gap(runtime) -> None:
    mem = runtime.memory("covered")
    mem.store_sync("The production server runs on port 8054.")

    result = mem.retrieve_sync("production server port")

    assert result.gap_report.severity == GapSeverity.NONE
    assert result.relevance and result.relevance[0] >= BUILTIN_POLICIES["balanced"].min_confidence


# ── Quality regression (default zero-dependency configuration) ────────────────

def test_default_config_retrieval_quality(runtime) -> None:
    mem = runtime.memory("quality")
    ids = [mem.store_sync(fact) for fact in FACTS]

    hit1 = hit5 = 0
    for query, gold in QUERIES:
        rank = _rank(mem.retrieve_sync(query), ids[gold])
        hit1 += rank == 1
        hit5 += bool(rank and rank <= 5)

    # a6 baseline: hit@1 8/14, hit@5 11/14.
    assert hit1 >= 12
    assert hit5 >= 13


def test_exact_identifiers_rank_first(runtime) -> None:
    mem = runtime.memory("ids")
    ids = [mem.store_sync(fact) for fact in FACTS]

    for query, gold in [("INV-2024-0087", 2), ("JIRA-4521", 11), ("A7731", 26)]:
        assert _rank(mem.retrieve_sync(query), ids[gold]) == 1, query


def test_unknown_dimension_fingerprint_does_not_force_reembedding(tmp_path: Path) -> None:
    """HTTP embedders learn their dimension on first call; a fresh process must not re-embed."""

    class LazyDimEmbedder(HashEmbeddingProvider):
        calls = 0

        def __init__(self) -> None:
            super().__init__(dim=64)
            self._known = False

        @property
        def fingerprint(self) -> str:
            return f"lazy-model:{64 if self._known else None}"

        def embed(self, text):
            self._known = True
            LazyDimEmbedder.calls += 1
            return super().embed(text)

        def embed_batch(self, texts):
            return [self.embed(text) for text in texts]

    path = tmp_path / "lazy.db"
    first = MarkRuntime.local(store_path=path, embedder=LazyDimEmbedder())
    mem = first.memory("lazy")
    for i in range(5):
        mem.store_sync(f"Lazy dimension fact {i}.")
    mem.retrieve_sync("lazy dimension")
    first.shutdown()

    LazyDimEmbedder.calls = 0
    second = MarkRuntime.local(store_path=path, embedder=LazyDimEmbedder())
    result = second.memory("lazy").retrieve_sync("lazy dimension fact")
    second.shutdown()

    assert result.fragments
    assert LazyDimEmbedder.calls == 1  # the query only; stored vectors reused


def test_simple_memory_block_filter_is_a_prefilter(tmp_path: Path) -> None:
    from mark import Mark

    with Mark.local(tmp_path) as mark:
        crowd = mark.memory.block("a")
        for i in range(40):
            crowd.write(f"Release checklist item {i} for the deploy pipeline.", importance=0.9)
        target = mark.memory.block("b").write("Release checklist owner is Priya.", importance=0.3)

        bundle = mark.memory.retrieve("release checklist", blocks=["b"])

        assert [m.id for m in bundle.memories] == [target.id]
