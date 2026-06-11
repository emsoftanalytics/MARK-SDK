# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

from pathlib import Path

import pytest

from mark import CachedEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from mark.embeddings import CachedEmbeddingProvider as CachedFromEmbeddings


def test_cached_importable_from_mark() -> None:
    assert CachedEmbeddingProvider is not None


def test_cached_importable_from_embeddings() -> None:
    assert CachedFromEmbeddings is CachedEmbeddingProvider


def test_is_embedding_provider() -> None:
    assert isinstance(CachedEmbeddingProvider(HashEmbeddingProvider(dim=32)), EmbeddingProvider)


def test_dim_delegates_to_inner() -> None:
    assert CachedEmbeddingProvider(HashEmbeddingProvider(dim=64)).dim == 64


def test_embed_returns_correct_dimension() -> None:
    vec = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32)).embed("hello")
    assert len(vec) == 32


def test_same_text_same_vector() -> None:
    p  = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32))
    v1 = p.embed("Elena enters the warehouse.")
    v2 = p.embed("Elena enters the warehouse.")
    assert v1 == v2


def test_first_call_is_miss() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32))
    p.embed("text")
    s = p.stats()
    assert s["misses"] == 1
    assert s["hits"]   == 0


def test_repeated_call_is_hit() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32))
    p.embed("text")
    p.embed("text")
    s = p.stats()
    assert s["hits"]   == 1
    assert s["misses"] == 1


def test_hit_rate() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32))
    p.embed("a")   # miss
    p.embed("a")   # hit
    p.embed("a")   # hit
    assert abs(p.stats()["hit_rate"] - 2 / 3) < 1e-9


def test_lru_evicts_oldest() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32), maxsize=2)
    p.embed("a")
    p.embed("b")
    p.embed("c")   # evicts "a"
    misses_before = p.stats()["misses"]
    p.embed("a")   # was evicted — miss
    assert p.stats()["misses"] == misses_before + 1


def test_recently_used_not_evicted() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32), maxsize=2)
    p.embed("a")
    p.embed("b")
    p.embed("a")   # re-access — now MRU
    p.embed("c")   # evicts "b", not "a"
    hits_before = p.stats()["hits"]
    p.embed("a")   # should be hit
    assert p.stats()["hits"] == hits_before + 1


def test_clear_empties_and_resets_stats() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32))
    p.embed("x")
    p.embed("x")
    p.clear()
    s = p.stats()
    assert s["size"]     == 0
    assert s["hits"]     == 0
    assert s["misses"]   == 0
    assert s["hit_rate"] == 0.0


def test_size_capped_at_maxsize() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32), maxsize=3)
    for i in range(10):
        p.embed(f"unique-{i}")
    assert p.stats()["size"] <= 3


def test_maxsize_zero_raises() -> None:
    with pytest.raises(ValueError):
        CachedEmbeddingProvider(HashEmbeddingProvider(), maxsize=0)


def test_embed_batch_populates_cache() -> None:
    p = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32))
    p.embed_batch(["x", "y", "z"])
    assert p.stats()["size"] == 3


def test_integration_with_runtime(tmp_path: Path) -> None:
    from mark.memory.runtime import MarkRuntime
    provider = CachedEmbeddingProvider(HashEmbeddingProvider(dim=32), maxsize=100)
    runtime  = MarkRuntime.local(store_path=tmp_path / "mem.db", embedder=provider)
    mem = runtime.memory("agent")

    mem.store_sync("Elena is left-handed.", importance=0.9)
    result = mem.retrieve_sync("Elena")
    assert len(result.fragments) >= 1

    hits_before = provider.stats()["hits"]
    mem.retrieve_sync("Elena")   # same query — embedding should be cached
    assert provider.stats()["hits"] > hits_before
    runtime.shutdown()
