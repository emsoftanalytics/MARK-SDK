# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Embedding providers: base fingerprinting, HashEmbeddingProvider v2, and the
thread-safe embedding cache.
"""
from __future__ import annotations

import threading
import time

import pytest

from mark.embeddings import CachedEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider

from _helpers_embeddings import _norm


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── base / hash v2 ───────────────────────────────────────────────────────────


def test_base_fingerprint_default() -> None:
    class Custom(EmbeddingProvider):
        def embed(self, text: str) -> list[float]:
            return [1.0]

        @property
        def dim(self) -> int:
            return 1

    assert Custom().fingerprint == f"{__name__}.test_base_fingerprint_default.<locals>.Custom:1"
    assert Custom().embed_batch(["a", "b"]) == [[1.0], [1.0]]


def test_hash_v2_defaults_and_fingerprint() -> None:
    h = HashEmbeddingProvider()
    assert h.dim == 256
    assert h.fingerprint == "mark.hash-v2:256"
    assert HashEmbeddingProvider(dim=64).fingerprint == "mark.hash-v2:64"
    with pytest.raises(ValueError):
        HashEmbeddingProvider(dim=0)


def test_hash_v2_deterministic_normalised_and_semantic() -> None:
    h = HashEmbeddingProvider()
    a = h.embed("Elena enters the warehouse at midnight.")
    assert a == HashEmbeddingProvider().embed("Elena enters the warehouse at midnight.")
    assert _norm(a) == pytest.approx(1.0)
    assert h.embed("") == [0.0] * 256
    # Possessive 's stripped, case-insensitive, punctuation ignored.
    assert h.embed("Elena's KEY!") == h.embed("elena key")
    # Stopwords dropped (but a stopword-only text still embeds).
    assert h.embed("the key of the warehouse") == h.embed("key warehouse")
    assert _norm(h.embed("the of and")) == pytest.approx(1.0)
    related = _cos(h.embed("brass key in the warehouse"), h.embed("warehouse keys made of brass"))
    unrelated = _cos(h.embed("brass key in the warehouse"), h.embed("quantum chromodynamics lecture"))
    assert related > 0.3 > unrelated
    # Char 3-grams give morphological overlap.
    assert _cos(h.embed("running"), h.embed("runner")) > 0.1
    # Word order matters via bigrams.
    assert h.embed("dog bites man") != h.embed("man bites dog")


def test_hash_v2_speed() -> None:
    h = HashEmbeddingProvider()
    text = "Elena carries a brass key into the warehouse."
    h.embed(text)
    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        h.embed(text)
    per_call_ms = (time.perf_counter() - start) / n * 1000
    assert per_call_ms < 0.5  # target < 0.1 ms; generous bound for slow CI


# ── cache ────────────────────────────────────────────────────────────────────


class CountingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    @property
    def dim(self) -> int:
        return 2

    @property
    def fingerprint(self) -> str:
        return "counting:2"


def test_cache_batch_only_computes_misses_once() -> None:
    inner = CountingProvider()
    cache = CachedEmbeddingProvider(inner, maxsize=10)
    cache.embed("aa")
    out = cache.embed_batch(["aa", "bbb", "c", "bbb"])
    assert out == [[2.0, 1.0], [3.0, 1.0], [1.0, 1.0], [3.0, 1.0]]
    assert inner.batch_calls == [["aa"], ["bbb", "c"]]
    assert cache.fingerprint == "counting:2"
    assert cache.stats()["misses"] == 3


def test_cache_returns_copies() -> None:
    cache = CachedEmbeddingProvider(CountingProvider())
    first = cache.embed("x")
    first[0] = 999.0
    assert cache.embed("x") == [1.0, 1.0]


def test_cache_thread_safe_and_bounded() -> None:
    cache = CachedEmbeddingProvider(HashEmbeddingProvider(dim=16), maxsize=50)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(300):
                text = f"text {(i * 7 + n) % 120}"
                assert cache.embed(text) == HashEmbeddingProvider(dim=16).embed(text)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cache.stats()["size"] <= 50
