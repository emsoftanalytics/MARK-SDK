# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""In-process LRU cache wrapper for embedding providers."""
from __future__ import annotations

from collections import OrderedDict

from mark.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """LRU in-process cache around any EmbeddingProvider.

    Avoids re-computing embeddings for repeated text inputs within a session.
    Memory-only; no persistence between processes.
    Thread safety: Python GIL protects OrderedDict mutations for single ops.
    """

    def __init__(self, inner: EmbeddingProvider, *, maxsize: int = 1000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._inner   = inner
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits   = 0
        self._misses = 0

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None when unknown."""
        return self._inner.dim

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        if text in self._cache:
            self._cache.move_to_end(text)
            self._hits += 1
            return self._cache[text]
        result = self._inner.embed(text)
        self._cache[text] = result
        self._cache.move_to_end(text)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        self._misses += 1
        return result

    def stats(self) -> dict[str, object]:
        """Return summary counters."""
        total = self._hits + self._misses
        return {
            "size":     len(self._cache),
            "maxsize":  self._maxsize,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    def clear(self) -> None:
        """Empty the cache."""
        self._cache.clear()
        self._hits   = 0
        self._misses = 0
