# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""In-process LRU cache wrapper for embedding providers."""
from __future__ import annotations

import threading
from collections import OrderedDict

from mark.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """Thread-safe LRU in-process cache around any EmbeddingProvider.

    Avoids re-computing embeddings for repeated text inputs within a session.
    Memory-only; no persistence between processes. Returned vectors are
    copies, so callers may mutate them without corrupting the cache.
    """

    def __init__(self, inner: EmbeddingProvider, *, maxsize: int = 1000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._inner   = inner
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock   = threading.Lock()
        self._hits   = 0
        self._misses = 0

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None when unknown."""
        return self._inner.dim

    @property
    def fingerprint(self) -> str:
        """The wrapped provider's fingerprint (caching does not change the space)."""
        return self._inner.fingerprint

    @property
    def semantic(self) -> bool:
        """Whether the wrapped provider embeds meaning (caching does not change it)."""
        return getattr(self._inner, "semantic", True)

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, computing only cache misses in one inner batch call."""
        results: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        with self._lock:
            for position, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    self._hits += 1
                    results[position] = list(cached)
                elif text in missing:
                    # Duplicate within this batch: computed once, counted as a hit.
                    self._hits += 1
                    missing[text].append(position)
                else:
                    missing[text] = [position]
        if missing:
            computed = self._inner.embed_batch(list(missing))
            with self._lock:
                for (text, positions), vector in zip(missing.items(), computed):
                    stored = [float(value) for value in vector]
                    self._cache[text] = stored
                    self._cache.move_to_end(text)
                    self._misses += 1
                    for position in positions:
                        results[position] = list(stored)
                while len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)
        return [vector if vector is not None else [] for vector in results]

    def stats(self) -> dict[str, object]:
        """Return summary counters."""
        with self._lock:
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
        with self._lock:
            self._cache.clear()
            self._hits   = 0
            self._misses = 0
