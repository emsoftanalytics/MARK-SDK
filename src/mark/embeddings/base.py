"""Embedding provider protocol and the deterministic hash fallback."""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for local or developer-provided embedding models."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an L2-normalized vector for one text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one call."""
        return [self.embed(text) for text in texts]

    @property
    @abstractmethod
    def dim(self) -> int | None:
        """Embedding dimension if known."""


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic CPU-only embedder for tests and zero-dependency local demos.

    Production local usage should pass a stronger provider, such as an adapter
    backed by sentence-transformers, Ollama, LangChain embeddings, Chroma, FAISS,
    or pgvector. The SDK keeps this fallback intentionally small.
    """

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        vector = [0.0] * self._dim
        for token in text.lower().split():
            digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            vector[digest % self._dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @property
    def dim(self) -> int:
        """Return the embedding dimension, or None when unknown."""
        return self._dim
