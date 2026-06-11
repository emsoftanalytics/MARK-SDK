"""Optional sentence-transformers embedding provider."""
from __future__ import annotations

from mark.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Optional sentence-transformers adapter, loaded only when instantiated."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from mark.embeddings.sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerEmbeddingProvider requires `sentence-transformers`."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self._dim: int | None = None

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        vector = self.model.encode(text, normalize_embeddings=True).tolist()
        self._dim = len(vector)
        return [float(value) for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one call."""
        vectors = self.model.encode(texts, normalize_embeddings=True).tolist()
        if vectors:
            self._dim = len(vectors[0])
        return [[float(value) for value in vector] for vector in vectors]

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None when unknown."""
        return self._dim
