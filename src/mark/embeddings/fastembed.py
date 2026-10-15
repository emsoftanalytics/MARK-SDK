"""Optional FastEmbed (ONNX, CPU) embedding provider (requires `mark-sdk[fastembed]`)."""
from __future__ import annotations

from mark.embeddings._http import l2_normalize
from mark.embeddings.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """Local ONNX embeddings via ``fastembed``; the model loads on construction."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 64,
        **model_kwargs: object,
    ) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                'FastEmbedProvider requires `fastembed`: install "mark-sdk[fastembed]".'
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_name, **model_kwargs)
        self._dim: int | None = None

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one model call (L2-normalised)."""
        if not texts:
            return []
        vectors = [
            l2_normalize([float(value) for value in vector])
            for vector in self.model.embed(list(texts), batch_size=self.batch_size)
        ]
        self._dim = len(vectors[0])
        return vectors

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None until the first call."""
        return self._dim

    @property
    def fingerprint(self) -> str:
        """Identity of this model's embedding space."""
        return f"fastembed:{self.model_name}:{self._dim}"
