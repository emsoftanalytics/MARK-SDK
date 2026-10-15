"""Optional sentence-transformers embedding provider (requires `mark-sdk[sentence-transformers]`)."""
from __future__ import annotations

from mark.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Optional sentence-transformers adapter, loaded only when instantiated."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
        **model_kwargs: object,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbeddingProvider requires `sentence-transformers`: "
                'install "mark-sdk[sentence-transformers]".'
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, **model_kwargs)
        self._dim: int | None = None
        # sentence-transformers >= 6 renamed the method; older releases only have the long name.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension", None
        )
        if callable(get_dim):
            dim = get_dim()
            self._dim = int(dim) if dim else None

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one model call (L2-normalised)."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = [[float(value) for value in vector] for vector in vectors.tolist()]
        self._dim = len(result[0])
        return result

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None when unknown."""
        return self._dim

    @property
    def fingerprint(self) -> str:
        """Identity of this model's embedding space."""
        return f"sentence-transformers:{self.model_name}:{self._dim}"
