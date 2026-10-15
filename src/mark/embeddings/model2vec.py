"""Optional Model2Vec static embedding provider (requires `mark-sdk[model2vec]`)."""
from __future__ import annotations

from mark.embeddings._http import l2_normalize
from mark.embeddings.base import EmbeddingProvider


class Model2VecProvider(EmbeddingProvider):
    """Very fast CPU static embeddings via ``model2vec``; the model loads on construction."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M", **model_kwargs: object) -> None:
        try:
            from model2vec import StaticModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                'Model2VecProvider requires `model2vec`: install "mark-sdk[model2vec]".'
            ) from exc
        self.model_name = model_name
        self.model = StaticModel.from_pretrained(model_name, **model_kwargs)
        dim = getattr(self.model, "dim", None)
        self._dim: int | None = int(dim) if isinstance(dim, int) and dim > 0 else None

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one model call (L2-normalised)."""
        if not texts:
            return []
        matrix = self.model.encode(list(texts))
        rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        vectors = [l2_normalize([float(value) for value in row]) for row in rows]
        self._dim = len(vectors[0])
        return vectors

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None until known."""
        return self._dim

    @property
    def fingerprint(self) -> str:
        """Identity of this model's embedding space."""
        return f"model2vec:{self.model_name}:{self._dim}"
