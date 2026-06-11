"""Optional Ollama-backed embedding provider (requires `mark-sdk[ollama]`)."""
from __future__ import annotations

from mark.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Optional adapter for developer-managed Ollama embeddings.

    The implementation is intentionally dependency-free and lazy-imports
    `requests` only when used. Applications may also pass any custom
    EmbeddingProvider into MarkRuntime.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim: int | None = None

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        try:
            import requests  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise ImportError("OllamaEmbeddingProvider requires `requests`.") from exc

        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        vector = [float(value) for value in response.json()["embedding"]]
        self._dim = len(vector)
        return _normalize(vector)

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None when unknown."""
        return self._dim


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]
