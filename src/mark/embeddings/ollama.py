"""Ollama-backed embedding provider (stdlib HTTP; no extra dependencies)."""
from __future__ import annotations

from typing import Any

from mark.embeddings._http import EmbeddingHTTPError, l2_normalize, post_json
from mark.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Adapter for developer-managed Ollama embeddings.

    Uses the batch endpoint ``POST /api/embed`` and falls back to the legacy
    single-text ``POST /api/embeddings`` when the server answers 404 for the
    route (Ollama < 0.3). Connection errors and HTTP 429/5xx are retried with
    exponential backoff and full jitter. Uses only the standard library.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        *,
        keep_alive: str | int | None = "5m",
        batch_size: int = 64,
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        max_attempts: int = 3,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.batch_size = batch_size
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_attempts = max_attempts
        self._dim: int | None = None
        self._legacy = False

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of ``batch_size`` via ``/api/embed``."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_chunk(list(texts[start:start + self.batch_size])))
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        if not self._legacy:
            payload: dict[str, Any] = {"model": self.model, "input": texts}
            if self.keep_alive is not None:
                payload["keep_alive"] = self.keep_alive
            try:
                data = self._post("/api/embed", payload)
            except EmbeddingHTTPError as exc:
                # A 404 naming the model means "model not pulled", not "old server".
                if exc.status != 404 or "model" in exc.body.lower():
                    raise
                self._legacy = True
            else:
                embeddings = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                    got = len(embeddings) if isinstance(embeddings, list) else 0
                    raise EmbeddingHTTPError(
                        f"Ollama /api/embed returned {got} embeddings for {len(texts)} inputs "
                        f"(model {self.model!r})."
                    )
                return [l2_normalize(vector) for vector in embeddings]
        vectors = []
        for text in texts:
            legacy_payload: dict[str, Any] = {"model": self.model, "prompt": text}
            if self.keep_alive is not None:
                legacy_payload["keep_alive"] = self.keep_alive
            data = self._post("/api/embeddings", legacy_payload)
            embedding = data.get("embedding") if isinstance(data, dict) else None
            if not isinstance(embedding, list):
                raise EmbeddingHTTPError(
                    f"Ollama /api/embeddings returned no embedding (model {self.model!r})."
                )
            vectors.append(l2_normalize(embedding))
        return vectors

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        return post_json(
            f"{self.base_url}{path}",
            payload,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            max_attempts=self.max_attempts,
        )

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None until the first call."""
        return self._dim

    @property
    def fingerprint(self) -> str:
        """Identity of this Ollama model's embedding space."""
        return f"ollama:{self.model}:{self._dim}"
