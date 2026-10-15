"""Embedding provider for any OpenAI-compatible ``/v1/embeddings`` server (stdlib only)."""
from __future__ import annotations

from typing import Any

from mark.embeddings._http import EmbeddingHTTPError, l2_normalize, post_json
from mark.embeddings.base import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """
    Local-first adapter for OpenAI-compatible embedding servers.

    Works with llama.cpp ``llama-server`` (default ``http://localhost:8080/v1``),
    LM Studio (``http://localhost:1234/v1``), vLLM, LocalAI, and Ollama
    (``http://localhost:11434/v1``). Sends ``POST {base_url}/embeddings`` with
    ``{"model", "input": [...]}``, orders results by ``index``, and
    re-normalises vectors client-side. Connection errors and HTTP 429/5xx are
    retried with exponential backoff and full jitter. The API key is only sent
    as a bearer header; it never appears in errors, logs, or ``repr``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "default",
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        dimensions: int | None = None,
        *,
        connect_timeout: float = 5.0,
        max_attempts: int = 3,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.dimensions = dimensions
        self.connect_timeout = connect_timeout
        self.max_attempts = max_attempts
        self._dim: int | None = dimensions

    def __repr__(self) -> str:
        key = "set" if self._api_key else "none"
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key=<{key}>, batch_size={self.batch_size})"
        )

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of ``batch_size`` (one request per batch)."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_chunk(list(texts[start:start + self.batch_size])))
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        data = post_json(
            f"{self.base_url}/embeddings",
            payload,
            headers=headers,
            connect_timeout=self.connect_timeout,
            read_timeout=self.timeout,
            max_attempts=self.max_attempts,
        )
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            got = len(items) if isinstance(items, list) else 0
            raise EmbeddingHTTPError(
                f"Embedding server at {self.base_url} returned {got} embeddings for "
                f"{len(texts)} inputs (model {self.model!r})."
            )
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        return [l2_normalize(item["embedding"]) for item in ordered]

    @property
    def dim(self) -> int | None:
        """Return the embedding dimension, or None until known."""
        return self._dim

    @property
    def fingerprint(self) -> str:
        """Identity of this model's embedding space."""
        return f"openai-compatible:{self.model}:{self._dim}"
