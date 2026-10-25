"""Writes survive a network embedder outage: store now, embed later.

With a network embedder (Ollama, an OpenAI-compatible server) a timeout,
refused connection or 5xx used to fail the write, losing the memory. Now a
*transient* embedder failure stores the memory anyway, with no embedding
(``embedding_blob`` NULL is the durable "pending" marker), and logs one
warning per outage. Meanwhile:

- recall still finds the memory through the lexical channel (and returns
  lexical-only results if even the query cannot be embedded);
- embedding is not retried for ``RETRY_SECONDS`` after a failure, so an
  outage costs one timeout, not one per call;
- every index sync re-embeds the pending memories once the embedder
  answers again (the next recall or ``warm_up``; in a new process the cold
  load finds the NULL embeddings and embeds them), for at most about
  ``CATCHUP_SECONDS`` per sync, so a slow server never stalls one recall
  on the whole backlog; the following recalls continue it.

Configuration errors (HTTP 4xx such as a bad key or model) are not
transient: they still fail the write, as before.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from mark.types import MemoryFragment

logger = logging.getLogger("mark.retrieval")
RETRY_SECONDS = 30.0
# A recall embeds a backlog for at most about this long (plus one batch);
# the next recalls continue it. Batches start small and grow while fast.
CATCHUP_SECONDS = 2.0
CATCHUP_FIRST_BATCH = 8


def catchup_deadline() -> float:
    """Monotonic time at which one index sync stops embedding a backlog."""
    return time.monotonic() + CATCHUP_SECONDS


def is_transient(exc: BaseException) -> bool:
    """A failure worth deferring: unreachable/slow server, rate limit or 5xx (not a 4xx config error)."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if type(exc).__name__ == "EmbeddingHTTPError":  # by name: never import the HTTP stack here
        status = getattr(exc, "status", None)
        return status is None or status == 429 or status >= 500
    return False


class _PendingEmbeddingsMixin:
    """Deferral of embeddings on transient embedder failures (see module docstring)."""

    def embed_for_write(self, fragments: Sequence[MemoryFragment], batch_size: int = 64) -> list[MemoryFragment]:
        """Fragments with embeddings; ``embedding=None`` (pending) when the embedder is transiently down."""
        out: list[MemoryFragment] = []
        for start in range(0, len(fragments), batch_size):
            chunk = list(fragments[start:start + batch_size])
            vectors = self._embed_batch_or_defer([f.content for f in chunk])
            if vectors is None:
                self.__dict__.setdefault("_pending_agents", set()).update(f.agent_id for f in chunk)
                vectors = [None] * len(chunk)
            elif len(vectors) != len(chunk):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunk)} texts")
            out.extend(f.model_copy(update={"embedding": v}) for f, v in zip(chunk, vectors))
        return out

    def _embed_batch_or_defer(self, texts: list[str]) -> list[Any] | None:
        """``embed_batch(texts)``, or None while the embedder is (transiently) unavailable."""
        if time.monotonic() < self.__dict__.get("_embed_retry_at", 0.0):
            return None
        try:
            vectors = self._embedder.embed_batch(texts)
        except Exception as exc:  # noqa: BLE001 - only transient failures are deferred
            if not is_transient(exc):
                raise
            self._embedding_failed(exc)
            return None
        if self.__dict__.pop("_embed_outage", False):
            logger.warning("embedder is reachable again; pending memories are being embedded")
        return vectors

    def _embed_query(self, text: str) -> list[float] | None:
        """The query vector, or None (lexical-only recall) while the embedder is down."""
        vectors = self._embed_batch_or_defer([text])
        return None if vectors is None else list(vectors[0])

    def _embedding_failed(self, exc: BaseException) -> None:
        self.__dict__["_embed_retry_at"] = time.monotonic() + RETRY_SECONDS
        if not self.__dict__.get("_embed_outage"):
            self.__dict__["_embed_outage"] = True
            logger.warning(
                "embedder unavailable (%s); memories are stored without embeddings and embedded when it "
                "answers again (recall uses keyword search meanwhile)", exc,
            )

    def _has_pending(self, agent_id: str) -> bool:
        """True when *agent_id* may have memories waiting for an embedding and a retry is allowed."""
        return (agent_id in self.__dict__.get("_pending_agents", ())
                and time.monotonic() >= self.__dict__.get("_embed_retry_at", 0.0))

    def _pending_done(self, agent_id: str) -> None:
        self.__dict__.get("_pending_agents", set()).discard(agent_id)
