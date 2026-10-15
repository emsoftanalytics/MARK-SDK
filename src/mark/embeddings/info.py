"""Embedder visibility: which model embeds memories, and how it was chosen.

``embedder_info(provider)`` reports the provider's name, dimension, whether it
is semantic, and its ``source``:

* ``"auto"``    picked by ``embedder="auto"`` (best cached local model, else hash);
* ``"spec"``    named explicitly (``embedder="model2vec"``, ``"ollama:<model>"``...);
* ``"env"``     selected by the ``MARK_EMBEDDER`` environment variable;
* ``"default"`` nothing configured: the zero-dependency hash embedder;
* ``"custom"``  a provider instance passed in by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from mark.embeddings.auto import _SOURCE_ATTR, SOURCE_CUSTOM, resolve_embedder
from mark.embeddings.base import EmbeddingProvider, HashEmbeddingProvider


@dataclass(frozen=True)
class EmbedderInfo:
    """What embeds memories: ``name``, ``dim``, ``semantic`` and how it was chosen (``source``)."""

    name: str
    dim: int | None
    semantic: bool
    source: str
    fingerprint: str

    def as_text(self) -> str:
        """One line for logs and ``mark stats``."""
        kind = "semantic" if self.semantic else "lexical"
        return f"{self.name} ({kind}, dim={self.dim}, source={self.source})"

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-able dict."""
        return asdict(self)


def embedder_info(embedder: object | None = None) -> EmbedderInfo:
    """Describe an embedder; with a spec/None, describe what ``Mark.local`` would use."""
    provider = embedder if isinstance(embedder, EmbeddingProvider) else resolve_embedder(embedder)
    fingerprint = str(getattr(provider, "fingerprint", type(provider).__name__))
    model = getattr(provider, "model_name", None) or getattr(provider, "model", None)
    kind = "hash" if isinstance(provider, HashEmbeddingProvider) else fingerprint.split(":", 1)[0]
    name = f"{kind}:{model}" if isinstance(model, str) and model else kind
    return EmbedderInfo(
        name=name,
        dim=getattr(provider, "dim", None),
        semantic=bool(getattr(provider, "semantic", True)),
        source=str(getattr(provider, _SOURCE_ATTR, SOURCE_CUSTOM)),
        fingerprint=fingerprint,
    )
