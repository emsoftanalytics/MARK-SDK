# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Image embedding providers for text->image and image->image search.

A provider embeds *both* images and texts into one shared space (CLIP-style),
so a caption query can be compared with image vectors. Its ``fingerprint``
names that space; image vectors are stored per fingerprint, so vectors from
different models (or from MARK's text embedder) are never compared.

``ClipImageEmbeddingProvider`` uses FastEmbed's ONNX CLIP models and needs the
optional ``vision`` extra. The import is lazy so ``import mark.media`` works
without it, and a missing install raises an error naming the extra.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ImageEmbeddingProvider(Protocol):
    """Embeds images and texts into one comparable vector space."""

    @property
    def fingerprint(self) -> str:
        """Stable identity of the vector space (model name + dimension)."""
        ...

    def embed_images(self, paths: Sequence[str | Path]) -> list[list[float]]:
        """Return one vector per image file."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in the same space as the images."""
        ...


def normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalise a vector (zero vectors are returned unchanged)."""
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    return values if norm == 0 else [v / norm for v in values]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for mismatched dimensions (never compare spaces)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class ClipImageEmbeddingProvider:
    """CLIP image + text embeddings via FastEmbed (requires ``mark-sdk[vision]``).

    The two halves of CLIP ship as separate FastEmbed models; both must come
    from the same CLIP checkpoint or their vectors are not comparable. Models
    load on first use, so constructing the provider is cheap.
    """

    def __init__(
        self,
        image_model: str = "Qdrant/clip-ViT-B-32-vision",
        text_model: str = "Qdrant/clip-ViT-B-32-text",
        **model_kwargs: object,
    ) -> None:
        try:
            import fastembed  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'ClipImageEmbeddingProvider requires `fastembed`: install "mark-sdk[vision]".'
            ) from exc
        self.image_model_name = image_model
        self.text_model_name = text_model
        self._kwargs = model_kwargs
        self._image_model: object | None = None
        self._text_model: object | None = None
        self._dim: int | None = None

    @property
    def fingerprint(self) -> str:
        """Identity of the shared CLIP space (named after the image model)."""
        return f"clip:{self.image_model_name}"

    @property
    def dim(self) -> int | None:
        """Vector dimension, or None until the first embedding call."""
        return self._dim

    def embed_images(self, paths: Sequence[str | Path]) -> list[list[float]]:
        """Embed image files; FastEmbed reads and preprocesses them itself."""
        if not paths:
            return []
        if self._image_model is None:
            from fastembed import ImageEmbedding  # type: ignore[import-not-found]

            self._image_model = ImageEmbedding(model_name=self.image_model_name, **self._kwargs)
        vectors = [normalize(v) for v in self._image_model.embed([str(p) for p in paths])]  # type: ignore[attr-defined]
        self._dim = len(vectors[0])
        return vectors

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with the matching CLIP text tower."""
        if not texts:
            return []
        if self._text_model is None:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]

            self._text_model = TextEmbedding(model_name=self.text_model_name, **self._kwargs)
        vectors = [normalize(v) for v in self._text_model.embed(list(texts))]  # type: ignore[attr-defined]
        self._dim = len(vectors[0])
        return vectors
