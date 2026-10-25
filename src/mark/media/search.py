# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Ranking remembered images for a text or image query.

Two paths, chosen by whether an :class:`ImageEmbeddingProvider` is set:

* **CLIP** -- the query (text or image file) is embedded in the provider's
  space and compared with the image vectors of that space only. Images whose
  vector is missing (remembered before the provider was configured) are
  embedded on demand, but only if the file still has its remembered hash, so
  a record is never indexed under a picture it does not describe.
* **Fallback** -- a text query runs MARK's normal recall restricted to image
  fragments (captions + subjects). An image query can only be answered by an
  exact SHA-256 match, because without a vision model there is nothing else
  honest to compare.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mark.media.embeddings import ImageEmbeddingProvider, cosine
from mark.media.files import sha256_file
from mark.media.records import IMAGE_TAG, MODALITY, ImageMatch, ImageRecord
from mark.media.vector_index import ImageVectorIndex

logger = logging.getLogger(__name__)


def index_missing(
    records: list[ImageRecord], embedder: ImageEmbeddingProvider, index: ImageVectorIndex,
) -> dict[str, list[float]]:
    """Return {record id: vector} in the embedder's space, embedding gaps first."""
    fingerprint = embedder.fingerprint
    stored = index.get_all(fingerprint)
    vectors: dict[str, list[float]] = {}
    todo: list[ImageRecord] = []
    for record in records:
        hit = stored.get(record.id)
        if hit is not None and hit[0] == record.sha256:
            vectors[record.id] = hit[1]
        else:
            todo.append(record)
    # Only embed files that still are the picture we remembered.
    fresh = [r for r in todo if _unchanged(r)]
    if fresh:
        for record, vector in zip(fresh, embedder.embed_images([r.path for r in fresh])):
            index.put(record.id, fingerprint, record.sha256, vector)
            vectors[record.id] = vector
    return vectors


def clip_search(
    query: str | Path,
    is_image: bool,
    records: list[ImageRecord],
    embedder: ImageEmbeddingProvider,
    index: ImageVectorIndex,
    k: int,
) -> list[ImageMatch]:
    """Rank ``records`` by cosine similarity to the embedded query."""
    vectors = index_missing(records, embedder, index)
    if is_image:
        probe = embedder.embed_images([str(query)])[0]
    else:
        probe = embedder.embed_texts([str(query)])[0]
    scored = [
        ImageMatch(record=r, score=cosine(probe, vectors[r.id]), method="clip")
        for r in records if r.id in vectors
    ]
    scored.sort(key=lambda match: match.score, reverse=True)
    return scored[:k]


def text_search(memory: Any, query: str, k: int, session_id: str | None) -> list[ImageMatch]:
    """MARK text recall over image captions/subjects (no vision model needed)."""
    result = memory.retrieve_sync(query, session_id=session_id, tags=[IMAGE_TAG])
    matches: list[ImageMatch] = []
    for fragment, score in zip(result.fragments, result.scores):
        if (fragment.metadata or {}).get("modality") == MODALITY:
            matches.append(ImageMatch(ImageRecord.from_fragment(fragment), float(score), "text"))
    return matches[:k]


def sha_search(query: str | Path, records: list[ImageRecord], k: int) -> list[ImageMatch]:
    """Exact-duplicate lookup for an image query when no embedder is set."""
    digest = sha256_file(query)
    return [ImageMatch(r, 1.0, "sha256") for r in records if r.sha256 == digest][:k]


def _unchanged(record: ImageRecord) -> bool:
    try:
        return sha256_file(record.path) == record.sha256
    except OSError:
        logger.debug("image %s is missing; not indexed", record.path)
        return False
