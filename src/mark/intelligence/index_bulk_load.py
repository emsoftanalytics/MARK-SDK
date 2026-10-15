"""Fast vector-index loading and scope bookkeeping for the retrieval pipeline.

Opening a large store used to rebuild the in-memory index one vector at a
time: decode a float32 blob into a Python list, normalise it in Python,
copy it into numpy (12 s at 100k memories, 48 s at 1M). Here stored
embeddings are read in pages of raw blobs and appended to the index as
whole numpy blocks (``VectorIndex.add_many``), so the load is dominated by
SQLite reading the rows.

``ScopeIds`` is the allowed-id set handed to the pipeline: a frozenset whose
equality test short-circuits on identity. The pipeline checks
``indexed_ids == allowed_ids`` on every recall; with a million ids a real
comparison costs ~80 ms even when both sides are the same object.
"""
from __future__ import annotations

import sys
from typing import Any, Iterable

from mark.index.vector_backends import _load_numpy
from mark.store._sqlite.codec import _decode_embedding


class ScopeIds(frozenset):
    """Immutable id set whose ``==`` is O(1) when both sides are the same object."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other or frozenset.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = frozenset.__hash__


def scope_ids(cache: dict, key: object, ids: frozenset[str]) -> ScopeIds:
    """Wrap the store's cached id set in a ScopeIds, once per distinct set object."""
    hit = cache.get(key)
    if hit is not None and hit[0] is ids:
        return hit[1]
    if len(cache) >= 64:
        cache.clear()
    cache[key] = (ids, ScopeIds(ids))
    return cache[key][1]


def bulk_index(
    pages: Iterable[list[tuple]],
    wanted: set[str] | frozenset[str],
    dim: int | None,
    add_many: Any,
    agent_id: str,
    use_numpy: bool,
) -> list[str]:
    """Index stored embeddings of *wanted* ids page by page; return ids that need embedding.

    ``pages`` yields lists of (id, float32 blob or None, legacy JSON or None).
    Rows without a usable stored vector (missing, legacy JSON, or a
    different dimension) are returned for re-embedding.
    """
    np = _load_numpy() if use_numpy else None
    to_embed: list[str] = []
    for page in pages:
        if dim is None:  # embedder dimension unknown: the first stored vector decides
            dim = next((len(blob) // 4 for _, blob, _ in page if blob and len(blob) % 4 == 0), None)
        size = None if dim is None else 4 * dim
        # One C-level pass keeps the per-row cost low at a million rows.
        good = [row for row in page if row[1] is not None and len(row[1]) == size and row[0] in wanted]
        if len(good) < len(page):
            to_embed.extend(_unusable(page, wanted, size, dim, add_many, agent_id))
        if good:
            ids, blobs, _ = zip(*good)
            add_many(list(ids), _decode_block(np, blobs, dim), agent_id=agent_id)
    return to_embed


def _unusable(page: list[tuple], wanted: Any, size: int | None, dim: int | None, add_many: Any, agent_id: str) -> list[str]:
    """Handle rows without a usable float32 blob: index legacy JSON vectors, return the rest."""
    to_embed: list[str] = []
    for fragment_id, blob, legacy in page:
        if fragment_id not in wanted or (blob is not None and len(blob) == size):
            continue
        vector = _decode_embedding(None, legacy) if blob is None and legacy else None
        if vector and (dim is None or len(vector) == dim):
            add_many([fragment_id], [vector], agent_id=agent_id)
        else:
            to_embed.append(fragment_id)
    return to_embed


def _decode_block(np: Any, blobs: tuple[bytes, ...], dim: int | None) -> Any:
    """float32 little-endian blobs -> a 2-D array (numpy) or a list of lists."""
    if np is not None:
        return np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), dim)
    from array import array

    rows = []
    for blob in blobs:
        values = array("f")
        values.frombytes(blob)
        if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
            values.byteswap()
        rows.append(values.tolist())
    return rows


def wants_numpy(index: Any, total: int) -> bool:
    """Decode blobs with numpy when the index is (or is about to become) numpy-backed."""
    from mark.index import vector as vector_module

    return getattr(index, "backend", "python") == "numpy" or total >= vector_module.AUTO_NUMPY_THRESHOLD


def changed_ids(store: Any, embedder: Any, agent_id: str, stored_fingerprint: str) -> list[str] | None:
    """Ids whose vectors differ in the embedder's space, or None when all of them do.

    Uses the embedder's optional ``reembed_filter(old_fingerprint)`` and the
    store's ``agent_contents``; without either, everything is re-embedded.
    """
    reembed_filter = getattr(embedder, "reembed_filter", None)
    contents = getattr(store, "agent_contents", None)
    needs = reembed_filter(stored_fingerprint) if reembed_filter is not None else None
    if needs is None or contents is None:
        return None
    return [fragment_id for page in contents(agent_id) for fragment_id, text in page if needs(text)]


def _same_embedding_space(stored: str, current: str) -> bool:
    """Fingerprints match, or match on model identity when a dimension is unknown."""
    if stored == current:
        return True
    stored_model, _, stored_dim = stored.rpartition(":")
    current_model, _, current_dim = current.rpartition(":")
    return stored_model == current_model and "None" in {stored_dim, current_dim}
