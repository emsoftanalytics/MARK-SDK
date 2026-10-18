"""Fast vector-index loading and scope bookkeeping for the retrieval pipeline.

Opening a large store used to rebuild the in-memory index one vector at a
time: decode a float32 blob into a Python list, normalise it in Python,
copy it into numpy (12 s at 100k memories, 48 s at 1M). Here stored
embeddings are read in pages of raw blobs and appended to the index as
whole numpy blocks (``VectorIndex.add_many``), so the load is dominated by
SQLite reading the rows.

``ScopeIds`` (defined next to the store that produces it) is the allowed-id
set handed to the pipeline: its equality test short-circuits on identity,
and a set the store derived from the previous one by a few writes carries
that delta, which ``sync_delta`` applies to the index instead of comparing
a million ids (see :mod:`mark.store._sqlite.scope_ids`).
"""
from __future__ import annotations

import itertools
import logging
import sys
from typing import Any, Iterable

from mark.index.vector_backends import _load_numpy
from mark.store._sqlite.codec import _decode_embedding
from mark.store._sqlite.scope_ids import ScopeIds  # re-exported: imported from here by callers

# A gap of less than 1/BY_ID_FRACTION of the agent is read by primary key
# instead of streaming every row (read at call time, so tests can patch it).
BY_ID_FRACTION = 16


def scope_ids(cache: dict, key: object, ids: frozenset[str]) -> ScopeIds:
    """Wrap the store's cached id set in a ScopeIds, once per distinct set object."""
    if isinstance(ids, ScopeIds):  # already immutable, identity-comparable (and may carry a delta)
        return ids
    hit = cache.get(key)
    if hit is not None and hit[0] is ids:
        return hit[1]
    if len(cache) >= 64:
        cache.clear()
    cache[key] = (ids, ScopeIds(ids))
    return cache[key][1]


def sync_delta(host: Any, agent_id: str, last: tuple | None, allowed: Any) -> tuple | None:
    """Bring the index from the last synced scope to *allowed* by its delta alone.

    Applies when the index partition equalled the last scope (``last`` is
    ``(scope, revision, result)`` with ``result is scope``) and the store
    derived *allowed* from that scope: removed ids leave the index, added
    ones are indexed from their stored vectors (or embedded). Returns the new
    ``_synced`` entry, or None to fall back to the full comparison. O(delta),
    where the full path costs two set operations over every id.
    """
    if last is None or last[2] is not last[0] or getattr(allowed, "parent", None) is None:
        return None
    index = host._index
    count, find_missing = getattr(index, "count", None), getattr(index, "missing", None)
    if allowed.parent != getattr(last[0], "token", None) or count is None or find_missing is None:
        return None
    for fragment_id in allowed.removed:  # no longer retrievable: drop from the partition
        index.remove(fragment_id)
    missing = find_missing(allowed.added, agent_id)
    if missing:
        stored = host._stored_embeddings(agent_id, set(missing))
        to_embed = index_each(stored, missing, getattr(host._embedder, "dim", None), host._index_add, agent_id)
        if to_embed:
            host._embed_and_index(agent_id, to_embed)
    # Partition ⊇ allowed (it held the old scope, now plus the added ids);
    # equal sizes therefore mean equal sets.
    if count(agent_id) != len(allowed) or find_missing(allowed.added, agent_id):
        return None
    return (allowed, getattr(index, "revision", None), allowed)


def bulk_index(
    pages: Iterable[list[tuple]],
    wanted: set[str] | frozenset[str],
    dim: int | None,
    add_many: Any,
    agent_id: str,
    use_numpy: bool,
) -> list[str]:
    """Index stored embeddings of *wanted* ids page by page; return ids that need embedding.

    ``pages`` yields lists of (id, float32 blob or None, legacy JSON or None, ...).
    Rows without a usable stored vector (missing, legacy JSON, or a
    different dimension) are returned for re-embedding.
    """
    np = _load_numpy() if use_numpy else None
    to_embed: list[str] = []
    for page in pages:
        if dim is None:  # embedder dimension unknown: the first stored vector decides
            dim = next((len(row[1]) // 4 for row in page if row[1] and len(row[1]) % 4 == 0), None)
        size = None if dim is None else 4 * dim
        # One C-level pass keeps the per-row cost low at a million rows.
        good = [row for row in page if row[1] is not None and len(row[1]) == size and row[0] in wanted]
        if len(good) < len(page):
            to_embed.extend(_unusable(page, wanted, size, dim, add_many, agent_id))
        if good:
            ids = [row[0] for row in good]
            block = _decode_block(np, [row[1] for row in good], dim)
            try:
                add_many(ids, block, agent_id=agent_id, expected=len(wanted))
            except TypeError:  # an index without the ``expected`` hint
                add_many(ids, block, agent_id=agent_id)
    return to_embed


def _unusable(page: list[tuple], wanted: Any, size: int | None, dim: int | None, add_many: Any, agent_id: str) -> list[str]:
    """Handle rows without a usable float32 blob: index legacy JSON vectors, return the rest."""
    to_embed: list[str] = []
    for fragment_id, blob, legacy, *_ in page:
        if fragment_id not in wanted or (blob is not None and len(blob) == size):
            continue
        vector = _decode_embedding(None, legacy) if blob is None and legacy else None
        if vector and (dim is None or len(vector) == dim):
            add_many([fragment_id], [vector], agent_id=agent_id)
        else:
            to_embed.append(fragment_id)
    return to_embed


def _decode_block(np: Any, blobs: list[bytes], dim: int | None) -> Any:
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


def restore_snapshot(
    store: Any, index: Any, embedder: Any, agent_id: str, missing: Any, before: int | None,
) -> tuple[Any, int | None, bool, int]:
    """Load the agent's vector snapshot into an empty index before any row is read.

    Returns (ids still missing, indexed count, whether a snapshot should be
    saved after the rows are loaded, write_version read before loading).
    Only for a cold start of at least ``MIN_ROWS`` memories with numpy;
    snapshot ids that are no longer allowed (expired, quarantined) are
    removed so the partition equals the scope. A snapshot that missed more
    than a tenth of the scope is rebuilt.
    """
    from mark.store._sqlite.vector_snapshot import MIN_ROWS

    version = getattr(store, "_write_version", lambda: -1)()
    load = getattr(store, "load_vector_snapshot", None)
    fingerprint = getattr(embedder, "fingerprint", None)
    if before != 0 or len(missing) < MIN_ROWS or load is None or not fingerprint or not hasattr(index, "missing"):
        return missing, before, False, version
    np = _load_numpy()
    snapshot = load(agent_id, fingerprint) if np is not None else None
    if snapshot is None or getattr(embedder, "dim", None) not in (None, snapshot[0]):
        return missing, before, np is not None, version
    dim, count, parts = snapshot
    wanted = missing.frozen() if hasattr(missing, "frozen") else missing
    gone: list[str] = []
    for ids, data in parts:
        if ids:
            # Stored unit-length: no renormalising copy; one copy into the pre-sized matrix.
            index.add_many(ids, np.frombuffer(data, dtype="<f4").reshape(len(ids), dim), agent_id=agent_id,
                           expected=count, normalised=True)
            # Deleted / no longer retrievable since the snapshot (C-level filter per part).
            gone.extend(itertools.filterfalse(wanted.__contains__, ids))
    for fragment_id in gone:
        index.remove(fragment_id)
    loaded = index.count(agent_id)
    # Every loaded id is wanted and distinct: equal counts mean nothing is left.
    left = set() if loaded == len(wanted) else index.missing(wanted, agent_id)
    return left, loaded, len(left) >= max(1_000, len(wanted) // 10), version


def save_snapshot(store: Any, index: Any, embedder: Any, agent_id: str, write_version: int) -> None:
    """Persist the agent's partition as a snapshot (numpy partitions only)."""
    from mark.index.vector_bulk import _export_rows
    from mark.store._sqlite.vector_snapshot import PART_ROWS

    save = getattr(store, "save_vector_snapshot", None)
    fingerprint = getattr(embedder, "fingerprint", None)
    exported = _export_rows(index, agent_id) if save is not None and fingerprint else None
    if exported is None or fingerprint.endswith(":None"):
        return
    ids, matrix = exported
    parts = (matrix[i:i + PART_ROWS].tobytes() for i in range(0, len(ids), PART_ROWS))
    try:
        save(agent_id, fingerprint, ids, parts, int(matrix.shape[1]), write_version)
    except Exception:  # noqa: BLE001 - a snapshot is an optimisation (e.g. disk full): never fail recall
        logging.getLogger("mark.retrieval").warning("could not save the vector snapshot", exc_info=True)


def index_each(stored: dict, missing: Iterable[str], dim: int | None, index_add: Any, agent_id: str) -> list[str]:
    """Index stored vectors one by one; return the ids that need (re-)embedding."""
    to_embed: list[str] = []
    for fragment_id in missing:
        vector = stored.get(fragment_id)
        if vector and (dim is None or len(vector) == dim):
            index_add(fragment_id, vector, "", agent_id)
        else:
            to_embed.append(fragment_id)
    return to_embed


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
