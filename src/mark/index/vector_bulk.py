"""Bulk loading for VectorIndex: many vectors of one agent in one call.

Loading a large store vector by vector costs a Python normalisation and a
row copy per vector (tens of seconds at a million). With numpy the whole
block is normalised with array operations and appended to the partition
matrix at once. Same semantics as calling ``VectorIndex.add`` per id: ids
that are already indexed (or repeated) take the per-vector path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mark.index.vector import VectorIndex


def _bulk_add(index: "VectorIndex", ids: list[str], embeddings: Any, agent_id: str | None) -> None:
    """Implementation of ``VectorIndex.add_many`` (takes the index lock)."""
    from mark.index import vector as vector_module

    if not ids:
        return
    if len(embeddings) != len(ids):
        raise ValueError("fragment_ids and embeddings differ in length")
    with index._lock:
        if index._auto and len(index._owner) + len(ids) >= vector_module.AUTO_NUMPY_THRESHOLD:
            if index._dim is None:
                # Promotion needs a dimension; a wrong one fails validation below.
                index._dim = len(embeddings[0])
            index._promote_to_numpy()
        np = index._np
        # C-level checks: a million-id load must not loop in Python here.
        fresh = np is not None and index._owner.keys().isdisjoint(ids) and len(set(ids)) == len(ids)
        if not fresh:
            for fragment_id, vector in zip(ids, embeddings):
                index.add(fragment_id, vector, agent_id=agent_id)
            return
        matrix = np.array(embeddings, dtype=np.float32, copy=True, ndmin=2)
        if index._dim is None:
            index._dim = int(matrix.shape[1])
        if matrix.shape[1] == 0:
            raise ValueError("Cannot index an empty embedding.")
        if matrix.shape[1] != index._dim:
            raise vector_module.DimensionMismatchError(
                f"Embeddings have dim {matrix.shape[1]}, index dim is {index._dim}."
            )
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0  # zero vectors stay zero, as in add()
        matrix /= norms[:, None]
        partition = index._partitions.get(agent_id)
        if partition is None:
            partition = index._new_partition()
            index._partitions[agent_id] = partition
        partition.add_matrix(ids, matrix)
        index._owner.update(dict.fromkeys(ids, agent_id))
        index.revision += 1
