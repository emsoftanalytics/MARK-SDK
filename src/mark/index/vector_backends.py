"""Scoring backends for VectorIndex: one partition (agent) of vectors each.

Both backends store unit-normalised vectors, so cosine similarity is a dot
product, and both return ``(score, insertion position, fragment id)``
triples ordered by score with ties broken by insertion order. That shared
contract is what lets ``backend="auto"`` switch from Python to numpy
mid-life without changing any search result.

Partitions are not thread-safe on their own: VectorIndex mutates them under
its lock and searches lock-free on a ``snapshot()`` taken under that lock.
"""
from __future__ import annotations

import heapq
import math
import operator
from typing import Any, Iterable, Sequence

_sumprod = getattr(math, "sumprod", None)  # Python 3.12+: fast, exactly-rounded dot product


class _PythonPartition:
    """Vectors for one agent, scored in pure Python."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def add(self, fragment_id: str, vector: list[float]) -> None:
        self.vectors[fragment_id] = vector

    def remove(self, fragment_id: str) -> None:
        self.vectors.pop(fragment_id, None)

    def get(self, fragment_id: str) -> list[float] | None:
        vector = self.vectors.get(fragment_id)
        return None if vector is None else list(vector)

    def snapshot(self) -> Any:
        # Vectors are never mutated in place, so a shallow copy is consistent.
        return list(self.vectors.items())

    @staticmethod
    def top(snapshot: Any, query: list[float], top_k: int, include_ids: set[str] | None) -> list[tuple[float, int, str]]:
        scored = [
            (_dot(query, vector), position, fragment_id)
            for position, (fragment_id, vector) in enumerate(snapshot)
            if include_ids is None or fragment_id in include_ids
        ]
        return heapq.nlargest(top_k, scored, key=lambda item: (item[0], -item[1]))

    def __len__(self) -> int:
        return len(self.vectors)


class _NumpyPartition:
    """Vectors for one agent, scored with a lazily (re)built float32 matrix."""

    def __init__(self, np: Any, dim: int) -> None:
        self._np = np
        self._dim = dim
        self.vectors: dict[str, Any] = {}
        self._matrix: Any = None
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._size = 0
        self._dirty = True

    def add(self, fragment_id: str, vector: list[float]) -> None:
        row = self._np.asarray(vector, dtype=self._np.float32)
        replacing = fragment_id in self.vectors
        self.vectors[fragment_id] = row
        if replacing or self._dirty or self._matrix is None or self._size >= self._matrix.shape[0]:
            self._dirty = True
            return
        # Append in place: rows below a snapshot's size are never touched, so
        # concurrent searches on an older snapshot stay consistent.
        self._matrix[self._size] = row
        self._ids.append(fragment_id)
        self._pos[fragment_id] = self._size
        self._size += 1

    def remove(self, fragment_id: str) -> None:
        if self.vectors.pop(fragment_id, None) is not None:
            self._dirty = True

    def get(self, fragment_id: str) -> list[float] | None:
        row = self.vectors.get(fragment_id)
        return None if row is None else [float(value) for value in row.tolist()]

    def _rebuild(self) -> None:
        np = self._np
        count = len(self.vectors)
        capacity = max(16, int(count * 1.5))
        matrix = np.zeros((capacity, self._dim), dtype=np.float32)
        if count:
            matrix[:count] = np.stack(list(self.vectors.values()))
        # New objects: snapshots holding the old ones remain valid.
        self._matrix = matrix
        self._ids = list(self.vectors)
        self._pos = {fragment_id: row for row, fragment_id in enumerate(self._ids)}
        self._size = count
        self._dirty = False

    def snapshot(self) -> Any:
        if self._dirty or self._matrix is None:
            self._rebuild()
        return (self._matrix, self._ids, self._pos, self._size)

    def top(self, snapshot: Any, query: Any, top_k: int, include_ids: set[str] | None) -> list[tuple[float, int, str]]:
        np = self._np
        matrix, ids, pos, size = snapshot
        if size == 0:
            return []
        if include_ids is None:
            rows = None
            scores = matrix[:size] @ query
        else:
            selected = [row for row in (pos.get(fid) for fid in include_ids) if row is not None and row < size]
            if not selected:
                return []
            rows = np.fromiter(sorted(selected), dtype=np.int64, count=len(selected))
            if len(selected) * 8 < size:
                scores = matrix[rows] @ query  # small subset: copy only those rows
            else:
                scores = (matrix[:size] @ query)[rows]  # large subset: one matvec is cheaper
        count = scores.shape[0]
        if top_k < count:
            # Keep every candidate tied with the k-th score so tie-breaking by
            # insertion order matches the pure-Python backend exactly.
            kth = np.partition(scores, count - top_k)[count - top_k]
            candidates = np.nonzero(scores >= kth)[0]
        else:
            candidates = np.arange(count)
        positions = candidates if rows is None else rows[candidates]
        order = np.lexsort((positions, -scores[candidates]))[:top_k]
        return [
            (float(scores[candidates[i]]), int(positions[i]), ids[int(positions[i])])
            for i in order
        ]

    def __len__(self) -> int:
        return len(self.vectors)


def _normalize(vector: Iterable[float]) -> list[float]:
    """Unit-length copy of *vector* as floats; a zero vector is returned unchanged."""
    values = [float(value) for value in vector]
    norm = math.sqrt(_dot(values, values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product; uses the C-level math.sumprod on Python 3.12+."""
    if _sumprod is not None:
        return float(_sumprod(left, right))
    return float(sum(map(operator.mul, left, right)))
