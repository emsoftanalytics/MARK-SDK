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

    def ids(self) -> list[str]:
        return list(self.vectors)

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
    """Vectors for one agent, scored with one float32 matrix (rows in insertion order).

    The matrix is the source of truth (no per-vector Python objects), so a
    million-vector partition costs one contiguous array plus an id list:

    - appends write the next free row in place; when the matrix is full a
      larger copy replaces it (snapshots keep the old object);
    - re-adding an id overwrites its row in place, keeping its position, which
      is what the pure-Python backend does with dict assignment;
    - removals only mark the row dead (scored as -inf and skipped); dead rows
      are compacted away once they exceed a quarter of the matrix.
    """

    def __init__(self, np: Any, dim: int) -> None:
        self._np = np
        self._dim = dim
        self._matrix: Any = np.zeros((16, dim), dtype=np.float32)
        self._ids: list[str | None] = []  # row -> id (None once removed)
        self._pos: dict[str, int] = {}  # live id -> row
        self._dead: Any = None  # bool mask over rows, allocated on first removal
        self._dead_count = 0

    @property
    def vectors(self) -> dict[str, list[float]]:
        """Live id -> vector mapping (a copy; used for promotion and persistence)."""
        return {fid: self._matrix[row].tolist() for fid, row in self._pos.items()}

    def ids(self) -> list[str]:
        return list(self._pos)

    def add(self, fragment_id: str, vector: Any) -> None:
        row = self._pos.get(fragment_id)
        if row is not None:
            # Same position as dict assignment in the Python backend. A search
            # scoring concurrently may see the old or the new vector.
            self._matrix[row] = vector
            return
        size = len(self._ids)
        if size >= self._matrix.shape[0]:
            self._grow(size + 1)
        self._matrix[size] = vector
        self._ids.append(fragment_id)
        self._pos[fragment_id] = size

    def add_matrix(self, ids: list[str], matrix: Any) -> None:
        """Append many unit-normalised rows at once (ids must not be indexed yet)."""
        size = len(self._ids)
        if size + len(ids) > self._matrix.shape[0]:
            self._grow(size + len(ids))
        self._matrix[size:size + len(ids)] = matrix
        self._ids.extend(ids)
        self._pos.update(zip(ids, range(size, size + len(ids))))

    def remove(self, fragment_id: str) -> None:
        row = self._pos.pop(fragment_id, None)
        if row is None:
            return
        np = self._np
        if self._dead is None or self._dead.shape[0] < self._matrix.shape[0]:
            dead = np.zeros(self._matrix.shape[0], dtype=bool)
            if self._dead is not None:
                dead[: self._dead.shape[0]] = self._dead
            self._dead = dead
        self._dead[row] = True
        self._ids[row] = None
        self._dead_count += 1
        if self._dead_count > max(64, len(self._ids) // 4):
            self._compact()

    def get(self, fragment_id: str) -> list[float] | None:
        row = self._pos.get(fragment_id)
        return None if row is None else [float(value) for value in self._matrix[row].tolist()]

    def _grow(self, needed: int) -> None:
        # Grow by ~1/8 for big partitions (1M x 256 floats is 1 GB: a 1.5x
        # policy would waste half a gigabyte), by doubling for small ones.
        capacity = max(16, needed + max(needed // 8, min(needed, 65_536)))
        matrix = self._np.zeros((capacity, self._dim), dtype=self._np.float32)
        size = len(self._ids)
        matrix[:size] = self._matrix[:size]
        if self._dead is not None:
            dead = self._np.zeros(capacity, dtype=bool)
            dead[: self._dead.shape[0]] = self._dead
            self._dead = dead
        self._matrix = matrix  # a new object: snapshots keep the old one

    def _compact(self) -> None:
        np = self._np
        live = list(self._pos.items())  # insertion (row) order
        rows = np.fromiter((row for _, row in live), dtype=np.int64, count=len(live))
        capacity = max(16, len(live) + max(len(live) // 8, min(len(live), 65_536)))
        matrix = np.zeros((capacity, self._dim), dtype=np.float32)
        matrix[: len(live)] = self._matrix[rows]
        # New objects throughout, so in-flight snapshots stay consistent.
        self._matrix = matrix
        self._ids = [fid for fid, _ in live]
        self._pos = {fid: row for row, fid in enumerate(self._ids)}
        self._dead = None
        self._dead_count = 0

    def snapshot(self) -> Any:
        return (self._matrix, self._ids, self._pos, len(self._ids), self._dead if self._dead_count else None)

    def top(self, snapshot: Any, query: Any, top_k: int, include_ids: set[str] | None) -> list[tuple[float, int, str]]:
        np = self._np
        matrix, ids, pos, size, dead = snapshot
        if size == 0:
            return []
        if include_ids is None:
            rows = None
            scores = matrix[:size] @ query
            if dead is not None:
                scores[dead[:size]] = -np.inf
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
        order = np.lexsort((positions, -scores[candidates]))
        hits: list[tuple[float, int, str]] = []
        for i in order:
            score = float(scores[candidates[i]])
            fragment_id = ids[int(positions[i])] if score != -np.inf else None
            if fragment_id is not None:  # None: removed after the snapshot was taken
                hits.append((score, int(positions[i]), fragment_id))
                if len(hits) == top_k:
                    break
        return hits

    def __len__(self) -> int:
        return len(self._pos)


def _load_numpy() -> Any | None:
    """Import numpy on demand; None when it is not installed (or fails to import)."""
    try:
        import numpy  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - exercised when numpy is absent
        return None
    return numpy


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
