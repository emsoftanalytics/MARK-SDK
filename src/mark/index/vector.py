"""In-memory cosine vector index with agent partitions, candidate restriction, and persistence.

Two interchangeable scoring backends are provided:

- ``numpy`` (used automatically when numpy is importable): one float32 matrix
  per agent partition, built lazily and scored with a single mat-vec product
  plus ``np.argpartition`` for top-k selection;
- ``python``: dependency-free scoring with ``heapq.nlargest``.

Both backends return the same hits in the same order (ties are broken by
insertion order); scores agree to float32 precision.
"""
from __future__ import annotations

import heapq
import json
import math
import operator
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_FORMAT = "mark.vector-index"
_FORMAT_VERSION = 2

_sumprod = getattr(math, "sumprod", None)  # Python 3.12+


class DimensionMismatchError(ValueError):
    """Raised when a vector's dimension differs from the index dimension."""


@dataclass(frozen=True)
class VectorSearchResult:
    """Single vector search hit: fragment id and similarity score."""
    fragment_id: str
    score: float
    content: str = ""


def _load_numpy() -> Any | None:
    try:
        import numpy  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - exercised when numpy is absent
        return None
    return numpy


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


class VectorIndex:
    """Thread-safe cosine index for local CPU-friendly MARK memory.

    Vectors are partitioned by ``agent_id`` (``None`` is its own partition), so
    ``search(..., agent_id=...)`` only scores that agent's vectors.
    """

    def __init__(self, *, backend: str = "auto") -> None:
        if backend not in {"auto", "numpy", "python"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'python'")
        np = None if backend == "python" else _load_numpy()
        if backend == "numpy" and np is None:
            raise ImportError("VectorIndex(backend='numpy') requires numpy: install `mark-sdk[fast]`.")
        self._np = np
        self._lock = threading.RLock()
        self._partitions: dict[str | None, _PythonPartition | _NumpyPartition] = {}
        self._owner: dict[str, str | None] = {}
        self._content: dict[str, str] = {}
        self._dim: int | None = None

    @property
    def backend(self) -> str:
        """Active scoring backend: "numpy" or "python"."""
        return "python" if self._np is None else "numpy"

    @property
    def dim(self) -> int | None:
        """Vector dimension (set by the first add), or None while empty."""
        return self._dim

    def add(
        self,
        fragment_id: str,
        embedding: Sequence[float],
        content: str = "",
        *,
        agent_id: str | None = None,
    ) -> None:
        """Index (or re-index) an embedding under a fragment id."""
        vector = _normalize(embedding)
        if not vector:
            raise ValueError("Cannot index an empty embedding.")
        with self._lock:
            if self._dim is None:
                self._dim = len(vector)
            elif len(vector) != self._dim:
                raise DimensionMismatchError(
                    f"Embedding for {fragment_id!r} has dim {len(vector)}, index dim is {self._dim}."
                )
            previous_owner = self._owner.get(fragment_id, agent_id)
            if fragment_id in self._owner and previous_owner != agent_id:
                self._partition_remove(previous_owner, fragment_id)
            partition = self._partitions.get(agent_id)
            if partition is None:
                partition = self._new_partition()
                self._partitions[agent_id] = partition
            partition.add(fragment_id, vector)
            self._owner[fragment_id] = agent_id
            self._content[fragment_id] = content

    def remove(self, fragment_id: str) -> None:
        """Drop a fragment from the index (no-op when absent)."""
        with self._lock:
            if fragment_id not in self._owner:
                return
            owner = self._owner.pop(fragment_id)
            self._content.pop(fragment_id, None)
            self._partition_remove(owner, fragment_id)
            if not self._owner:
                self._dim = None

    def get(self, fragment_id: str) -> list[float] | None:
        """Return the stored (normalised) embedding for a fragment id, or None."""
        with self._lock:
            if fragment_id not in self._owner:
                return None
            return self._partitions[self._owner[fragment_id]].get(fragment_id)

    def ids(self, agent_id: str | None = None) -> list[str]:
        """Return indexed ids — all of them, or only one agent's when agent_id is given."""
        with self._lock:
            if agent_id is None:
                return list(self._owner)
            partition = self._partitions.get(agent_id)
            return [] if partition is None else list(partition.vectors)

    def clear(self, agent_id: str | None = None) -> None:
        """Remove every vector, or only one agent's when agent_id is given."""
        with self._lock:
            if agent_id is None:
                self._partitions.clear()
                self._owner.clear()
                self._content.clear()
                self._dim = None
                return
            partition = self._partitions.pop(agent_id, None)
            if partition is None:
                return
            for fragment_id in partition.vectors:
                self._owner.pop(fragment_id, None)
                self._content.pop(fragment_id, None)
            if not self._owner:
                self._dim = None

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 10,
        min_score: float = -1.0,
        include_ids: set[str] | None = None,
        agent_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search the index.

        agent_id    — when provided, only that agent's vectors are scored.
        include_ids — when provided, only fragments in this set are scored.
        Use these to pre-restrict vector search to an agent, session, or tag
        scope so unrelated memories cannot crowd relevant ones out of the top-k.

        Raises DimensionMismatchError when the query dimension differs from
        the index dimension.
        """
        if top_k <= 0 or (include_ids is not None and not include_ids):
            return []
        with self._lock:
            if self._dim is None:
                return []
            if len(query_embedding) != self._dim:
                raise DimensionMismatchError(
                    f"Query has dim {len(query_embedding)}, index dim is {self._dim}."
                )
            if agent_id is not None:
                partition = self._partitions.get(agent_id)
                partitions = [] if partition is None else [partition]
            else:
                partitions = list(self._partitions.values())
            snapshots = [(partition, partition.snapshot()) for partition in partitions]
        query = _normalize(query_embedding)
        if self._np is not None:
            query = self._np.asarray(query, dtype=self._np.float32)
        hits: list[tuple[float, int, int, str]] = []
        for order, (partition, snapshot) in enumerate(snapshots):
            for score, position, fragment_id in partition.top(snapshot, query, top_k, include_ids):
                hits.append((score, order, position, fragment_id))
        hits.sort(key=lambda item: (-item[0], item[1], item[2]))
        content = self._content
        return [
            VectorSearchResult(fragment_id=fragment_id, score=score, content=content.get(fragment_id, ""))
            for score, _, _, fragment_id in hits[:top_k]
            if score >= min_score
        ]

    def save(self, path: str | Path) -> None:
        """Write the index to a JSON file atomically (temp file + os.replace)."""
        with self._lock:
            data = {
                "format": _FORMAT,
                "version": _FORMAT_VERSION,
                "dim": self._dim,
                "vectors": {
                    fragment_id: self._partitions[owner].get(fragment_id)
                    for fragment_id, owner in self._owner.items()
                },
                "content": dict(self._content),
                "agents": dict(self._owner),
            }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path, *, backend: str = "auto") -> "VectorIndex":
        """Load an index from a JSON file (current or legacy v1 format)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(backend=backend)
        content = {str(key): str(value) for key, value in data.get("content", {}).items()}
        agents = data.get("agents", {}) or {}
        for fragment_id, vector in data.get("vectors", {}).items():
            owner = agents.get(fragment_id)
            index.add(
                str(fragment_id),
                [float(value) for value in vector],
                content.get(str(fragment_id), ""),
                agent_id=None if owner is None else str(owner),
            )
        return index

    def __contains__(self, fragment_id: object) -> bool:
        return fragment_id in self._owner

    def __len__(self) -> int:
        return len(self._owner)

    def __bool__(self) -> bool:
        # An empty index is still a real index: never let `index or VectorIndex()`
        # silently replace a shared (empty) index with a private one.
        return True

    def _new_partition(self) -> _PythonPartition | _NumpyPartition:
        if self._np is None:
            return _PythonPartition()
        assert self._dim is not None
        return _NumpyPartition(self._np, self._dim)

    def _partition_remove(self, owner: str | None, fragment_id: str) -> None:
        partition = self._partitions.get(owner)
        if partition is None:
            return
        partition.remove(fragment_id)
        if not len(partition):
            del self._partitions[owner]


def _normalize(vector: Iterable[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(_dot(values, values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if _sumprod is not None:
        return float(_sumprod(left, right))
    return float(sum(map(operator.mul, left, right)))
