"""In-memory cosine vector index with agent partitions, candidate restriction, and persistence.

Two interchangeable scoring backends are provided:

- ``numpy``: one float32 matrix per agent partition (the matrix is the
  source of truth; ``add_many`` appends whole blocks), scored with a single
  mat-vec product plus ``np.partition`` for top-k selection;
- ``python``: dependency-free scoring with ``heapq.nlargest``.

Both backends return the same hits in the same order (ties are broken by
insertion order); scores agree to float32 precision.

``backend="auto"`` (the default) starts in pure Python and switches to numpy
once the index holds ``AUTO_NUMPY_THRESHOLD`` vectors, if numpy is installed.
Importing numpy costs 0.2–0.7 s, far more than scoring a few thousand vectors
in Python, so short-lived processes with small stores never pay for it.

Backends live in ``vector_backends``, bulk loading in ``vector_bulk`` and the
file format in ``vector_storage``; this module owns partitioning, locking,
auto-promotion and the public API. ``revision`` changes on every mutation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# _dot and _sumprod are not used here; they stay importable from this module.
from mark.index.vector_backends import _dot, _load_numpy, _normalize, _NumpyPartition, _PythonPartition, _sumprod
from mark.index.vector_bulk import _bulk_add
from mark.index.vector_storage import _index_payload, _read_index_file, _write_json_atomic

#: Vector count at which ``backend="auto"`` switches to numpy (when installed).
#: Read at call time from this module's globals, so it can be patched in tests.
AUTO_NUMPY_THRESHOLD = 2048


class DimensionMismatchError(ValueError):
    """Raised when a vector's dimension differs from the index dimension."""


@dataclass(frozen=True)
class VectorSearchResult:
    """Single vector search hit: fragment id and similarity score."""
    fragment_id: str
    score: float
    content: str = ""


class VectorIndex:
    """Thread-safe cosine index for local CPU-friendly MARK memory.

    Vectors are partitioned by ``agent_id`` (``None`` is its own partition), so
    ``search(..., agent_id=...)`` only scores that agent's vectors.
    """

    def __init__(self, *, backend: str = "auto") -> None:
        if backend not in {"auto", "numpy", "python"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'python'")
        self._np = _load_numpy() if backend == "numpy" else None
        if backend == "numpy" and self._np is None:
            raise ImportError("VectorIndex(backend='numpy') requires numpy: install `mark-sdk[fast]`.")
        # "auto" decides lazily: numpy only once the index is large enough.
        self._auto = backend == "auto"
        self._lock = threading.RLock()
        self._partitions: dict[str | None, _PythonPartition | _NumpyPartition] = {}
        self._owner: dict[str, str | None] = {}
        # Only non-empty contents are kept (the pipeline indexes without content).
        self._content: dict[str, str] = {}
        self._dim: int | None = None
        # Bumped by every mutation, so callers can cache "already in sync".
        self.revision = 0

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
            self._set_content(fragment_id, content)
            self.revision += 1
            # Promotion is checked on the total vector count, across all agents.
            if self._auto and len(self._owner) >= AUTO_NUMPY_THRESHOLD:
                self._promote_to_numpy()

    def add_many(
        self,
        fragment_ids: Sequence[str],
        embeddings: Any,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Index many embeddings (vectors or a 2-D array) of one agent, without contents.

        Same result as :meth:`add` per id; with numpy it is one block append.
        """
        _bulk_add(self, list(fragment_ids), embeddings, agent_id)

    def remove(self, fragment_id: str) -> None:
        """Drop a fragment from the index (no-op when absent)."""
        with self._lock:
            if fragment_id not in self._owner:
                return
            owner = self._owner.pop(fragment_id)
            self._content.pop(fragment_id, None)
            self._partition_remove(owner, fragment_id)
            self.revision += 1
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
            return [] if partition is None else partition.ids()

    def count(self, agent_id: str | None = None) -> int:
        """Number of indexed vectors — all of them, or one agent's."""
        with self._lock:
            if agent_id is None:
                return len(self._owner)
            partition = self._partitions.get(agent_id)
            return 0 if partition is None else len(partition)

    def missing(self, fragment_ids: Iterable[str], agent_id: str | None = None) -> set[str]:
        """The given ids that are not indexed under *agent_id* (one C-level set difference)."""
        with self._lock:
            partition = self._partitions.get(agent_id)
            return set(fragment_ids).difference(partition.keys() if partition is not None else ())

    def clear(self, agent_id: str | None = None) -> None:
        """Remove every vector, or only one agent's when agent_id is given."""
        with self._lock:
            self.revision += 1
            if agent_id is None:
                self._partitions.clear()
                self._owner.clear()
                self._content.clear()
                self._dim = None
                return
            partition = self._partitions.pop(agent_id, None)
            if partition is None:
                return
            for fragment_id in partition.ids():
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
        """Search the index (only *agent_id*'s vectors, only *include_ids*, when given).

        Pre-restricting to an agent/session/tag scope keeps unrelated memories
        from crowding the top-k. Raises DimensionMismatchError on a query of
        another dimension.
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
            np = self._np
        query = _normalize(query_embedding)
        np_query = np.asarray(query, dtype=np.float32) if np is not None else None
        hits: list[tuple[float, int, int, str]] = []
        for order, (partition, snapshot) in enumerate(snapshots):
            partition_query = np_query if isinstance(partition, _NumpyPartition) else query
            for score, position, fragment_id in partition.top(snapshot, partition_query, top_k, include_ids):
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
            data = _index_payload(self._dim, self._owner, self._content, self._partitions)
        _write_json_atomic(Path(path), data)

    @classmethod
    def load(cls, path: str | Path, *, backend: str = "auto") -> "VectorIndex":
        """Load an index from a JSON file (current or legacy v1 format)."""
        index = cls(backend=backend)
        for fragment_id, vector, content, owner in _read_index_file(Path(path)):
            index.add(fragment_id, vector, content, agent_id=owner)
        return index

    def __contains__(self, fragment_id: object) -> bool:
        return fragment_id in self._owner

    def __len__(self) -> int:
        return len(self._owner)

    def __bool__(self) -> bool:
        # Empty is still real: `index or VectorIndex()` must not swap in a private one.
        return True

    def _promote_to_numpy(self) -> None:
        """Switch every partition to the numpy backend (caller holds the lock).

        Searches already running keep scoring their Python snapshots; since
        both backends share the same ordering contract, results don't change.
        """
        self._auto = False  # decide once, whether or not numpy is available
        np = _load_numpy()
        if np is None or self._dim is None:
            return
        self._np = np
        for owner, partition in list(self._partitions.items()):
            if isinstance(partition, _PythonPartition):
                promoted = _NumpyPartition(np, self._dim)
                if len(partition):
                    promoted.add_matrix(
                        list(partition.vectors), np.asarray(list(partition.vectors.values()), dtype=np.float32),
                    )
                self._partitions[owner] = promoted

    def _set_content(self, fragment_id: str, content: str) -> None:
        if content:
            self._content[fragment_id] = content
        else:
            self._content.pop(fragment_id, None)

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
