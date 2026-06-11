"""In-memory cosine vector index with candidate restriction and persistence."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VectorSearchResult:
    """Single vector search hit: fragment id and similarity score."""
    fragment_id: str
    score: float
    content: str = ""


class VectorIndex:
    """Small pure-Python cosine index for local CPU-friendly MARK memory."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._content: dict[str, str] = {}

    def add(self, fragment_id: str, embedding: list[float], content: str = "") -> None:
        """Index an embedding under a fragment id."""
        self._vectors[fragment_id] = _normalize(embedding)
        self._content[fragment_id] = content

    def remove(self, fragment_id: str) -> None:
        """Drop a fragment from the index."""
        self._vectors.pop(fragment_id, None)
        self._content.pop(fragment_id, None)

    def get(self, fragment_id: str) -> list[float] | None:
        """Return the stored embedding for a fragment id, or None."""
        vector = self._vectors.get(fragment_id)
        return None if vector is None else list(vector)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        min_score: float = 0.0,
        include_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search the index.

        include_ids — when provided, only fragments in this set are scored.
        Use this to pre-restrict vector search to a session or tag scope so
        that unrelated memories cannot crowd out relevant ones in the top-k.
        """
        query = _normalize(query_embedding)
        vectors = (
            self._vectors.items()
            if include_ids is None
            else ((fid, v) for fid, v in self._vectors.items() if fid in include_ids)
        )
        results = [
            VectorSearchResult(
                fragment_id=fragment_id,
                score=_cosine(query, vector),
                content=self._content.get(fragment_id, ""),
            )
            for fragment_id, vector in vectors
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return [result for result in results if result.score >= min_score][:top_k]

    def save(self, path: str | Path) -> None:
        """Write the index to a JSON file."""
        data = {
            "vectors": self._vectors,
            "content": self._content,
        }
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        """Load an index from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        index._vectors = {
            str(fragment_id): [float(value) for value in vector]
            for fragment_id, vector in data.get("vectors", {}).items()
        }
        index._content = {str(key): str(value) for key, value in data.get("content", {}).items()}
        return index

    def __len__(self) -> int:
        return len(self._vectors)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    return sum(left[index] * right[index] for index in range(size))
