"""Multi-factor reranking of retrieval candidates."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mark.index import VectorSearchResult
from mark.store import LocalMemoryStore


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate handed to the reranker with its vector score."""
    fragment_id: str
    content: str
    score: float
    importance: float = 0.5
    node_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """Reranked candidate with its final combined score."""
    fragment_id: str
    content: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)


class MemoryReranker:
    """MIT fallback reranker: cosine + importance + graph node weight."""

    _W_COSINE = 0.60
    _W_IMPORTANCE = 0.25
    _W_NODE = 0.15
    _CONTRADICTION_PENALTY = 0.20

    def __init__(self, store: LocalMemoryStore) -> None:
        self._store = store

    def rerank(
        self,
        query: str,
        candidates: list[object],
        top_k: int,
        agent_id: str,
        contradiction_ids: set[str] | None = None,
    ) -> list[RerankResult]:
        """Rerank candidates and return the top results."""
        del query
        contradiction_ids = contradiction_ids or set()
        results: list[RerankResult] = []
        for candidate in self._normalize(candidates, agent_id):
            contradiction = candidate.fragment_id in contradiction_ids
            score = (
                self._W_COSINE * candidate.score
                + self._W_IMPORTANCE * candidate.importance
                + self._W_NODE * candidate.node_weight
                - (self._CONTRADICTION_PENALTY if contradiction else 0.0)
            )
            score = max(0.0, round(score, 6))
            results.append(
                RerankResult(
                    fragment_id=candidate.fragment_id,
                    content=candidate.content,
                    score=score,
                    signals={
                        "cosine": round(candidate.score, 4),
                        "importance": round(candidate.importance, 4),
                        "node_weight": round(candidate.node_weight, 4),
                        "contradiction_penalty": self._CONTRADICTION_PENALTY if contradiction else 0.0,
                        "blended": score,
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    async def arerank(
        self,
        query: str,
        candidates: list[object],
        top_k: int,
        agent_id: str,
        contradiction_ids: set[str] | None = None,
    ) -> list[RerankResult]:
        """Async variant of rerank()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.rerank,
            query,
            candidates,
            top_k,
            agent_id,
            contradiction_ids,
        )

    def _normalize(self, candidates: list[object], agent_id: str) -> list[RerankCandidate]:
        output: list[RerankCandidate] = []
        for item in candidates:
            if isinstance(item, RerankCandidate):
                output.append(item)
                continue
            if isinstance(item, VectorSearchResult):
                fragment_id = item.fragment_id
                score = item.score
                content = item.content
            elif isinstance(item, dict):
                fragment_id = str(item.get("fragment_id") or item.get("id") or "")
                score = float(item.get("score", 0.0))
                content = str(item.get("content", ""))
            else:
                continue
            if not fragment_id:
                continue
            fragment = self._store.get(fragment_id)
            if fragment is None or fragment.agent_id != agent_id:
                continue
            node_weight = max((node.weight for node in self._store.nodes_for_fragment(fragment_id)), default=1.0)
            output.append(
                RerankCandidate(
                    fragment_id=fragment_id,
                    content=content or fragment.content,
                    score=score,
                    importance=fragment.importance,
                    node_weight=node_weight,
                )
            )
        return output
