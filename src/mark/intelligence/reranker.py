"""Multi-factor reranking of retrieval candidates.

Relevance first, bounded priors (importance, confidence, graph weight,
optionally recency) second, then two penalties: memories contradicted by a
retrieved memory, and memories retired through the API (tag
``mark:superseded`` / ``mark:resolved`` or metadata ``superseded_by`` /
``resolved_by``) unless the query asks about the past.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from mark.index import VectorSearchResult
from mark.intelligence.supersession import is_retired
from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryFragment


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate handed to the reranker with its retrieval score."""
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
    """Apache-2.0 fallback reranker: relevance first, bounded priors second.

    ``final = relevance * (1 - W_PRIOR + W_PRIOR * prior) * contradiction``

    ``relevance`` is the candidate's retrieval score (fused lexical + vector
    evidence). ``prior`` blends importance, confidence, graph node weight, and
    optionally recency. Priors can only reorder candidates of comparable
    relevance: an irrelevant high-importance memory can never outrank a
    relevant one, which keeps ranking stable under Hebbian reinforcement.
    """

    W_PRIOR = 0.25
    _W_IMPORTANCE = 0.55
    _W_CONFIDENCE = 0.20
    _W_NODE = 0.25
    _RECENCY_HALF_LIFE_DAYS = 30.0
    _CONTRADICTION_FACTOR = 0.6
    # A retired memory still answers "what was it before?", so it is demoted, not dropped.
    _RETIRED_FACTOR = 0.5

    def __init__(self, store: LocalMemoryStore) -> None:
        self._store = store

    def rerank(
        self,
        query: str,
        candidates: list[object],
        top_k: int,
        agent_id: str,
        contradiction_ids: set[str] | None = None,
        *,
        fragments: Mapping[str, MemoryFragment] | None = None,
        recency_weight: float = 0.0,
        now: datetime | None = None,
        demote_retired: bool = True,
    ) -> list[RerankResult]:
        """Rerank candidates and return the top results.

        fragments       prefetched fragments by id (avoids one lookup per candidate)
        recency_weight  0.0–1.0 share of the prior given to recency (temporal queries)
        demote_retired  halve the score of superseded / resolved memories
                        (False for "original" / "previously" questions)
        """
        del query
        contradiction_ids = contradiction_ids or set()
        recency_weight = min(1.0, max(0.0, recency_weight))
        current = now or datetime.now(timezone.utc)
        results: list[RerankResult] = []
        for candidate, fragment in self._normalize(candidates, agent_id, fragments):
            contradiction = candidate.fragment_id in contradiction_ids
            confidence = float(getattr(fragment, "confidence", 0.5) if fragment else 0.5)
            prior = (
                self._W_IMPORTANCE * candidate.importance
                + self._W_CONFIDENCE * confidence
                + self._W_NODE * candidate.node_weight
            )
            recency = _recency(fragment, current, self._RECENCY_HALF_LIFE_DAYS) if fragment else 0.0
            if recency_weight:
                prior = (1.0 - recency_weight) * prior + recency_weight * recency
            factor = (1.0 - self.W_PRIOR) + self.W_PRIOR * min(1.0, max(0.0, prior))
            score = max(0.0, candidate.score) * factor
            if contradiction:
                score *= self._CONTRADICTION_FACTOR
            retired = demote_retired and fragment is not None and is_retired(fragment)
            if retired:
                score *= self._RETIRED_FACTOR
            score = round(score, 6)
            results.append(
                RerankResult(
                    fragment_id=candidate.fragment_id,
                    content=candidate.content,
                    score=score,
                    signals={
                        "relevance": round(candidate.score, 4),
                        "importance": round(candidate.importance, 4),
                        "confidence": round(confidence, 4),
                        "node_weight": round(candidate.node_weight, 4),
                        "recency": round(recency, 4),
                        "prior": round(prior, 4),
                        "contradiction_penalty": (1.0 - self._CONTRADICTION_FACTOR) if contradiction else 0.0,
                        "retired_penalty": (1.0 - self._RETIRED_FACTOR) if retired else 0.0,
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
        return await asyncio.to_thread(
            self.rerank, query, candidates, top_k, agent_id, contradiction_ids,
        )

    def _normalize(
        self,
        candidates: list[object],
        agent_id: str,
        fragments: Mapping[str, MemoryFragment] | None,
    ) -> list[tuple[RerankCandidate, MemoryFragment | None]]:
        output: list[tuple[RerankCandidate, MemoryFragment | None]] = []
        for item in candidates:
            if isinstance(item, RerankCandidate):
                output.append((item, (fragments or {}).get(item.fragment_id)))
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
            fragment = fragments.get(fragment_id) if fragments is not None else self._store.get(fragment_id)
            if fragment is None or fragment.agent_id != agent_id:
                continue
            # A fragment without graph nodes is neutral (0.5), not maximal:
            # attaching a node must never lower a fragment's rank by default.
            node_weight = max(
                (node.weight for node in self._store.nodes_for_fragment(fragment_id)),
                default=0.5,
            )
            output.append((
                RerankCandidate(
                    fragment_id=fragment_id,
                    content=content or fragment.content,
                    score=score,
                    importance=fragment.importance,
                    node_weight=node_weight,
                ),
                fragment,
            ))
        return output


def _recency(fragment: MemoryFragment, now: datetime, half_life_days: float) -> float:
    stamp = fragment.updated_at or fragment.created_at
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - stamp).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / half_life_days)
