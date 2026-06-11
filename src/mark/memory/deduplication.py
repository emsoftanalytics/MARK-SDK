# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# DeduplicationConsolidator — embedding-similarity-based near-duplicate clustering.
#
# Biological analogy: schema consolidation — repeated exposures to nearly identical
# content should merge into a single stronger memory rather than cluttering recall.
#
# Design adapted from mark-v0/v4 clustering.py (MIT-safe subset).
# Cloud replacement: HOOK_CONSOLIDATION adds LLM-backed content summarisation,
# contradiction review, canonical entity resolution, and nightly scheduling.
"""Embedding-cluster deduplication of stored fragments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClusterResult:
    """Provenance record for one merged cluster."""
    kept_id:         str
    merged_from_ids: list[str]
    cluster_size:    int
    centroid:        list[float] | None = None


@dataclass
class DeduplicationResult:
    """Summary of one deduplication pass over stored embeddings."""
    merged:   int                = 0
    kept:     int                = 0
    skipped:  int                = 0
    clusters: list[ClusterResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure-Python cosine similarity — no numpy dependency.
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))   # y iterates over b — not x
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    """Mean vector of a list of embeddings."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    result = [0.0] * dim
    for emb in embeddings:
        for k in range(dim):
            result[k] += emb[k]
    n = len(embeddings)
    return [v / n for v in result]


def _build_clusters(
    fragments: list[Any],
    *,
    threshold: float,
) -> list[list[Any]]:
    """
    Greedy seed-based clustering.

    Iterates fragments in the order given (caller sorts by importance DESC so
    the first member of each cluster is the most important one).  Each fragment
    is compared against the seed (first member) of every open cluster; if it
    falls within ``threshold`` cosine similarity it joins that cluster.
    Otherwise it starts a new cluster.

    Complexity: O(N × C) where C is the number of distinct clusters found.
    Suitable for agent memory sizes (<10k fragments).
    """
    clusters: list[list[Any]] = []

    for frag in fragments:
        placed = False
        for cluster in clusters:
            seed = cluster[0]
            if _cosine(frag.embedding, seed.embedding) >= threshold:
                cluster.append(frag)
                placed = True
                break
        if not placed:
            clusters.append([frag])

    return clusters


class DeduplicationConsolidator:
    """
    Identifies near-duplicate fragments by embedding cosine similarity and retains
    only the highest-importance member of each cluster, deleting the rest.

    Clustering is seed-based greedy (adapted from mark-v0/v4 clustering.py):
    fragments are sorted by importance DESC so the most important copy becomes
    the cluster seed and is always kept.

    Fragments without stored embeddings are skipped (appear in ``skipped`` count).
    The kept fragment's metadata gains ``merged_from_ids`` and ``cluster_size``
    so provenance is traceable.

    Local SDK: O(N × C) pass — suitable for agent memory sizes (<10k fragments).
    Cloud replacement: HOOK_CONSOLIDATION handles larger corpora, LLM-summarised
    merges, contradiction review, and nightly scheduling.

    Usage::

        result = DeduplicationConsolidator(store, agent_id="director").run()
        # result.merged   — duplicate fragments deleted
        # result.kept     — canonical fragments retained
        # result.clusters — per-cluster provenance records
    """

    def __init__(
        self,
        store: Any,
        agent_id: str,
        *,
        similarity_threshold: float = 0.95,
    ) -> None:
        self._store     = store
        self._agent_id  = agent_id
        self._threshold = similarity_threshold

    def run(
        self,
        *,
        similarity_threshold: float | None = None,
        session_id: str | None = None,
    ) -> DeduplicationResult:
        """
        Run a deduplication pass and return a DeduplicationResult.

        Parameters
        ----------
        similarity_threshold:
            Override the instance threshold for this run.
        session_id:
            When set, only fragments matching this session are considered.
            Pass ``None`` to deduplicate across the full agent namespace.
        """
        threshold = similarity_threshold if similarity_threshold is not None else self._threshold
        result    = DeduplicationResult()

        all_fragments = self._store.list_by_agent(self._agent_id, limit=10_000)

        # Optional session filter
        if session_id is not None:
            all_fragments = [f for f in all_fragments if f.session_id == session_id]

        # Separate fragments with embeddings
        embedded = [f for f in all_fragments if f.embedding]
        result.skipped = len(all_fragments) - len(embedded)

        if not embedded:
            return result

        # Sort descending by importance → cluster seeds are the most important copies
        embedded.sort(key=lambda f: f.importance, reverse=True)

        clusters = _build_clusters(embedded, threshold=threshold)

        for cluster in clusters:
            seed      = cluster[0]   # highest-importance fragment in this cluster
            rest      = cluster[1:]

            if not rest:
                # Singleton cluster — just keep it
                result.kept += 1
                result.clusters.append(
                    ClusterResult(
                        kept_id         = seed.id,
                        merged_from_ids = [],
                        cluster_size    = 1,
                        centroid        = None,
                    )
                )
                continue

            # Multiple-member cluster — merge rest into seed
            merged_ids = [f.id for f in rest]
            prev_merged = list(seed.metadata.get("merged_from_ids", []))
            all_merged  = prev_merged + merged_ids

            centroid = _centroid([f.embedding for f in cluster])

            updated_meta = {
                **seed.metadata,
                "merged_from_ids": all_merged,
                "cluster_size":    len(cluster),
            }
            self._store.store(seed.model_copy(update={"metadata": updated_meta}))

            for dup in rest:
                if hasattr(self._store, "nodes_for_fragment") and hasattr(self._store, "store_node"):
                    for node in self._store.nodes_for_fragment(dup.id):
                        self._store.store_node(node.model_copy(update={"fragment_id": seed.id}))
                self._store.delete(dup.id)
                result.merged += 1

            result.kept += 1
            result.clusters.append(
                ClusterResult(
                    kept_id         = seed.id,
                    merged_from_ids = merged_ids,
                    cluster_size    = len(cluster),
                    centroid        = centroid,
                )
            )

        return result
