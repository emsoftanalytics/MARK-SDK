"""Maintenance operations for :class:`~mark.memory.runtime.MarkRuntime`.

Consolidation, deduplication, pruning and index warm-up live here so the
runtime module stays focused on wiring. The mixin relies on the runtime's ``store``, ``tracer()``
and ``governance_audit_log()``; heavy modules are imported on first use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mark.memory.consolidation import ConsolidationResult
    from mark.memory.deduplication import DeduplicationResult
    from mark.plasticity.pruner import PrunerStats


class MaintenanceMixin:
    """Memory hygiene: promote, merge duplicates, decay and prune; warm the index."""

    def warm_up(self, agent_ids: "list[str] | None" = None, *, sketches: bool = False) -> int:
        """Do now the work the first recall would do; returns the vectors indexed.

        Loads the embedder (one probe embedding) and, for every agent (or
        just ``agent_ids``), the stored vectors into the in-memory index and
        the retrievable-id cache. Read-only: unlike a dummy recall it does not
        bump access counts. Long-running hosts (``mark serve``, MCP servers)
        call it at start so the first real recall is fast. ``sketches=True``
        also builds an encrypted store's token sketches (one decrypt per
        memory), so the first timeline or identifier recall is fast too.
        """
        from mark.intelligence.retrieval_policy import SessionFilter

        self.embedder.embed("warm-up")
        if agent_ids is None:
            rows = self.store._fetchall("SELECT DISTINCT agent_id FROM fragments")
            agent_ids = [str(row[0]) for row in rows]
        pipeline = self.pipeline
        allowed_ids = getattr(pipeline, "_allowed_ids", None)
        sync_index = getattr(pipeline, "_sync_index", None)
        if allowed_ids is None or sync_index is None:  # custom pipeline: nothing to preload
            return 0
        indexed = 0
        warm_sketches = getattr(self.store, "warm_token_sketches", None) if sketches else None
        for agent_id in agent_ids:
            indexed += len(sync_index(agent_id, allowed_ids(agent_id, SessionFilter())))
            if warm_sketches is not None:
                warm_sketches(agent_id)
        return indexed

    def save_vector_snapshots(self) -> int:
        """Save a fresh vector snapshot for every large agent whose snapshot is stale; returns how many.

        Long-running hosts call it on a graceful stop (``mark serve`` does),
        so the next process starts from a snapshot that already covers every
        write: its cold start then reads no rows and scans no scope. The index
        is first brought in sync with the store (writes from other processes
        included); a write racing with the save simply makes it skip.
        """
        import json

        from mark.intelligence.index_bulk_load import save_snapshot
        from mark.intelligence.retrieval_policy import SessionFilter
        from mark.store._sqlite import vector_snapshot

        store, pipeline, index = self.store, self.pipeline, self.index
        if not getattr(store, "fts_available", False) or not hasattr(pipeline, "_sync_index"):
            return 0  # encrypted / custom stores never use snapshots
        saved = 0
        for agent_id in [a for a in list(getattr(index, "_partitions", {})) if a is not None]:
            if index.count(agent_id) < vector_snapshot.MIN_ROWS:
                continue
            version = store._write_version()
            raw = store.get_meta(vector_snapshot.SCOPE_META_PREFIX + agent_id)
            if raw and json.loads(raw).get("version") == version:
                continue  # the snapshot already describes the store as it is
            pipeline._sync_index(agent_id, pipeline._allowed_ids(agent_id, SessionFilter()))
            save_snapshot(store, index, self.embedder, agent_id, version)
            saved += 1
        return saved

    def consolidate(self, agent_id: str, *, promote_threshold: float = 0.7) -> "ConsolidationResult":
        """Promote high-importance working memory to LTM; delete expired fragments."""
        from mark.memory.consolidation import ConsolidationManager
        tracer = self.tracer()
        tracer.emit("memory.consolidate.start", agent_id=agent_id)
        result = ConsolidationManager(
            self.store,
            agent_id=agent_id,
            audit_log=self.governance_audit_log(),
        ).run(
            promote_threshold=promote_threshold
        )
        tracer.emit(
            "memory.consolidate.finish",
            agent_id=agent_id,
            promoted=result.promoted,
            expired=result.expired,
            skipped=result.skipped,
            blocked=result.blocked,
            sanitized=result.sanitized,
        )
        return result

    def deduplicate(
        self,
        agent_id: str,
        *,
        similarity_threshold: float = 0.95,
    ) -> "DeduplicationResult":
        """Remove near-duplicate fragments, keeping the highest-importance copy."""
        from mark.memory.deduplication import DeduplicationConsolidator, DeduplicationResult
        result = DeduplicationConsolidator(
            self.store, agent_id, similarity_threshold=similarity_threshold
        ).run()
        self.tracer().emit(
            "memory.deduplicate",
            agent_id=agent_id,
            merged=result.merged,
            kept=result.kept,
            skipped=result.skipped,
        )
        return result

    def prune(self, agent_id: str) -> "PrunerStats":
        """Apply decay, cull forgotten fragments, and dissolve weak edges."""
        from mark.plasticity.pruner import MemoryPruner
        stats = MemoryPruner(self.store).run(agent_id)
        self.tracer().emit(
            "memory.prune",
            agent_id=agent_id,
            fragments_pruned=stats.fragments_pruned,
            edges_dissolved=stats.edges_dissolved,
            importance_updated=stats.importance_updated,
        )
        return stats

    def run_cycle(self, agent_id: str, *, promote_threshold: float = 0.7) -> dict:
        """Full maintenance cycle: consolidation, deduplication, pruning; returns a summary."""
        consolidation = self.consolidate(agent_id, promote_threshold=promote_threshold)
        deduplication = self.deduplicate(agent_id)
        pruning       = self.prune(agent_id)
        return {
            "agent_id":          agent_id,
            "promoted":          consolidation.promoted,
            "expired":           consolidation.expired,
            "dedup_merged":      deduplication.merged,
            "dedup_kept":        deduplication.kept,
            "dedup_skipped":     deduplication.skipped,
            "fragments_pruned":  pruning.fragments_pruned,
            "edges_dissolved":   pruning.edges_dissolved,
            "importance_updated": pruning.importance_updated,
            "blocked":           consolidation.blocked,
            "sanitized":         consolidation.sanitized,
        }
