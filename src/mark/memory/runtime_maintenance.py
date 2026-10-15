"""Maintenance operations for :class:`~mark.memory.runtime.MarkRuntime`.

Consolidation, deduplication and pruning live here so the runtime module stays
focused on wiring. The mixin relies on the runtime's ``store``, ``tracer()``
and ``governance_audit_log()``; heavy modules are imported on first use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mark.memory.consolidation import ConsolidationResult
    from mark.memory.deduplication import DeduplicationResult
    from mark.plasticity.pruner import PrunerStats


class MaintenanceMixin:
    """Memory hygiene: promote, merge duplicates, decay and prune."""

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
