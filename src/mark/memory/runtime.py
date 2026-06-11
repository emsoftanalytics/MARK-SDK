"""MarkRuntime: wiring and lifecycle for the local memory engine."""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mark.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mark.index import VectorIndex
from mark.intelligence import GraphExpander, RetrievalPipeline
from mark.intelligence.compressor import ContextualCompressor
from mark.intelligence.query_expander import QueryExpander
from mark.middleware import MarkMiddleware, MiddlewareStack
from mark.observability import LocalTracer, SessionTrace
from mark.plugins import MarkCorePlugin, PluginRegistry
from mark.security import EncryptionProvider, NoOpEncryptionProvider
from mark.store import LocalMemoryStore
from mark.types.llm import LLMProvider

from mark.governance import GovernanceAuditLog
from mark.memory.global_bus import GlobalMemoryBus
from mark.memory.mark_memory import MarkMemory
from mark.memory.trust_bus import TrustAwareGlobalMemoryBus


class MarkRuntime:
    """Stronger local hippocampus/cortex runtime for agent applications."""

    def __init__(
        self,
        *,
        store: LocalMemoryStore,
        embedder: EmbeddingProvider,
        plugins: PluginRegistry,
        index: VectorIndex | None = None,
        workers: int = 4,
        llm: LLMProvider | None = None,
        compressor: ContextualCompressor | None = None,
        query_expander: QueryExpander | None = None,
        enable_activity_log: bool = False,
        tracer: LocalTracer | None = None,
        retrieval_top_k: int = 20,
        middleware: Iterable[MarkMiddleware] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.plugins = plugins
        self.index = index or VectorIndex()
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self._llm: LLMProvider | None = llm
        self._enable_activity_log = enable_activity_log
        self._global_bus: GlobalMemoryBus | None = None
        self._trust_bus: TrustAwareGlobalMemoryBus | None = None
        self._governance_audit_log = GovernanceAuditLog()
        self._tracer = tracer or LocalTracer()
        self.pipeline = RetrievalPipeline(
            store=store,
            index=self.index,
            embedder=embedder,
            expander=GraphExpander(store),
            plugins=plugins,
            compressor=compressor,
            query_expander=query_expander,
            top_k_with_llm=retrieval_top_k,
        )
        self.middleware = MiddlewareStack(middleware)
        self.middleware.bind(self)

    @classmethod
    def local(
        cls,
        *,
        store_path: str | Path = ":memory:",
        embedder: EmbeddingProvider | None = None,
        encryption: EncryptionProvider | None = None,
        plugins: Iterable[MarkCorePlugin] | None = None,
        workers: int = 4,
        llm: LLMProvider | None = None,
        compressor: ContextualCompressor | None = None,
        query_expander: QueryExpander | None = None,
        enable_activity_log: bool = False,
        tracer: LocalTracer | None = None,
        retrieval_top_k: int = 20,
        middleware: Iterable[MarkMiddleware] | None = None,
    ) -> "MarkRuntime":
        """Construct a local instance with default wiring."""
        registry = PluginRegistry()
        runtime = cls(
            store=LocalMemoryStore(store_path, encryption=encryption or NoOpEncryptionProvider()),
            embedder=embedder or HashEmbeddingProvider(),
            plugins=registry,
            workers=workers,
            llm=llm,
            compressor=compressor,
            query_expander=query_expander,
            enable_activity_log=enable_activity_log,
            tracer=tracer,
            retrieval_top_k=retrieval_top_k,
            middleware=middleware,
        )
        for plugin in plugins or []:
            registry.register(plugin, runtime=runtime)
        return runtime

    def configure_llm(self, llm: LLMProvider) -> None:
        """Bind an LLM provider at runtime (late-binding; overrides constructor value)."""
        self._llm = llm

    def configure_compressor(self, compressor: ContextualCompressor) -> None:
        """Replace the active compressor at runtime (late-binding)."""
        self.pipeline.configure_compressor(compressor)

    def configure_query_expander(self, expander: QueryExpander) -> None:
        """Replace the active query expander at runtime (late-binding)."""
        self.pipeline.configure_query_expander(expander)

    def configure_retrieval_top_k(self, top_k: int) -> None:
        """Set the LLM-aware candidate pool size for the retrieval pipeline."""
        self.pipeline.configure_top_k_with_llm(top_k)

    def add_middleware(self, middleware: MarkMiddleware) -> MarkMiddleware:
        """Register middleware for future memory operations."""
        return self.middleware.add(middleware)

    def use(self, *middleware: MarkMiddleware) -> "MarkRuntime":
        """Register middleware and return this runtime for chaining."""
        self.middleware.extend(middleware)
        return self

    def session_log(self, agent_id: str, session_id: str | None = None) -> "SessionActivityLog":
        """Return a SessionActivityLog scoped to agent_id (and optionally session_id)."""
        from mark.store.activity_log import SessionActivityLog
        return SessionActivityLog(self.store, agent_id, session_id)

    def tracer(self) -> LocalTracer:
        """Return the local runtime tracer."""
        return self._tracer

    def session_trace(self, session_id: str) -> SessionTrace:
        """Return a session-scoped view over local runtime events."""
        return SessionTrace(self._tracer, session_id)

    def governance_audit_log(self) -> GovernanceAuditLog:
        """Return the in-process local governance audit log."""
        return self._governance_audit_log

    def memory(self, agent_id: str) -> MarkMemory:
        """Return (creating if needed) the MarkMemory facade for an agent id."""
        return MarkMemory(
            agent_id=agent_id,
            store=self.store,
            pipeline=self.pipeline,
            embedder=self.embedder,
            executor=self.executor,
            llm=self._llm,
            middleware_stack=self.middleware,
            runtime=self,
        )

    def global_bus(self) -> GlobalMemoryBus:
        """Return the shared global memory bus."""
        if self._global_bus is None:
            self._global_bus = GlobalMemoryBus(
                store=self.store,
                pipeline=self.pipeline,
                embedder=self.embedder,
                executor=self.executor,
            )
        return self._global_bus

    def trust_bus(self) -> TrustAwareGlobalMemoryBus:
        """Return the trust-aware global memory bus."""
        if self._trust_bus is None:
            self._trust_bus = TrustAwareGlobalMemoryBus(
                store=self.store,
                pipeline=self.pipeline,
                embedder=self.embedder,
                executor=self.executor,
            )
        return self._trust_bus

    def working_memory(self, agent_id: str) -> "WorkingMemoryManager":
        """Return a WorkingMemoryManager scoped to agent_id."""
        from mark.memory.working import WorkingMemoryManager
        return WorkingMemoryManager(self.store, agent_id=agent_id)

    def consolidate(self, agent_id: str, *, promote_threshold: float = 0.7) -> "ConsolidationResult":
        """Promote high-importance working memory to LTM; delete expired fragments."""
        from mark.memory.consolidation import ConsolidationManager
        self._tracer.emit("memory.consolidate.start", agent_id=agent_id)
        result = ConsolidationManager(
            self.store,
            agent_id=agent_id,
            audit_log=self._governance_audit_log,
        ).run(
            promote_threshold=promote_threshold
        )
        self._tracer.emit(
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
        self._tracer.emit(
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
        self._tracer.emit(
            "memory.prune",
            agent_id=agent_id,
            fragments_pruned=stats.fragments_pruned,
            edges_dissolved=stats.edges_dissolved,
            importance_updated=stats.importance_updated,
        )
        return stats

    def run_cycle(self, agent_id: str, *, promote_threshold: float = 0.7) -> dict:
        """
        Full hippocampus maintenance cycle for agent_id.

        Runs consolidation (working→LTM promotion + expiry) then pruning
        (decay + edge dissolution). Returns a summary dict.
        """
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

    def integrity_check(self) -> list[str]:
        """Run store integrity check. Returns [] on a healthy store."""
        if hasattr(self.store, "integrity_check"):
            return self.store.integrity_check()
        return []

    def shutdown(self) -> None:
        """Release resources held by the runtime."""
        self.executor.shutdown(wait=True)
        self.store.close()
