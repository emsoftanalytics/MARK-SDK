"""MarkRuntime: wiring and lifecycle for the local memory engine."""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mark.embeddings import EmbeddingProvider, resolve_embedder
from mark.index import VectorIndex
from mark.intelligence.retrieval import GraphExpander, RetrievalPipeline
from mark.intelligence.compressor import ContextualCompressor
from mark.intelligence.fact_extraction import FactExtractionService, resolve_extractor
from mark.intelligence.query_expander import QueryExpander
from mark.middlewares.base import MarkMiddleware, MiddlewareStack
from mark.plugins import MarkCorePlugin, PluginRegistry
from mark.security import EncryptionProvider, NoOpEncryptionProvider
from mark.store.sqlite import LocalMemoryStore
from mark.types.llm import LLMProvider

from mark.memory.global_bus import GlobalMemoryBus
from mark.memory.mark_memory import MarkMemory
from mark.memory.runtime_maintenance import MaintenanceMixin


class MarkRuntime(MaintenanceMixin):
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
        tracer: object | None = None,
        retrieval_top_k: int = 20,
        middleware: Iterable[MarkMiddleware] | None = None,
        fact_extractor: object = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.plugins = plugins
        # Explicit None check: an empty VectorIndex has len() == 0, so
        # `index or VectorIndex()` would silently swap in a private index.
        self.index = index if index is not None else VectorIndex()
        # Keep the vector index consistent with the store: deleted or
        # no-longer-retrievable fragments must stop surfacing from search.
        if hasattr(store, "add_listener"):
            store.add_listener(self._on_store_event)
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self._llm: LLMProvider | None = llm
        self._enable_activity_log = enable_activity_log
        self._global_bus: GlobalMemoryBus | None = None
        self._trust_bus: object | None = None
        self._governance_audit_log: object | None = None
        self._tracer = tracer
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
        # Default fact extractor for writes (None = off; background ones get a durable queue).
        extractor = resolve_extractor(fact_extractor)
        self.fact_extraction = FactExtractionService(store, extractor) if extractor is not None else None

    def _on_store_event(self, event: str, fragment_id: str) -> None:
        """Store listener: drop deleted/unretrievable fragments from the index."""
        if event in {"delete", "unretrievable"}:
            self.index.remove(fragment_id)

    @classmethod
    def local(
        cls,
        *,
        store_path: str | Path = ":memory:",
        embedder: EmbeddingProvider | str | None = None,
        encryption: EncryptionProvider | None = None,
        plugins: Iterable[MarkCorePlugin] | None = None,
        workers: int = 4,
        llm: LLMProvider | None = None,
        compressor: ContextualCompressor | None = None,
        query_expander: QueryExpander | None = None,
        enable_activity_log: bool = False,
        tracer: object | None = None,
        retrieval_top_k: int = 20,
        middleware: Iterable[MarkMiddleware] | None = None,
        fact_extractor: object = None,
    ) -> "MarkRuntime":
        """Construct a local instance.

        ``embedder``: a provider, a spec such as ``"auto"`` (see ``mark.embeddings.auto``)
        or None (hash, or ``MARK_EMBEDDER``). ``fact_extractor``: None (off),
        ``"patterns"`` or a FactExtractor.
        """
        registry = PluginRegistry()
        runtime = cls(
            store=LocalMemoryStore(store_path, encryption=encryption or NoOpEncryptionProvider()),
            embedder=resolve_embedder(embedder),
            plugins=registry,
            workers=workers,
            llm=llm,
            compressor=compressor,
            query_expander=query_expander,
            enable_activity_log=enable_activity_log,
            tracer=tracer,
            retrieval_top_k=retrieval_top_k,
            middleware=middleware,
            fact_extractor=fact_extractor,
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

    def tracer(self):
        """Return the local runtime tracer."""
        if self._tracer is None:
            from mark.middlewares.observability import LocalTracer
            self._tracer = LocalTracer()
        return self._tracer

    def session_trace(self, session_id: str):
        """Return a session-scoped view over local runtime events."""
        from mark.middlewares.observability import SessionTrace

        return SessionTrace(self.tracer(), session_id)

    def governance_audit_log(self):
        """Return the in-process local governance audit log."""
        if self._governance_audit_log is None:
            from mark.middlewares.governance import GovernanceAuditLog

            self._governance_audit_log = GovernanceAuditLog()
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
            fact_extraction=self.fact_extraction,
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

    def trust_bus(self):
        """Return the trust-aware global memory bus."""
        if self._trust_bus is None:
            from mark.middlewares.trust_bus import TrustAwareGlobalMemoryBus

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

    def integrity_check(self) -> list[str]:
        """Run store integrity check. Returns [] on a healthy store."""
        if hasattr(self.store, "integrity_check"):
            return self.store.integrity_check()
        return []

    def shutdown(self) -> None:
        """Release resources held by the runtime."""
        if self.fact_extraction is not None:
            self.fact_extraction.close()  # queued jobs stay in the store
        self.executor.shutdown(wait=True)
        self.store.close()
