"""MarkRuntime: wiring and lifecycle for the local memory engine.

Lifecycle: :meth:`MarkRuntime.shutdown` is bounded and final. It runs the
registered closers (``register_closer``), stops the background fact worker
(unfinished jobs go back to the durable queue), drains the executor, retires
the adapter offload pool and closes the store (then ``StoreClosedError``).
"""
from __future__ import annotations

import sys

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mark.embeddings import EmbeddingProvider
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

from mark.memory.facade.write_validation import DEFAULT_MAX_MEMORY_BYTES
from mark.memory.global_bus import GlobalMemoryBus
from mark.memory.mark_memory import MarkMemory
from mark.memory.runtime_closers import ClosersMixin
from mark.runtime_embedder import open_store, quiet_unavailable
from mark.memory.runtime_maintenance import MaintenanceMixin


class MarkRuntime(ClosersMixin, MaintenanceMixin):
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
        fact_job_lease: float = 300.0,
        max_memory_bytes: int | None = DEFAULT_MAX_MEMORY_BYTES,
        redact: object = True,
        injection: str = "tag",
    ) -> None:
        self.store = store
        self._closed = False
        # Largest single memory (UTF-8 bytes) a write accepts; None = no limit.
        self.max_memory_bytes = max_memory_bytes
        # Secret redaction + prompt-injection screening applied to every write.
        from mark.security import resolve_write_policy

        self.write_policy = resolve_write_policy(redact, injection)  # type: ignore[arg-type]
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
        self.fact_extraction = (
            FactExtractionService(store, extractor, lease_seconds=fact_job_lease) if extractor is not None else None
        )

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
        fact_job_lease: float = 300.0,
        max_memory_bytes: int | None = DEFAULT_MAX_MEMORY_BYTES,
        redact: object = True,
        injection: str = "tag",
    ) -> "MarkRuntime":
        """Construct a local instance.

        ``embedder``: a provider, a spec such as ``"auto"`` (see ``mark.embeddings.auto``)
        or None: the store's own embedder (``mark.runtime_embedder``), else hash
        or ``MARK_EMBEDDER``; an explicit other one re-embeds. ``fact_extractor``: None (off),
        ``"patterns"`` or a FactExtractor. ``fact_job_lease``: seconds a
        background extraction job may run before another worker may take it
        over (jobs of a crashed process on this host are reclaimed at once).
        ``max_memory_bytes``: largest single memory a write accepts (default
        1 MB; None = no limit); longer text belongs in ``store_document``.
        ``redact``: True (default secret redaction), False, a RedactionPolicy or
        a WritePolicy. ``injection``: "tag" (default), "quarantine" or "off".
        """
        registry = PluginRegistry()
        store, provider = open_store(store_path, encryption or NoOpEncryptionProvider(), embedder)
        runtime = cls(
            store=store,
            embedder=provider,
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
            fact_job_lease=fact_job_lease,
            max_memory_bytes=max_memory_bytes,
            redact=redact,
            injection=injection,
        )
        for plugin in plugins or []:
            registry.register(plugin, runtime=runtime)
        # Direct users (scripts, the LangChain backend) get the user subject saved
        # with the store (``mark config``), like Mark.local and the CLI do.
        from mark.runtime_config import apply_user_subject

        apply_user_subject(runtime)
        return quiet_unavailable(runtime)  # the stored-embedder warning was already given

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

    def configure_user_subject(self, subject: str) -> None:
        """Name the user: the subject of "I/my" facts and of "my ..." questions (default "user")."""
        name = str(subject or "").strip()
        if not name:
            raise ValueError("user subject must be a non-empty name")
        self.pipeline.configure_user_subject(name)
        extractor = getattr(self.fact_extraction, "extractor", None)
        if hasattr(extractor, "user_subject"):
            extractor.user_subject = name  # pattern extractor: first-person facts

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

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop background work and close the store; bounded by ``timeout`` seconds.

        The fact worker finishes its current job if it can within
        ``timeout``; otherwise that job (and any it claimed but had not
        started) is returned to the durable queue for the next run. The store
        is closed for good: later calls raise ``StoreClosedError``.
        Idempotent.
        """
        if self._closed:
            return
        self._run_closers()  # background readers (ContextWatch workers) stop before the store closes
        self._closed = True
        if self.fact_extraction is not None:
            self.fact_extraction.close(timeout)  # queued jobs stay in the store
        self.executor.shutdown(wait=True)
        backend = sys.modules.get("mark.adapters.backend")
        if backend is not None:  # only if adapters were used; never import them here
            backend.shutdown_offload_pool()
        self.store.close()
