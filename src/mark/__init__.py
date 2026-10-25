"""Local-first memory for AI agents.

Quickstart::

    from mark import Mark

    mark = Mark.local("./.mark")          # ./.mark/memory.db (as `mark --db .mark`)
    mark.memory.remember("We chose SQLite for storage.", kind="decision")
    print(mark.memory.recall("which database?").as_context())

Every memory object (``mark.memory`` and ``mark.runtime.memory("agent-id")``)
has the same verbs: remember, remember_many, recall, forget, get, list, count,
update, supersede, resolve, facts, timeline, brief, store_document,
remember_file, documents, images, export_jsonl, import_jsonl; plain names are
synchronous and an ``a`` prefix marks the async twin (``arecall``).

The top level exports only the curated names in ``__all__``. Every other name
that used to live here still works but emits a ``DeprecationWarning`` naming
the submodule to import it from. Names load lazily (PEP 562), so
``import mark`` and the ``mark`` command line start in milliseconds.
"""

from mark._version import __version__

# Curated public API: name -> defining module (loaded on first access).
_LAZY_EXPORTS = {
    "Mark": "mark.runtime",
    "MarkMemory": "mark.memory",
    "MarkRuntime": "mark.memory",
    "MarkAgent": "mark.agent",
    "AgentResult": "mark.agent",
    "ContextWatch": "mark.context",
    "MarkChat": "mark.context",
    "Briefing": "mark.context",
    "MarkMiddleware": "mark.middlewares.base",
    "RetrievalResult": "mark.intelligence",
    "RetrievalPolicy": "mark.intelligence",
    "ContextBundle": "mark.context",
    "Timeline": "mark.memory.timeline",
    "Fact": "mark.memory.facts",
    "MemoryFragment": "mark.types",
    "MemoryState": "mark.memory",
    "EmbeddingProvider": "mark.embeddings",
    "HashEmbeddingProvider": "mark.embeddings",
    "embedder_info": "mark.embeddings.info",
    "EmbedderInfo": "mark.embeddings.info",
    "KINDS": "mark.kinds",
    "MarkError": "mark.errors",
    "InvalidInputError": "mark.errors",
    "MemoryNotFoundError": "mark.errors",
    "UnknownBlockError": "mark.errors",
    "InvalidMemoryError": "mark.errors",
    "MemoryTooLargeError": "mark.errors",
    "StoreClosedError": "mark.errors",
    "StoreVersionError": "mark.errors",
}

# Former top-level names, kept importable for compatibility (deprecated here).
_DEPRECATED_EXPORTS = {
    "AgentPersona": "mark.middlewares.skills",
    "AgentRun": "mark.agent",
    "AuditEntry": "mark.middlewares.governance",
    "BaseMiddleware": "mark.middlewares.base",
    "BlockChain": "mark.memory",
    "BlockGraph": "mark.memory",
    "BlockLink": "mark.types",
    "BlockStatus": "mark.types",
    "BlockVerification": "mark.memory",
    "BusMessage": "mark.middlewares.trust_bus",
    "BusSnapshot": "mark.middlewares.trust_bus",
    "BusSubscription": "mark.middlewares.trust_bus",
    "CachedEmbeddingProvider": "mark.embeddings",
    "ChainVerification": "mark.memory",
    "CharacterMemory": "mark.middlewares.media_continuity",
    "CloudSync": "mark.middlewares.sync",
    "ClusterResult": "mark.memory",
    "CompressionMiddleware": "mark.middlewares.compression",
    "ConsolidationGate": "mark.middlewares.governance",
    "ConsolidationGateResult": "mark.middlewares.governance",
    "ContentSanitizer": "mark.middlewares.governance",
    "ContextBudget": "mark.context",
    "ContextBuilder": "mark.context",
    "ContextualCompressor": "mark.intelligence",
    "ConversationMemory": "mark.memory",
    "DeduplicationConsolidator": "mark.memory",
    "DeduplicationResult": "mark.memory",
    "DeterministicExtractor": "mark.intelligence",
    "DuplicateHashGate": "mark.middlewares.governance",
    "EventReplayer": "mark.middlewares.observability",
    "ExtractedEntity": "mark.intelligence",
    "ExtractedRelation": "mark.intelligence",
    "ExtractionMerger": "mark.intelligence",
    "ExtractionResult": "mark.intelligence",
    "FailureFilter": "mark.middlewares.governance",
    "GapHealingMiddleware": "mark.middlewares.gap_healing",
    "GateResult": "mark.middlewares.governance",
    "GlobalMemoryBus": "mark.memory",
    "GovernanceAuditLog": "mark.middlewares.governance",
    "GovernanceMiddleware": "mark.middlewares.governance",
    "GraphNeighborhood": "mark.memory",
    "HOOK_OBSERVE_EVENT": "mark.plugins",
    "JsonMemoryStore": "mark.store",
    "KeywordQueryExpander": "mark.intelligence",
    "LLMContextualCompressor": "mark.intelligence",
    "LLMProvider": "mark.types.llm",
    "LLMQueryExpander": "mark.intelligence",
    "LLMStructuredExtractor": "mark.intelligence",
    "LifecycleMiddleware": "mark.middlewares.lifecycle",
    "LocalMemoryStore": "mark.store",
    "LocalTracer": "mark.middlewares.observability",
    "LocationMemory": "mark.middlewares.media_continuity",
    "MarkConfig": "mark.runtime",
    "MediaContinuityMiddleware": "mark.middlewares.media_continuity",
    "MemoryBlock": "mark.memory",
    "MemoryManager": "mark.memory",
    "MemoryRecord": "mark.memory",
    "MemoryWriteRejection": "mark.memory",
    "MiddlewareContext": "mark.middlewares.base",
    "MiddlewareStack": "mark.middlewares.base",
    "NoopCompressor": "mark.intelligence",
    "NoopQueryExpander": "mark.intelligence",
    "ObjectMemory": "mark.middlewares.media_continuity",
    "ObservabilityMiddleware": "mark.middlewares.observability",
    "ObserveEvent": "mark.memory",
    "ObserveMiddleware": "mark.middlewares.observe",
    "ObserveResult": "mark.memory",
    "PolicyDecision": "mark.policies",
    "PolicyRegistry": "mark.policies",
    "PublisherTrust": "mark.middlewares.trust_bus",
    "QueryExpander": "mark.intelligence",
    "QueryExpansionMiddleware": "mark.middlewares.query_expansion",
    "RecallMiddleware": "mark.middlewares.recall",
    "RetrievalPipeline": "mark.intelligence",
    "RuntimeEvent": "mark.middlewares.observability",
    "RuntimeEventLog": "mark.middlewares.observability",
    "SandboxMiddleware": "mark.middlewares.sandbox",
    "SessionActivityLog": "mark.store",
    "SessionFilter": "mark.intelligence",
    "SessionMemory": "mark.middlewares.media_continuity",
    "SessionTrace": "mark.middlewares.observability",
    "SimpleBlock": "mark.memory",
    "SimpleMemory": "mark.memory",
    "SimpleWindowCompressor": "mark.intelligence",
    "SkillMiddleware": "mark.middlewares.skills",
    "SkillRegistry": "mark.middlewares.skills",
    "StructuredExtraction": "mark.intelligence",
    "SyncDelta": "mark.middlewares.sync",
    "SyncMiddleware": "mark.middlewares.sync",
    "SyncMode": "mark.middlewares.sync",
    "SyncOptions": "mark.middlewares.sync",
    "SyncStats": "mark.middlewares.sync",
    "TrustAwareGlobalMemoryBus": "mark.middlewares.trust_bus",
    "TrustBusMiddleware": "mark.middlewares.trust_bus",
    "ValidationGate": "mark.middlewares.governance",
    "VectorIndex": "mark.index",
    "VectorSearchResult": "mark.index",
    "WorldBibleMemory": "mark.middlewares.media_continuity",
    "agent_skill": "mark.middlewares.skills",
    "list_packaged_skills": "mark.middlewares.skills",
    "load_skill_text": "mark.middlewares.skills",
}

# Subpackages reachable as attributes (``import mark; mark.memory.MarkMemory``),
# exactly as when they were imported eagerly.
_SUBPACKAGES = {
    "adapters", "agent", "cli", "context", "embeddings", "errors", "index", "intelligence", "kinds",
    "memory", "middlewares", "plasticity", "plugins", "policies", "rl", "runtime", "security", "store",
    "types",
}


def __getattr__(name: str):
    from importlib import import_module

    if name in _SUBPACKAGES:
        module = import_module(f"mark.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if name in _DEPRECATED_EXPORTS:
        import warnings

        source = _DEPRECATED_EXPORTS[name]
        warnings.warn(
            f"`mark.{name}` is no longer a top-level export; import it from `{source}` "
            f"(from {source} import {name})",
            DeprecationWarning, stacklevel=2,
        )
        # Not cached, so every new importer sees the warning once per call site.
        return getattr(import_module(source), name)
    raise AttributeError(f"module 'mark' has no attribute {name!r}")


__all__ = [
    "Mark",
    "MarkMemory",
    "MarkRuntime",
    "MarkAgent",
    "AgentResult",
    "ContextWatch",
    "MarkChat",
    "Briefing",
    "MarkMiddleware",
    "RetrievalResult",
    "RetrievalPolicy",
    "ContextBundle",
    "Timeline",
    "Fact",
    "MemoryFragment",
    "MemoryState",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "embedder_info",
    "EmbedderInfo",
    "KINDS",
    "MarkError",
    "InvalidInputError",
    "MemoryNotFoundError",
    "UnknownBlockError",
    "InvalidMemoryError",
    "MemoryTooLargeError",
    "StoreClosedError",
    "StoreVersionError",
    "__version__",
]


def __dir__() -> list[str]:
    return sorted(set(__all__) | _SUBPACKAGES)
