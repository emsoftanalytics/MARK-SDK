"""Per-agent memory facade: store, retrieve, observe, graph, and blocks.

:class:`MarkMemory` is what ``MarkRuntime.memory(agent_id)`` returns. Its
behaviour is composed from focused mixins in :mod:`mark.memory.facade`:

==========================  ==============================================
mixin                       responsibility
==========================  ==============================================
``WriteMixin``              store / store_sync / store_many_sync, facts
``GuardMixin``              consolidation gate, quarantine, last_rejection
``RetrieveMixin``           retrieve / retrieve_sync / retrieve_many_sync
``HealingMixin``            gap healing via an explicit web lookup
``TimelineMixin``           timeline(): a subject's whole ordered history
``ObserveMixin``            observe() and optional graph structuring
``GraphMixin``              node / attach_node / link_nodes / neighborhood
``LifecycleMixin``          list / forget / promote / clear
``ViewsMixin``              character / session / blocks / chain helpers
==========================  ==============================================

This module remains the public home of ``MarkMemory`` and re-exports the
helper names that historically lived here.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mark.embeddings import EmbeddingProvider
from mark.intelligence.retrieval import RetrievalPipeline
from mark.memory.facade.graph_ops import _NODE_TYPE_ALIASES, GraphMixin, _resolve_node_type  # noqa: F401
from mark.memory.facade.lifecycle import LifecycleMixin
from mark.memory.facade.observing import ObserveMixin
from mark.memory.facade.retrieve import RetrieveMixin
from mark.memory.facade.timeline import TimelineMixin
from mark.memory.facade.views import ViewsMixin
from mark.memory.facade.write import WriteMixin
from mark.memory.facade.write_guard import GuardMixin, MemoryWriteRejection, _with_occurred_at  # noqa: F401
from mark.middlewares.base import MiddlewareStack
from mark.store.sqlite import LocalMemoryStore
from mark.types.llm import LLMProvider

# Backward-compatible re-exports: these names used to be imported into (and
# were therefore importable from) this module.
from mark.intelligence.extractor import (  # noqa: F401
    DeterministicExtractor,
    ExtractionMerger,
    LLMStructuredExtractor,
)
from mark.intelligence.retrieval import RetrievalPolicy, RetrievalResult, SessionFilter  # noqa: F401
from mark.memory.graph import GraphNeighborhood  # noqa: F401
from mark.memory.observe import ObserveEvent, ObserveResult  # noqa: F401
from mark.middlewares.skills.base import Skill  # noqa: F401
from mark.plugins import HOOK_OBSERVE_EVENT  # noqa: F401
from mark.types import MemoryEdge, MemoryFragment, MemoryNode, MemoryScope, MemoryState, MemoryTier  # noqa: F401
from mark.types.graph import EdgeRelation, NodeType  # noqa: F401

logger = logging.getLogger(__name__)


class MarkMemory(
    RetrieveMixin,
    TimelineMixin,
    ObserveMixin,
    LifecycleMixin,
    ViewsMixin,
    GraphMixin,
    WriteMixin,
    GuardMixin,
):
    """Per-agent memory facade backed by local SQLite, embeddings, and retrieval.

    Public operations (store, retrieve, observe) are routed through the
    runtime middleware stack when one is attached, so governance, redaction,
    and observability apply uniformly to sync, async, and batch callers.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        store: LocalMemoryStore,
        pipeline: RetrievalPipeline,
        embedder: EmbeddingProvider,
        executor: ThreadPoolExecutor,
        llm: LLMProvider | None = None,
        consolidation_gate: Any = None,
        middleware_stack: MiddlewareStack | None = None,
        runtime: Any = None,
    ) -> None:
        self.agent_id = agent_id
        self._store = store
        self._pipeline = pipeline
        self._embedder = embedder
        self._executor = executor
        self._llm: LLMProvider | None = llm
        self._consolidation_gate = consolidation_gate
        self._last_rejection: MemoryWriteRejection | None = None
        # Middleware routing needs both the stack and the runtime it is bound to.
        self._middleware = middleware_stack
        self._runtime = runtime
