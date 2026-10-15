# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared attribute contract for the MarkMemory mixins.

``MemoryCore`` only declares (annotates) the state that
:meth:`mark.memory.mark_memory.MarkMemory.__init__` sets up, so each mixin
module can be read on its own and type checkers know what ``self`` carries.
It defines no behaviour and assigns no values.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.embeddings import EmbeddingProvider
    from mark.intelligence.retrieval import RetrievalPipeline
    from mark.memory.facade.write_guard import MemoryWriteRejection
    from mark.middlewares.base import MiddlewareStack
    from mark.store.sqlite import LocalMemoryStore
    from mark.types.llm import LLMProvider


class MemoryCore:
    """Annotations for the per-agent state shared by every MarkMemory mixin."""

    agent_id: str
    #: Persistent fragment / node / edge / block storage.
    _store: "LocalMemoryStore"
    #: Embeds, indexes, and retrieves fragments.
    _pipeline: "RetrievalPipeline"
    _embedder: "EmbeddingProvider"
    #: Shared runtime pool; async variants hop onto it so they run the sync path.
    _executor: ThreadPoolExecutor
    #: Optional LLM used for structured extraction in observe().
    _llm: "LLMProvider | None"
    #: Optional write guard (``check(content, confidence=...)``).
    _consolidation_gate: Any
    _last_rejection: "MemoryWriteRejection | None"
    #: Runtime middleware stack; when set (with a runtime) public operations
    #: are routed through it so governance/redaction cannot be bypassed.
    _middleware: "MiddlewareStack | None"
    _runtime: Any
    #: Default FactExtractionService for writes (None: extraction off).
    _fact_extraction: Any
