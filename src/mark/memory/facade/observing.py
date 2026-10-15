# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""observe() for MarkMemory: store an observation, optionally structure it.

Plain observation stores the text as a fragment and emits an
``HOOK_OBSERVE_EVENT``. With ``auto_structure=True`` the text is also run
through deterministic extraction (plus LLM extraction when an LLM is bound)
and the resulting entities/relations are written as graph nodes and edges.
Extraction and hook failures never block the write itself.
"""
from __future__ import annotations

from typing import Any

from mark.intelligence.extractor import DeterministicExtractor, ExtractionMerger, LLMStructuredExtractor
from mark.memory.facade.graph_ops import GraphMixin
from mark.memory.facade.write import WriteMixin
from mark.memory.observe import ObserveEvent, ObserveResult
from mark.plugins import HOOK_OBSERVE_EVENT
from mark.types import MemoryEdge, MemoryNode, MemoryState


class ObserveMixin(GraphMixin, WriteMixin):
    """Observation capture and optional graph structuring."""

    def observe(
        self,
        text: str,
        *,
        session_id:  str | None = None,
        importance:  float = 0.5,
        tags:        list[str] | None = None,
        source:      str | None = None,
        metadata:    dict[str, Any] | None = None,
        memory_type: str | None = None,
        auto_structure: bool = False,
    ) -> ObserveResult:
        """Store an observation.

        Bare MARK stores the text as a fragment and emits an observe event.
        Automatic structuring is a middleware battery: pass
        ``auto_structure=True`` through middleware to create tags, nodes, and
        edges from deterministic or developer-provided LLM extraction.

        memory_type — override the context node type (default: "scene").
                      Use "episode", "session", "shot", or any NodeType value.
                      Applies to the context entity derived from session_id.

        Falls back gracefully — extraction failures never block the write.
        Returns ObserveResult with fragment_id and, when enabled, nodes, edges,
        and auto_tags.
        """
        if self._middleware is not None and self._runtime is not None:
            return self._middleware.run(
                "observe",
                runtime=self._runtime,
                agent_id=self.agent_id,
                payload={
                    "text": text,
                    "session_id": session_id,
                    "importance": importance,
                    "tags": tags,
                    "source": source,
                    "metadata": metadata,
                    "memory_type": memory_type,
                    "auto_structure": auto_structure,
                },
                handler=lambda ctx: self._observe_impl(**ctx.payload),
            )
        return self._observe_impl(
            text,
            session_id=session_id,
            importance=importance,
            tags=tags,
            source=source,
            metadata=metadata,
            memory_type=memory_type,
            auto_structure=auto_structure,
        )

    def _emit_observe_event(self, event: ObserveEvent) -> None:
        """Send an event to the HOOK_OBSERVE_EVENT plugin, if one is registered.

        Observability must never break a write, so hook errors are swallowed.
        """
        hook = self._pipeline._plugins.get(HOOK_OBSERVE_EVENT) if hasattr(self._pipeline, "_plugins") else None
        if hook is not None:
            try:
                hook(event)
            except Exception:
                pass

    def _observe_impl(
        self,
        text: str,
        *,
        session_id:  str | None = None,
        importance:  float = 0.5,
        tags:        list[str] | None = None,
        source:      str | None = None,
        metadata:    dict[str, Any] | None = None,
        memory_type: str | None = None,
        auto_structure: bool = False,
        state: MemoryState = MemoryState.UNVERIFIED,
        confidence: float = 1.0,
    ) -> ObserveResult:
        """Middleware-free observe; ``state``/``confidence`` may be set by middleware."""
        # Governance middleware marks rejected observations QUARANTINED: store
        # them for audit only — never indexed, never structured into the graph.
        if state == MemoryState.QUARANTINED or not auto_structure:
            fragment_id = self._store_sync_impl(
                text,
                importance=importance,
                tags=tags,
                source=source or "observe",
                state=state,
                confidence=confidence,
                session_id=session_id,
                metadata=metadata,
            )
            self._emit_observe_event(ObserveEvent(
                agent_id=self.agent_id,
                fragment_id=fragment_id,
                content=text,
                session_id=session_id,
                node_labels=[],
                auto_tags=[],
                edge_count=0,
                inferred=False,
            ))
            return ObserveResult(
                fragment_id=fragment_id,
                content=text,
                session_id=session_id,
                nodes=[],
                edges=[],
                auto_tags=[],
                inferred=False,
            )

        # 1. Deterministic extraction (always runs, no LLM required)
        det_result = DeterministicExtractor().extract(
            text, session_id=session_id, memory_type=memory_type,
        )

        # 2. Optional LLM structured extraction — merges additively with deterministic
        llm_found = False
        if self._llm is not None:
            llm_struct = LLMStructuredExtractor(self._llm).extract_structured(text)
            merged     = ExtractionMerger().merge(det_result, llm_struct)
            llm_found  = bool(llm_struct.entities) or bool(llm_struct.relations)
        else:
            merged = det_result

        # 3. Store fragment — all merged tags (deterministic + LLM) applied at write time
        auto_tags   = merged.tags
        all_tags    = [*(tags or []), *auto_tags]
        fragment_id = self._store_sync_impl(
            text,
            importance = importance,
            tags       = all_tags,
            source     = source or "observe",
            state      = state,
            confidence = confidence,
            session_id = session_id,
            metadata   = metadata,
        )

        nodes: list[MemoryNode] = []
        edges: list[MemoryEdge] = []
        seen_labels: set[str]   = set()

        # Graph writes are best effort: one bad entity/relation must not
        # abort the rest. Labels are de-duplicated case-insensitively.
        def _safe_node(type_str: str, label: str) -> MemoryNode | None:
            if label.lower() in seen_labels:
                return None
            try:
                n = self.node(label, type_str, fragment_id=fragment_id)
                seen_labels.add(label.lower())
                nodes.append(n)
                return n
            except Exception:
                return None

        def _safe_edge(a: str, rel: str, b: str) -> None:
            try:
                edges.append(self.link_nodes(a, b, rel))
            except Exception:
                pass

        # 4. Create nodes and edges from the merged extraction result
        for type_str, label in merged.entities:
            _safe_node(type_str, label)
        for a, rel, b in merged.relations:
            _safe_edge(a, rel, b)

        inferred = bool(merged.entities) or bool(merged.relations) or llm_found

        # 5. Emit observability event
        self._emit_observe_event(ObserveEvent(
            agent_id    = self.agent_id,
            fragment_id = fragment_id,
            content     = text,
            session_id  = session_id,
            node_labels = [n.label for n in nodes],
            auto_tags   = auto_tags,
            edge_count  = len(edges),
            inferred    = inferred,
        ))

        return ObserveResult(
            fragment_id = fragment_id,
            content     = text,
            session_id  = session_id,
            nodes       = nodes,
            edges       = edges,
            auto_tags   = auto_tags,
            inferred    = inferred,
        )
