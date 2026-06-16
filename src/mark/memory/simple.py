# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""SimpleMemory: the ergonomic block/write/retrieve surface for scripts."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from mark.context.builder import ContextBundle
from mark.memory.graph import GraphNeighborhood
from mark.memory.observe import ObserveResult
from mark.memory.record import MemoryRecord, MemoryState
from mark.types import MemoryTier
from mark.types.graph import EdgeRelation, NodeType

_SIMPLE_AGENT = "__mark__"


def _fragment_to_record(frag: Any, block_id: str) -> MemoryRecord:
    """Convert a MemoryFragment to a MemoryRecord for backward-compat API surface."""
    return MemoryRecord(
        id         = frag.id,
        block_id   = block_id,
        content    = frag.content,
        importance = frag.importance,
        confidence = frag.confidence,
        state      = MemoryState.RAW,
        source     = frag.source or "local",
        metadata   = dict(frag.metadata),
        created_at = frag.created_at,
    )


class SimpleBlock:
    """
    A named memory container in the simple API.

    Writes are stored as MemoryFragments tagged with the block label,
    so they can be scoped during retrieval with blocks=[...].
    """

    def __init__(self, label: str, mark_memory: Any) -> None:
        self._label  = label
        self._memory = mark_memory

    @property
    def label(self) -> str:
        """Return the display label."""
        return self._label

    @property
    def id(self) -> str:
        """Block identifier — equals the label in the simple API."""
        return self._label

    def write(
        self,
        content:    str,
        *,
        importance: float = 0.5,
        confidence: float = 0.5,
        metadata:   dict[str, Any] | None = None,
        source:     str = "local",
        state:      MemoryState = MemoryState.RAW,
    ) -> MemoryRecord:
        """Write content to this block. Returns a MemoryRecord."""
        frag_id = self._memory.store_sync(
            content,
            importance = importance,
            confidence = confidence,
            tags       = [f"block:{self._label}"],
            source     = source,
            metadata   = metadata or {},
        )
        # Fetch the stored fragment so created_at matches what is persisted
        stored = self._memory._store.get(frag_id)
        if stored is not None:
            return _fragment_to_record(stored, self._label)

        # Fallback — should not happen, but never raise on write
        from datetime import datetime, timezone
        return MemoryRecord(
            id         = frag_id,
            block_id   = self._label,
            content    = content,
            importance = importance,
            confidence = confidence,
            state      = state,
            source     = source,
            metadata   = metadata or {},
            created_at = datetime.now(timezone.utc),
        )

    def records(self) -> list[MemoryRecord]:
        """Return all records stored in this block."""
        tag = f"block:{self._label}"
        return [
            _fragment_to_record(f, self._label)
            for f in self._memory.list()
            if tag in f.tags
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all blocks and records to a plain dictionary."""
        return {"id": self._label, "label": self._label, "kind": "simple"}


class SimpleMemory:
    """
    Simple memory API — the surface exposed by Mark.local().

    Backed by the full SQLite + retrieval pipeline of MarkRuntime.
    All writes go to a single reserved agent (__mark__) and are tagged
    with their block label so block-scoped retrieval works correctly.

    Simple path:
        mark.memory.block("project").write("FastAPI is used.", importance=0.9)
        bundle = mark.memory.retrieve("Which framework?")
        print(bundle.as_text())

    Advanced path (bypass SimpleMemory, use MarkRuntime directly):
        agent = mark.runtime.memory("coding-agent")
        bus   = mark.runtime.global_bus()
    """

    def __init__(self, runtime: Any) -> None:
        from mark.memory.mark_memory import MarkMemory
        self._runtime:     Any       = runtime
        self._mark_memory: MarkMemory = runtime.memory(_SIMPLE_AGENT)

    # ── block API ─────────────────────────────────────────────────────────────

    def block(self, label: str, *, kind: str = "custom",
              description: str = "") -> SimpleBlock:
        """Return (creating if needed) the named block."""
        return SimpleBlock(label=label, mark_memory=self._mark_memory)

    def list_blocks(self) -> list[SimpleBlock]:
        """Return unique blocks discovered from stored fragment tags."""
        labels: set[str] = set()
        for f in self._mark_memory.list():
            for tag in f.tags:
                if tag.startswith("block:"):
                    labels.add(tag[6:])
        return [SimpleBlock(label=lb, mark_memory=self._mark_memory)
                for lb in sorted(labels)]

    # ── retrieval API ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:          str,
        *,
        blocks:         list[str] | None = None,
        top_k:          int = 5,
        max_chars:      int = 6000,
        session_id:     str | None = None,
        session_prefix: str | None = None,
        tags:           list[str] | None = None,
        tier:           MemoryTier | None = None,
        escalate_on_gap: bool = False,
        heal_gaps:      bool = False,
        compress:       bool = False,
        expand:         bool = False,
    ) -> ContextBundle:
        """Retrieve relevant memory and return a ContextBundle.

        blocks          — restrict to named memory blocks (OR semantics)
        session_id      — restrict to an exact session  (e.g. "episode-01")
        session_prefix  — restrict to a session prefix  (e.g. "season-01/")
        tags            — all tags must be present (AND semantics)
        tier            — restrict to a MemoryTier (WORKING / EPISODIC / ...)
        escalate_on_gap — on a session miss, retry the whole agent namespace
        heal_gaps       — after session and agent misses, run explicit basic web lookup
        compress        — apply contextual compressor (requires compressor configured)
        expand          — apply query expansion for improved recall
        """
        result    = self._mark_memory.retrieve_sync(
            query,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            tier=tier,
            escalate_on_gap=escalate_on_gap,
            heal_gaps=heal_gaps,
            compress=compress,
            expand=expand,
        )
        fragments = list(result.fragments)
        scores    = list(result.scores)

        # Filter to requested blocks if specified
        if blocks:
            allowed_tags = {f"block:{b}" for b in blocks}
            pairs = [(f, s) for f, s in zip(fragments, scores)
                     if any(t in allowed_tags for t in f.tags)]
            fragments, scores = (list(x) for x in zip(*pairs)) if pairs else ([], [])

        fragments = fragments[:top_k]
        scores    = scores[:top_k]

        if not fragments:
            text = "MARK context: no relevant local memory found."
            return ContextBundle(
                query          = query,
                memories       = [],
                text           = text,
                token_estimate = max(1, len(text) // 4),
            )

        lines:    list[str]              = ["MARK context:"]
        memories: list[MemoryRecord]     = []
        score_map: dict[str, float]      = {}

        for frag, score in zip(fragments, scores):
            # Determine the block label from tags so MemoryRecord.block_id is correct
            block_label = next(
                (t[6:] for t in frag.tags if t.startswith("block:")), "default"
            )
            record = _fragment_to_record(frag, block_label)
            line = (
                f"- [{record.source}] {record.content} "
                f"(importance={record.importance:.2f}, "
                f"confidence={record.confidence:.2f}, score={score:.3f})"
            )
            lines.append(line)
            memories.append(record)
            score_map[record.id] = score

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars]

        return ContextBundle(
            query          = query,
            memories       = memories,
            text           = text,
            token_estimate = max(1, len(text) // 4),
            scores         = score_map,
        )

    # ── observe API ───────────────────────────────────────────────────────────

    def observe(
        self,
        text:       str,
        *,
        session_id: str | None = None,
        importance: float = 0.5,
        tags:       list[str] | None = None,
        source:     str | None = None,
        metadata:   dict[str, Any] | None = None,
    ) -> ObserveResult:
        """Store an observation and optionally infer entities and relationships.

        MARK runs deterministic local extraction by default. If an LLM is
        configured on this Mark instance (via Mark.local(llm=...) or
        mark.configure_llm(...)), MARK enriches the deterministic pass with
        structured character, object, location, and relation extraction and
        wires the result into the memory graph.

        Example::

            result = mark.memory.observe(
                "Elena enters the North Warehouse wearing the red scarf.",
                session_id="season-01/ep-02/scene-04",
            )
            # result.nodes — extracted entity nodes
            # result.edges — inferred graph edges
            # result.inferred — True if LLM extraction ran
        """
        return self._mark_memory.observe(
            text,
            session_id = session_id,
            importance = importance,
            tags       = [f"block:observe", *(tags or [])],
            source     = source or "observe",
            metadata   = metadata,
        )

    # ── feedback API ──────────────────────────────────────────────────────────

    def feedback(self, record_id: str, score: float) -> None:
        """Signal quality feedback. Register HOOK_FEEDBACK for custom collection."""
        pass

    # ── graph API (delegates to the underlying MarkMemory) ───────────────────

    def node(
        self,
        label: str,
        node_type: "NodeType | str" = NodeType.CONCEPT,
        *,
        fragment_id: str | None = None,
        weight: float = 1.0,
    ) -> "Any":
        """Get or create a named concept node in the simple agent's graph."""
        return self._mark_memory.node(label, node_type,
                                      fragment_id=fragment_id, weight=weight)

    def attach_node(
        self,
        fragment_id: str,
        label: str,
        node_type: "NodeType | str" = NodeType.ENTITY,
        *,
        weight: float = 1.0,
    ) -> "Any":
        """Associate a fragment with a named concept node."""
        return self._mark_memory.attach_node(fragment_id, label, node_type,
                                             weight=weight)

    def link_nodes(
        self,
        label_a: str,
        label_b: str,
        relation: "EdgeRelation | str",
        *,
        weight: float = 1.0,
    ) -> "Any":
        """Create a directed edge between two named nodes."""
        return self._mark_memory.link_nodes(label_a, label_b, relation, weight=weight)

    def unlink(
        self,
        label_a: str,
        label_b: str,
        relation: "EdgeRelation | str | None" = None,
    ) -> int:
        """Remove directed edges from label_a → label_b. Returns count removed."""
        return self._mark_memory.unlink(label_a, label_b, relation)

    def neighborhood(self, label: str, *, depth: int = 2) -> GraphNeighborhood:
        """Return the graph neighbourhood around a named concept node."""
        return self._mark_memory.neighborhood(label, depth=depth)
