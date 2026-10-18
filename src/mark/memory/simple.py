# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""SimpleMemory: ``mark.memory``, the ergonomic surface for scripts and apps.

Everything is stored under one reserved agent (``__mark__``) of the full
MarkRuntime. The canonical verbs (remember / recall / forget / list / get /
update / supersede / resolve / facts / timeline / export ...) come from
:mod:`mark.memory.simple_verbs` and match ``runtime.memory(id)`` exactly;
blocks are tag scopes (see :mod:`mark.memory.simple_block`) and graph helpers
are forwarded by :mod:`mark.memory.simple_graph`. ``retrieve`` (returning a
:class:`ContextBundle`) is the historical read: it keeps working but is
deprecated for application code in favour of ``recall`` (one read verb on
both memories; ``runtime.memory(id).retrieve`` is a different, async call).
``SimpleBlock`` and ``_fragment_to_record`` are re-exported for compatibility.
"""
from __future__ import annotations

import sys
import warnings
from typing import Any

from mark.context.builder import ContextBundle
from mark.memory.graph import GraphNeighborhood  # noqa: F401  (re-export)
from mark.memory.observe import ObserveResult
from mark.memory.record import MemoryRecord, MemoryState  # noqa: F401  (MemoryState re-export)
from mark.memory.simple_block import SimpleBlock, _fragment_to_record
from mark.memory.simple_graph import SimpleGraphMixin
from mark.memory.simple_verbs import SimpleVerbsMixin
from mark.types import MemoryTier
from mark.types.graph import EdgeRelation, NodeType  # noqa: F401  (re-export)

#: Reserved agent id that owns every SimpleMemory write.
_SIMPLE_AGENT = "__mark__"


class SimpleMemory(SimpleVerbsMixin, SimpleGraphMixin):
    """
    Simple memory API — the surface exposed by Mark.local().

    Backed by the full SQLite + retrieval pipeline of MarkRuntime.
    All writes go to a single reserved agent (__mark__) and are tagged
    with their block label so block-scoped retrieval works correctly.

    Simple path:
        mark.memory.remember("We use FastAPI.", kind="decision")
        print(mark.memory.recall("Which framework?").as_context())
        mark.memory.block("project").write("FastAPI is used.", importance=0.9)

    Advanced path (bypass SimpleMemory, use MarkRuntime directly):
        agent = mark.runtime.memory("coding-agent")
        bus   = mark.runtime.global_bus()
    """

    #: Maximum importance change applied by one feedback() call.
    FEEDBACK_STEP = 0.05

    def __init__(self, runtime: Any) -> None:
        from mark.memory.mark_memory import MarkMemory
        self._runtime:     Any       = runtime
        self._mark_memory: MarkMemory = runtime.memory(_SIMPLE_AGENT)
        # Block names created in this process (block()) or seen in the store;
        # retrieve/recall(blocks=[...]) rejects names outside this set.
        self._known_blocks: set[str] = set()

    # ── block API ─────────────────────────────────────────────────────────────

    def block(self, label: str, *, kind: str = "custom",
              description: str = "") -> SimpleBlock:
        """Return (creating if needed) the named block.

        ``kind`` labels the block (reported by ``to_dict()`` and stored as
        ``metadata["block_kind"]`` on its writes); when it names a memory kind
        (``"preference"``, ``"todo"``... see :data:`mark.kinds.KINDS`) every
        write in the block is also stored with that kind.
        """
        self._known_blocks.add(label)
        return SimpleBlock(label=label, mark_memory=self._mark_memory,
                           kind=kind, description=description)

    def list_blocks(self) -> list[SimpleBlock]:
        """Return every block that holds at least one memory (no cap)."""
        labels = {tag[6:] for tag in self._mark_memory._store.tag_values(_SIMPLE_AGENT, "block:")}
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

        Unknown block names raise :class:`mark.errors.UnknownBlockError`.

        .. deprecated:: 0.2.0a10
           Use :meth:`recall` (same filters; returns a RetrievalResult whose
           ``as_context()`` is the LLM-ready text). Calls from application
           code warn; MARK's own agent and skills still use this method.
        """
        if _called_by_application():
            warnings.warn(
                "mark.memory.retrieve() is deprecated; use mark.memory.recall(query, ...).as_context() "
                "(the same verb as runtime.memory(id).recall)", DeprecationWarning, stacklevel=2)
        if blocks:
            self._check_blocks(blocks)

        def _retrieve(tag_filter: list[str] | None) -> Any:
            return self._mark_memory.retrieve_sync(
                query,
                session_id=session_id,
                session_prefix=session_prefix,
                tags=tag_filter,
                tier=tier,
                escalate_on_gap=escalate_on_gap,
                heal_gaps=heal_gaps,
                compress=compress,
                expand=expand,
            )

        if blocks:
            # Block membership is a pre-filter (one scoped retrieval per block,
            # merged by score) so other blocks can't take every top-k slot.
            best: dict[str, tuple[Any, float]] = {}
            for block in dict.fromkeys(blocks):
                scoped = _retrieve([*(tags or []), f"block:{block}"])
                for fragment, score in zip(scoped.fragments, scoped.scores):
                    if fragment.id not in best or score > best[fragment.id][1]:
                        best[fragment.id] = (fragment, score)
            ranked = sorted(best.values(), key=lambda pair: pair[1], reverse=True)
            fragments = [fragment for fragment, _ in ranked]
            scores    = [score for _, score in ranked]
        else:
            result    = _retrieve(tags)
            fragments = list(result.fragments)
            scores    = list(result.scores)

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

        used = len(lines[0])
        for frag, score in zip(fragments, scores):
            # Determine the block label from tags so MemoryRecord.block_id is correct
            block_label = next(
                (t[6:] for t in frag.tags if t.startswith("block:")), "default"
            )
            record = _fragment_to_record(frag, block_label)
            # Content only: source/importance/confidence/score labels cost
            # tokens and confused readers; scores stay in ``bundle.scores``.
            line = f"- {record.content}"
            if used + 1 + len(line) > max_chars:
                if len(lines) == 1:  # never return an empty context for one long memory
                    lines.append(line[: max(0, max_chars - used - 1)])
                break
            used += 1 + len(line)
            lines.append(line)
            memories.append(record)
            score_map[record.id] = score

        text = "\n".join(lines)

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

    def feedback(self, record_id: str, score: float) -> bool:
        """Record quality feedback for a memory record.

        ``score`` is clamped to ``[-1.0, 1.0]``. Positive feedback nudges the
        record's importance up, negative feedback down, by at most
        ``FEEDBACK_STEP`` per call; running totals are kept in metadata
        (``feedback_count`` / ``feedback_sum``). A registered HOOK_FEEDBACK
        plugin also receives the signal. Returns False for unknown records.
        """
        score = max(-1.0, min(1.0, float(score)))
        store = self._runtime.store
        fragment = store.get(record_id)
        if fragment is None:
            return False
        metadata = dict(fragment.metadata)
        metadata["feedback_count"] = int(metadata.get("feedback_count", 0)) + 1
        metadata["feedback_sum"] = round(float(metadata.get("feedback_sum", 0.0)) + score, 6)
        importance = max(0.0, min(1.0, fragment.importance + self.FEEDBACK_STEP * score))
        store.store(fragment.model_copy(update={"metadata": metadata, "importance": importance}))
        from mark.plugins import HOOK_FEEDBACK

        hook = self._runtime.plugins.get(HOOK_FEEDBACK) if hasattr(self._runtime, "plugins") else None
        if hook:
            hook(record_id=record_id, score=score, agent_id=fragment.agent_id)
        return True

    # Graph API (node / attach_node / link_nodes / unlink / neighborhood) is
    # inherited from SimpleGraphMixin and forwards to the underlying MarkMemory.


#: Callers that are not "application code": MARK itself, and the executors
#: MARK's agent uses to run retrieve() off the event loop (asyncio.to_thread).
_INTERNAL_CALLERS = ("mark.", "concurrent.", "asyncio.", "threading", "functools", "contextvars")


def _called_by_application() -> bool:
    """True when ``SimpleMemory.retrieve`` was called from outside MARK (warn only there)."""
    caller = sys._getframe(2).f_globals.get("__name__", "")
    return caller != "mark" and not caller.startswith(_INTERNAL_CALLERS)
