"""LangChain tool helpers for MARK."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mark.adapters.backend import MarkBackend


_SYNC_EXECUTOR: ThreadPoolExecutor | None = None
_SYNC_EXECUTOR_LOCK = threading.Lock()


def _require_langchain_tool():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "LangChain adapters require optional dependencies. "
            'Install with: pip install "mark-sdk[langchain]"'
        ) from exc
    return StructuredTool


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        with _SYNC_EXECUTOR_LOCK:
            if _SYNC_EXECUTOR is None:
                _SYNC_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mark-tools-sync-bridge"
                )
    return _SYNC_EXECUTOR.submit(asyncio.run, coro).result()


def create_mark_tools(
    backend: MarkBackend,
    *,
    default_agent_id: str,
    default_blocks: list[str] | None = None,
    compress: bool = False,
) -> list[Any]:
    """Create basic LangChain tools backed by MARK.

    compress — opt-in LLM contextual compression for mark_retrieve results.
    Off by default: compression is lossy and requires a configured LLM.
    """

    structured_tool = _require_langchain_tool()

    async def amark_retrieve(query: str) -> str:
        """Call this as your VERY FIRST action before planning or writing any code.

        MARK stores project memory: coding conventions, API patterns, architectural
        decisions, and past task outcomes. The returned context contains requirements
        you MUST follow. Always call before starting work.

        Args:
            query: Phrase describing your current task or topic, e.g.
                   "FastAPI endpoint conventions and project structure".
        """
        result = await backend.retrieve(
            query,
            agent_id=default_agent_id,
            blocks=default_blocks,
            compress=compress,
        )
        if not result.ok:
            return f"MARK retrieve failed: {result.error}"
        value = result.value
        if hasattr(value, "as_context"):
            return value.as_context()
        if hasattr(value, "as_text"):
            return value.as_text()
        return str(value)

    def mark_retrieve(query: str) -> str:
        """Call this as your VERY FIRST action before planning or writing any code.

        MARK stores project memory: coding conventions, API patterns, architectural
        decisions, and past task outcomes. The returned context contains requirements
        you MUST follow. Always call before starting work.

        Args:
            query: Phrase describing your current task or topic, e.g.
                   "FastAPI endpoint conventions and project structure".
        """
        return _run_sync(amark_retrieve(query))

    async def amark_write(content: str, canonical: bool = False) -> str:
        """Write a durable, reusable memory fragment to MARK.

        Use for recording conventions, patterns, or facts that should always
        be remembered across future agent runs.  Unlike mark_observe (episodic
        events), mark_write stores persistent reference knowledge.

        Args:
            content:   The fact, pattern, or convention to store permanently.
            canonical: Set True for authoritative reference material (character
                       sheets, project rules, world canon). Canonical memory is
                       stored at maximum importance so retrieval always ranks
                       it above ordinary observations.
        """
        # Reference knowledge must outrank episodic notes in the reranker
        # (importance carries 25% of the score) — store it accordingly.
        options: dict[str, Any] = {"importance": 0.95 if canonical else 0.8}
        if canonical:
            options["tags"] = ["canonical"]
        result = await backend.write(content, agent_id=default_agent_id, **options)
        if not result.ok:
            return f"MARK write failed: {result.error}"
        return "MARK canonical memory written." if canonical else "MARK memory written."

    def mark_write(content: str, canonical: bool = False) -> str:
        """Write a durable, reusable memory fragment to MARK.

        Use for recording conventions, patterns, or facts that should always
        be remembered across future agent runs.  Unlike mark_observe (episodic
        events), mark_write stores persistent reference knowledge.

        Args:
            content:   The fact, pattern, or convention to store permanently.
            canonical: Set True for authoritative reference material (character
                       sheets, project rules, world canon). Canonical memory is
                       stored at maximum importance so retrieval always ranks
                       it above ordinary observations.
        """
        return _run_sync(amark_write(content, canonical=canonical))

    async def amark_observe(content: str) -> str:
        """Record an episodic observation after completing meaningful work.

        Call this after finishing a task or making an important decision.
        MARK structures the observation into memory so future agents can
        learn from it.

        Args:
            content: Brief description of what you implemented or discovered.
        """
        result = await backend.observe(content, agent_id=default_agent_id)
        if not result.ok:
            return f"MARK observe failed: {result.error}"
        fragment_id = result.metadata.get("fragment_id", "")
        return f"MARK observation stored{f' as {fragment_id}' if fragment_id else ''}."

    def mark_observe(content: str) -> str:
        """Record an episodic observation after completing meaningful work.

        Call this after finishing a task or making an important decision.
        MARK structures the observation into memory so future agents can
        learn from it.

        Args:
            content: Brief description of what you implemented or discovered.
        """
        return _run_sync(amark_observe(content))

    return [
        structured_tool.from_function(
            func=mark_retrieve,
            coroutine=amark_retrieve,
            name="mark_retrieve",
            description=mark_retrieve.__doc__,
        ),
        structured_tool.from_function(
            func=mark_write,
            coroutine=amark_write,
            name="mark_write",
            description=mark_write.__doc__,
        ),
        structured_tool.from_function(
            func=mark_observe,
            coroutine=amark_observe,
            name="mark_observe",
            description=mark_observe.__doc__,
        ),
    ]
