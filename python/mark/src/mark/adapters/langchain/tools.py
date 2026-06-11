"""LangChain tool helpers for MARK."""

from __future__ import annotations

import asyncio
from typing import Any

from mark.adapters.backend import MarkBackend


def _require_langchain_tool():
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise ImportError(
            "LangChain adapters require optional dependencies. "
            'Install with: pip install "mark-sdk[langchain]"'
        ) from exc
    return tool


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("MARK LangChain sync tools cannot be called inside a running event loop")


def create_mark_tools(
    backend: MarkBackend,
    *,
    default_agent_id: str,
    default_blocks: list[str] | None = None,
) -> list[Any]:
    """Create basic LangChain tools backed by MARK."""

    tool = _require_langchain_tool()

    @tool("mark_retrieve")
    def mark_retrieve(query: str) -> str:
        """Call this as your VERY FIRST action before planning or writing any code.

        MARK stores project memory: coding conventions, API patterns, architectural
        decisions, and past task outcomes. The returned context contains requirements
        you MUST follow. Always call before starting work.

        Args:
            query: Phrase describing your current task or topic, e.g.
                   "FastAPI endpoint conventions and project structure".
        """
        result = _run(backend.retrieve(query, agent_id=default_agent_id, blocks=default_blocks, compress=True))
        if not result.ok:
            return f"MARK retrieve failed: {result.error}"
        value = result.value
        if hasattr(value, "as_context"):
            return value.as_context()
        if hasattr(value, "as_text"):
            return value.as_text()
        return str(value)

    @tool("mark_write")
    def mark_write(content: str) -> str:
        """Write a durable, reusable memory fragment to MARK.

        Use for recording conventions, patterns, or facts that should always
        be remembered across future agent runs.  Unlike mark_observe (episodic
        events), mark_write stores persistent reference knowledge.

        Args:
            content: The fact, pattern, or convention to store permanently.
        """
        result = _run(backend.write(content, agent_id=default_agent_id))
        if not result.ok:
            return f"MARK write failed: {result.error}"
        return "MARK memory written."

    @tool("mark_observe")
    def mark_observe(content: str) -> str:
        """Record an episodic observation after completing meaningful work.

        Call this after finishing a task or making an important decision.
        MARK structures the observation into memory so future agents can
        learn from it.

        Args:
            content: Brief description of what you implemented or discovered.
        """
        result = _run(backend.observe(content, agent_id=default_agent_id))
        if not result.ok:
            return f"MARK observe failed: {result.error}"
        fragment_id = result.metadata.get("fragment_id", "")
        return f"MARK observation stored{f' as {fragment_id}' if fragment_id else ''}."

    return [mark_retrieve, mark_write, mark_observe]
