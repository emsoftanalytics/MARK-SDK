"""Core MCP tools: ``mark_retrieve``, ``mark_observe`` and ``mark_write``.

Descriptions are neutral (what the tool does, when it is useful) rather
than pushy ("call this as your VERY FIRST action"): a tool description is a
prompt, and over-eager ones made small models call memory on every turn
(usage test m2). Every parameter is documented with its bounds.
"""
from __future__ import annotations

from typing import Annotated, Any, Callable

from mark.adapters.backend import call_backend_off_loop as _backend_call
from mark.adapters.mcp._common import (
    MAX_CONTENT,
    MAX_QUERY,
    AgentId,
    bounded,
    clean_error,
    fail,
    render_retrieval,
    require_range,
    require_text,
)

Query = Annotated[str, bounded("What to look for, in plain words or exact identifiers.",
                               minLength=1, maxLength=MAX_QUERY)]
Content = Annotated[str, bounded("The text to remember (one fact, decision or outcome).",
                                 minLength=1, maxLength=MAX_CONTENT)]
Tags = Annotated[list[str] | None, bounded("Optional labels, e.g. ['decision'] or ['kind:todo'].")]
Session = Annotated[str | None, bounded("Optional session id to scope the memory.")]
Importance = Annotated[float, bounded("0.0-1.0; higher ranks first in retrieval.", minimum=0.0, maximum=1.0)]


def register_core_tools(server: Any, backend: Any, resolve: Callable[[str], str], *,
                        default_blocks: list[str] | None, compress: bool, enabled: set[str]) -> None:
    """Add the enabled core tools to *server*. *resolve* maps agent_id or fails."""

    if "mark_retrieve" in enabled:
        @server.tool()
        async def mark_retrieve(
            query: Query,
            agent_id: AgentId = "",
            max_results: Annotated[int, bounded("Memories to return (1-50).", minimum=1, maximum=50)] = 10,
            budget_tokens: Annotated[int, bounded("Cap on the reply size in tokens (chars/4); 0 = no cap, "
                                                  "else 50-8000.", minimum=0, maximum=8000)] = 0,
            session_id: Session = None,
            session_prefix: Annotated[str | None, bounded("Only sessions whose id starts with this.")] = None,
            tags: Annotated[list[str] | None, bounded("Only memories carrying all these tags.")] = None,
            block_id: Annotated[str | None, bounded("Only this memory block.")] = None,
            block_ids: Annotated[list[str] | None, bounded("Only these memory blocks.")] = None,
        ) -> str:
            """Search memory for notes relevant to a query.

            Returns current facts and the most relevant memories (with date and
            short id), most relevant first. Useful before answering questions
            about earlier work, decisions, conventions or the user.
            """
            text = require_text("query", query, MAX_QUERY)
            require_range("max_results", max_results, 1, 50)
            if budget_tokens and not 50 <= budget_tokens <= 8000:
                fail("budget_tokens must be 0 (no cap) or between 50 and 8000")
            result = await _backend_call(
                backend, "retrieve", text, agent_id=resolve(agent_id), session_id=session_id,
                session_prefix=session_prefix, tags=tags, block_id=block_id,
                block_ids=block_ids or default_blocks, compress=compress,
            )
            if not result.ok:
                fail(f"mark_retrieve failed: {clean_error(result.error)}")
            value = result.value
            if hasattr(value, "fragments"):
                return render_retrieval(value, max_results=max_results, budget_tokens=budget_tokens)
            return str(value.as_text() if hasattr(value, "as_text") else value)

    if "mark_observe" in enabled:
        @server.tool()
        async def mark_observe(
            content: Content,
            agent_id: AgentId = "",
            session_id: Session = None,
            tags: Tags = None,
            source: Annotated[str, bounded("Who produced it: user, tool, document, mcp ...")] = "mcp",
            importance: Importance = 0.5,
        ) -> str:
            """Record something that happened (an event, outcome or observation).

            Use for episodic notes such as "Implemented Product CRUD in
            routers/products.py". Use mark_write for reference knowledge.
            """
            text = require_text("content", content, MAX_CONTENT)
            require_range("importance", importance, 0.0, 1.0)
            result = await _backend_call(backend, "observe", text, agent_id=resolve(agent_id),
                                         session_id=session_id, tags=tags, source=source,
                                         importance=importance)
            if not result.ok:
                fail(f"mark_observe failed: {clean_error(result.error)}")
            fragment_id = result.metadata.get("fragment_id", "")
            return f"MARK observation stored{f' as {fragment_id}' if fragment_id else ''}."

    if "mark_write" in enabled:
        @server.tool()
        async def mark_write(
            content: Content,
            agent_id: AgentId = "",
            session_id: Session = None,
            tags: Tags = None,
            source: Annotated[str, bounded("Who produced it: user, tool, document, mcp ...")] = "mcp",
            importance: Importance = 0.8,
            canonical: Annotated[bool, bounded("Mark as authoritative reference (ranks first).")] = False,
        ) -> str:
            """Store durable reference knowledge: a convention, decision or fact.

            Prefix with DECISION:, TODO:, BUG: or FIX: (or tag kind:<name>) so
            mark_brief can list it. Unlike mark_observe, this is meant to be
            recalled across future sessions.
            """
            text = require_text("content", content, MAX_CONTENT)
            require_range("importance", importance, 0.0, 1.0)
            write_tags = list(tags or [])
            if canonical and "canonical" not in write_tags:
                write_tags.append("canonical")
            result = await _backend_call(backend, "write", text, agent_id=resolve(agent_id),
                                         session_id=session_id, tags=write_tags or None, source=source,
                                         importance=0.95 if canonical else importance)
            if not result.ok:
                fail(f"mark_write failed: {clean_error(result.error)}")
            fragment_id = result.metadata.get("fragment_id", "")
            label = "canonical memory" if canonical else "memory"
            return f"MARK {label} written{f' as {fragment_id}' if fragment_id else ''}."
