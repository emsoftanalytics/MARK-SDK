"""MARK MCP Server adapter.

Exposes MARK memory operations as a Model Context Protocol (MCP) server.
Install optional dependencies: pip install "mark-sdk[mcp]"

Usage (standalone server script)::

    from mark.runtime import Mark
    from mark.adapters.backend import LocalMarkBackend
    from mark.adapters.mcp import create_mark_mcp_server

    mark = Mark.local(".")
    backend = LocalMarkBackend(mark)
    server = create_mark_mcp_server(backend)
    server.run()  # starts stdio MCP server

Usage (from a local Mark instance directly)::

    from mark.runtime import Mark
    from mark.adapters.mcp import create_mark_mcp_server_from_local

    mark = Mark.local(".")
    server = create_mark_mcp_server_from_local(mark)
    server.run()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from mark.adapters.backend import call_backend_off_loop as _backend_call

if TYPE_CHECKING:
    from mark.adapters.backend import MarkBackend


#: Importance is a normalised score; the JSON schema advertises the bounds so
#: MCP clients see them, and pydantic rejects out-of-range values.
Importance = Annotated[float, Field(ge=0.0, le=1.0)]



def _require_mcp() -> Any:
    """Return the high-level MCP server class for the installed ``mcp`` SDK.

    mcp 2.x renamed ``FastMCP`` to ``MCPServer``; mcp 1.x ships ``FastMCP``.
    Both expose the same ``tool()`` decorator / ``run()`` surface used here.
    """
    try:
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError as exc:
        raise ImportError(
            "MCP adapter requires optional dependencies. "
            'Install with: pip install "mark-sdk[mcp]"'
        ) from exc


def create_mark_mcp_server(
    backend: "MarkBackend",
    *,
    name: str = "mark-memory",
    default_agent_id: str = "__mark__",
    default_blocks: list[str] | None = None,
    compress: bool = False,
    allowed_agent_ids: Iterable[str] | None = None,
    briefing: bool = False,
    briefing_sessions: int = 64,
    briefing_preset: str = "default",
) -> Any:
    """
    Create a FastMCP server that exposes MARK memory as MCP tools.

    The server exposes three tools:

    - ``mark_retrieve(query)`` — semantic search over MARK memory.
    - ``mark_observe(content)`` — store a new observation.
    - ``mark_write(content)`` — write a durable memory fragment.

    Args:
        backend:          A :class:`~mark.adapters.backend.MarkBackend` instance.
        name:             Human-readable name for the MCP server.
        default_agent_id: Agent namespace for all memory operations.
        default_blocks:   Optional graph block ids used when retrieval omits blocks.
        compress:         Opt-in local/developer-LLM compression for retrieval.
        allowed_agent_ids: Optional allow-list of agent namespaces. Every tool
                          accepts an ``agent_id`` argument, so by default an
                          MCP client may read and write *any* namespace in the
                          store. Pass a set of ids to restrict clients; calls
                          naming any other namespace are rejected.
                          ``default_agent_id`` is always allowed.
        briefing:         Opt in to the ``mark_briefing`` tool (and the
                          ``mark://briefing/{session_id}`` resource): push
                          context from a per-session ContextWatch. Needs a
                          backend with ``memory_for`` (``LocalMarkBackend``).
        briefing_sessions: Sessions kept (LRU; evicted watches are closed).
        briefing_preset:  ``"default"`` or ``"small"`` (small-model preset).

    Namespaces:
        The ``agent_id`` tool parameter lets MCP clients choose the memory
        namespace (empty string → ``default_agent_id``). Use
        ``allowed_agent_ids`` when the server is exposed to untrusted clients.

    Returns:
        A ``FastMCP`` server instance. Call ``.run()`` to start it
        (stdio transport by default) or use it programmatically in tests.

    Example::

        server = create_mark_mcp_server(backend)
        server.run()               # blocks, speaks MCP stdio
    """
    FastMCP = _require_mcp()
    server = FastMCP(name)
    allowed: set[str] | None = (
        None if allowed_agent_ids is None else {*allowed_agent_ids, default_agent_id}
    )

    def _resolve_agent(agent_id: str) -> tuple[str, str | None]:
        resolved = agent_id or default_agent_id
        if allowed is not None and resolved not in allowed:
            return resolved, f"agent_id {resolved!r} is not allowed on this MARK MCP server"
        return resolved, None

    @server.tool()
    async def mark_retrieve(
        query: str,
        agent_id: str = "",
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
    ) -> str:
        """Call this as your VERY FIRST action before planning or writing any code.

        MARK stores project memory: coding conventions, API patterns, architectural
        decisions, and past task outcomes. The returned context contains local
        evidence the agent should use before starting work.
        Example query: "FastAPI endpoint conventions and project structure".
        """
        resolved, denied = _resolve_agent(agent_id)
        if denied:
            return f"MARK retrieve failed: {denied}"
        result = await _backend_call(
            backend,
            "retrieve",
            query,
            agent_id=resolved,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            block_id=block_id,
            block_ids=block_ids or default_blocks,
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

    @server.tool()
    async def mark_observe(
        content: str,
        agent_id: str = "",
        session_id: str | None = None,
        tags: list[str] | None = None,
        source: str = "mcp",
        importance: Importance = 0.5,
    ) -> str:
        """Record an episodic observation after completing meaningful work.

        Call this after finishing a task or making an important decision.
        MARK structures the observation into memory so future agents can learn from it.
        Example: "Implemented Product CRUD with Pydantic model in routers/products.py".
        """
        resolved, denied = _resolve_agent(agent_id)
        if denied:
            return f"MARK observe failed: {denied}"
        result = await _backend_call(
            backend,
            "observe",
            content,
            agent_id=resolved,
            session_id=session_id,
            tags=tags,
            source=source,
            importance=importance,
        )
        if not result.ok:
            return f"MARK observe failed: {result.error}"
        fragment_id = result.metadata.get("fragment_id", "")
        return f"MARK observation stored{f' as {fragment_id}' if fragment_id else ''}."

    @server.tool()
    async def mark_write(
        content: str,
        agent_id: str = "",
        session_id: str | None = None,
        tags: list[str] | None = None,
        source: str = "mcp",
        importance: Importance = 0.8,
        canonical: bool = False,
    ) -> str:
        """Write a durable, reusable memory fragment to MARK.

        Use for recording conventions, patterns, or facts that should always
        be remembered across future agent runs. Unlike mark_observe (episodic),
        mark_write stores persistent reference knowledge.
        """
        write_tags = list(tags or [])
        if canonical and "canonical" not in write_tags:
            write_tags.append("canonical")
        resolved, denied = _resolve_agent(agent_id)
        if denied:
            return f"MARK write failed: {denied}"
        result = await _backend_call(
            backend,
            "write",
            content,
            agent_id=resolved,
            session_id=session_id,
            tags=write_tags or None,
            source=source,
            importance=0.95 if canonical else importance,
        )
        if not result.ok:
            return f"MARK write failed: {result.error}"
        fragment_id = result.metadata.get("fragment_id", "")
        label = "canonical memory" if canonical else "memory"
        return f"MARK {label} written{f' as {fragment_id}' if fragment_id else ''}."

    if briefing:
        from mark.adapters.mcp._briefing import register_briefing

        server.mark_briefing_sessions = register_briefing(
            server, backend, _resolve_agent, default_agent_id=default_agent_id,
            max_sessions=briefing_sessions, preset=briefing_preset,
        )
    return server


def create_mark_mcp_server_from_local(
    mark: Any,
    *,
    name: str = "mark-memory",
    agent_id: str = "__mark__",
    default_blocks: list[str] | None = None,
    compress: bool = False,
    allowed_agent_ids: Iterable[str] | None = None,
    **briefing_options: Any,
) -> Any:
    """
    Convenience wrapper: create a MARK MCP server directly from a
    :class:`~mark.runtime.Mark` instance.

    Args:
        mark:     A ``Mark.local(...)`` instance.
        name:     Human-readable name for the MCP server.
        agent_id: Agent namespace for all memory operations.
        briefing_options: ``briefing``, ``briefing_sessions`` and
                  ``briefing_preset`` as for :func:`create_mark_mcp_server`.

    Example::

        mark = Mark.local(".")
        server = create_mark_mcp_server_from_local(mark)
        server.run()
    """
    from mark.adapters.backend import LocalMarkBackend

    backend = LocalMarkBackend(mark, default_agent_id=agent_id)
    return create_mark_mcp_server(
        backend,
        name=name,
        default_agent_id=agent_id,
        default_blocks=default_blocks,
        compress=compress,
        allowed_agent_ids=allowed_agent_ids,
        **briefing_options,
    )


__all__ = [
    "create_mark_mcp_server",
    "create_mark_mcp_server_from_local",
]
