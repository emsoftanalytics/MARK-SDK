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

Usage (command line; the same ``--db`` file the CLI uses)::

    python -m mark.adapters.mcp --db .mark/memory.db --briefing

Usage (from a local Mark instance directly)::

    from mark.runtime import Mark
    from mark.adapters.mcp import create_mark_mcp_server_from_local

    mark = Mark.local(".")
    server = create_mark_mcp_server_from_local(mark)
    server.run()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from mark.adapters.mcp._common import fail

if TYPE_CHECKING:
    from mark.adapters.backend import MarkBackend

#: Every tool the server can expose (``tools=`` selects a subset).
CORE_TOOLS = ("mark_retrieve", "mark_observe", "mark_write")
MEMORY_TOOLS = ("mark_fact", "mark_timeline", "mark_brief", "mark_list", "mark_resolve",
                "mark_supersede", "mark_forget")
#: The opt-in push-context tool: exposed with ``briefing=True`` or when ``tools=`` names it.
BRIEFING_TOOL = "mark_briefing"
ALL_TOOLS = CORE_TOOLS + MEMORY_TOOLS + (BRIEFING_TOOL,)


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
    tools: Iterable[str] | None = None,
) -> Any:
    """
    Create an MCP server that exposes MARK memory as MCP tools.

    Tools (``tools=`` limits the set; default all):

    - ``mark_retrieve(query)`` — search memory (facts + memories, bounded plain text).
    - ``mark_observe(content)`` / ``mark_write(content, kind=...)`` — store an
      event / durable reference knowledge (kind also from a "TODO:" label).
    - ``mark_fact(action, subject, ...)`` — get / set / history / list of facts.
    - ``mark_timeline(subject)`` — every event about a subject, oldest first.
    - ``mark_brief()`` — one-call session brief (facts, TODOs, decisions, bugs).
    - ``mark_list(kind, open)`` — memories by kind (e.g. open TODOs) with short ids.
    - ``mark_resolve(id)`` / ``mark_supersede(id, content)`` — close or replace one.
    - ``mark_forget(memory_id | match, secure)`` — delete memories.
    - ``mark_briefing`` (opt-in: ``briefing=True`` or named in ``tools=``) — push
      context per session.

    Ids in every output are 8-character short ids, accepted back as input.

    Failures the caller can act on (empty query, unknown id, out-of-range
    value, denied namespace, store error) come back as ``isError=True``
    results with one clear sentence. The fact/timeline/brief/list/resolve/
    supersede/forget tools need a backend with ``memory_for`` (``LocalMarkBackend``).

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
                          ``mark://briefing/{session_id}`` resource).
        briefing_sessions: Sessions kept (LRU; evicted watches are closed).
        briefing_preset:  ``"default"``, ``"small"`` or ``"tiny"`` (model size).
        tools:            Names of the tools to expose (default: ``ALL_TOOLS``
                          except ``mark_briefing``, which needs ``briefing=True``
                          or its name in this list). Unknown names raise ValueError.

    Returns:
        The SDK's server instance. Call ``.run()`` to start it (stdio by
        default), or ``python -m mark.adapters.mcp --db PATH`` (see ``launcher``).
    """
    server = _require_mcp()(name)
    allowed: set[str] | None = (
        None if allowed_agent_ids is None else {*allowed_agent_ids, default_agent_id}
    )
    enabled = set(ALL_TOOLS if tools is None else tools)
    unknown = enabled - set(ALL_TOOLS)
    if unknown:
        raise ValueError(f"unknown MCP tool(s) {', '.join(sorted(unknown))}; "
                         f"expected some of: {', '.join(ALL_TOOLS)}")
    # mark_briefing is opt-in: briefing=True, or named in an explicit tools= list.
    briefing = briefing or (tools is not None and BRIEFING_TOOL in enabled)

    def _resolve_agent(agent_id: str) -> tuple[str, str | None]:
        resolved = agent_id or default_agent_id
        if allowed is not None and resolved not in allowed:
            return resolved, f"agent_id {resolved!r} is not allowed on this MARK MCP server"
        return resolved, None

    def resolve(agent_id: str) -> str:
        """The namespace to use, or a tool error when it is not allowed."""
        resolved, denied = _resolve_agent(agent_id)
        if denied:
            fail(denied)
        return resolved

    from mark.adapters.mcp._common import install_error_cleaner
    from mark.adapters.mcp._tools import register_core_tools
    from mark.adapters.mcp._tools_more import register_memory_tools

    install_error_cleaner(server)  # schema errors: one sentence, no pydantic dump

    register_core_tools(server, backend, resolve, default_blocks=default_blocks, compress=compress,
                        enabled=enabled)
    register_memory_tools(server, backend, resolve, enabled=enabled)
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


def run_mcp(*args: Any, **kwargs: Any) -> None:
    """Start a stdio MCP server over a store (see :func:`mark.adapters.mcp.launcher.run_mcp`)."""
    from mark.adapters.mcp.launcher import run_mcp as _run

    _run(*args, **kwargs)


__all__ = [
    "ALL_TOOLS",
    "BRIEFING_TOOL",
    "create_mark_mcp_server",
    "create_mark_mcp_server_from_local",
    "run_mcp",
]
