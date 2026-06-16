# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Functional tests for the MCP server adapter and LangChain tools adapter.

Both adapters run against the real local backend — no network, no provider
keys — proving the shipped adapter surfaces work end to end.
"""

import asyncio

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")
pytest.importorskip("langchain_core", reason="langchain extra not installed")

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain import create_mark_tools
from mark.adapters.mcp import create_mark_mcp_server, create_mark_mcp_server_from_local
from mark.runtime import Mark

AGENT = "adapter-test-agent"


@pytest.fixture()
def mark(tmp_path):
    with Mark.local(project_path=str(tmp_path)) as m:
        yield m


# ── MCP server adapter ────────────────────────────────────────────────────────

def _call(server, tool, args):
    async def _run():
        return await server.call_tool(tool, args)
    return asyncio.run(_run())


def test_mcp_server_exposes_three_tools(mark):
    server = create_mark_mcp_server(LocalMarkBackend(mark, default_agent_id=AGENT))

    async def _names():
        return {t.name for t in await server.list_tools()}

    assert asyncio.run(_names()) == {"mark_retrieve", "mark_observe", "mark_write"}


def test_mcp_write_then_retrieve_round_trip(mark):
    server = create_mark_mcp_server(
        LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT
    )
    write_result = _call(server, "mark_write", {"content": "The API uses FastAPI dependency injection."})
    assert "written" in str(write_result)

    retrieved = _call(server, "mark_retrieve", {"query": "Which API framework do we use?"})
    assert "FastAPI" in str(retrieved)


def test_mcp_observe_stores_fragment(mark):
    server = create_mark_mcp_server(
        LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT
    )
    result = _call(server, "mark_observe", {"content": "Implemented Product CRUD in routers/products.py."})
    assert "stored" in str(result)
    fragments = mark.runtime.memory(AGENT).list()
    assert any("Product CRUD" in f.content for f in fragments)


def test_mcp_tools_accept_scope_and_canonical_options(mark):
    server = create_mark_mcp_server(
        LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT
    )
    write_result = _call(
        server,
        "mark_write",
        {
            "content": "CANON: MCP agents use tenant scoped sessions.",
            "session_id": "release-1",
            "tags": ["mcp"],
            "canonical": True,
        },
    )
    assert "canonical memory written" in str(write_result)

    retrieved = _call(
        server,
        "mark_retrieve",
        {
            "query": "tenant scoped sessions",
            "session_id": "release-1",
            "tags": ["canonical"],
        },
    )
    assert "CANON" in str(retrieved)


def test_mcp_server_from_local_convenience(mark):
    server = create_mark_mcp_server_from_local(mark, agent_id=AGENT)
    result = _call(server, "mark_write", {"content": "Convenience wrapper works."})
    assert "written" in str(result)


# ── LangChain tools adapter ───────────────────────────────────────────────────

def test_langchain_tools_round_trip(mark):
    tools = create_mark_tools(
        LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT
    )
    by_name = {t.name: t for t in tools}
    assert "mark_retrieve" in by_name and "mark_write" in by_name

    by_name["mark_write"].invoke({"content": "Webhooks retry five times with backoff."})
    answer = by_name["mark_retrieve"].invoke({"query": "How many webhook retries?"})
    assert "five times" in answer


def test_langchain_tools_sync_invoke_inside_running_loop(mark):
    tools = {
        t.name: t for t in create_mark_tools(
            LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT
        )
    }

    async def _drive():
        tools["mark_write"].invoke({"content": "Notebook sync tools use a thread bridge."})
        return tools["mark_retrieve"].invoke({"query": "What bridge do notebook sync tools use?"})

    answer = asyncio.run(_drive())
    assert "thread bridge" in answer


def test_langchain_tools_async_round_trip(mark):
    tools = {
        t.name: t for t in create_mark_tools(
            LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT
        )
    }

    async def _drive():
        await tools["mark_write"].ainvoke({"content": "Async tools call MARK without blocking."})
        return await tools["mark_retrieve"].ainvoke({"query": "How do async tools call MARK?"})

    answer = asyncio.run(_drive())
    assert "without blocking" in answer


def test_mark_write_canonical_outranks_observations(mark):
    """Canonical reference memory must rank above episodic notes in retrieval."""
    tools = {t.name: t for t in create_mark_tools(
        LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT)}

    tools["mark_write"].invoke({
        "content": "CANON: the hero wears a magenta mandarin jacket and jade pendant.",
        "canonical": True,
    })
    for i in range(4):
        tools["mark_observe"].invoke({
            "content": f"scene note {i}: hero seen near magenta neon signs with a jacket."})

    context = str(tools["mark_retrieve"].invoke({"query": "hero magenta jacket appearance"}))
    lines = [line for line in context.splitlines() if "] " in line]
    assert lines, context
    assert "CANON" in lines[0], f"canonical memory should rank first:\n{context}"

    memory = mark.runtime.memory(AGENT)
    canon = next(f for f in memory.list(limit=50) if "CANON" in f.content)
    assert canon.importance >= 0.95
    assert "canonical" in canon.tags
