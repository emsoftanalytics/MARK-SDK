# SPDX-License-Identifier: MIT
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
