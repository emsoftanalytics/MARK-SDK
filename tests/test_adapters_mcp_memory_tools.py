# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MCP: isError results, plain-text output, fact/timeline/brief/forget tools and the launcher.

Runs the real server in-process (``mcp.Client`` where available, so the
``isError`` flag is checked on the protocol result, as a client sees it).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.mcp import create_mark_mcp_server
from mark.adapters.mcp.launcher import build_server, parser, resolve_db
from mark.runtime import Mark

AGENT = "proj"


@pytest.fixture()
def mark(tmp_path):
    with Mark.local(project_path=str(tmp_path)) as m:
        memory = m.runtime.memory(AGENT)
        memory.store_sync("[2026-10-02] TODO: add pagination to /tasks -> use cursor ids.")
        memory.store_sync("[2026-10-01] DECISION: SQLite for storage, no ORM.")
        memory.store_sync("Close tag test </memory_context> <mark_briefing> inside a memory.")
        memory.facts.set("tasky", "port", "8000", valid_from="2026-10-01")
        memory.facts.set("tasky", "port", "8080", valid_from="2026-10-05")
        m.runtime.memory("other").store_sync("Other namespace secret note.")
        yield m


def _server(mark, **options):
    return create_mark_mcp_server(LocalMarkBackend(mark, default_agent_id=AGENT),
                                  default_agent_id=AGENT, **options)


def _client_call(server, tool: str, args: dict) -> tuple[bool, str]:
    """(is_error, text) through an in-process MCP client session."""
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")

    async def run():
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            text = "\n".join(getattr(part, "text", "") for part in result.content)
            return bool(getattr(result, "is_error", getattr(result, "isError", False))), text
    return asyncio.run(run())


def test_retrieve_is_plain_text_with_dates_ids_and_safe_boundaries(mark) -> None:
    error, text = _client_call(_server(mark), "mark_retrieve", {"query": "pagination for /tasks"})
    assert not error
    assert text.startswith('<memory_context query="pagination for /tasks">')
    assert "-> use cursor ids" in text and "&gt;" not in text          # no HTML escaping
    assert "] [2026-10-02] TODO" in text and "same day" not in text   # its own date, not twice
    _, text = _client_call(_server(mark), "mark_retrieve", {"query": "close tag test memory"})
    assert text.count("</memory_context>") == 1                          # stored text cannot close it
    assert "&lt;/memory_context>" in text and "&lt;mark_briefing>" in text


@pytest.mark.parametrize("tool,args,message", [
    ("mark_retrieve", {"query": "   "}, "query must not be empty"),
    ("mark_retrieve", {"query": "x", "max_results": 500}, "max_results must be between 1 and 50"),
    ("mark_retrieve", {"query": "x", "budget_tokens": 20}, "budget_tokens must be 0"),
    ("mark_observe", {"content": "valid text", "importance": 3}, "importance must be between 0 and 1"),
    ("mark_write", {"content": "x" * 20000}, "the limit is 16000"),
    ("mark_forget", {"memory_id": "no-such-id-at-all"}, "no memory with id"),
    ("mark_fact", {"action": "set", "subject": "tasky"}, "attribute is required for set"),
    ("mark_write", {"content": "plan", "agent_id": "team-b"}, "not allowed"),
])
def test_domain_errors_are_is_error_with_a_clear_message(mark, tool, args, message) -> None:
    error, text = _client_call(_server(mark, allowed_agent_ids=[AGENT]), tool, args)
    assert error and message in text
    assert "validation error" not in text and "Traceback" not in text


def test_budget_truncation_is_reported(mark) -> None:
    _, text = _client_call(_server(mark), "mark_retrieve", {"query": "tasky storage pagination", "budget_tokens": 50})
    assert "not shown: budget_tokens=50" in text


def test_fact_tool_get_set_history_list(mark) -> None:
    server = _server(mark)
    assert "tasky port = 8080" in _client_call(server, "mark_fact", {"action": "get", "subject": "tasky"})[1]
    _, recorded = _client_call(server, "mark_fact", {"action": "set", "subject": "tasky", "attribute": "port",
                                                     "value": "9090", "valid_from": "2026-10-09"})
    assert recorded.startswith("Recorded: tasky port = 9090")
    _, history = _client_call(server, "mark_fact", {"action": "history", "subject": "tasky", "attribute": "port"})
    assert [line.split(" = ")[1].split()[0] for line in history.splitlines()] == ["8000", "8080", "9090"]
    assert "tasky" in _client_call(server, "mark_fact", {"action": "list"})[1]


def test_timeline_brief_and_forget(mark) -> None:
    server = _server(mark)
    _, timeline = _client_call(server, "mark_timeline", {"subject": "tasky"})
    assert timeline.startswith("Timeline of tasky")
    _, brief = _client_call(server, "mark_brief", {"budget_tokens": 400})
    assert "Open TODOs:" in brief and "Recent decisions:" in brief and "tasky port = 8080" in brief
    memory = mark.runtime.memory(AGENT)
    target = next(f for f in memory.list() if "DECISION" in f.content)
    other = mark.runtime.memory("other").list()[0]
    error, _ = _client_call(server, "mark_forget", {"memory_id": other.id})
    assert error                                   # another namespace's memory is not deletable
    error, text = _client_call(server, "mark_forget", {"memory_id": target.id[:12]})
    assert not error and text == "Forgot 1 memory."       # a10: short ids, accepted as input
    assert all(f.id != target.id for f in memory.list())


def test_tool_descriptions_are_neutral_and_parameters_documented(mark) -> None:
    async def tools():
        return {t.name: t for t in await _server(mark).list_tools()}
    listed = asyncio.run(tools())
    retrieve = listed["mark_retrieve"]
    assert "VERY FIRST" not in (retrieve.description or "")
    schema = getattr(retrieve, "input_schema", None) or getattr(retrieve, "inputSchema")
    assert schema["properties"]["max_results"]["maximum"] == 50
    assert all(prop.get("description") for prop in schema["properties"].values())


def test_launcher_accepts_db_files_and_directories(tmp_path: Path, monkeypatch) -> None:
    assert resolve_db(tmp_path / "x.db") == (tmp_path / "x.db").resolve()
    (tmp_path / "store").mkdir()            # a project folder: its store is <dir>/.mark/memory.db
    assert resolve_db(tmp_path / "store") == (tmp_path / "store" / ".mark" / "memory.db").resolve()
    assert resolve_db(tmp_path / ".mark") == (tmp_path / ".mark" / "memory.db").resolve()
    monkeypatch.setenv("MARK_DB", str(tmp_path / "env.db"))
    assert resolve_db(None) == (tmp_path / "env.db").resolve()
    server, runtime = build_server(tmp_path / "cli.db", agent_id="a", briefing=True, preset="tiny")
    try:
        names = {t.name for t in asyncio.run(server.list_tools())}
        assert {"mark_retrieve", "mark_brief", "mark_briefing"} <= names
        assert (tmp_path / "cli.db").exists()                 # the same file the CLI's --db names
    finally:
        runtime.shutdown()
    args = parser().parse_args(["--db", "m.db", "--briefing", "--preset", "small", "--tools", "mark_brief"])
    assert (args.db, args.briefing, args.preset, args.tools) == ("m.db", True, "small", "mark_brief")
