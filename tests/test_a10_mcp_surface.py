# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""a10 MCP surface: kinds, lifecycle tools, short ids, clean errors, bounded retrieve."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.mcp import ALL_TOOLS, create_mark_mcp_server
from mark.runtime import Mark

AGENT = "proj"
SHORT = re.compile(r"\b[0-9a-f]{8}\b")


@pytest.fixture()
def mark(tmp_path: Path):
    with Mark.local(tmp_path) as m:
        yield m


def _call(server, tool: str, args: dict) -> tuple[bool, str]:
    """(is_error, text) as an MCP client sees it."""
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")

    async def run():
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            text = "\n".join(getattr(part, "text", "") for part in result.content)
            return bool(getattr(result, "is_error", getattr(result, "isError", False))), text
    return asyncio.run(run())


def _server(mark, **options):
    return create_mark_mcp_server(LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT,
                                  **options)


def _id(text: str) -> str:
    return SHORT.findall(text)[-1]


def test_tool_list_includes_the_lifecycle_tools() -> None:
    assert {"mark_list", "mark_resolve", "mark_supersede", "mark_forget"} <= set(ALL_TOOLS)


def test_write_kind_list_resolve_supersede_and_short_ids(mark) -> None:
    server = _server(mark)
    error, text = _call(server, "mark_write", {"content": "add retries to the uploader", "kind": "todo"})
    assert not error and re.search(r"written as [0-9a-f]{8}\.$", text)
    todo = _id(text)
    _, text = _call(server, "mark_write", {"content": "DECISION: SQLite for storage"})   # label -> kind
    decision = _id(text)
    error, text = _call(server, "mark_list", {"kind": "todo"})
    assert not error and f"[{todo}]" in text and "[todo]" in text and "SQLite" not in text
    assert f"[{decision}]" in _call(server, "mark_list", {"kind": "decision"})[1]
    assert _call(server, "mark_resolve", {"memory_id": todo}) == (False, f"Resolved memory {todo}.")
    assert "No memories match." == _call(server, "mark_list", {"kind": "todo"})[1]
    assert "(resolved)" in _call(server, "mark_list", {"kind": "todo", "open": False})[1]
    error, text = _call(server, "mark_supersede", {"memory_id": decision, "content": "DECISION: Postgres"})
    assert not error and text.startswith(f"Superseded memory {decision} with ")
    _, text = _call(server, "mark_retrieve", {"query": "storage decision Postgres"})
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", text)                   # no full UUIDs
    error, text = _call(server, "mark_write", {"content": "x", "kind": "chore"})
    assert error and "unknown memory kind 'chore'" in text


def test_forget_by_match_and_secure(mark) -> None:
    server = _server(mark)
    for text in ("token abc in the log", "another token abc line", "unrelated"):
        _call(server, "mark_write", {"content": text})
    error, text = _call(server, "mark_forget", {"match": "token abc", "secure": True})
    assert not error and text.startswith("2 memories match 'token abc'; nothing deleted yet")
    error, text = _call(server, "mark_forget", {"match": "token abc", "secure": True, "confirm": True})
    assert not error and text == "Forgot 2 memories and erased the text from disk."
    assert _call(server, "mark_forget", {})[0]                                  # needs id or match
    assert "at least 4" in _call(server, "mark_forget", {"memory_id": "ab"})[1]


def test_schema_errors_are_one_mark_sentence(mark) -> None:
    error, text = _call(_server(mark), "mark_retrieve", {"query": "x", "max_results": "lots"})
    assert error and "pydantic" not in text and "http" not in text and "\n" not in text
    assert text.startswith("mark_retrieve got invalid arguments: max_results:")


def test_retrieve_is_bounded_and_drops_noise_when_memory_lacks_the_answer(mark) -> None:
    server = _server(mark)
    huge = "Release checklist. " + "Filler words about nothing in particular. " * 300 + \
        "The signing key lives in the vault under ops/signing."
    assert not _call(server, "mark_write", {"content": huge})[0]
    _call(server, "mark_write", {"content": "The staging database is db-7."})
    error, text = _call(server, "mark_retrieve", {"query": "where does the signing key live"})
    assert not error and "ops/signing" in text and len(text) < 3000          # best passage, capped
    error, text = _call(server, "mark_retrieve", {"query": "signing key vault", "budget_tokens": 60,
                                                  "max_item_chars": 4000})
    assert not error and "Memories" in text and len(text) <= 60 * 4 + 120     # cut, not emptied
    _, text = _call(server, "mark_retrieve", {"query": "kubernetes helm chart values for the ingress"})
    assert "Filler words" not in text                                        # below the floor
    _call(server, "mark_write", {"content": "NUL\x00here and bell\x07 too"})
    _, text = _call(server, "mark_retrieve", {"query": "NUL here bell"})
    assert "\x00" not in text and "\x07" not in text and "NULhere" in text


def test_fact_set_natural_dates_and_scheduled(mark) -> None:
    server = _server(mark)
    error, text = _call(server, "mark_fact", {"action": "set", "subject": "atlas", "attribute": "port",
                                              "value": "9191", "valid_from": "2099-01-01"})
    assert not error and text.startswith("Scheduled") and "scheduled from 2099-01-01" in text
    error, text = _call(server, "mark_fact", {"action": "set", "subject": "Atlas Core", "attribute": "port",
                                              "value": "9090", "valid_from": "yesterday"})
    assert not error and text.startswith("Recorded")
    assert "Atlas Core" in _call(server, "mark_fact", {"action": "list"})[1]
    error, text = _call(server, "mark_fact", {"action": "set", "subject": "a", "attribute": "b",
                                              "value": "c", "valid_from": "someday"})
    assert error and "could not read a date" in text and "\n" not in text


def test_briefing_parameters_are_documented_and_bounded(mark) -> None:
    server = _server(mark, briefing=True)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    schema = getattr(tools["mark_briefing"], "input_schema", None) or tools["mark_briefing"].inputSchema
    assert all(p.get("description") for p in schema["properties"].values())
    assert schema["properties"]["wait"]["maximum"] == 10
    assert "between 50 and 4000" in _call(server, "mark_briefing", {"budget_tokens": 5})[1]
    assert "wait must be between 0 and 10" in _call(server, "mark_briefing", {"wait": 1e9})[1]
