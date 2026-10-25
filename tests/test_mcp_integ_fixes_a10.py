# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""MCP and CLI fixes from the integrations re-test: --tools mark_briefing, strict
briefing input, unknown block ids, --extract-facts on the launcher, --no-server anywhere."""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from mark import cli
from mark.errors import UnknownBlockError
from mark.memory.runtime import MarkRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_sdk_unknown_block_id_raises(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "b.db")
    try:
        memory = runtime.memory("a")
        memory.remember("Warehouse lighting is LED.")
        with pytest.raises(UnknownBlockError, match="nope"):
            memory.retrieve_sync("lighting", block_id="nope")
        with pytest.raises(UnknownBlockError):
            memory.retrieve_many_sync(["lighting"], block_ids=["nope"])
    finally:
        runtime.shutdown()


def test_no_server_after_the_command(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "n.db")
    assert cli.main(["--db", db, "remember", "Ports are 4444.", "--no-server"]) == 0
    assert cli.main(["--db", db, "recall", "ports", "--no-server"]) == 0
    assert "4444" in capsys.readouterr().out


def test_launcher_rejects_unknown_tools_in_one_line(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    proc = subprocess.run([sys.executable, "-m", "mark.adapters.mcp", "--db", str(tmp_path / "x.db"),
                           "--tools", "bogus,mark_retrieve"], capture_output=True, text=True, timeout=60,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
    assert "unknown tool(s) bogus" in proc.stderr


def _client(server, calls):
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")

    async def run():
        out = []
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {t.name for t in getattr(listed, "tools", listed)}
            for tool, args in calls:
                r = await client.call_tool(tool, args)
                out.append((bool(getattr(r, "is_error", getattr(r, "isError", False))),
                            "\n".join(getattr(p, "text", "") for p in r.content)))
        return names, out
    return asyncio.run(run())


def test_tools_can_name_mark_briefing_and_briefing_input_is_checked(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from mark.adapters.mcp.launcher import build_server

    server, runtime = build_server(tmp_path / "m.db", tools=["mark_retrieve", "mark_briefing"])
    try:
        names, out = _client(server, [
            ("mark_briefing", {}),
            ("mark_briefing", {"messages": [{"role": "user"}]}),
            ("mark_briefing", {"text": "hi", "session_id": "s" * 500}),
            ("mark_briefing", {"text": "What port does billing use?", "wait": 0}),
            ("mark_retrieve", {"query": "billing", "block_id": "no-such-block"}),
        ])
    finally:
        runtime.shutdown()
    assert names == {"mark_retrieve", "mark_briefing"}
    (e0, t0), (e1, t1), (e2, t2), (e3, _), (e4, t4) = out
    assert e0 and "needs the latest message" in t0
    assert e1 and "messages[0]" in t1
    assert e2 and "the limit is 200" in t2
    assert not e3
    assert e4 and "unknown block" in t4


def test_launcher_extract_facts_uses_the_user_subject(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from mark.adapters.mcp.launcher import build_server

    server, runtime = build_server(tmp_path / "f.db", agent_id="proj", user_subject="Sam", extract_facts=True)
    try:
        _, out = _client(server, [("mark_write", {"content": "I live in Lisbon."})])
        runtime.fact_extraction.extract_pending(wait=True) if hasattr(runtime.fact_extraction, "extract_pending") else None
        facts = {(f.subject, str(f.value)) for f in runtime.memory("proj").facts.current()}
    finally:
        runtime.shutdown()
    assert not out[0][0] and ("Sam", "Lisbon") in facts
