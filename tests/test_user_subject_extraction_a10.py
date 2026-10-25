# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Fact extraction uses the stored user subject everywhere (dev re-test major 2).

``mark remember/ingest --extract-facts`` used the built-in pattern
extractor with the default subject "user", so facts split between "user"
and the configured name. MCP mark_write/mark_observe gain ``extract_facts``,
and unknown MCP arguments are an error naming the argument.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mark.cli import main
from mark.runtime import Mark


def _subjects(db: Path) -> set[str]:
    with Mark.local(db) as mark:
        return {f.subject for f in mark.memory.facts.current() if str(f.value) == "Lisbon"}


def test_cli_remember_and_ingest_use_the_stored_subject(tmp_path: Path, capsys) -> None:
    db = tmp_path / "m.db"
    assert main(["--no-server", "--db", str(db), "config", "user-subject", "Sam"]) == 0
    assert main(["--no-server", "--db", str(db), "remember", "I live in Lisbon.", "--extract-facts"]) == 0
    assert "1 fact extracted" in capsys.readouterr().out
    assert _subjects(db) == {"Sam"}
    lines = tmp_path / "notes.txt"
    lines.write_text("My editor is Helix.\nI work at Acme.\n")
    assert main(["--no-server", "--db", str(db), "ingest", str(lines), "--extract-facts"]) == 0
    with Mark.local(db) as mark:
        subjects = {f.subject for f in mark.memory.facts.current()}
    assert "user" not in subjects and "Sam" in subjects


def test_python_extract_facts_true_uses_the_stored_subject(tmp_path: Path) -> None:
    with Mark.local(tmp_path, user_subject="Sam") as mark:
        mark.memory.remember("I live in Lisbon.", extract_facts=True)
        runtime_mem = mark.runtime.memory("other")
        runtime_mem.remember("I live in Lisbon.", extract_facts=True)
        assert {f.subject for f in runtime_mem.facts.current()} == {"Sam"}
    assert _subjects(tmp_path / ".mark" / "memory.db") == {"Sam"}


def _mcp(tmp_path: Path, calls: list[tuple[str, dict]]) -> tuple[list[tuple[bool, str]], set]:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")
    from mark.adapters.backend import LocalMarkBackend
    from mark.adapters.mcp import create_mark_mcp_server

    with Mark.local(tmp_path, user_subject="Sam") as mark:
        server = create_mark_mcp_server(LocalMarkBackend(mark, default_agent_id="proj"),
                                        default_agent_id="proj")

        async def run():
            out = []
            async with Client(server) as client:
                for tool, args in calls:
                    r = await client.call_tool(tool, args)
                    out.append((bool(getattr(r, "is_error", getattr(r, "isError", False))),
                                "\n".join(getattr(p, "text", "") for p in r.content)))
            return out
        results = asyncio.run(run())
        facts = {(f.subject, f.attribute, str(f.value)) for f in mark.runtime.memory("proj").facts.current()}
    return results, facts


def test_mcp_write_and_observe_extract_facts_with_the_user_subject(tmp_path: Path) -> None:
    results, facts = _mcp(tmp_path, [
        ("mark_write", {"content": "I live in Lisbon.", "extract_facts": True}),
        ("mark_observe", {"content": "I work at Acme.", "extract_facts": True}),
        ("mark_write", {"content": "My editor is Vim."}),
    ])
    assert all(not error for error, _ in results)
    assert "1 fact extracted" in results[0][1]
    assert any(s == "Sam" and v == "Lisbon" for s, _, v in facts)
    assert any(s == "Sam" and v == "Acme" for s, _, v in facts)
    assert not any(v == "Vim" for _, _, v in facts)          # extraction is opt-in


def test_mcp_unknown_arguments_are_an_error_naming_them(tmp_path: Path) -> None:
    results, facts = _mcp(tmp_path, [
        ("mark_write", {"content": "I live in Lisbon.", "extract_fact": True}),
        ("mark_retrieve", {"query": "Lisbon", "limit": 3}),
    ])
    (write_error, write_text), (read_error, read_text) = results
    assert write_error and "extract_fact" in write_text and "mark_write" in write_text
    assert read_error and "'limit'" in read_text and "max_results" in read_text  # lists valid names
