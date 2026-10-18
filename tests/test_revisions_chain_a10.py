# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Revisions never fork history, and resolve only closes TODOs and bugs (re-test B, C).

Superseding an already-superseded memory used to leave two "current"
versions; resolve accepted a DECISION (which then vanished from the brief)
and reported success twice for the same TODO.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mark import cli
from mark.errors import InvalidInputError
from mark.memory.runtime import MarkRuntime


@pytest.fixture()
def mem(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "r.db")
    yield runtime.memory("agent")
    runtime.shutdown()


def test_supersede_of_a_superseded_memory_follows_the_chain(mem) -> None:
    v1 = mem.remember("Storage backend is JSON files.", kind="decision")
    v2 = mem.supersede(v1, "Storage backend is SQLite.")
    assert mem.current_version(v1) == v2
    with pytest.warns(UserWarning, match="already superseded"):
        v3 = mem.supersede(v1, "Storage backend is Postgres.")
    assert mem.get(v3).metadata["supersedes"] == v2
    assert mem.get(v3).metadata["history"] == [v1, v2]
    assert [f.id for f in mem.list(kind="decision", open=True)] == [v3]   # no fork
    with pytest.warns(UserWarning):
        v4 = mem.update(v2, "Storage backend is Postgres 16.")
    assert mem.get(v4).metadata["supersedes"] == v3
    assert [f.id for f in mem.list(kind="decision", open=True)] == [v4]


def test_resolve_only_closes_todos_and_bugs(mem) -> None:
    decision = mem.remember("DECISION: keep the CLI stdlib-only")
    note = mem.remember("The office plant needs water.")
    bug = mem.remember("BUG: export crashes on empty lists.")
    with pytest.raises(InvalidInputError, match="decision.*supersede"):
        mem.resolve(decision)
    with pytest.raises(InvalidInputError, match="supersede.*forget"):
        mem.resolve(note)
    assert "mark:resolved" not in mem.get(decision).tags
    assert mem.resolve(bug) == bug


def test_resolving_twice_is_a_no_op(mem) -> None:
    todo = mem.remember("TODO: write the changelog", kind="todo")
    mem.resolve(todo, note="first")
    first = mem.get(todo).metadata["resolved_at"]
    assert mem.resolve(todo, note="second") == todo
    after = mem.get(todo).metadata
    assert after["resolved_at"] == first and after["resolution"] == "first"


def test_resolve_of_a_superseded_todo_closes_its_current_version(mem) -> None:
    v1 = mem.remember("TODO: add retries", kind="todo")
    v2 = mem.supersede(v1, "TODO: add retries with backoff")
    with pytest.warns(UserWarning, match="already superseded"):
        assert mem.resolve(v1) == v2
    assert mem.list(kind="todo", open=True) == []


def _cli(capsys, db: str, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["--db", db, "--no-server", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_says_what_it_did(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "c.db")
    _cli(capsys, db, "remember", "DECISION: use JSON files")
    _cli(capsys, db, "remember", "TODO: add retries", "--kind", "todo")
    rows = {m["content"]: m["short_id"] for m in json.loads(_cli(capsys, db, "list", "--json")[1])["memories"]}
    decision, todo = rows["DECISION: use JSON files"], rows["TODO: add retries"]
    code, out, _ = _cli(capsys, db, "supersede", decision, "DECISION: use SQLite")
    assert code == 0 and out.startswith(f"superseded {decision} with ")
    code, out, err = _cli(capsys, db, "supersede", decision, "DECISION: use Postgres")
    assert code == 0 and "already superseded" in out and "warn" not in err.lower()
    code, _, err = _cli(capsys, db, "resolve", decision)
    assert code == 1 and "supersede" in err
    assert _cli(capsys, db, "resolve", todo)[1].startswith(f"resolved {todo}")
    code, out, _ = _cli(capsys, db, "resolve", todo)
    assert code == 0 and out.startswith(f"already resolved: {todo}")


def test_mcp_resolve_and_supersede_messages(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")
    from mark.adapters.backend import LocalMarkBackend
    from mark.adapters.mcp import create_mark_mcp_server
    from mark.runtime import Mark

    with Mark.local(project_path=str(tmp_path)) as mark:
        memory = mark.runtime.memory("proj")
        todo = memory.remember("TODO: add retries", kind="todo")[:8]
        note = memory.remember("The plant needs water.")[:8]
        v1 = memory.remember("DECISION: JSON files")
        memory.supersede(v1, "DECISION: SQLite")
        server = create_mark_mcp_server(LocalMarkBackend(mark, default_agent_id="proj"),
                                        default_agent_id="proj")

        async def calls():
            out = []
            async with Client(server) as client:
                for tool, args in (("mark_resolve", {"memory_id": todo}), ("mark_resolve", {"memory_id": todo}),
                                   ("mark_resolve", {"memory_id": note}),
                                   ("mark_supersede", {"memory_id": v1[:8], "content": "DECISION: Postgres"})):
                    r = await client.call_tool(tool, args)
                    out.append((bool(getattr(r, "is_error", getattr(r, "isError", False))),
                                "\n".join(getattr(p, "text", "") for p in r.content)))
            return out
        first, again, plain, chained = asyncio.run(calls())
    assert first == (False, f"Resolved memory {todo}.")
    assert not again[0] and "already resolved" in again[1]
    assert plain[0] and "mark_supersede" in plain[1]
    assert not chained[0] and "already superseded" in chained[1]


def test_get_accepts_unique_short_ids(mem) -> None:
    fid = mem.remember("Short ids work in get().")
    assert mem.get(fid[:8]).id == fid and mem.get(fid[:4]).id == fid
    assert mem.get(fid[:3]) is None
    other = mem._runtime.memory("other").remember("not mine")
    assert mem.get(other[:8]) is None


def test_date_errors_use_the_right_article(mem) -> None:
    with pytest.raises(InvalidInputError, match="could not read an occurred_at"):
        mem.remember("x happened", occurred_at="not a date at all")
