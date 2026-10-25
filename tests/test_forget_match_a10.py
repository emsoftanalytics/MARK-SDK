# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""forget(match=...) safety (release re-test A): whole words, a minimum length,
dry runs, a cap, and confirmation on the CLI and over MCP.

``--match e`` used to wipe a whole store and ``spendlog`` deleted 37 of 41
memories without a question.
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest

from mark import cli
from mark.errors import InvalidInputError
from mark.memory.facade.forget_match import FORGET_MATCH_CAP, ForgetPreview
from mark.memory.runtime import MarkRuntime


@pytest.fixture()
def mem(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    yield runtime.memory("agent")
    runtime.shutdown()


def test_match_is_whole_word_or_phrase_not_substring(mem) -> None:
    mem.remember("The cat sleeps on the sofa.")
    mem.remember("Concatenate the logs before upload.")
    mem.remember("The black cat is called Ink.")
    assert mem.forget(match="cat") == 2
    assert [f.content for f in mem.list(limit=None)] == ["Concatenate the logs before upload."]
    mem.remember("Run the deploy script on Friday.")
    assert mem.forget(match="deploy script") == 1          # a phrase
    assert mem.forget(match="script on monday") == 0


@pytest.mark.parametrize("needle", ["e", "ab", "  x ", "the", "and the", "of"])
def test_short_or_stopword_matches_are_refused(mem, needle: str) -> None:
    mem.remember("The end of the line.")
    with pytest.raises(InvalidInputError):
        mem.forget(match=needle)
    assert mem.count() == 1


def test_dry_run_returns_count_and_previews_without_deleting(mem) -> None:
    ids = [mem.remember(f"spendlog note number {i} about the ledger") for i in range(3)]
    preview = mem.forget(match="spendlog", dry_run=True)
    assert isinstance(preview, ForgetPreview)
    assert preview.count == 3 and sorted(preview.ids) == sorted(ids)
    assert all(line.startswith(i[:8]) for line, i in zip(preview.previews, preview.ids))
    assert "spendlog note" in preview.previews[0]
    assert mem.count() == 3


def test_cap_needs_all_true(mem) -> None:
    mem.remember_many([f"bulk item {i}" for i in range(FORGET_MATCH_CAP + 1)])
    with pytest.raises(InvalidInputError, match=str(FORGET_MATCH_CAP + 1)):
        mem.forget(match="bulk item")
    assert mem.count() == FORGET_MATCH_CAP + 1
    assert mem.forget(match="bulk item", all=True) == FORGET_MATCH_CAP + 1


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cli(db: str, *argv: str) -> int:
    return cli.main(["--db", db, "--no-server", *argv])


def _seed(capsys, db: str) -> None:
    for text in ("spendlog uses EUR.", "spendlog stores data in SQLite.", "Unrelated note."):
        assert _cli(db, "remember", text) == 0
    capsys.readouterr()


def test_cli_match_without_yes_refuses_when_not_interactive(tmp_path: Path, capsys, monkeypatch) -> None:
    db = str(tmp_path / "c.db")
    _seed(capsys, db)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))       # not a terminal
    assert _cli(db, "forget", "--match", "spendlog") == 1
    captured = capsys.readouterr()
    assert "2 memories match" in captured.out + captured.err and "--yes" in captured.err
    assert _cli(db, "forget", "--match", "spendlog", "--yes") == 0
    assert "forgot 2 memories" in capsys.readouterr().out


def test_cli_match_asks_on_a_terminal(tmp_path: Path, capsys, monkeypatch) -> None:
    db = str(tmp_path / "t.db")
    _seed(capsys, db)

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("n\n"))
    assert _cli(db, "forget", "--match", "spendlog") == 1
    assert "nothing forgotten" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", Tty("y\n"))
    assert _cli(db, "forget", "--match", "spendlog") == 0
    out = capsys.readouterr().out
    assert "2 memories match" in out and "forgot 2 memories" in out


def test_cli_dry_run_and_short_match(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "d.db")
    _seed(capsys, db)
    assert _cli(db, "forget", "--match", "spendlog", "--dry-run") == 0
    assert "2 memories match" in capsys.readouterr().out
    assert _cli(db, "forget", "--match", "e", "--yes") == 1
    assert "at least 3 characters" in capsys.readouterr().err
    assert _cli(db, "list", "--json") == 0 and capsys.readouterr().out.count('"id"') == 3


# ── MCP ──────────────────────────────────────────────────────────────────────


def _mcp(tmp_path: Path, tool: str, args: dict) -> tuple[bool, str]:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")
    from mark.adapters.backend import LocalMarkBackend
    from mark.adapters.mcp import create_mark_mcp_server
    from mark.runtime import Mark

    with Mark.local(project_path=str(tmp_path)) as mark:
        memory = mark.runtime.memory("proj")
        if not memory.count():
            for text in ("spendlog uses EUR.", "spendlog stores data in SQLite.", "Cats are nice."):
                memory.remember(text)
        server = create_mark_mcp_server(LocalMarkBackend(mark, default_agent_id="proj"),
                                        default_agent_id="proj")

        async def run():
            async with Client(server) as client:
                result = await client.call_tool(tool, args)
                text = "\n".join(getattr(p, "text", "") for p in result.content)
                return bool(getattr(result, "is_error", getattr(result, "isError", False))), text
        out = asyncio.run(run())
        out = (*out, memory.count())
    return out


def test_mcp_forget_match_needs_confirm_for_several(tmp_path: Path) -> None:
    error, text, left = _mcp(tmp_path, "mark_forget", {"match": "spendlog"})
    assert not error and "2 memories match" in text and "confirm=true" in text and left == 3
    error, text, left = _mcp(tmp_path, "mark_forget", {"match": "spendlog", "confirm": True})
    assert not error and text.startswith("Forgot 2 memories") and left == 1
    error, text, left = _mcp(tmp_path, "mark_forget", {"match": "cats"})
    assert not error and text.startswith("Forgot 1 memory") and left == 0


def test_mcp_forget_match_too_short_is_an_error(tmp_path: Path) -> None:
    error, text, left = _mcp(tmp_path, "mark_forget", {"match": "e", "confirm": True})
    assert error and "at least 3 characters" in text and left == 3
