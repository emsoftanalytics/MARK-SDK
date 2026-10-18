# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The user subject ("who is 'I'?") is stored with the store and applied on every open.

Set once with ``Mark.local(user_subject=...)``, ``mark.configure_user_subject``,
``mark config user-subject NAME`` or ``mark mcp --user-subject NAME``; every
later open (Python, CLI, MCP) links "I/my" facts and questions to that name.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mark.cli import main
from mark.runtime import Mark


def _lives(mark: Mark) -> list[tuple[str, str]]:
    return [(f.subject, str(f.value)) for f in mark.memory.recall("Where do I live now?").facts]


def test_user_subject_is_persisted_and_applied_on_reopen(tmp_path: Path) -> None:
    with Mark.local(tmp_path, fact_extractor="patterns", user_subject="ana") as mark:
        assert mark.user_subject == "ana"
        mark.memory.remember("I moved to Porto in 2026.", extract_facts=True)
    with Mark.local(tmp_path, fact_extractor="patterns") as mark:      # no argument: stored value
        assert mark.user_subject == "ana"
        assert ("ana", "Porto") in _lives(mark)                         # "my" questions -> ana
        mark.memory.remember("I work at Acme.", extract_facts=True)       # "I" facts -> ana
        assert all(f.subject == "ana" for f in mark.memory.facts.current() if str(f.value) == "Acme")


def test_configure_user_subject_persists_and_default_is_user(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        assert mark.user_subject == "user"
        mark.configure_user_subject("Somto")
    with Mark.local(tmp_path) as mark:
        assert mark.user_subject == "Somto"
        with pytest.raises(ValueError):
            mark.configure_user_subject("   ")


def test_cli_config_user_subject_and_show(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "m.db")
    assert main(["--no-server", "--db", db, "config", "user-subject", "ana"]) == 0
    assert "ana" in capsys.readouterr().out
    assert main(["--no-server", "--db", db, "config", "show"]) == 0
    out = capsys.readouterr().out
    assert "user subject: ana" in out and db in out
    assert main(["--no-server", "--db", db, "config", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["user_subject"] == "ana"
    with Mark.local(db) as mark:
        assert mark.user_subject == "ana"


def test_mcp_launcher_user_subject(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from mark.adapters.mcp.launcher import build_server, parser

    assert parser().parse_args(["--user-subject", "ana"]).user_subject == "ana"
    server, runtime = build_server(tmp_path / "m.db", user_subject="ana")
    runtime.shutdown()
    with Mark.local(tmp_path / "m.db") as mark:
        assert mark.user_subject == "ana"


def test_mark_mcp_accepts_every_launcher_option(tmp_path: Path, monkeypatch) -> None:
    """`mark mcp` parity with `python -m mark.adapters.mcp` (integ report N2)."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    import mark.adapters.mcp.launcher as launcher

    seen: list[list[str]] = []
    monkeypatch.setattr(launcher, "main", lambda argv: seen.append(list(argv)))
    db = tmp_path / "x.db"
    assert main(["mcp", "--db", str(db), "--agent", "proj", "--briefing", "--preset", "small",
                 "--tools", "mark_brief,mark_retrieve", "--allow-agent", "b", "--user-subject", "ana"]) == 0
    args = launcher.parser().parse_args(seen[0])
    assert args.db == str(db.resolve()) and args.agent == "proj" and args.briefing
    assert (args.preset, args.tools, args.allowed, args.user_subject) == (
        "small", "mark_brief,mark_retrieve", ["b"], "ana")
    assert main(["--db", str(db), "--agent", "g", "mcp"]) == 0       # global options still work
    args = launcher.parser().parse_args(seen[1])
    assert (args.db, args.agent, args.briefing) == (str(db.resolve()), "g", False)
