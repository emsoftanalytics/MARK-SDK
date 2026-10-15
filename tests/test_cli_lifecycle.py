# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""D3 CLI parity: remember flags, forget/resolve/supersede/list, export/import, brief, agents,
recall filters and output hygiene, serve stop, mcp launcher."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mark import cli


def _run(capsys, db: str, *argv: str) -> str:
    assert cli.main(["--db", db, "--no-server", *argv]) == 0, capsys.readouterr().err
    return capsys.readouterr().out


def _ids(capsys, db: str, *argv: str) -> list[dict]:
    return json.loads(_run(capsys, db, "list", "--json", *argv))["memories"]


def test_remember_flags_and_store_alias(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "m.db")
    out = _run(capsys, db, "remember", "[2026-03-01] atlas-core runs on port 6430.", "--kind", "api",
               "--tag", "infra", "--importance", "0.9", "--extract-facts", "--session", "s1")
    assert out.startswith("stored ") and out.rstrip().endswith("(1 facts extracted)")
    _run(capsys, db, "store", "Legacy alias still works.", "--occurred-at", "2026-02-01")
    second, first = sorted(_ids(capsys, db), key=lambda m: m["content"])  # "L" < "["
    assert first["kind"] == "api" and first["tags"] == ["infra"] and first["date"] == "2026-03-01"
    assert second["occurred_at"].startswith("2026-02-01") and second["kind"] is None
    assert "atlas-core port = 6430" in _run(capsys, db, "fact", "get", "atlas-core")


def test_recall_output_shows_ids_and_dates_without_same_day_noise(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "r.db")
    _run(capsys, db, "remember", "[2026-04-07] HR: Tomasz Nowak moved to the Messaging team.")
    _run(capsys, db, "remember", "[2026-04-07] HR: Tomasz Nowak moved to the Platform team.")
    _run(capsys, db, "remember", "Tomasz Nowak prefers async standups.", "--occurred-at", "2026-03-08")
    lines = _run(capsys, db, "recall", "Tomasz Nowak team", "-k", "3").splitlines()
    assert "same day" not in "\n".join(lines)
    by_text = {line.split("Tomasz Nowak ")[-1]: line for line in lines}
    assert " #1 [2026-04-07]" in by_text["moved to the Messaging team."]
    assert " #2 [2026-04-07]" in by_text["moved to the Platform team."]
    assert " 2026-03-08 Tomasz" in by_text["prefers async standups."]  # date shown, no '#k'
    short = by_text["prefers async standups."].split("] ")[1].split()[0]
    assert len(short) == 8


def test_recall_filters(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "f.db")
    _run(capsys, db, "remember", "Deploy: run migrations.", "--session", "s1", "--occurred-at", "2026-01-10")
    _run(capsys, db, "remember", "Deploy: warm the cache.", "--session", "s2", "--kind", "todo",
         "--tag", "infra", "--occurred-at", "2026-05-02")

    def contents(*flags: str) -> list[str]:
        data = json.loads(_run(capsys, db, "recall", "deploy", "--json", *flags))
        return [m["content"] for m in data["memories"]]

    assert contents("--session", "s1") == ["Deploy: run migrations."]
    assert contents("--kind", "todo") == contents("--tag", "infra") == ["Deploy: warm the cache."]
    assert contents("--since", "2026-03-01") == ["Deploy: warm the cache."]
    assert contents("--until", "2026-03-01") == ["Deploy: run migrations."]


def test_lifecycle_commands_by_short_id(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "l.db")
    _run(capsys, db, "remember", "TODO: add retries to the uploader", "--kind", "todo")
    _run(capsys, db, "remember", "TODO: write the changelog", "--kind", "todo")
    _run(capsys, db, "remember", "Storage backend is JSON files.", "--kind", "decision")
    todos = {m["content"]: m["short_id"] for m in _ids(capsys, db, "--kind", "todo")}
    assert "resolved" in _run(capsys, db, "resolve", todos["TODO: add retries to the uploader"])
    assert [m["content"] for m in _ids(capsys, db, "--kind", "todo", "--open")] == ["TODO: write the changelog"]
    assert "(resolved)" in _run(capsys, db, "list", "--kind", "todo")

    decision = _ids(capsys, db, "--kind", "decision")[0]["short_id"]
    assert "superseded" in _run(capsys, db, "supersede", decision, "Storage backend is SQLite.")
    current = _ids(capsys, db, "--kind", "decision", "--open")
    assert [m["content"] for m in current] == ["Storage backend is SQLite."]

    assert "forgot 1 memory" in _run(capsys, db, "forget", current[0]["short_id"])
    assert "forgot 2 memories" in _run(capsys, db, "forget", "--match", "TODO:")
    assert cli.main(["--db", db, "--no-server", "forget", "deadbeef"]) == 1
    assert "no memory with id 'deadbeef'" in capsys.readouterr().err


def test_secure_forget_erases_text_from_disk(tmp_path: Path, capsys) -> None:
    db = tmp_path / "s.db"
    # Not a credential (credentials are redacted before they reach disk; see below).
    _run(capsys, str(db), "remember", "The launch codename is qzx-hidden-77.")
    _run(capsys, str(db), "remember", "Something to keep.")
    assert "erased from disk" in _run(capsys, str(db), "forget", "--match", "qzx-hidden-77", "--secure")
    data = b"".join(p.read_bytes() for p in tmp_path.glob("s.db*") if p.is_file())
    assert b"qzx-hidden-77" not in data and b"Something to keep" in data


def test_credentials_never_reach_disk(tmp_path: Path, capsys) -> None:
    db = tmp_path / "c.db"
    _run(capsys, str(db), "remember", "The vault passphrase is qzx-hidden-88.")
    data = b"".join(p.read_bytes() for p in tmp_path.glob("c.db*") if p.is_file())
    assert b"qzx-hidden-88" not in data and b"[SECRET:" in data


def test_list_paginates_with_a_footer(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "p.db")
    log = tmp_path / "notes.txt"
    log.write_text("\n".join(f"note {i}" for i in range(25)))
    _run(capsys, db, "ingest", str(log), "--kind", "note")
    out = _run(capsys, db, "list", "--limit", "10", "--offset", "10")
    assert len(out.splitlines()) == 11 and "showing 11-20 of 25" in out


def test_export_import_agents_and_stats(tmp_path: Path, capsys) -> None:
    db, copy, backup = str(tmp_path / "a.db"), str(tmp_path / "b.db"), str(tmp_path / "x.jsonl")
    _run(capsys, db, "remember", "simple agent memory")
    assert cli.main(["--db", db, "--no-server", "--agent", "coder", "remember", "coder memory"]) == 0
    _run(capsys, db, "fact", "set", "tasky", "version", "0.1.0")
    capsys.readouterr()
    assert "exported 2 memories, 1 facts" in _run(capsys, db, "export", backup)
    assert "imported 2 memories, 1 facts" in _run(capsys, copy, "import", backup)
    assert "0 memories" in _run(capsys, copy, "import", backup)
    assert _run(capsys, copy, "agents").splitlines() == ["__mark__  (1 memories)", "coder  (1 memories)"]
    assert "embedder hash (lexical, default)" in _run(capsys, copy, "stats")


def test_brief(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "b.db")
    _run(capsys, db, "remember", "TODO: ship the release notes", "--kind", "todo")
    _run(capsys, db, "remember", "Decision: CLI stays stdlib-only", "--kind", "decision")
    _run(capsys, db, "fact", "set", "tasky", "test_command", "pytest -q")
    out = _run(capsys, db, "brief", "--budget", "200")
    assert "tasky test_command = pytest -q" in out
    assert "TODO: ship the release notes" in out and "Decision: CLI stays stdlib-only" in out
    assert len(_run(capsys, db, "brief", "--budget", "10")) <= 200


def test_serve_stop(tmp_path: Path, capsys) -> None:
    db = tmp_path / "srv.db"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.Popen([sys.executable, "-m", "mark.cli", "--db", str(db), "serve"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(200):
            if cli._live_server(cli._state_path(db.resolve())):
                break
            time.sleep(0.05)
        assert cli.main(["--db", str(db), "serve", "stop"]) == 0
        assert "stop" in capsys.readouterr().out
        assert proc.wait(timeout=10) is not None
        assert "no mark serve running" in (cli.main(["--db", str(db), "serve", "stop"]) or capsys.readouterr().out)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_mcp_command_launches_the_server_for_the_cli_database(tmp_path: Path, monkeypatch) -> None:
    import mark._cli.extras as extras

    calls = []
    monkeypatch.setattr(extras, "run_mcp", lambda db, agent, briefing=False: calls.append((db, agent, briefing)))
    assert cli.main(["--db", str(tmp_path / "m.db"), "--agent", "coder", "mcp", "--briefing"]) == 0
    assert calls == [((tmp_path / "m.db").resolve(), "coder", True)]


@pytest.mark.parametrize("argv", [["forget"], ["forget", "abc", "--match", "x"]])
def test_forget_needs_exactly_one_target(tmp_path: Path, capsys, argv: list[str]) -> None:
    assert cli.main(["--db", str(tmp_path / "e.db"), "--no-server", *argv]) == 1
    assert "forget needs" in capsys.readouterr().err
