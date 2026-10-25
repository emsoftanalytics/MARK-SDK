# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""a10 CLI polish: short ids, honest counts, missing stores, kinds from labels, timeline dates."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mark.cli import main


@pytest.fixture()
def db(tmp_path: Path) -> str:
    return str(tmp_path / "m.db")


def run(db: str, *argv: str) -> int:
    return main(["--no-server", "--db", db, *argv])


def test_remember_prints_the_short_id(db: str, capsys) -> None:
    assert run(db, "remember", "We chose SQLite.") == 0
    out = capsys.readouterr().out.strip()
    assert re.fullmatch(r"stored [0-9a-f]{8}", out), out
    assert run(db, "remember", "Second note.", "--json") == 0
    assert len(json.loads(capsys.readouterr().out)["id"]) == 36


def test_reingest_reports_new_and_present(db: str, tmp_path: Path, capsys) -> None:
    lines = tmp_path / "notes.txt"
    lines.write_text("TODO: add retries\nDECISION: use SQLite\nplain note\n", encoding="utf-8")
    assert run(db, "ingest", str(lines)) == 0
    assert capsys.readouterr().out.strip() == "stored 3 new memories"
    assert run(db, "ingest", str(lines)) == 0
    assert capsys.readouterr().out.strip() == "stored 0 new memories (3 already present)"
    assert run(db, "list", "--kind", "todo", "--json") == 0      # the label set the kind
    assert [m["content"] for m in json.loads(capsys.readouterr().out)["memories"]] == ["TODO: add retries"]


@pytest.mark.parametrize("argv", [["list"], ["stats"], ["brief"], ["recall", "x"], ["config", "show"],
                                  ["fact", "list"], ["timeline", "x"], ["agents"]])
def test_read_commands_never_create_a_missing_store(tmp_path: Path, capsys, argv: list[str]) -> None:
    db = tmp_path / "nowhere" / "m.db"
    assert run(str(db), *argv) == 1
    assert f"no store at {db}" in capsys.readouterr().err
    assert not db.exists() and not db.parent.exists()


def test_serve_stop_without_a_server_fails(db: str, capsys) -> None:
    assert run(db, "serve", "stop") == 1
    assert "no mark serve running" in capsys.readouterr().err


def test_brief_on_an_empty_store(db: str, capsys) -> None:
    assert run(db, "remember", "xyzzy") == 0 and run(db, "forget", "--match", "xyzzy", "--yes") == 0
    capsys.readouterr()
    assert run(db, "brief") == 0
    assert capsys.readouterr().out.strip() == "(memory is empty)"


def test_recall_lines_have_no_rank_suffix_and_kind_filters_hide_retired(db: str, capsys) -> None:
    for text in ("TODO: add retries to the uploader", "TODO: add retries to the downloader"):
        run(db, "remember", text)
    capsys.readouterr()
    run(db, "list", "--json")
    first = json.loads(capsys.readouterr().out)["memories"][0]
    assert run(db, "resolve", first["short_id"]) == 0
    capsys.readouterr()
    assert run(db, "recall", "retries uploader downloader") == 0
    out = capsys.readouterr().out
    assert "#1" not in out and "#2" not in out
    assert run(db, "recall", "add retries", "--kind", "todo", "--json") == 0
    shown = [m["id"] for m in json.loads(capsys.readouterr().out)["memories"]]
    assert first["id"] not in shown and len(shown) == 1
    assert run(db, "recall", "add retries", "--kind", "todo", "--all", "--json") == 0
    assert first["id"] in [m["id"] for m in json.loads(capsys.readouterr().out)["memories"]]
    assert run(db, "recall", "add retries", "--open", "--json") == 0
    assert first["id"] not in [m["id"] for m in json.loads(capsys.readouterr().out)["memories"]]


def test_timeline_shows_the_stored_date_instead_of_undated(db: str, capsys) -> None:
    run(db, "remember", "atlas-core moved to port 9191", "--occurred-at", "2026-10-01")
    run(db, "remember", "atlas-core runs on port 9090")
    run(db, "remember", "atlas-core restarted twice", "--occurred-at", "2026-10-01")
    capsys.readouterr()
    assert run(db, "timeline", "atlas-core") == 0
    out = capsys.readouterr().out
    assert "undated" not in out and "same day" not in out
    lines = out.strip().splitlines()[1:]
    assert lines[0].startswith("2026-10-01") and lines[1].startswith("2026-10-01")
    assert re.match(r"stored \d{4}-\d{2}-\d{2} ", lines[2]), lines[2]


def test_fact_cli_list_help_scheduled_and_natural_dates(db: str, capsys) -> None:
    with pytest.raises(SystemExit):
        main(["fact", "--help"])
    help_text = capsys.readouterr().out
    for verb in ("set", "get", "history", "list", "retract"):
        assert re.search(rf"^\s+{verb}\s+\S", help_text, re.M), verb
    assert run(db, "fact", "set", "atlas", "port", "9090", "--valid-from", "yesterday") == 0
    assert run(db, "fact", "set", "atlas", "owner", "ana") == 0
    assert run(db, "fact", "set", "atlas", "port", "9191", "--valid-from", "2099-01-01") == 0
    out = capsys.readouterr().out
    assert "scheduled from 2099-01-01" in out
    assert run(db, "fact", "list") == 0
    out = capsys.readouterr().out
    assert "atlas port = 9090" in out and "atlas owner = ana" in out
    assert run(db, "fact", "list", "nobody") == 0
    assert capsys.readouterr().out.strip() == "(no facts)"
    assert run(db, "fact", "set", "atlas", "port", "1", "--valid-from", "not a date at all") == 1
    assert "could not read a date" in capsys.readouterr().err
    assert run(db, "remember", "x", "--occurred-at", "last tuesday") == 0


def test_serve_client_never_imports_ssl_or_urllib(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = ("import sys; from pathlib import Path; import mark.cli; "
            "from mark._cli.client import _live_server; _live_server(Path(sys.argv[1])); "
            "print(sorted(m for m in ('ssl', 'urllib.request', 'http.client') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "x.db.server.json")],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_brief_never_calls_a_non_empty_store_empty(db: str, capsys) -> None:
    for i in range(30):
        assert run(db, "remember", f"Plain note number {i} about the garden shed.") == 0
    capsys.readouterr()
    assert run(db, "brief", "--budget", "10") == 0
    out = capsys.readouterr().out
    assert "(memory is empty)" not in out and out.strip()
