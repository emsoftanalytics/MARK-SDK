# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The `mark` CLI: in-process round trips, recall_many input formats, server
lifecycle/auth/stale state, loopback-only serving, and lightweight imports.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mark import cli


def _run(capsys, *argv: str) -> str:
    assert cli.main(list(argv)) == 0
    return capsys.readouterr().out


def test_in_process_round_trip(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "m.db")
    log = tmp_path / "log.txt"
    log.write_text("[2026-03-01] atlas-core listens on port 6430.\n\n[2026-03-30] atlas-core moved to port 4444.\n")

    assert "stored 2 memories" in _run(capsys, "--db", db, "--no-server", "ingest", str(log))
    _run(capsys, "--db", db, "--no-server", "fact", "set", "atlas-core", "port", "4444", "--valid-from", "2026-03-30")
    out = _run(capsys, "--db", db, "--no-server", "recall", "Which port does atlas-core listen on now?", "-k", "2")
    assert out.splitlines()[0].startswith(" F. atlas-core port = 4444")
    assert "port 4444" in out.splitlines()[1]
    data = json.loads(_run(capsys, "--db", db, "--no-server", "recall", "Who is Marcus Webb?", "--json"))
    assert data["gap"] == "high" and data["missing"] == ["marcus webb"]
    assert "2 memories, 1 current facts" in _run(capsys, "--db", db, "--no-server", "stats")


def test_recall_many_accepts_lines_and_json(tmp_path: Path, capsys, monkeypatch) -> None:
    db = str(tmp_path / "m.db")
    _run(capsys, "--db", db, "--no-server", "store", "The guest Wi-Fi is halcyon-guest-7.")
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"id": "a", "question": "guest Wi-Fi name?"}, "unrelated zebra?"]))
    out = json.loads(_run(capsys, "--db", db, "--no-server", "recall-many", str(questions), "--json"))
    assert [r["query"] for r in out["results"]] == ["guest Wi-Fi name?", "unrelated zebra?"]
    assert "halcyon-guest-7" in out["results"][0]["memories"][0]["content"]


def test_server_lifecycle_auth_and_stale_state(tmp_path: Path) -> None:
    db = (tmp_path / "s.db").resolve()
    stop = threading.Event()
    thread = threading.Thread(target=cli.serve, args=(db,), kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    state = cli._state_path(db)
    for _ in range(100):
        if cli._live_server(state):
            break
        time.sleep(0.05)
    server = cli._live_server(state)
    assert server is not None
    assert oct(state.stat().st_mode & 0o777) == "0o600"

    assert cli.execute(db, "store", {"agent": "x", "text": "served memory about kestrel-core"})["id"]
    result = cli.execute(db, "recall", {"agent": "x", "query": "kestrel-core"})
    assert result["memories"][0]["content"] == "served memory about kestrel-core"
    with pytest.raises(RuntimeError, match="unauthorized"):
        cli._call_server({**server, "token": "wrong"}, "stats", {"agent": "x"})
    with pytest.raises(RuntimeError, match="unknown command"):
        cli._call_server(server, "drop_tables", {})

    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive() and not state.exists()

    # A stale state file (server gone) falls back to in-process execution.
    state.write_text(json.dumps({**server, "port": 1}))
    assert cli.execute(db, "stats", {"agent": "x"})["memories"] == 1


def test_serve_refuses_non_loopback(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.serve(tmp_path / "x.db", host="0.0.0.0")


def test_cli_client_does_not_load_the_runtime_when_served(tmp_path: Path) -> None:
    db = (tmp_path / "c.db").resolve()
    stop = threading.Event()
    thread = threading.Thread(target=cli.serve, args=(db,), kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    for _ in range(100):
        if cli._live_server(cli._state_path(db)):
            break
        time.sleep(0.05)
    code = (
        "import sys; from mark import cli; "
        f"cli.main(['--db', {str(db)!r}, 'stats']); "
        "print('LOADED' if 'mark.memory.runtime' in sys.modules else 'LIGHT')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    stop.set()
    thread.join(timeout=10)
    assert out.strip().splitlines()[-1] == "LIGHT"


def test_import_mark_is_lightweight() -> None:
    code = "import sys, mark; print(sorted(m for m in sys.modules if m.startswith('mark')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "['mark', 'mark._version']"


def test_mismatched_server_build_falls_back_in_process(tmp_path: Path, monkeypatch, capsys) -> None:
    """A client from a different build must not send arguments the server can't take."""
    db = (tmp_path / "skew.db").resolve()
    stop = threading.Event()
    threading.Thread(target=cli.serve, args=(db,), kwargs={"stop_event": stop}, daemon=True).start()
    for _ in range(100):
        if cli._live_server(cli._state_path(db)):
            break
        time.sleep(0.05)
    try:
        import mark._cli.client as client

        monkeypatch.setattr(client, "protocol_fingerprint", lambda: "other-build")
        assert cli._live_server(cli._state_path(db)) is None
        assert "different MARK build" in capsys.readouterr().err
        assert cli.execute(db, "stats", {"agent": "x"})["memories"] == 0  # in-process
    finally:
        stop.set()
