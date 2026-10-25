"""a10 quick fixes: ``mark --version`` and the other-build server notice.

The notice ("server runs a different MARK build; running in-process") is
shown at most once per process, and only once the in-process store has
actually opened: when an older server holds the store and the open is
refused (StoreInUseError), only the refusal is shown.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import mark._cli.client as client
from mark import cli
from mark._version import __version__
from mark.errors import StoreInUseError


def _other_build_server(monkeypatch, tmp_path: Path) -> Path:
    """A state file whose server answers the handshake with another build's fingerprint."""
    db = tmp_path / "m.db"
    db.touch()  # the store exists; only its server is from another build
    state = cli._state_path(db)
    state.write_text('{"host": "127.0.0.1", "port": 1, "token": "t", "pid": 1}')

    def fake_call(host, port, method, path, *a, **k):
        nonce = path.split("nonce=")[1]
        return 200, {"proof": client._proof("t", nonce), "protocol": "older-build"}

    import mark._cli.wire as wire

    monkeypatch.setattr(wire, "call_json", fake_call)
    monkeypatch.setattr(client, "_OTHER_BUILD", {"seen": False, "shown": False})
    return db


def test_version_flag_prints_package_version() -> None:
    proc = subprocess.run([sys.executable, "-m", "mark.cli", "--version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"mark-sdk {__version__}"


def test_version_flag_in_process(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"mark-sdk {__version__}"


def test_probe_alone_prints_nothing(tmp_path, monkeypatch, capsys) -> None:
    db = _other_build_server(monkeypatch, tmp_path)
    from mark._cli import daemon

    for _ in range(9):  # what `serve --detach` does while it waits
        assert daemon.status(db)[0] is False
    assert capsys.readouterr().err == ""


def test_refused_open_shows_only_the_refusal(tmp_path, monkeypatch, capsys) -> None:
    db = _other_build_server(monkeypatch, tmp_path)

    def refuse(_db):
        raise StoreInUseError(str(_db), 4242)

    monkeypatch.setattr(client, "_Engine", refuse)
    assert cli.main(["--db", str(db), "stats"]) == 1
    err = capsys.readouterr().err
    assert "different MARK build" not in err
    assert "being served by a MARK server" in err


def test_notice_is_shown_once_when_running_in_process(tmp_path, monkeypatch, capsys) -> None:
    db = _other_build_server(monkeypatch, tmp_path)
    for _ in range(3):
        assert cli.execute(db, "stats", {"agent": "x"})["memories"] == 0
    assert capsys.readouterr().err.count("different MARK build") == 1
