# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""``mark migrate`` and ``mark backups``: dry run, upgrade, list, restore round trip."""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from mark.cli import main
from _helpers_upgrade import backups, build_store, counts, dump, stamp

Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def old(tmp_path: Path):
    db = tmp_path / "memory.db"
    before = build_store(db, version=5)
    return db, before, dump(db)


def test_dry_run_shows_the_plan_and_changes_nothing(old, capsys) -> None:
    db, _before, original = old
    assert main(["--db", str(db), "migrate", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "schema:  5 (mark-sdk 0.2.0a9) -> 6 (mark-sdk 0.2.0a10)" in out
    assert f"would create {db}.v5-backup-" in out
    assert "store " in out and "WAL " in out and "10% margin" in out and "-> PASS" in out
    assert "server:  none" in out
    assert dump(db) == original and backups(db.parent) == []


def test_dry_run_reports_a_space_failure(old, capsys, monkeypatch) -> None:
    from mark.store._sqlite import upgrade_backup

    db, _before, _original = old
    monkeypatch.setattr(upgrade_backup.shutil, "disk_usage", lambda p: Usage(10**9, 10**9 - 10, 10))
    assert main(["--db", str(db), "migrate", "--dry-run"]) == 1
    assert "-> FAIL (short by" in capsys.readouterr().out


def test_migrate_upgrades_once_and_reports_sizes(old, capsys) -> None:
    db, before, _original = old
    assert main(["--db", str(db), "migrate"]) == 0
    out = capsys.readouterr().out
    (backup,) = backups(db.parent)
    assert f"upgraded {db}: schema 5 -> 6" in out and f"backup kept at {backup}" in out
    assert "backup (" in out and " = store " in out and "WAL checkpointed" in out and "migrate " in out
    assert stamp(db) == 6 and counts(db) == before
    assert main(["--db", str(db), "migrate"]) == 0
    assert "already current" in capsys.readouterr().out
    assert backups(db.parent) == [backup]


def test_migrate_options_and_failures_print_plain_messages(old, tmp_path: Path, capsys, monkeypatch) -> None:
    from mark.store._sqlite import upgrade_backup

    db, _before, original = old
    monkeypatch.setattr(upgrade_backup.shutil, "disk_usage", lambda p: Usage(10**9, 10**9 - 10, 10))
    assert main(["--db", str(db), "migrate"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("mark: Not enough free space") and "Traceback" not in err
    assert "Your store was not changed." in err and dump(db) == original
    monkeypatch.undo()
    target = tmp_path / "elsewhere"
    assert main(["--db", str(db), "migrate", "--backup-dir", str(target)]) == 0
    assert len(backups(target)) == 1 and backups(db.parent) == []


def test_migrate_without_backup(old, capsys) -> None:
    db, _before, _original = old
    assert main(["--db", str(db), "migrate", "--no-backup"]) == 0
    assert "no backup taken" in capsys.readouterr().out
    assert backups(db.parent) == [] and stamp(db) == 6


def test_backups_list_and_restore_round_trip(old, capsys) -> None:
    db, before, original = old
    assert main(["--db", str(db), "backups"]) == 0
    assert "no backups of memory.db" in capsys.readouterr().out
    assert main(["--db", str(db), "migrate"]) == 0
    capsys.readouterr()
    (backup,) = backups(db.parent)
    assert main(["--db", str(db), "backups"]) == 0
    out = capsys.readouterr().out
    assert str(backup) in out and "schema 5 (mark-sdk 0.2.0a9)" in out and "KB" in out and "UTC" in out
    assert "last upgrade: schema 5 -> 6" in out

    # No terminal and no --yes: refuses, changes nothing.
    assert main(["--db", str(db), "backups", "restore", str(backup)]) == 1
    assert stamp(db) == 6
    assert main(["--db", str(db), "backups", "restore", str(backup), "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"restored {db} from {backup}" in out and "pre-restore" in out
    assert stamp(db) == 5 and counts(db) == before and dump(db) == original
    (kept,) = db.parent.glob("memory.db.pre-restore-*.db")
    assert stamp(kept) == 6 and counts(kept) == before
    assert main(["--db", str(db), "backups"]) == 0
    assert "kept before a restore" in capsys.readouterr().out


def test_restore_refuses_while_a_server_holds_the_store(old, capsys, monkeypatch) -> None:
    from mark.store._sqlite import server_guard

    db, _before, _original = old
    assert main(["--db", str(db), "migrate"]) == 0
    (backup,) = backups(db.parent)
    monkeypatch.setattr(server_guard, "foreign_server", lambda path: (True, 4242))
    assert main(["--db", str(db), "backups", "restore", str(backup), "--yes"]) == 1
    assert "mark serve (pid 4242)" in capsys.readouterr().err
    assert stamp(db) == 6


def test_restore_refuses_a_file_that_is_not_a_store(tmp_path: Path, capsys) -> None:
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not sqlite at all" * 100)
    assert main(["--db", str(tmp_path / "memory.db"), "backups", "restore", str(junk), "--yes"]) == 1
    assert "not a readable SQLite database" in capsys.readouterr().err


def test_parser_stays_light() -> None:
    import subprocess
    import sys

    code = ("import sys; from mark._cli.parser import build_parser; build_parser(); "
            "print(sorted(m for m in sys.modules if m.startswith('mark.store')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_serve_engine_and_mcp_launcher_upgrade_with_a_backup(tmp_path: Path) -> None:
    from mark._cli.engine import _Engine

    served, mcp = tmp_path / "served" / "memory.db", tmp_path / "mcp" / "memory.db"
    served.parent.mkdir()
    mcp.parent.mkdir()
    build_store(served)
    build_store(mcp)
    engine = _Engine(served)  # what `mark serve` opens
    engine.close()
    assert stamp(served) == 6 and len(backups(served.parent)) == 1
    launcher = pytest.importorskip("mark.adapters.mcp.launcher")
    try:
        _server, runtime = launcher.build_server(mcp)
    except ImportError as exc:  # the mcp extra is not installed
        pytest.skip(str(exc))
    runtime.shutdown()
    assert stamp(mcp) == 6 and len(backups(mcp.parent)) == 1
