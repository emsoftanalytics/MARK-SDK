# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""``mark serve --detach`` / ``mark serve status`` (agent re-test K): an agent's shell
must not block on a foreground server."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from mark import cli


def test_detach_status_and_stop(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "d.db")
    assert cli.main(["--db", db, "--no-server", "remember", "Warm me up."]) == 0
    assert cli.main(["--db", db, "serve", "status"]) == 1
    assert "no mark serve running" in capsys.readouterr().out
    try:
        assert cli.main(["--db", db, "serve", "--detach"]) == 0
        out = capsys.readouterr().out
        assert "mark serve started" in out and "pid" in out
        assert cli.main(["--db", db, "serve", "status"]) == 0
        assert "mark serve running" in capsys.readouterr().out
        assert cli.main(["--db", db, "serve", "--detach"]) == 0          # idempotent
        assert "already running" in capsys.readouterr().out
        assert cli.main(["--db", db, "recall", "warm"]) == 0              # answered by the server
        assert "Warm me up." in capsys.readouterr().out
        log = Path(db + ".server.log")
        assert log.exists() and stat.S_IMODE(os.stat(log).st_mode) == 0o600
    finally:
        cli.main(["--db", db, "serve", "stop"])
    capsys.readouterr()
    assert cli.main(["--db", db, "serve", "status"]) == 1
