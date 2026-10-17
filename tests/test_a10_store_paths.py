# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""One store-path rule for Mark.local, the ``mark`` CLI and the MCP launcher (a10 blocker B1).

* a path ending in ``.mark`` (or an existing folder holding ``memory.db``) is the store folder;
* any other folder is a project folder: its store is ``<dir>/.mark/memory.db``;
* a ``.db``/``.sqlite`` path (or an existing file) is the database file itself.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from mark.cli import main
from mark.runtime import Mark
from mark.runtime_data import resolve_store_location, store_db_file


@pytest.mark.parametrize("given,expected", [
    (".mark", ".mark/memory.db"),
    ("proj/.mark", "proj/.mark/memory.db"),
    ("proj", "proj/.mark/memory.db"),
    ("notes.db", "notes.db"),
    ("data/notes.sqlite", "data/notes.sqlite"),
])
def test_missing_paths_are_decided_by_their_suffix(tmp_path: Path, given: str, expected: str) -> None:
    assert store_db_file(tmp_path / given) == (tmp_path / expected).resolve()


def test_existing_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    assert store_db_file(project) == project.resolve() / ".mark" / "memory.db"
    legacy = tmp_path / "store"                     # a folder that already holds a store
    legacy.mkdir()
    (legacy / "memory.db").write_bytes(b"")
    assert store_db_file(legacy) == legacy.resolve() / "memory.db"
    custom = tmp_path / "custom.bin"                # any existing file is the database
    custom.write_bytes(b"")
    assert store_db_file(custom) == custom.resolve()


def test_mark_local_dot_mark_opens_dot_mark_memory_db(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with Mark.local("./.mark") as mark:
        assert mark.db_path == (tmp_path / ".mark" / "memory.db").resolve()
        assert mark.config.project_path == tmp_path.resolve()
        assert mark.config.store_path == (tmp_path / ".mark").resolve()
    assert not (tmp_path / ".mark" / ".mark").exists()
    project, mark_dir, db = resolve_store_location(tmp_path, None)
    assert (project, mark_dir, db) == (tmp_path.resolve(), (tmp_path / ".mark").resolve(),
                                       (tmp_path / ".mark" / "memory.db").resolve())


def test_python_cli_and_mcp_share_one_store(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARK_DB", raising=False)
    with Mark.local("./.mark") as mark:
        mark.memory.remember("We chose SQLite for storage.", kind="decision")
    for db in (".mark", "./.mark/memory.db", "."):
        assert main(["--no-server", "--db", db, "list", "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert [m["content"] for m in listed["memories"]] == ["We chose SQLite for storage."], db
    assert main(["--no-server", "list", "--json"]) == 0           # the default store
    assert json.loads(capsys.readouterr().out)["total"] == 1

    pytest.importorskip("mcp", reason="mcp extra not installed")
    from mark.adapters.mcp.launcher import build_server

    server, runtime = build_server(".mark")
    try:
        result = asyncio.run(server.call_tool("mark_retrieve", {"query": "which database storage"}))
        text = "\n".join(getattr(part, "text", "") for part in getattr(result, "content", result))
        assert "SQLite" in text
    finally:
        runtime.shutdown()


def test_mcp_launcher_never_raises_not_a_directory(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from mark.adapters.mcp.launcher import build_server, resolve_db

    assert resolve_db(tmp_path / "new-project") == (tmp_path / "new-project" / ".mark" / "memory.db").resolve()
    server, runtime = build_server(tmp_path / "new-project")
    runtime.shutdown()
    assert (tmp_path / "new-project" / ".mark" / "memory.db").exists()
    assert os.path.isdir(tmp_path / "new-project" / ".mark")
