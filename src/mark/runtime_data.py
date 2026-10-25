# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Store-wide data management for :class:`mark.runtime.Mark`.

``Mark`` mixes in :class:`MarkDataMixin` for the operations that span every
agent namespace in one store — listing namespaces, whole-store backup and
restore, vacuum, and reporting the active embedder — plus
:func:`store_db_file` / :func:`resolve_store_location`, the single
store-path rule shared by ``Mark.local``, ``mark --db`` and ``mark mcp --db``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.embeddings.info import EmbedderInfo
    from mark.memory.mark_memory import MarkMemory

#: File suffixes that mark a path as a database file rather than a folder.
DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")
#: The database file inside a store folder.
DB_NAME = "memory.db"
#: Folder suffix that marks a path as the store folder itself (``./.mark``).
STORE_DIR_SUFFIX = ".mark"


def store_db_file(path: str | Path) -> Path:
    """The SQLite file a store path names: the one rule shared by Python, CLI and MCP.

    * an existing file, or a path ending in .db/.sqlite/.sqlite3, is the file;
    * a folder whose name ends in ``.mark``, or an existing folder that
      already holds ``memory.db``, is the store folder: ``<dir>/memory.db``;
    * any other folder (existing or not) is a project folder:
      ``<dir>/.mark/memory.db``.

    Missing paths are decided by their name alone, so a new path never fails
    with NotADirectoryError. Standard library only (the CLI calls it).
    """
    target = Path(path).expanduser().resolve()
    if target.is_file():
        return target
    if not target.is_dir() and target.suffix.lower() in DB_SUFFIXES:
        return target
    # Why both tests: `Mark.local("./.mark")` once created ./.mark/.mark/memory.db
    # while `mark --db .mark` opened ./.mark/memory.db (two stores, blocker B1).
    if target.name.endswith(STORE_DIR_SUFFIX) or (target / DB_NAME).is_file():
        return target / DB_NAME
    return target / STORE_DIR_SUFFIX / DB_NAME


def resolve_store_location(project_path: str | Path, store_path: str | Path | None) -> tuple[Path, Path, Path]:
    """Return ``(project_dir, mark_dir, db_file)`` for ``Mark.local`` arguments.

    ``project_path`` follows :func:`store_db_file`, so ``Mark.local(".")``,
    ``Mark.local("./.mark")``, ``mark --db .mark`` and ``mark mcp --db .mark``
    all open ``./.mark/memory.db``. ``store_path`` (explicit) is the store
    folder itself, or a database file.
    """
    project = Path(project_path).expanduser().resolve()
    if store_path is not None:
        store = Path(store_path).expanduser().resolve()
        if store.is_file() or (store.suffix.lower() in DB_SUFFIXES and not store.is_dir()):
            return project, store.parent, store
        return project, store, store / DB_NAME
    db = store_db_file(project)
    if db == project:                # a database file: its folder is the project
        return project.parent, project.parent, project
    if db.parent == project:         # the store folder itself
        return project.parent, project, db
    return project, db.parent, db


class MarkDataMixin:
    """agents / export_jsonl / import_jsonl / vacuum / embedder_info for a whole store."""

    _runtime_engine: Any

    def memory_for(self, agent_id: str) -> "MarkMemory":
        """The memory of one agent namespace (same verbs as ``mark.memory``)."""
        return self._runtime_engine.memory(agent_id)

    def agents(self) -> list[str]:
        """Every agent namespace in this store that holds memories or facts."""
        return self._runtime_engine.store.agent_ids()

    def export_jsonl(self, path: str | Path, *, agents: list[str] | None = None,
                     decrypt: bool = False) -> dict[str, int]:
        """Back up every namespace (or only *agents*) to a JSON Lines file; returns counts.

        Encrypted stores export plaintext only with ``decrypt=True``.
        """
        from mark.memory.facade.portability import export_jsonl

        store = self._runtime_engine.store
        return export_jsonl(store, path, agents if agents is not None else store.agent_ids(),
                            decrypt=decrypt)

    def import_jsonl(self, path: str | Path, *, agent_id: str | None = None) -> dict[str, int]:
        """Restore an export into the namespaces it came from (or all into *agent_id*)."""
        from mark.memory.facade.portability import import_jsonl

        return import_jsonl(path, self.memory_for, agent_id=agent_id)

    def vacuum(self) -> None:
        """Purge deleted content from the database file and its WAL."""
        self._runtime_engine.store.vacuum()

    def embedder_info(self) -> "EmbedderInfo":
        """The active embedder: name, dim, semantic, and how it was chosen (source)."""
        from mark.embeddings.info import embedder_info

        return embedder_info(self._runtime_engine.embedder)
