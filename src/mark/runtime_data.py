# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Store-wide data management for :class:`mark.runtime.Mark`.

``Mark`` mixes in :class:`MarkDataMixin` for the operations that span every
agent namespace in one store — listing namespaces, whole-store backup and
restore, vacuum, and reporting the active embedder — plus
:func:`resolve_store_location`, which lets ``Mark.local`` accept a folder or
a database file exactly like the ``mark --db`` command line.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.embeddings.info import EmbedderInfo
    from mark.memory.mark_memory import MarkMemory

#: File suffixes that mark a path as a database file rather than a folder.
DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")


def resolve_store_location(project_path: str | Path, store_path: str | Path | None) -> tuple[Path, Path, Path]:
    """Return ``(project_dir, mark_dir, db_file)`` for ``Mark.local`` arguments.

    Either argument may name a database file (an existing file, or a path
    ending in .db/.sqlite/.sqlite3): ``Mark.local("notes.db")`` and
    ``mark --db notes.db`` then open the same store. Otherwise the folder
    rules are unchanged: ``store_path`` is the ``.mark`` directory, else
    ``<project_path>/.mark``, holding ``memory.db``.
    """
    def is_db_file(path: Path) -> bool:
        return path.is_file() or (path.suffix.lower() in DB_SUFFIXES and not path.is_dir())

    project = Path(project_path).expanduser().resolve()
    if store_path is not None:
        store = Path(store_path).expanduser().resolve()
        if is_db_file(store):
            return project, store.parent, store
        return project, store, store / "memory.db"
    if is_db_file(project):
        return project.parent, project.parent, project
    return project, project / ".mark", project / ".mark" / "memory.db"


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
