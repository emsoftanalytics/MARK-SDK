# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Find and restore the backups taken before upgrades (``mark backups``).

``restore_backup`` writes a backup's content into the live database with
SQLite's backup API rather than copying the file over it: SQLite takes its
own locks and goes through the WAL, so a process that still has the store
open sees a consistent database (the restored one) instead of a file swapped
under it, and no ``-wal``/``-shm`` file is ever deleted under a reader. The
result is checked: same key-table row counts as the backup, and an
integrity check (quick check above 200 MB).
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mark.store._sqlite import upgrade_backup as ub

# <db>.v5-backup-20261025T101500Z.db, <db>.pre-restore-20261025T101500Z-1.db
_NAME = re.compile(r"\.(v(?P<v>\d+)-backup|pre-restore)-(?P<ts>\d{8}T\d{6}Z)(?:-\d+)?\.db$")


@dataclass
class BackupInfo:
    """One backup file found next to a store (or in a backup folder)."""

    path: Path
    kind: str  # "upgrade" or "pre-restore"
    schema: int | None
    bytes: int
    taken: datetime


def read_schema(path: str | Path) -> int | None:
    """The schema stamp of a database file, read-only; None when it has none or is unreadable."""
    try:
        conn = sqlite3.connect(ub.read_only_uri(path), uri=True, timeout=5.0)
        try:
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return None if row is None else int(row[0])


def list_backups(db_path: str | Path, folders: list[str | Path] = ()) -> list[BackupInfo]:
    """Backups of *db_path* in its folder, *folders* and the last upgrade's folder; oldest first."""
    db = Path(db_path).expanduser().resolve()
    places = {db.parent, *(Path(f).expanduser().resolve() for f in folders)}
    last = _last_backup(db)
    if last is not None:
        places.add(last.parent)
    found: dict[Path, BackupInfo] = {}
    for place in places:
        for path in place.glob(f"{db.name}.*.db"):
            match = _NAME.search(path.name[len(db.name):])
            if match is None or not path.name.startswith(db.name + "."):
                continue
            taken = datetime.strptime(match["ts"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            kind = "upgrade" if match["v"] else "pre-restore"
            found[path] = BackupInfo(path, kind, read_schema(path), path.stat().st_size, taken)
    return sorted(found.values(), key=lambda b: (b.taken, b.path.name))


def last_upgrade(db_path: str | Path) -> dict | None:
    """The ``last_upgrade`` record stored in the store, if any."""
    try:
        conn = sqlite3.connect(ub.read_only_uri(db_path), uri=True, timeout=5.0)
        try:
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'last_upgrade'").fetchone()
        finally:
            conn.close()
        return None if row is None else json.loads(str(row[0]))
    except (sqlite3.Error, ValueError):
        return None


def _last_backup(db: Path) -> Path | None:
    record = last_upgrade(db) if db.exists() else None
    path = (record or {}).get("backup")
    return Path(path) if path else None


def check_backup(path: str | Path) -> tuple[int | None, dict[str, int]]:
    """(schema, counts) of a backup file; raises ValueError when it is not a healthy MARK store."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    try:
        conn = sqlite3.connect(ub.read_only_uri(path), uri=True, timeout=5.0)
        try:
            _check, problems = ub.health_check(conn, path.stat().st_size)
            counts = ub.table_counts(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ValueError(f"{path} is not a readable SQLite database: {exc}") from exc
    schema = read_schema(path)
    if problems or schema is None or "fragments" not in counts:
        reason = "; ".join(problems) or "it has no MARK schema stamp or memories table"
        raise ValueError(f"{path} is not a healthy MARK store backup: {reason}")
    return schema, counts


def restore_backup(db_path: str | Path, backup_path: str | Path) -> dict[str, int]:
    """Replace the content of *db_path* with *backup_path*'s; returns the verified counts."""
    _schema, expected = check_backup(backup_path)
    src = sqlite3.connect(ub.read_only_uri(backup_path), uri=True, timeout=5.0)
    try:
        dst = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            src.backup(dst)
            dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # the restored pages into the main file
            found = ub.table_counts(dst)
            _check, problems = ub.health_check(dst, ub._size(Path(db_path)))
        finally:
            dst.close()
    finally:
        src.close()
    differences = ub.count_differences(expected, found, "backup", "restored store") + problems
    if differences:
        raise RuntimeError("the restored store does not match the backup: " + "; ".join(differences))
    return found
