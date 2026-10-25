# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""The verified safety backup taken before a store is upgraded to a newer schema.

``create_backup`` copies the store into ``<db>.v<N>-backup-<UTC time>.db``
(next to it, or in ``backup_dir``) and checks the copy before any migration
runs. Steps, in order, each failing with a plain-words error that leaves the
store untouched:

1. free space: ``(store + WAL) * 1.10`` (and at least 1 MB extra) must be free
   in the target folder (:class:`~mark.errors.InsufficientSpaceError`);
2. copy: a passive WAL checkpoint, then SQLite's online backup API in one
   step into ``<name>.partial`` created 0600 with ``O_EXCL``; the copy is
   switched to a self-contained rollback-journal file and fsynced
   (:class:`~mark.errors.BackupWriteError`);
3. verify: ``COUNT(*)`` of every key table must equal the store's, and
   ``PRAGMA integrity_check`` (``quick_check`` above 200 MB) must say ok
   (:class:`~mark.errors.BackupVerificationError`); only then is the file
   linked to its final name, which never replaces an existing file.

Why the backup API and not ``VACUUM INTO``: it copies pages as they are (as
fast as the disk), while VACUUM INTO rebuilds every table and index; on a
1M-memory store it was several times slower (benchmarks/scale/upgrade_bench.py).
The caller must hold the store's write lock, so nothing changes between the
counts and the copy. Encrypted stores are copied as they are: the backup
holds the same ciphertext and needs the same key.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mark._private_fs import FILE_MODE, ensure_private_dir
from mark.errors import BackupVerificationError, BackupWriteError, InsufficientSpaceError
from mark.store._sqlite import upgrade_messages as messages

#: Tables whose rows are the user's data: a backup or a migration must keep every one.
KEY_TABLES = ("fragments", "facts", "documents", "nodes", "edges", "blocks", "block_nodes",
              "block_links", "session_events", "fact_extractions")
MARGIN = 0.10
MIN_MARGIN = 1_000_000
#: Above this size the cheaper ``quick_check`` replaces the full ``integrity_check``.
FULL_CHECK_LIMIT = 200_000_000


@dataclass
class SpaceCheck:
    """Free space in the backup folder versus what the backup needs."""

    db_name: str
    directory: str
    mount: str
    store_bytes: int
    wal_bytes: int
    required_bytes: int
    free_bytes: int
    margin_percent: int = int(MARGIN * 100)

    @property
    def ok(self) -> bool:
        return self.free_bytes >= self.required_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_bytes - self.free_bytes)


@dataclass
class Backup:
    """A verified backup file and how it was made."""

    path: str
    version: int
    bytes: int
    store_bytes: int
    wal_bytes: int
    seconds: float
    verify_seconds: float
    check: str
    checkpointed: bool
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Bytes copied per second."""
        return self.bytes / self.seconds if self.seconds > 0 else float(self.bytes)


def target_dir(db_path: str | Path, backup_dir: str | Path | None) -> Path:
    return Path(backup_dir).expanduser() if backup_dir else Path(db_path).expanduser().resolve().parent


def space_check(db_path: str | Path, backup_dir: str | Path | None = None) -> SpaceCheck:
    """How much the backup needs and how much is free where it would go."""
    db = Path(db_path)
    store, wal = _size(db), _size(Path(f"{db}-wal"))
    folder = target_dir(db, backup_dir)
    probe = folder
    while not probe.exists() and probe != probe.parent:  # a backup_dir not created yet
        probe = probe.parent
    required = int((store + wal) * (1 + MARGIN)) + MIN_MARGIN
    free = shutil.disk_usage(probe).free
    return SpaceCheck(db.name, str(folder), _mount(probe), store, wal, required, free)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_name(db_path: str | Path, label: str, folder: Path) -> Path:
    """The first free ``<db>.<label>-<time>[-N].db`` in *folder* (nothing is created)."""
    base = f"{Path(db_path).name}.{label}-{_timestamp()}"
    for n in range(1000):
        path = folder / (f"{base}.db" if n == 0 else f"{base}-{n}.db")
        if not path.exists() and not Path(f"{path}.partial").exists():
            return path
    raise FileExistsError(f"no free backup name for {base} in {folder}")


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """``COUNT(*)`` of each key table present in the database *conn* is open on."""
    present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return {t: int(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in KEY_TABLES if t in present}


def health_check(conn: sqlite3.Connection, size: int) -> tuple[str, list[str]]:
    """(check name, problems): full integrity_check for small files, quick_check for big ones."""
    pragma = "integrity_check" if size < FULL_CHECK_LIMIT else "quick_check"
    rows = [str(row[0]) for row in conn.execute(f"PRAGMA {pragma}(20)")]
    return pragma, ([] if rows == ["ok"] else [f"{pragma}: {row}" for row in rows])


def ordinary_tables(conn: sqlite3.Connection) -> set[str]:
    """Names of the ordinary (not virtual) tables; FTS shadow tables are ordinary."""
    return {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'")}


def check_tables(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
    """``quick_check`` of only *tables* (and their indexes): cheap after a migration.

    Why only these: the migration's own changes are the new tables; the
    existing data is covered by the row counts, and its structure was just
    checked on the backup, so a second full scan would only cost time.
    Needs SQLite 3.33+ (the table argument); older versions skip it.
    """
    if sqlite3.sqlite_version_info < (3, 33, 0):
        return []
    problems: list[str] = []
    for table in tables:
        rows = [str(row[0]) for row in conn.execute(f'PRAGMA quick_check("{table}")')]
        problems += [] if rows == ["ok"] else [f"quick_check {table}: {row}" for row in rows]
    return problems


def count_differences(expected: dict[str, int], found: dict[str, int], first: str, second: str) -> list[str]:
    return [f"{t}: {first} {n:,} rows, {second} {found.get(t, 0):,}"
            for t, n in expected.items() if found.get(t, 0) != n]


def create_backup(db_path: str | Path, version: int, *, backup_dir: str | Path | None = None,
                  label: str | None = None, counts: dict[str, int] | None = None,
                  purpose: str | None = None) -> Backup:
    """Copy and verify the store (see the module docstring); the caller holds the write lock.

    *label* names the file (default ``v<version>-backup``); *purpose* completes
    the space message ("... back up memory.db <purpose>: ...").
    """
    db = str(db_path)
    space = space_check(db, backup_dir)
    if not space.ok:
        raise InsufficientSpaceError(
            messages.no_space(space, purpose or f"before upgrading it from schema {version} to "
                                                f"{max(messages.RELEASES)}"), path=db,
            store_bytes=space.store_bytes, wal_bytes=space.wal_bytes, required_bytes=space.required_bytes,
            free_bytes=space.free_bytes, shortfall_bytes=space.shortfall_bytes, directory=space.directory,
            mount=space.mount)
    folder = target_dir(db, backup_dir)
    final = folder / f"{Path(db).name}.{label or f'v{version}-backup'}-pending.db"  # for messages only
    try:
        ensure_private_dir(folder)  # 0700 when created here; an existing folder is left alone
        final = backup_name(db, label or f"v{version}-backup", folder)
        partial = Path(f"{final}.partial")
        os.close(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE))
    except OSError as exc:
        raise _write_error(db, str(final), exc) from exc
    try:
        started = time.perf_counter()
        checkpointed, expected = _copy(db, partial, counts)
        copied = time.perf_counter()
        size = _size(partial)
        check, problems, found = _verify(partial, size)
    except (OSError, sqlite3.Error) as exc:
        _remove(partial)
        raise _write_error(db, str(final), exc) from exc
    differences = count_differences(expected, found, "store", "backup") + problems
    if differences:
        _remove(partial)
        raise BackupVerificationError(messages.verify_failed(Path(db).name, str(final), differences),
                                      path=db, backup_path=str(final), differences=differences)
    try:
        _publish(partial, final)
    except OSError as exc:
        _remove(partial)
        raise _write_error(db, str(final), exc) from exc
    return Backup(str(final), version, size, _size(Path(db)), space.wal_bytes, copied - started,
                  time.perf_counter() - copied, check, checkpointed, expected)


def _copy(db: str, partial: Path, counts: dict[str, int] | None) -> tuple[bool, dict[str, int]]:
    """Checkpoint, count and copy; returns (WAL fully checkpointed?, the store's counts)."""
    src = sqlite3.connect(db, timeout=5.0)
    try:
        busy, _log, _done = src.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        expected = dict(counts) if counts is not None else table_counts(src)
        dst = sqlite3.connect(partial)
        try:
            dst.execute("PRAGMA synchronous=OFF")  # one fsync at the end instead (below)
            src.backup(dst)  # pages=-1: the whole store in one step
            dst.execute("PRAGMA journal_mode=DELETE")  # one self-contained file, no -wal/-shm
        finally:
            dst.close()
    finally:
        src.close()
    fd = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return busy == 0, expected


def _verify(path: Path, size: int) -> tuple[str, list[str], dict[str, int]]:
    conn = sqlite3.connect(read_only_uri(path), uri=True)
    try:
        check, problems = health_check(conn, size)
        return check, problems, table_counts(conn)
    finally:
        conn.close()


def _publish(partial: Path, final: Path) -> None:
    """Give the verified copy its final name without ever replacing an existing file."""
    try:
        os.link(partial, final)  # fails with FileExistsError instead of overwriting
    except (AttributeError, NotImplementedError, PermissionError):  # no hard links (some filesystems)
        if final.exists():
            raise FileExistsError(17, "File exists", str(final)) from None
        os.rename(partial, final)
        return
    os.unlink(partial)


def _write_error(db: str, final: str, exc: BaseException) -> BackupWriteError:
    return BackupWriteError(messages.write_failed(Path(db).name, final, exc), path=db, backup_path=final,
                            errno=getattr(exc, "errno", None), strerror=getattr(exc, "strerror", None) or str(exc))


def _remove(path: Path) -> None:
    for item in (path, Path(f"{path}-journal"), Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            item.unlink()
        except OSError:
            pass


def read_only_uri(path: str | Path) -> str:
    """A ``mode=ro`` URI for *path* (percent-encoded: spaces, ``?`` and ``#`` are safe)."""
    return Path(path).expanduser().resolve().as_uri() + "?mode=ro"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mount(path: Path) -> str:
    """The mount point holding *path* (what ``df`` names), for messages."""
    path = path.resolve()
    while not os.path.ismount(path) and path != path.parent:
        path = path.parent
    return str(path)
