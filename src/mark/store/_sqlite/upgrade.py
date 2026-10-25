# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Safe schema upgrades: guards, a verified backup first, checks after, undo on failure.

``_init_schema`` (see :mod:`mark.store._sqlite.schema`) calls these hooks:

* before its transaction: ``_refuse_newer_schema`` (never touch a store
  from a newer MARK) and ``_refuse_migration_under_server`` (never migrate
  under an older ``mark serve``);
* right after ``BEGIN IMMEDIATE``, before any DDL: ``_upgrade_begin``. When
  a file store's schema is older than this build (or it is pre-versioned and
  holds data) it counts the key tables and, unless switched off, takes the
  verified backup of :mod:`~mark.store._sqlite.upgrade_backup`. Holding the
  write lock means the backup is exactly what gets migrated;
* at the end of the transaction: ``_upgrade_verify`` (no key table lost
  rows, the tables it created pass quick_check, ``last_upgrade`` recorded);
* after the commit: ``_upgrade_confirm`` (counts re-read, success logged);
* on any failure: ``_upgrade_failed``. The transaction was rolled back; if
  the upgrade had already been committed (the post-commit check failed),
  the store is restored from the backup through
  SQLite's backup API, which is safe with other open connections (no file
  swap, no deleting -wal/-shm under a live reader). Raises
  :class:`~mark.errors.UpgradeFailedError` naming the backup.

Backups are skipped for in-memory stores, with ``backup_on_upgrade=False``
or with ``MARK_BACKUP_ON_UPGRADE=0``; verification and rollback still apply.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mark.errors import UpgradeBackupError, UpgradeFailedError
from mark.store._sqlite import upgrade_backup
from mark.store._sqlite import upgrade_messages as messages

logger = logging.getLogger("mark.store.sqlite")
ENV_VAR = "MARK_BACKUP_ON_UPGRADE"
_OFF = {"0", "false", "no", "off"}


def backups_enabled(flag: bool = True) -> bool:
    """*flag*, unless ``MARK_BACKUP_ON_UPGRADE`` switches backups off."""
    return bool(flag) and os.environ.get(ENV_VAR, "1").strip().lower() not in _OFF


@dataclass
class UpgradePlan:
    """One upgrade in progress: where it starts, what it must keep, and its backup."""

    from_version: int
    counts: dict[str, int]
    backup: upgrade_backup.Backup | None
    started: float  # before the backup: the upgrade's total time starts here
    tables: set[str] = field(default_factory=set)  # ordinary tables before migrating
    migrate_from: float = field(default_factory=time.perf_counter)  # backup done, DDL next
    migrate_seconds: float = 0.0
    committed: bool = False  # set once the migration transaction has been saved


class _VerificationFailed(RuntimeError):
    """The migrated store failed its checks; the transaction is rolled back."""


class UpgradeMixin:
    """Backup-first, verified, undoable schema upgrades for file stores."""

    db_path: str
    CURRENT_SCHEMA_VERSION: int
    #: Class defaults, so stores built with ``__new__`` (tests) still work.
    backup_on_upgrade: bool = True
    backup_dir: str | Path | None = None
    #: Set after an upgrade by this process: the ``last_upgrade`` record.
    last_upgrade: dict | None = None
    _upgrade_plan: UpgradePlan | None = None

    def _refuse_newer_schema(self) -> None:
        """Raise StoreVersionError if the store was written by a newer MARK."""
        from mark.errors import StoreVersionError

        with self._reading() as conn:
            try:
                row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
            except sqlite3.OperationalError:  # new database: no schema_info yet
                return
        if row is not None and int(row[0]) > self.CURRENT_SCHEMA_VERSION:
            self.close()
            raise StoreVersionError(str(self.db_path), int(row[0]), self.CURRENT_SCHEMA_VERSION)

    def _refuse_migration_under_server(self) -> None:
        """Raise StoreInUseError if a migration is due while another process serves the store.

        See :mod:`mark.store._sqlite.server_guard`: an older ``mark serve``
        would keep writing into the migrated layout with its old code.
        """
        from mark.store._sqlite.server_guard import StoreInUseError, foreign_server

        if self.db_path == ":memory:":
            return
        try:
            row = self._fetchone("SELECT value FROM schema_info WHERE key = 'schema_version'")
        except sqlite3.OperationalError:  # new database: nothing to migrate
            return
        if row is None or int(row[0]) >= self.CURRENT_SCHEMA_VERSION:
            return
        held, pid = foreign_server(self.db_path)
        if held:
            self.close()
            raise StoreInUseError(self.db_path, pid)

    def _upgrade_begin(self, conn: sqlite3.Connection) -> UpgradePlan | None:
        """Inside the write transaction, before any DDL: plan (and back up for) a due upgrade."""
        if self.db_path == ":memory:":
            return None
        started = time.perf_counter()
        try:
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
        except sqlite3.OperationalError:  # no schema_info: brand-new or pre-versioned
            row = None
        version = 0 if row is None else int(row[0])
        if version >= self.CURRENT_SCHEMA_VERSION:
            return None
        counts = upgrade_backup.table_counts(conn)
        if row is None and not any(counts.values()):
            return None  # a new, empty database: nothing to upgrade or keep
        backup = None
        if backups_enabled(self.backup_on_upgrade):
            backup = upgrade_backup.create_backup(self.db_path, version, backup_dir=self.backup_dir,
                                                  counts=counts)
        return UpgradePlan(version, counts, backup, started,
                           upgrade_backup.ordinary_tables(conn))

    def _upgrade_verify(self, conn: sqlite3.Connection, plan: UpgradePlan) -> None:
        """End of the transaction: no key table lost rows, new tables sound; record the upgrade."""
        after = upgrade_backup.table_counts(conn)
        lost = [f"{t}: {n:,} rows before, {after.get(t, 0):,} after"
                for t, n in plan.counts.items() if after.get(t, 0) < n]
        # Only what the migration created gets a structure check (see check_tables).
        problems = upgrade_backup.check_tables(conn, sorted(upgrade_backup.ordinary_tables(conn) - plan.tables))
        if lost or problems:
            raise _VerificationFailed("the upgraded store did not pass its checks: " + "; ".join(lost + problems))
        plan.migrate_seconds = time.perf_counter() - plan.migrate_from  # migrations + checks
        backup = plan.backup
        record = {
            "from_schema": plan.from_version, "to_schema": self.CURRENT_SCHEMA_VERSION,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backup": backup.path if backup else None, "backup_bytes": backup.bytes if backup else None,
            "store_bytes": backup.store_bytes if backup else None,
            "backup_seconds": round(backup.seconds, 3) if backup else None,
            "verify_seconds": round(backup.verify_seconds, 3) if backup else None,
            "migrate_seconds": round(plan.migrate_seconds, 3),
            "check": backup.check if backup else None, "checkpointed": backup.checkpointed if backup else None,
        }
        sql = "INSERT INTO schema_info (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        conn.execute(sql, ("last_upgrade", json.dumps(record)))
        if backup is not None:
            conn.execute(sql, ("last_upgrade_backup", backup.path))
        self.last_upgrade = record

    def _upgrade_confirm(self, plan: UpgradePlan) -> None:
        """After the commit: re-read the counts (cheap) and log the one success line."""
        with self._reading() as conn:
            after = upgrade_backup.table_counts(conn)
        lost = [f"{t}: {n:,} rows before, {after.get(t, 0):,} after"
                for t, n in plan.counts.items() if after.get(t, 0) < n]
        if lost:
            raise _VerificationFailed("the saved upgrade lost rows: " + "; ".join(lost))
        total = time.perf_counter() - plan.started
        text = messages.success(self.db_path, plan.from_version, self.CURRENT_SCHEMA_VERSION, total, plan)
        if self.last_upgrade is not None:
            self.last_upgrade.update(seconds=round(total, 3), message=text)
        logger.warning("%s", text)  # once per upgrade: says where the full copy of the data now lives

    def _upgrade_failed(self, plan: UpgradePlan | None, exc: BaseException) -> BaseException | None:
        """After a failed open: close, undo a committed upgrade; return the error to raise (None: re-raise)."""
        if plan is None:
            if isinstance(exc, UpgradeBackupError):
                self.close()  # the backup failed before any change: release the store
            return None
        self.close()
        backup = plan.backup.path if plan.backup else None
        # Known, not guessed from the file: a stamp read could see another process's upgrade.
        committed = plan.committed
        restored, restore_error = False, None
        if committed and backup:
            try:
                from mark.store._sqlite.upgrade_restore import restore_backup

                restore_backup(self.db_path, backup)
                restored = True
            except Exception as error:  # noqa: BLE001 - reported in the message below
                restore_error = str(error)
        if not isinstance(exc, Exception):
            return None  # KeyboardInterrupt and friends: undone above, re-raised as they are
        reason = str(exc) or type(exc).__name__
        text = messages.upgrade_failed(Path(self.db_path).name, plan.from_version, self.CURRENT_SCHEMA_VERSION,
                                       reason, backup=backup, committed=committed, restored=restored,
                                       restore_error=restore_error)
        return UpgradeFailedError(text, path=self.db_path, store_changed=committed and not restored,
                                  backup_path=backup, from_version=plan.from_version,
                                  to_version=self.CURRENT_SCHEMA_VERSION, restored=restored, reason=reason)

