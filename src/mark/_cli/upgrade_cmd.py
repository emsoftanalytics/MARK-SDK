# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""``mark migrate`` and ``mark backups``: explicit upgrades and their backups.

* ``mark migrate [--dry-run] [--no-backup] [--backup-dir DIR]`` upgrades a
  store written by an older MARK now (any other command would do it on
  open, the same way); ``--dry-run`` prints the schemas, the backup it would
  create and the free-space math with PASS/FAIL, and changes nothing.
* ``mark backups`` lists the store's backups (schema, size, date) and its
  last upgrade; ``mark backups restore FILE [--yes]`` puts one back, after
  keeping the current store as ``<db>.pre-restore-<time>.db``. It refuses
  while a ``mark serve`` holds the store and asks first unless ``--yes``.

Importing this module is standard library only (the parser imports it on
every ``mark`` call); MARK's store modules load inside the commands.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

COMMANDS = ("migrate", "backups")


def add_parsers(sub: Any) -> None:
    """Register ``migrate`` and ``backups`` on the ``mark`` sub-parsers."""
    p = sub.add_parser("migrate", help="upgrade a store written by an older MARK (a verified backup first)")
    p.add_argument("--dry-run", action="store_true",
                   help="only show the schemas, the backup it would make and the free-space check")
    p.add_argument("--no-backup", action="store_true", help="upgrade without taking a backup first")
    p.add_argument("--backup-dir", help="put the backup in this folder (default: next to the store)")
    p = sub.add_parser("backups", help="list the store's upgrade backups; `mark backups restore FILE` restores one")
    p.add_argument("action", nargs="?", choices=("list", "restore"), default="list")
    p.add_argument("file", nargs="?", help="with restore: the backup file to put back")
    p.add_argument("--yes", "-y", action="store_true", help="with restore: do not ask for confirmation")
    p.add_argument("--backup-dir", help="also look for backups in this folder")


def run(args: argparse.Namespace, db: Path) -> int:
    """Run ``migrate`` or ``backups``; returns the exit code (errors propagate to ``mark``'s handler)."""
    if args.cmd == "migrate":
        return _migrate(args, db)
    if args.action == "restore":
        if not args.file:
            raise ValueError("mark backups restore needs the backup FILE (see `mark backups`)")
        return _restore(args, db, Path(args.file).expanduser().resolve())
    return _list(args, db)


def _migrate(args: argparse.Namespace, db: Path) -> int:
    from mark.errors import StoreEncryptionError
    from mark.store._sqlite import upgrade_backup as ub
    from mark.store._sqlite import upgrade_messages as msg
    from mark.store._sqlite.ddl import SchemaDDL
    from mark.store._sqlite.server_guard import foreign_server
    from mark.store._sqlite.upgrade import backups_enabled
    from mark.store._sqlite.upgrade_restore import last_upgrade, read_schema

    if not db.exists():
        raise FileNotFoundError(f"no store at {db} (--db or $MARK_DB picks another)")
    current, target = read_schema(db), SchemaDDL.CURRENT_SCHEMA_VERSION
    print(f"store:   {db}")
    if current is not None and current >= target:
        if current > target:
            from mark.errors import StoreVersionError

            raise StoreVersionError(str(db), current, target)
        print(f"schema:  {msg.schema_label(current)}: already current, nothing to upgrade")
        return 0
    print(f"schema:  {msg.schema_label(current or 0)} -> {msg.schema_label(target)}")
    if _marked_encrypted(db):
        # The command line opens stores without a key, and an encrypted store is
        # refused before any backup or change: say so instead of "upgrading".
        print("encrypted: yes -> the mark command cannot upgrade an encrypted store; open it once from "
              "Python with its key (Mark.local(..., encryption=...)) to upgrade it with a backup", file=sys.stderr)
        return 1
    with_backup = not args.no_backup and backups_enabled()
    space = ub.space_check(db, args.backup_dir)
    held, pid = foreign_server(str(db))
    if args.dry_run:
        label = f"v{current or 0}-backup"
        where = ub.backup_name(db, label, ub.target_dir(db, args.backup_dir)) if with_backup else None
        print(f"backup:  would create {where}" if where else
              "backup:  none (--no-backup or MARK_BACKUP_ON_UPGRADE=0)")
        verdict = "PASS" if space.ok else f"FAIL (short by {msg.fmt_size(space.shortfall_bytes)})"
        print(f"space:   {msg.size_math(space)} needed; {msg.fmt_size(space.free_bytes)} free on "
              f"{space.mount} (folder {space.directory}) -> {verdict if with_backup else 'not needed'}")
        print(f"server:  mark serve (pid {pid or '?'}) has the store open: stop it first (mark serve stop)"
              if held else "server:  none has the store open")
        return 0 if (space.ok or not with_backup) and not held else 1
    from mark.store.sqlite import LocalMemoryStore

    quiet = logging.getLogger("mark.store.sqlite")
    level = quiet.level
    quiet.setLevel(logging.ERROR)  # the success line is printed below, on stdout
    try:
        LocalMemoryStore(db, backup_on_upgrade=with_backup, backup_dir=args.backup_dir).close()
    except StoreEncryptionError as exc:
        # Never report an upgrade that did not happen: re-read the schema.
        print(f"not upgraded: {exc} (schema is still {msg.schema_label(read_schema(db) or 0)})", file=sys.stderr)
        return 1
    finally:
        quiet.setLevel(level)
    record = last_upgrade(db) or {}
    print(f"upgraded {db}: {msg.record_summary(record, with_path=False)}")
    if record.get("backup"):
        print(f"backup kept at {record['backup']}")
    return 0


def _marked_encrypted(db: Path) -> bool:
    """True when the store says it is encrypted at rest (read-only, no MARK import)."""
    import sqlite3

    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'encrypted_at_rest'").fetchone()
    except sqlite3.Error:
        return False
    return row is not None and str(row[0]) == "1"


def _list(args: argparse.Namespace, db: Path) -> int:
    from mark.store._sqlite import upgrade_messages as msg
    from mark.store._sqlite.upgrade_restore import last_upgrade, list_backups, read_schema

    if db.exists():
        print(f"store: {db} (schema {read_schema(db)}, {msg.fmt_size(db.stat().st_size)})")
    found = list_backups(db, [args.backup_dir] if args.backup_dir else [])
    if not found:
        print(f"no backups of {db.name} found")
    for item in found:
        schema = msg.schema_label(item.schema) if item.schema is not None else "unreadable"
        kind = "" if item.kind == "upgrade" else "  (kept before a restore)"
        print(f"  {item.path}  schema {schema}  {msg.fmt_size(item.bytes)}  "
              f"{item.taken:%Y-%m-%d %H:%M:%S} UTC{kind}")
    record = last_upgrade(db) if db.exists() else None
    if record:
        print(f"last upgrade: {msg.record_summary(record)}")
    return 0


def _restore(args: argparse.Namespace, db: Path, backup: Path) -> int:
    import sqlite3

    from mark._private_fs import ensure_private_db_file
    from mark.store._sqlite import upgrade_backup as ub
    from mark.store._sqlite.ddl import SchemaDDL
    from mark.store._sqlite.server_guard import foreign_server
    from mark.store._sqlite.upgrade_restore import check_backup, read_schema, restore_backup

    schema, counts = check_backup(backup)  # ValueError: not a healthy MARK store
    if schema is not None and schema > SchemaDDL.CURRENT_SCHEMA_VERSION:
        raise ValueError(f"{backup} has schema {schema}, newer than this mark-sdk supports; not restoring it")
    held, pid = foreign_server(str(db))
    if held:
        raise RuntimeError(f"{db} is open in mark serve (pid {pid or '?'}); stop it first "
                           "(mark serve stop), then run the restore again")
    print(f"restore {db} from {backup} (schema {schema}, {_n(counts, 'fragments', 'memory', 'memories')});"
          " the current store is kept as a .pre-restore copy first")
    if not args.yes and not _confirmed():
        return 1
    kept = None
    if db.exists():
        holder = sqlite3.connect(str(db), timeout=30.0)
        try:
            holder.execute("BEGIN IMMEDIATE")  # nothing changes while the current store is copied
            kept = ub.create_backup(db, read_schema(db) or 0, label="pre-restore",
                                    purpose="before restoring a backup over it")
        finally:
            holder.rollback()
            holder.close()
    else:
        ensure_private_db_file(db)
    restored = restore_backup(db, backup)
    print(f"restored {db} from {backup}: {_n(restored, 'fragments', 'memory', 'memories')}, "
          f"{_n(restored, 'facts', 'fact', 'facts')}, schema {schema}")
    if kept is not None:
        print(f"the store as it was is kept at {kept.path}")
    if schema is not None and schema < SchemaDDL.CURRENT_SCHEMA_VERSION:
        print(f"note: this mark-sdk upgrades it again (with a new backup) the next time it opens it; "
              f"use the release that wrote schema {schema} to keep it as it is")
    return 0


def _n(counts: dict[str, int], table: str, one: str, many: str) -> str:
    n = counts.get(table, 0)
    return f"{n:,} {one if n == 1 else many}"


def _confirmed() -> bool:
    """Ask on a terminal; refuse (saying how to proceed) when nobody can answer."""
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        print("mark: not restoring without confirmation; rerun with --yes", file=sys.stderr)
        return False
    print("Restore it? [y/N] ", end="", flush=True)
    if stdin.readline().strip().lower() in {"y", "yes"}:
        return True
    print("mark: nothing restored", file=sys.stderr)
    return False
