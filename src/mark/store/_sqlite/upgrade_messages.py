# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Plain-words messages for store upgrades and their safety backups.

Every failure message says what happened, the numbers behind it (sizes,
free space, row counts), what the user can do, and whether the store was
changed. The CLI prints these messages as they are (no traceback), so they
are written for people, not for logs. Standard library only.
"""
from __future__ import annotations

#: Which mark-sdk release writes each schema version (shown next to the number).
RELEASES = {3: "0.2.0a7", 4: "0.2.0a8", 5: "0.2.0a9", 6: "0.2.0a10"}
UNCHANGED = "Your store was not changed."
_ISSUES = "https://github.com/emsoftanalytics/MARK-SDK/issues"


def fmt_size(n: float) -> str:
    """Decimal units, like ``df -H``: 2.61 GB, 412.3 MB, 18.0 KB, 512 bytes."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.{2 if unit == 'GB' else 1}f} {unit}"
    return f"{int(n)} bytes"


def schema_label(version: int) -> str:
    """``5 (mark-sdk 0.2.0a9)``; pre-versioned stores are ``0 (before 0.2.0a7)``."""
    if version <= 0:
        return "0 (unversioned, before mark-sdk 0.2.0a7)"
    release = RELEASES.get(version)
    return f"{version} (mark-sdk {release})" if release else str(version)


def size_math(space) -> str:
    """``store 2.61 GB + WAL 0.10 GB + 10% margin = 2.98 GB`` for a SpaceCheck."""
    return (f"store {fmt_size(space.store_bytes)} + WAL {fmt_size(space.wal_bytes)} + "
            f"{space.margin_percent}% margin = {fmt_size(space.required_bytes)}")


def no_space(space, purpose: str) -> str:
    """*purpose*: "before upgrading it from schema 5 to 6", "before restoring a backup over it"."""
    return (f"Not enough free space to back up {space.db_name} {purpose}: the backup needs about {fmt_size(space.required_bytes)} "
            f"(store {fmt_size(space.store_bytes)} + WAL {fmt_size(space.wal_bytes)} + "
            f"{space.margin_percent}% margin) but only {fmt_size(space.free_bytes)} is free on "
            f"{space.mount} (folder {space.directory}; short by {fmt_size(space.shortfall_bytes)}). "
            "Free some space, or put the backup on another disk with backup_dir= "
            "(CLI: mark migrate --backup-dir DIR), or upgrade without a backup with "
            "backup_on_upgrade=False (MARK_BACKUP_ON_UPGRADE=0, CLI: mark migrate --no-backup). "
            f"{UNCHANGED}")


def write_failed(db_name: str, backup_path: str, exc: BaseException) -> str:
    reason = _os_reason(exc)
    return (f"Could not write the backup of {db_name} to {backup_path}: {reason}. "
            "The incomplete backup was removed. Make sure the folder is writable and has room, or "
            "choose another folder with backup_dir= (CLI: mark migrate --backup-dir DIR). "
            f"{UNCHANGED}")


def verify_failed(db_name: str, backup_path: str, differences: list[str]) -> str:
    return (f"The backup of {db_name} made before upgrading did not match the store, so the upgrade "
            f"was not started: {'; '.join(differences)}. The incomplete backup {backup_path} "
            "was removed. Try again; if it keeps failing, check the disk and run "
            "`sqlite3 <store> 'PRAGMA integrity_check'` on the store. "
            f"{UNCHANGED}")


def upgrade_failed(db_name: str, from_version: int, to_version: int, reason: str, *,
                   backup: str | None, committed: bool, restored: bool, restore_error: str | None) -> str:
    head = f"Upgrading {db_name} from schema {schema_label(from_version)} to {to_version} failed: {reason}."
    if not committed:
        state = f"The changes were rolled back; your store is unchanged at schema {from_version}."
    elif restored:
        state = f"Your store was restored from the backup and is back at schema {from_version}."
    else:
        state = ("The upgrade had already been saved and could not be undone"
                 + (f" ({restore_error})" if restore_error else "") + ".")
    kept = (f"The backup taken before the upgrade is kept at {backup}." if backup
            else "no backup was taken (backup_on_upgrade=False or MARK_BACKUP_ON_UPGRADE=0).")
    kept = kept[0].upper() + kept[1:]
    advice = (f"Keep using mark-sdk {RELEASES.get(from_version, 'the version that wrote it')} with this "
              f"store for now and report the message at {_ISSUES}.")
    if committed and not restored and backup:
        advice = f"Restore it with `mark backups restore {backup}` and report the message at {_ISSUES}."
    return " ".join((head, state, kept, advice))


def success(db_path: str, from_version: int, to_version: int, total: float, plan) -> str:
    """The one line logged (and printed by ``mark migrate``) after an upgrade."""
    backup = plan.backup
    if backup is None:
        return (f"upgraded {db_path} from schema {from_version} to {to_version} in {total:.2f} s "
                f"(migrate {plan.migrate_seconds:.2f} s); no backup taken (backup_on_upgrade off)")
    return (f"upgraded {db_path} from schema {from_version} to {to_version} in {total:.2f} s "
            f"(backup {backup.seconds:.2f} s at {fmt_size(backup.throughput)}/s, verify "
            f"{backup.verify_seconds:.2f} s, migrate {plan.migrate_seconds:.2f} s); backup kept at "
            f"{backup.path} ({backup_vs_store(backup)})")


def backup_vs_store(backup) -> str:
    """``backup 2.61 GB = store 2.61 GB (WAL checkpointed)``."""
    sign = "=" if abs(backup.bytes - backup.store_bytes) <= max(4096, backup.store_bytes // 100) else "vs"
    wal = "WAL checkpointed" if backup.checkpointed else f"WAL {fmt_size(backup.wal_bytes)} not checkpointed"
    return f"backup {fmt_size(backup.bytes)} {sign} store {fmt_size(backup.store_bytes)}, {wal}"


def _os_reason(exc: BaseException) -> str:
    """``[Errno 13] Permission denied`` for OS errors, the SQLite text otherwise."""
    if isinstance(exc, OSError) and exc.strerror:
        return f"[Errno {exc.errno}] {exc.strerror}" + (f" ({exc.filename})" if exc.filename else "")
    return str(exc) or type(exc).__name__


def record_summary(record: dict, *, with_path: bool = True) -> str:
    """One line for a stored ``last_upgrade`` record (``mark migrate`` / ``mark backups``)."""
    keys = (("backup", "backup_seconds"), ("verify", "verify_seconds"), ("migrate", "migrate_seconds"))
    parts = {name: record[key] for name, key in keys if isinstance(record.get(key), (int, float))}
    total = sum(parts.values())
    if record.get("backup") and parts.get("backup"):
        parts["backup"] = f"{parts['backup']:.2f} s at {fmt_size((record.get('backup_bytes') or 0) / parts['backup'])}/s"
    timing = ", ".join(f"{name} {value if isinstance(value, str) else f'{value:.2f} s'}"
                       for name, value in parts.items())
    head = (f"schema {record.get('from_schema')} -> {record.get('to_schema')} on {record.get('at')} "
            f"in {total:.2f} s ({timing})")
    if not record.get("backup"):
        return f"{head}; no backup taken"
    backup, store = record.get("backup_bytes") or 0, record.get("store_bytes") or 0
    sign = "=" if abs(backup - store) <= max(4096, store // 100) else "vs"
    wal = ", WAL checkpointed" if record.get("checkpointed") else ""
    where = f" {record['backup']}" if with_path else ""
    return f"{head}; backup{where} ({fmt_size(backup)} {sign} store {fmt_size(store)}{wal})"
