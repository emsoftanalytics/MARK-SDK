"""Create MARK's on-disk store files privately (owner-only on POSIX).

Memory databases hold whatever agents remember, so the ``.mark/`` directory
is created 0700 and the SQLite file 0600. SQLite creates its -wal/-shm files
with the database file's permissions, so pre-creating the database 0600 is
enough for all three. Only paths MARK itself creates are affected: existing
directories and database files keep the permissions their owner chose.
On non-POSIX systems these helpers just create the paths.

Standard library only: the ``mark`` CLI imports this on its fast path.
"""
from __future__ import annotations

import os
from pathlib import Path

_POSIX = os.name == "posix"
DIR_MODE = 0o700
FILE_MODE = 0o600


def ensure_private_dir(path: str | Path) -> Path:
    """Create *path* (and missing parents) if needed; the leaf, if created, is 0700.

    Missing parents get default permissions: they may be ordinary project
    folders, not MARK's private directory.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=DIR_MODE)
    except FileExistsError:  # created concurrently; leave its mode alone
        return path
    if _POSIX:
        os.chmod(path, DIR_MODE)  # mkdir's mode is filtered by the umask
    return path


def ensure_private_db_file(path: str | Path) -> None:
    """Pre-create an empty SQLite file 0600 if it does not exist yet (POSIX only).

    An empty file is a valid, empty SQLite database. O_EXCL makes this a
    no-op for an existing file, so its owner's chosen mode is never changed.
    """
    if not _POSIX:
        return
    try:
        fd = os.open(Path(path).expanduser(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    except FileExistsError:
        return
    os.close(fd)


def tighten_mark_store(db_file: str | Path) -> list[Path]:
    """Make an existing store in MARK's own ``.mark`` folder private again; returns what changed.

    Stores written by 0.2.0a8 and earlier were left 0644 (file) and 0775
    (folder). On open, a store whose folder is named ``.mark`` gets the
    folder tightened to 0700 and the database (and its -wal/-shm) to 0600,
    but only paths owned by the current user and only bits for group/other
    are removed. Stores elsewhere keep the permissions their owner chose.
    """
    if not _POSIX:
        return []
    path = Path(db_file).expanduser()
    if path.parent.name != ".mark":
        return []
    changed: list[Path] = []
    uid = os.getuid()
    candidates = [(path.parent, DIR_MODE)] + [
        (path.with_name(path.name + suffix), FILE_MODE) for suffix in ("", "-wal", "-shm")]
    for item, wanted in candidates:
        try:
            info = item.lstat()
        except OSError:
            continue
        if info.st_uid != uid or (info.st_mode & 0o077) == 0 or os.path.islink(item):
            continue
        try:
            os.chmod(item, info.st_mode & wanted & 0o7777)
            changed.append(item)
        except OSError:
            pass
    return changed
