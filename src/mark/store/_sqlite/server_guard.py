"""Never migrate a store under a running ``mark serve`` from an older build.

A server keeps its store open for hours. If a newer MARK opened the same
file in-process and migrated the schema, the old server would keep writing
with its old code: rows without the newer tables' entries (multilingual
FTS, write log) that nothing ever backfills, because the backfill markers
say it already ran.

``mark serve`` holds an OS lock on ``<db>.server.lock`` for its lifetime
(see :mod:`mark._cli.server`). A store that needs a migration checks that
lock first: if another process holds it, the open is refused with
:class:`StoreInUseError`, which tells the user to stop that server. A
server of the *same* build never triggers this: it migrated the store when
it opened it, so no migration is pending. The lock is only probed when a
migration is due (once per upgrade), never on a normal open.
"""
from __future__ import annotations

import os
import sys

from mark.errors import StoreInUseError  # noqa: F401 - re-exported for callers

# Lock files this process holds as a server (``mark serve`` registers its own):
# its engine opens the store after taking the lock and must not refuse itself.
OWN_LOCKS: set[str] = set()




def foreign_server(db_path: str) -> tuple[bool, int | None]:
    """(held, pid): whether another process holds *db_path*'s server lock, and its pid if known.

    Probing takes the lock for an instant when it is free (released at
    once), which can only make a server starting in that same instant
    report "already running"; it is done only before a migration.
    """
    lock = f"{db_path}.server.lock"
    if os.path.abspath(lock) in OWN_LOCKS:
        return False, os.getpid()
    try:
        fd = os.open(lock, os.O_RDWR)
    except OSError:
        return False, None  # no lock file: no server
    try:
        try:
            _try_lock(fd)
        except OSError:
            pid = _owner(fd)
            return pid != os.getpid(), pid  # our own server (serve opens its engine after locking)
        _unlock(fd)
        return False, None
    finally:
        os.close(fd)


def _owner(fd: int) -> int | None:
    try:
        text = os.pread(fd, 32, 0).decode("ascii", "replace").strip() if hasattr(os, "pread") else ""
        return int(text) if text.isdigit() else None
    except OSError:
        return None


def _try_lock(fd: int) -> None:
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if sys.platform == "win32":  # pragma: no cover
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
