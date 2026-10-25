# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""File-level helpers for upgrade backups: sizes, names, publishing, page compare.

``same_pages`` is how a large backup is verified. SQLite's backup API copies
pages verbatim, so once the store's WAL is fully checkpointed (and the
caller holds the write lock) the backup file must equal the store file byte
for byte, except the 100-byte header (change counter, journal mode). Reading
both files sequentially is far cheaper than a b-tree walk such as
``quick_check``, which reads pages in key order: at 1M memories (1.7 GB) on a
loaded machine whose page cache could not hold the file, ``quick_check`` of
the memories table alone took 78 s and of the whole file about 50-100 s;
the sequential compare of both files took 19 s. It also checks every byte,
not just the structure.
Standard library only.
"""
from __future__ import annotations

import os
from pathlib import Path

_HEADER = 100  # SQLite file header: may differ between a store and its backup
_CHUNK = 8 << 20


def read_only_uri(path: str | Path) -> str:
    """A ``mode=ro`` URI for *path* (percent-encoded: spaces, ``?`` and ``#`` are safe)."""
    return Path(path).expanduser().resolve().as_uri() + "?mode=ro"


def file_size(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def mount_of(path: Path) -> str:
    """The mount point holding *path* (what ``df`` names), for messages."""
    path = path.resolve()
    while not os.path.ismount(path) and path != path.parent:
        path = path.parent
    return str(path)


def fsync_file(path: str | Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remove_db_files(path: Path) -> None:
    """Remove a (partial) database and any journal files SQLite left next to it."""
    for item in (path, Path(f"{path}-journal"), Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            item.unlink()
        except OSError:
            pass


def publish(partial: Path, final: Path) -> None:
    """Give the verified copy its final name without ever replacing an existing file."""
    try:
        os.link(partial, final)  # fails with FileExistsError instead of overwriting
    except (AttributeError, NotImplementedError, PermissionError):  # no hard links (some filesystems)
        if final.exists():
            raise FileExistsError(17, "File exists", str(final)) from None
        os.rename(partial, final)
        return
    os.unlink(partial)


def same_pages(store: str | Path, backup: str | Path) -> str | None:
    """None when *backup* holds the same pages as *store*; else what differs (for the error)."""
    size = file_size(backup)
    if file_size(store) != size:
        return f"file size: store {file_size(store):,} bytes, backup {size:,}"
    with open(store, "rb", buffering=0) as a, open(backup, "rb", buffering=0) as b:
        left, right = bytearray(_CHUNK), bytearray(_CHUNK)
        offset = 0
        while offset < size:
            n, m = a.readinto(left), b.readinto(right)
            if n != m or n == 0:
                return f"short read at byte {offset:,}"
            skip = _HEADER if offset == 0 else 0
            if memoryview(left)[skip:n] != memoryview(right)[skip:n]:
                return f"content differs within bytes {offset:,}-{offset + n:,}"
            offset += n
    return None
