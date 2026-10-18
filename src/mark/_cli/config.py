"""CLI configuration: constants and the on-disk paths derived from a database.

A database ``X.db`` owns two sibling files: ``X.db.server.json`` (0600 state
file with the server's address and token) and ``X.db.server.lock`` (pid of
the one server allowed per database). Standard library only.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

DEFAULT_AGENT = "__mark__"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
INGEST_CHUNK_BYTES = 2 * 1024 * 1024  # client-side split so requests stay well under the cap
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _db_path(arg: str | None) -> Path:
    """Resolve the database: ``--db``, else ``$MARK_DB``, else ``./.mark/memory.db``.

    ``--db`` follows the store-path rule of :func:`mark.runtime_data.store_db_file`
    (``.mark`` folder, project folder or database file), so the CLI opens the
    same file as ``Mark.local(path)`` and ``mark mcp --db path``.
    """
    from mark.runtime_data import store_db_file  # standard library only

    return store_db_file(arg or os.environ.get("MARK_DB") or str(Path(".mark") / "memory.db"))


def _state_path(db: Path) -> Path:
    """Server state file (host, port, token, pid) for *db*."""
    return db.with_name(db.name + ".server.json")


def _lock_path(db: Path) -> Path:
    """One-server-per-database lock file for *db*."""
    return db.with_name(db.name + ".server.lock")


def _proof(token: str, nonce: str) -> str:
    """What a genuine server answers to a nonce: HMAC-SHA256(token, nonce)."""
    return hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()
