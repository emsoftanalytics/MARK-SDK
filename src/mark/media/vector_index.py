# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""A separate vector partition for image embeddings.

Image vectors (e.g. 512-d CLIP) must never mix with MARK's text embeddings
(the ``fragments.embedding_blob`` column, indexed by the retrieval pipeline).
They live in their own table, keyed by (fragment id, fingerprint), together
with the SHA-256 of the file they were computed from:

* a different fingerprint is a different space -> separate rows, never compared;
* a different sha256 means the file changed -> the vector is stale and ignored.

The table sits in the same SQLite database as the memory (one file to back
up). Stores that do not expose SQLite transactions get an in-process dict,
which is correct but not persistent. Vectors are JSON float lists; no image
bytes are ever written.
"""
from __future__ import annotations

import json
import threading
from typing import Any

_DDL = (
    "CREATE TABLE IF NOT EXISTS mark_image_vectors ("
    " fragment_id TEXT NOT NULL, agent_id TEXT NOT NULL, fingerprint TEXT NOT NULL,"
    " sha256 TEXT NOT NULL, dim INTEGER NOT NULL, vector TEXT NOT NULL,"
    " PRIMARY KEY (fragment_id, fingerprint))"
)


class ImageVectorIndex:
    """Per-agent image vectors, partitioned by embedding-space fingerprint."""

    def __init__(self, store: Any, agent_id: str) -> None:
        self._store = store
        self._agent_id = agent_id
        # SQLite-backed when the store offers transactions; dict otherwise.
        self._sqlite = hasattr(store, "_writing") and hasattr(store, "_reading")
        self._memory: dict[tuple[str, str], tuple[str, list[float]]] = {}
        self._lock = threading.Lock()
        self._ready = False

    def _ensure(self) -> None:
        if self._sqlite and not self._ready:
            # Not a structural write: it never changes which fragments are retrievable.
            with self._store._writing(structural=False) as conn:
                conn.execute(_DDL)
            self._ready = True

    def put(self, fragment_id: str, fingerprint: str, sha256: str, vector: list[float]) -> None:
        """Insert or replace the vector for one image in one space."""
        if not self._sqlite:
            with self._lock:
                self._memory[(fragment_id, fingerprint)] = (sha256, list(vector))
            return
        self._ensure()
        with self._store._writing(structural=False) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO mark_image_vectors"
                " (fragment_id, agent_id, fingerprint, sha256, dim, vector) VALUES (?, ?, ?, ?, ?, ?)",
                (fragment_id, self._agent_id, fingerprint, sha256, len(vector), json.dumps(vector)),
            )

    def get_all(self, fingerprint: str) -> dict[str, tuple[str, list[float]]]:
        """Return {fragment_id: (sha256, vector)} for one space."""
        if not self._sqlite:
            with self._lock:
                return {fid: value for (fid, fp), value in self._memory.items() if fp == fingerprint}
        self._ensure()
        with self._store._reading() as conn:
            rows = conn.execute(
                "SELECT fragment_id, sha256, vector FROM mark_image_vectors"
                " WHERE agent_id = ? AND fingerprint = ?",
                (self._agent_id, fingerprint),
            ).fetchall()
        return {str(row[0]): (str(row[1]), json.loads(row[2])) for row in rows}

    def delete(self, fragment_id: str) -> None:
        """Drop every vector of one image (all spaces)."""
        if not self._sqlite:
            with self._lock:
                for key in [k for k in self._memory if k[0] == fragment_id]:
                    del self._memory[key]
            return
        self._ensure()
        with self._store._writing(structural=False) as conn:
            conn.execute("DELETE FROM mark_image_vectors WHERE fragment_id = ?", (fragment_id,))
