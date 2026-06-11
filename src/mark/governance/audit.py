# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""In-process audit log for local governance decisions."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditEntry:
    """One recorded governance decision."""
    id:           str
    content_hash: str    # first 16 hex chars of SHA-256
    passed:       bool
    reason:       str
    gate:         str
    agent_id:     str | None
    timestamp:    str


class GovernanceAuditLog:
    """
    In-process audit log for local governance decisions.

    Stores AuditEntry records in memory. Pass ``db_path`` to persist to SQLite
    (planned for a future release). Cloud governance uses an immutable ledger
    with compliance export support.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self,
        content: str,
        *,
        passed:   bool,
        reason:   str          = "",
        gate:     str          = "ConsolidationGate",
        agent_id: str | None   = None,
    ) -> str:
        """Persist an entry and return it."""
        entry_id = str(uuid.uuid4())
        self._entries.append(
            AuditEntry(
                id           = entry_id,
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16],
                passed       = passed,
                reason       = reason,
                gate         = gate,
                agent_id     = agent_id,
                timestamp    = datetime.now(timezone.utc).isoformat(),
            )
        )
        return entry_id

    def list_entries(
        self,
        *,
        agent_id: str | None  = None,
        passed:   bool | None = None,
        limit:    int         = 100,
    ) -> list[AuditEntry]:
        """Return recorded entries, newest first."""
        entries: list[AuditEntry] = self._entries
        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]
        if passed is not None:
            entries = [e for e in entries if e.passed == passed]
        return list(reversed(entries[-limit:]))

    def stats(self) -> dict[str, Any]:
        """Return summary counters."""
        total   = len(self._entries)
        passed  = sum(1 for e in self._entries if e.passed)
        blocked = total - passed
        by_gate: dict[str, int] = {}
        for e in self._entries:
            by_gate[e.gate] = by_gate.get(e.gate, 0) + 1
        return {
            "total":   total,
            "passed":  passed,
            "blocked": blocked,
            "by_gate": by_gate,
        }

    def clear(self) -> None:
        """Delete all audit entries."""
        self._entries.clear()
