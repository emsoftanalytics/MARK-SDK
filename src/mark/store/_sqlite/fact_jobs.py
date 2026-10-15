"""Durable fact-extraction jobs and the ledger of extracted assertions.

``fact_jobs`` is a small work queue for slow extractors (e.g. an LLM). Each
row is one (fragment, extractor) pair, so enqueueing is idempotent by
fragment id. Status moves ``pending -> running -> done``; a failure records
the error and returns the job to ``pending`` until ``max_attempts`` is
reached, then it stays ``failed`` for inspection. A ``running`` job whose
lease expired (the process died mid-job) is claimed again, so jobs survive
crashes and restarts.

``fact_extractions`` records every assertion already turned into a fact,
keyed by a hash of (source text, occurrence, subject, attribute, value,
valid_from). Re-ingesting the same log therefore never re-appends values to
a same-day sequence, which ``fact_set``'s latest-value check alone cannot
guarantee ("A->B, B->C" re-ingested would otherwise become B, C, B, C).

These writes are non-structural: they never change which fragments are
retrievable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


class FactJobMixin:
    """Queue, claim and settle fact-extraction jobs; record extracted assertions."""

    def fact_job_enqueue(self, agent_id: str, items: Iterable[tuple[str, int]], extractor: str) -> int:
        """Queue ``(fragment_id, occurrence)`` items; returns how many were new."""
        now = datetime.now(timezone.utc).isoformat()
        added = 0
        with self._writing(structural=False) as conn:
            for fragment_id, occurrence in items:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO fact_jobs (fragment_id, extractor, agent_id, occurrence, status,"
                    " attempts, error, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?)",
                    (fragment_id, extractor, agent_id, int(occurrence), now, now),
                )
                added += cursor.rowcount
        return added

    def fact_job_claim(
        self,
        extractor: str,
        *,
        agent_id: str | None = None,
        limit: int = 32,
        lease_seconds: float = 300.0,
        max_attempts: int = 3,
        not_after: str | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically mark up to ``limit`` runnable jobs ``running`` and return them.

        Runnable: ``pending``, or ``running`` with an expired lease. Every
        claim counts as an attempt, so a job that keeps crashing its worker
        ends up ``failed`` instead of looping forever. ``not_after`` (ISO
        time) skips pending jobs touched later — a worker pass passes its
        start time so a job that just failed waits for the next pass.
        """
        now = datetime.now(timezone.utc)
        stale = (now - timedelta(seconds=lease_seconds)).isoformat()
        sql = ("SELECT fragment_id, agent_id, occurrence, attempts, status FROM fact_jobs WHERE extractor = ?"
               " AND ((status = 'pending' AND updated_at <= ?) OR (status = 'running' AND updated_at < ?))")
        params: list[object] = [extractor, not_after or "9999", stale]
        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        sql += " ORDER BY created_at, rowid LIMIT ?"
        params.append(int(limit))
        claimed: list[dict[str, Any]] = []
        with self._writing(structural=False) as conn:
            for row in conn.execute(sql, params).fetchall():
                if row["status"] == "running" and int(row["attempts"]) >= max_attempts:
                    conn.execute(
                        "UPDATE fact_jobs SET status = 'failed', error = COALESCE(error, ?), updated_at = ?"
                        " WHERE fragment_id = ? AND extractor = ?",
                        ("abandoned: worker stopped mid-job", now.isoformat(), row["fragment_id"], extractor),
                    )
                    continue
                conn.execute(
                    "UPDATE fact_jobs SET status = 'running', attempts = attempts + 1, updated_at = ?"
                    " WHERE fragment_id = ? AND extractor = ?",
                    (now.isoformat(), row["fragment_id"], extractor),
                )
                claimed.append({"fragment_id": str(row["fragment_id"]), "agent_id": str(row["agent_id"]),
                                "occurrence": int(row["occurrence"] or 0), "attempts": int(row["attempts"]) + 1})
        return claimed

    def fact_job_complete(self, fragment_id: str, extractor: str) -> None:
        """Mark a job done (clearing any error from earlier attempts)."""
        self._settle_job(fragment_id, extractor, "done", None)

    def fact_job_fail(self, fragment_id: str, extractor: str, error: str, *, max_attempts: int = 3) -> str:
        """Record a failure; the job is retried until ``max_attempts``. Returns the new status."""
        row = self._fetchone("SELECT attempts FROM fact_jobs WHERE fragment_id = ? AND extractor = ?",
                             (fragment_id, extractor))
        status = "failed" if row is None or int(row["attempts"]) >= max_attempts else "pending"
        self._settle_job(fragment_id, extractor, status, error[:2000])
        return status

    def fact_job_counts(self, agent_id: str | None = None, extractor: str | None = None) -> dict[str, int]:
        """Number of jobs per status (``pending``, ``running``, ``done``, ``failed``)."""
        sql, params = "SELECT status, COUNT(*) FROM fact_jobs WHERE 1 = 1", []
        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if extractor is not None:
            sql += " AND extractor = ?"
            params.append(extractor)
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for row in self._fetchall(sql + " GROUP BY status", params):
            counts[str(row[0])] = int(row[1])
        return counts

    def fact_job_list(self, *, status: str | None = None, agent_id: str | None = None,
                      limit: int = 100) -> list[dict[str, Any]]:
        """Inspect jobs (oldest first), e.g. ``status="failed"`` with their errors."""
        sql, params = "SELECT * FROM fact_jobs WHERE 1 = 1", []
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        rows = self._fetchall(sql + " ORDER BY created_at, rowid LIMIT ?", [*params, int(limit)])
        return [{key: row[key] for key in row.keys()} for row in rows]

    def fact_set_extracted(self, agent_id: str, key: str, subject: str, attribute: str, value: str, *,
                           valid_from: datetime, source_fragment_id: str | None,
                           metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Record an extracted fact once per ``key``; None when it was already recorded.

        The ledger row and the fact are written in one transaction: if
        ``fact_set`` rejects the fact, the ledger row is rolled back too.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO fact_extractions (agent_id, key, fragment_id, fact_id, created_at)"
                " VALUES (?, ?, ?, NULL, ?)", (agent_id, key, source_fragment_id, now),
            )
            if cursor.rowcount == 0:
                return None
            fact = self.fact_set(agent_id, subject, attribute, value, valid_from=valid_from,
                                 source_fragment_id=source_fragment_id, metadata=metadata)
            conn.execute("UPDATE fact_extractions SET fact_id = ? WHERE agent_id = ? AND key = ?",
                         (fact["id"], agent_id, key))
            return fact

    def _settle_job(self, fragment_id: str, extractor: str, status: str, error: str | None) -> None:
        with self._writing(structural=False) as conn:
            conn.execute(
                "UPDATE fact_jobs SET status = ?, error = ?, updated_at = ? WHERE fragment_id = ? AND extractor = ?",
                (status, error, datetime.now(timezone.utc).isoformat(), fragment_id, extractor),
            )
