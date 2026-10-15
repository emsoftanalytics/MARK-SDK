"""Durable fact-extraction jobs and the ledger of extracted assertions.

``fact_jobs`` is a small work queue for slow extractors (e.g. an LLM). Each
row is one (fragment, extractor) pair, so enqueueing is idempotent by
fragment id. Status moves ``pending -> running -> done``; a failure records
the error and returns the job to ``pending`` until ``max_attempts`` is
reached, then it stays ``failed`` for inspection. Jobs survive crashes and
restarts:

  - each claim records its owner process in ``fact_job_owners`` (see
    :mod:`.job_owner`); a ``running`` job whose owner process is dead (same
    host) is reclaimed at once, on open or on the next claim;
  - otherwise a ``running`` job is reclaimed when its lease expires. The lease
    is renewed when the job actually starts (:meth:`fact_job_start`), so a job
    that waited behind others in its batch is never taken over mid-run;
  - a worker that stops returns the jobs it claimed but did not start
    (:meth:`fact_job_release`), so they do not wait for any lease.

``fact_job_owners`` is a side table created on first use (``CREATE TABLE IF
NOT EXISTS``): stores written by older versions simply have no owners, and
their running jobs fall back to the lease.

``fact_extractions`` records every assertion already turned into a fact,
keyed by a hash of (source text, occurrence, subject, attribute, value,
valid_from). Re-ingesting the same log therefore never re-appends values to
a same-day sequence, which ``fact_set``'s latest-value check alone cannot
guarantee ("A->B, B->C" re-ingested would otherwise become B, C, B, C).

These writes are non-structural: they never change which fragments are
retrievable.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from mark.store._sqlite.job_owner import current_owner, owner_is_dead

_OWNERS_DDL = ("CREATE TABLE IF NOT EXISTS fact_job_owners (fragment_id TEXT NOT NULL, extractor TEXT NOT NULL,"
               " owner TEXT NOT NULL, PRIMARY KEY (fragment_id, extractor))")


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
        limit: int = 4,
        lease_seconds: float = 300.0,
        max_attempts: int = 3,
        not_after: str | None = None,
        owner: str | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically mark up to ``limit`` runnable jobs ``running`` and return them.

        Runnable: ``pending``, ``running`` with a dead owner process, or
        ``running`` with an expired lease. Every claim counts as an attempt,
        so a job that keeps crashing its worker ends up ``failed`` instead of
        looping forever. ``not_after`` (ISO time) skips pending jobs touched
        later — a worker pass passes its start time so a job that just failed
        waits for the next pass. ``owner`` defaults to this process. Small
        batches (default 4) keep a slow extractor from sitting on many jobs.
        """
        now = datetime.now(timezone.utc)
        owner = owner or current_owner()
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
            self._reclaim_dead(conn, extractor, max_attempts)
            for row in conn.execute(sql, params).fetchall():
                if row["status"] == "running" and int(row["attempts"]) >= max_attempts:
                    self._abandon(conn, str(row["fragment_id"]), extractor, "abandoned: worker stopped mid-job")
                    continue
                conn.execute(
                    "UPDATE fact_jobs SET status = 'running', attempts = attempts + 1, updated_at = ?"
                    " WHERE fragment_id = ? AND extractor = ?",
                    (now.isoformat(), row["fragment_id"], extractor),
                )
                conn.execute("INSERT OR REPLACE INTO fact_job_owners (fragment_id, extractor, owner)"
                             " VALUES (?, ?, ?)", (row["fragment_id"], extractor, owner))
                claimed.append({"fragment_id": str(row["fragment_id"]), "agent_id": str(row["agent_id"]),
                                "occurrence": int(row["occurrence"] or 0), "attempts": int(row["attempts"]) + 1})
        return claimed

    def fact_job_reclaim(self, extractor: str | None = None, *, max_attempts: int = 3) -> int:
        """Return ``running`` jobs whose owner process died to ``pending``; returns how many.

        Called when an extraction service starts, so jobs orphaned by a crash
        are visible as pending right away instead of after the lease.
        """
        with self._writing(structural=False) as conn:
            return self._reclaim_dead(conn, extractor, max_attempts)

    def fact_job_start(self, fragment_id: str, extractor: str) -> None:
        """Renew a claimed job's lease as it starts running (lease counts from the start)."""
        with self._writing(structural=False) as conn:
            conn.execute("UPDATE fact_jobs SET updated_at = ? WHERE fragment_id = ? AND extractor = ?"
                         " AND status = 'running'", (datetime.now(timezone.utc).isoformat(), fragment_id, extractor))

    def fact_job_release(self, fragment_ids: Iterable[str], extractor: str, *, started: bool = False,
                         owner: str | None = None) -> int:
        """Give claimed jobs back to the queue (e.g. on shutdown); returns how many.

        Only jobs still ``running`` under ``owner`` (default: this process) are
        released, so a job another process already reclaimed is left alone.
        Unstarted jobs get their attempt back; a job abandoned mid-run
        (``started=True``) keeps it, so a job that always hangs still fails.
        """
        owner = owner or current_owner()
        refund = 0 if started else 1
        released = 0
        with self._writing(structural=False) as conn:
            self._ensure_owner_table(conn)
            for fragment_id in fragment_ids:
                cursor = conn.execute(
                    "UPDATE fact_jobs SET status = 'pending', attempts = MAX(0, attempts - ?)"
                    " WHERE fragment_id = ? AND extractor = ? AND status = 'running' AND EXISTS (SELECT 1"
                    " FROM fact_job_owners o WHERE o.fragment_id = fact_jobs.fragment_id"
                    " AND o.extractor = fact_jobs.extractor AND o.owner = ?)",
                    (refund, fragment_id, extractor, owner),
                )
                if cursor.rowcount:
                    released += 1
                    conn.execute("DELETE FROM fact_job_owners WHERE fragment_id = ? AND extractor = ?",
                                 (fragment_id, extractor))
        return released

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
            self._ensure_owner_table(conn)
            conn.execute("DELETE FROM fact_job_owners WHERE fragment_id = ? AND extractor = ?",
                         (fragment_id, extractor))

    # ── owners ────────────────────────────────────────────────────────────────

    def _ensure_owner_table(self, conn: sqlite3.Connection) -> None:
        """Create the side table if missing.

        Not cached on the store: if the surrounding transaction rolls back,
        the CREATE rolls back with it, and IF NOT EXISTS is a cheap no-op.
        """
        conn.execute(_OWNERS_DDL)

    def _reclaim_dead(self, conn: sqlite3.Connection, extractor: str | None, max_attempts: int) -> int:
        """Inside a write transaction: requeue running jobs whose owner process is gone."""
        self._ensure_owner_table(conn)
        sql = ("SELECT j.fragment_id, j.extractor, j.attempts, o.owner FROM fact_jobs j JOIN fact_job_owners o"
               " ON o.fragment_id = j.fragment_id AND o.extractor = j.extractor WHERE j.status = 'running'")
        rows = conn.execute(sql + (" AND j.extractor = ?" if extractor else ""),
                            (extractor,) if extractor else ()).fetchall()
        verdicts: dict[str, bool] = {}
        reclaimed = 0
        for row in rows:
            owner = str(row["owner"])
            if owner not in verdicts:
                verdicts[owner] = owner_is_dead(owner)
            if not verdicts[owner]:
                continue
            fragment_id, job_extractor = str(row["fragment_id"]), str(row["extractor"])
            if int(row["attempts"]) >= max_attempts:
                self._abandon(conn, fragment_id, job_extractor, "abandoned: worker process died mid-job")
                continue
            # updated_at is kept, so a pass that started before the crash can still pick it up.
            conn.execute("UPDATE fact_jobs SET status = 'pending' WHERE fragment_id = ? AND extractor = ?",
                         (fragment_id, job_extractor))
            conn.execute("DELETE FROM fact_job_owners WHERE fragment_id = ? AND extractor = ?",
                         (fragment_id, job_extractor))
            reclaimed += 1
        return reclaimed

    def _abandon(self, conn: sqlite3.Connection, fragment_id: str, extractor: str, reason: str) -> None:
        conn.execute(
            "UPDATE fact_jobs SET status = 'failed', error = COALESCE(error, ?), updated_at = ?"
            " WHERE fragment_id = ? AND extractor = ?",
            (reason, datetime.now(timezone.utc).isoformat(), fragment_id, extractor),
        )
        conn.execute("DELETE FROM fact_job_owners WHERE fragment_id = ? AND extractor = ?",
                     (fragment_id, extractor))
