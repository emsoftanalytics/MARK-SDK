"""Durable job-queue half of :class:`FactExtractionService` (background extractors).

Split from ``service.py`` to keep both modules small. The mixin expects the
host to provide ``store``, ``name``, ``lease_seconds``, ``max_attempts``,
``batch_size``, ``_worker``, ``_stopping``, ``_held``, ``_started``, ``_held_lock`` and
``extract_fragment``.

Lifecycle guarantees:
  - jobs are claimed ``batch_size`` at a time and every job's lease is renewed
    when it starts, so late jobs in a batch are never taken over mid-run;
  - when stopping, unstarted claimed jobs are released at once, and
    :meth:`close` waits a bounded time for the job in flight before handing it
    back too;
  - waiting for other threads' jobs is bounded (:data:`DEFAULT_WAIT_SECONDS`).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from mark.errors import StoreClosedError

logger = logging.getLogger("mark.intelligence.fact_extraction.service")

#: How long ``drain(wait=True)`` / ``facts.extract_pending()`` waits for jobs
#: other threads hold. Bounded so a stuck extractor never hangs the caller.
DEFAULT_WAIT_SECONDS = 30.0


class JobQueueMixin:
    """Claim, run, drain, release and stop: the background side of the service."""

    def run_pending(self, *, agent_id: str | None = None, limit: int | None = None) -> int:
        """Process queued jobs now (in this thread); returns how many were processed."""
        processed = 0
        # Jobs that fail during this pass are retried on a later pass, not in a tight loop.
        started = datetime.now(timezone.utc).isoformat()
        while (limit is None or processed < limit) and not self._stopping.is_set():
            batch = min(self.batch_size, limit - processed) if limit is not None else self.batch_size
            jobs = self.store.fact_job_claim(self.name, agent_id=agent_id, limit=batch, not_after=started,
                                             lease_seconds=self.lease_seconds, max_attempts=self.max_attempts)
            if not jobs:
                break
            with self._held_lock:
                self._held.update(job["fragment_id"] for job in jobs)
            for index, job in enumerate(jobs):
                if self._stopping.is_set():
                    # Shutting down: hand the rest back now rather than after the lease.
                    self._release([j["fragment_id"] for j in jobs[index:]], started=False)
                    return processed
                self._run_job(job)
                processed += 1
        return processed

    def drain(self, *, agent_id: str | None = None, limit: int | None = None,
              wait: float | bool = True) -> int:
        """Process queued jobs here, then wait for jobs another thread is running.

        A background worker claims jobs in small batches, so a synchronous
        caller can find the queue empty while the facts are not written yet.
        With ``wait`` (True = up to :data:`DEFAULT_WAIT_SECONDS` (30 s), a
        float = that many seconds, False/0 = do not wait) this also blocks
        until no job for *agent_id* is running. Returns the jobs processed here plus the jobs other threads
        finished meanwhile; on timeout, those still running are not counted.
        """
        timeout = float(DEFAULT_WAIT_SECONDS if wait is True else (wait or 0.0))
        before = self._finished(agent_id)
        own_runs, own_finished = self._run_and_count(agent_id, limit)
        deadline = time.monotonic() + timeout
        while timeout > 0 and self.store.fact_job_counts(agent_id, self.name)["running"]:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
            if limit is None or own_runs < limit:
                # Jobs enqueued meanwhile, or released by an expired lease.
                runs, finished = self._run_and_count(agent_id, None if limit is None else limit - own_runs)
                own_runs, own_finished = own_runs + runs, own_finished + finished
        by_others = max(0, self._finished(agent_id) - before - own_finished)
        return own_runs + by_others

    def _run_and_count(self, agent_id: str | None, limit: int | None) -> tuple[int, int]:
        """run_pending, also returning how many of those jobs reached done/failed."""
        before = self._finished(agent_id)
        runs = self.run_pending(agent_id=agent_id, limit=limit)
        return runs, self._finished(agent_id) - before

    def _finished(self, agent_id: str | None) -> int:
        counts = self.store.fact_job_counts(agent_id, self.name)
        return counts["done"] + counts["failed"]

    def pending(self, agent_id: str | None = None) -> int:
        """Jobs not yet finished (pending or running) for this extractor."""
        counts = self.store.fact_job_counts(agent_id, self.name)
        return counts["pending"] + counts["running"]

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until the queue is empty (useful in tests and scripts)."""
        deadline = time.monotonic() + timeout
        while self.pending() and time.monotonic() < deadline:
            time.sleep(0.02)
        return not self.pending()

    def close(self, timeout: float = 5.0) -> bool:
        """Stop the worker; returns True when it stopped within ``timeout`` seconds.

        Cooperative: the worker finishes the job it is running (if that takes
        less than ``timeout``) and claims nothing new. Jobs it claimed but did
        not finish — including one still running when the timeout expires —
        go back to the queue durably, for this or another process to redo.
        Idempotent; calls after the first only re-check.
        """
        self._stopping.set()
        stopped = True
        if self._worker is not None:
            stopped = self._worker.stop(timeout)
            self._worker = None
        with self._held_lock:
            running = [fid for fid in self._held if fid in self._started]
            waiting = [fid for fid in self._held if fid not in self._started]
        if waiting:
            self._release(waiting, started=False)
        if running:
            self._release(running, started=True)
        return stopped

    def _reclaim_orphans(self) -> None:
        """Requeue jobs whose worker process died (best effort; never fails construction)."""
        reclaim = getattr(self.store, "fact_job_reclaim", None)
        if reclaim is None:
            return
        try:
            count = reclaim(self.name, max_attempts=self.max_attempts)
        except Exception:  # noqa: BLE001 - e.g. a read-only store; the lease still applies
            logger.debug("could not reclaim orphaned fact jobs", exc_info=True)
            return
        if count:
            logger.info("requeued %d fact-extraction job(s) left by a stopped process", count)

    def _release(self, fragment_ids: list[str], *, started: bool) -> None:
        with self._held_lock:
            self._held.difference_update(fragment_ids)
        try:
            self.store.fact_job_release(fragment_ids, self.name, started=started)
        except StoreClosedError:
            pass  # the owner process is exiting; the dead-owner check requeues them next time
        except Exception:  # noqa: BLE001 - the lease still returns them eventually
            logger.debug("could not release fact jobs", exc_info=True)

    def _run_job(self, job: dict[str, Any]) -> None:
        fragment_id = job["fragment_id"]
        with self._held_lock:
            self._started.add(fragment_id)
        try:
            # The lease counts from here, not from when the batch was claimed.
            self.store.fact_job_start(fragment_id, self.name)
            fragment = self.store.get(fragment_id)
            # Deleted or no-longer-eligible memories simply have nothing to extract
            # (extract_fragment returns [] for them).
            if fragment is not None:
                self.extract_fragment(job["agent_id"], fragment, job["occurrence"])
            self.store.fact_job_complete(fragment_id, self.name)
        except StoreClosedError:
            # Shut down mid-job: close() already handed the job back to the queue.
            logger.debug("fact extraction job %s interrupted by shutdown", fragment_id)
        except Exception as exc:  # noqa: BLE001 - recorded on the job and retried
            try:
                status = self.store.fact_job_fail(fragment_id, self.name, f"{type(exc).__name__}: {exc}",
                                                  max_attempts=self.max_attempts)
            except StoreClosedError:
                return
            logger.warning("fact extraction job %s failed (%s): %s", fragment_id, status, exc)
        finally:
            with self._held_lock:
                self._held.discard(fragment_id)
                self._started.discard(fragment_id)

