"""FactExtractionService: run an extractor over stored memories and record facts.

The service is the only place where candidates become facts. It is given
fragments *as stored* (after governance, redaction and the consolidation
gate), so it extracts exactly what memory kept, and it ignores fragments
that are not retrievable (quarantined, contradicted, raw).

Two modes, chosen by the extractor's ``background`` flag:
  - inline (pattern extractors): facts are written during the store call;
  - background (LLM extractors): the store call only enqueues a durable job
    per fragment; a worker thread (or :meth:`run_pending`) drains the queue,
    and :meth:`drain` also waits for jobs the worker is running.

Invariants:
  - ``valid_from`` is the candidate's own date, else the memory's event time
    (``occurred_at`` metadata or a leading "[YYYY-MM-DD]" stamp), else when
    the memory was stored — never the (later) time a job happened to run;
  - every fact links ``source_fragment_id`` to the memory it came from;
  - extraction is idempotent per assertion (see ``fact_set_extracted``):
    re-ingesting a log or re-running a job never duplicates history;
  - :meth:`submit` never raises — a failing extractor must not fail a write.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Iterable

from mark.intelligence.fact_extraction.base import FactExtractor, extractor_name, runs_in_background
from mark.intelligence.fact_extraction.patterns import PatternFactExtractor
from mark.intelligence.temporal_order import declared_event_time
from mark.types import MemoryFragment, MemoryState

logger = logging.getLogger(__name__)

# Same set as the store's retrievable states: only memories that can surface
# in retrieval may produce facts.
_ELIGIBLE = frozenset({MemoryState.UNVERIFIED, MemoryState.VERIFIED, MemoryState.PROMOTED})


def resolve_extractor(spec: Any) -> FactExtractor | None:
    """``None``/``False`` -> off, ``"patterns"``/``True`` -> built-in packs, else the object itself."""
    if spec is None or spec is False:
        return None
    if spec is True or spec == "patterns":
        return PatternFactExtractor()
    if isinstance(spec, str):
        raise ValueError(f"unknown fact extractor {spec!r} (use 'patterns' or an extractor object)")
    if not callable(getattr(spec, "extract", None)):
        raise TypeError("fact_extractor must provide extract(text, *, occurred_at=None)")
    return spec


def eligible(fragment: MemoryFragment | None) -> bool:
    """True when a stored fragment may produce facts."""
    return (fragment is not None and fragment.state in _ELIGIBLE
            and "mark:quarantined" not in (fragment.tags or []))


class FactExtractionService:
    """Bind one extractor to one store; extract inline or through the job queue."""

    def __init__(
        self,
        store: Any,
        extractor: FactExtractor,
        *,
        start_worker: bool = True,
        max_attempts: int = 3,
        lease_seconds: float = 300.0,
        poll_interval: float = 5.0,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.name = extractor_name(extractor)
        self.background = runs_in_background(extractor)
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self._worker = None
        if self.background and start_worker:
            from mark.intelligence.fact_extraction.worker import FactExtractionWorker

            # Starts by draining jobs left over from earlier runs (durability).
            self._worker = FactExtractionWorker(self, poll_interval=poll_interval)
            self._worker.start()

    # ── write path ────────────────────────────────────────────────────────────

    def submit(self, agent_id: str, fragments: Iterable[MemoryFragment | None]) -> int:
        """Extract (inline) or enqueue (background) stored fragments.

        Returns the number of facts written inline (0 when queued). Never
        raises: errors are logged and the write that triggered them stands.
        """
        try:
            items = _with_occurrences(f for f in fragments if eligible(f))
            if not items:
                return 0
            if self.background:
                self.store.fact_job_enqueue(agent_id, [(f.id, n) for f, n in items], self.name)
                if self._worker is not None:
                    self._worker.wake()
                return 0
            written = 0
            for fragment, occurrence in items:
                try:
                    written += len(self.extract_fragment(agent_id, fragment, occurrence))
                except Exception:  # noqa: BLE001 - extraction must never fail a write
                    logger.warning("fact extraction failed for fragment %s", fragment.id, exc_info=True)
            return written
        except Exception:  # noqa: BLE001
            logger.warning("fact extraction skipped after an error", exc_info=True)
            return 0

    def extract_fragment(self, agent_id: str, fragment: MemoryFragment, occurrence: int = 0) -> list[dict[str, Any]]:
        """Run the extractor on one fragment and record new facts (may raise)."""
        if not eligible(fragment):
            return []
        event = declared_event_time(fragment)
        default_when = event or _aware(fragment.created_at)
        candidates = self.extractor.extract(fragment.content, occurred_at=event)
        written: list[dict[str, Any]] = []
        text_hash = hashlib.sha256(" ".join(fragment.content.split()).encode("utf-8")).hexdigest()
        for candidate in candidates:
            valid_from = _aware(candidate.valid_from) if candidate.valid_from else default_when
            key = hashlib.sha256("\x1f".join((
                text_hash, str(occurrence), candidate.subject.lower(), candidate.attribute.lower(),
                candidate.value, valid_from.astimezone(timezone.utc).isoformat(),
            )).encode("utf-8")).hexdigest()
            metadata = {"extracted_by": self.name, "rule": candidate.rule,
                        "confidence": round(float(candidate.confidence), 3)}
            try:
                fact = self.store.fact_set_extracted(
                    agent_id, key, candidate.subject, candidate.attribute, candidate.value,
                    valid_from=valid_from, source_fragment_id=fragment.id, metadata=metadata,
                )
            except ValueError:
                logger.debug("rejected fact candidate %r", candidate)
                continue
            if fact is not None:
                written.append(fact)
        return written

    # ── queue ─────────────────────────────────────────────────────────────────

    def run_pending(self, *, agent_id: str | None = None, limit: int | None = None) -> int:
        """Process queued jobs now (in this thread); returns how many were processed."""
        processed = 0
        # Jobs that fail during this pass are retried on a later pass, not in a tight loop.
        started = datetime.now(timezone.utc).isoformat()
        while limit is None or processed < limit:
            batch = min(32, limit - processed) if limit is not None else 32
            jobs = self.store.fact_job_claim(self.name, agent_id=agent_id, limit=batch, not_after=started,
                                             lease_seconds=self.lease_seconds, max_attempts=self.max_attempts)
            if not jobs:
                break
            for job in jobs:
                self._run_job(job)
                processed += 1
        return processed

    def drain(self, *, agent_id: str | None = None, limit: int | None = None,
              wait: float | bool = True) -> int:
        """Process queued jobs here, then wait for jobs another thread is running.

        A background worker claims jobs in batches, so a synchronous caller
        can find the queue empty while the facts are not written yet. With
        ``wait`` (True = up to ``lease_seconds``, a float = that many seconds,
        False/0 = do not wait) this also blocks until no job for *agent_id* is
        running. Returns the jobs processed here plus the jobs other threads
        finished meanwhile; on timeout, those still running are not counted.
        """
        timeout = float(self.lease_seconds if wait is True else (wait or 0.0))
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

    def close(self) -> None:
        """Stop the background worker (queued jobs stay durable for next time)."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def _run_job(self, job: dict[str, Any]) -> None:
        fragment_id = job["fragment_id"]
        try:
            fragment = self.store.get(fragment_id)
            # Deleted or no-longer-eligible memories simply have nothing to extract.
            if eligible(fragment):
                self.extract_fragment(job["agent_id"], fragment, job["occurrence"])
            self.store.fact_job_complete(fragment_id, self.name)
        except Exception as exc:  # noqa: BLE001 - recorded on the job and retried
            status = self.store.fact_job_fail(fragment_id, self.name, f"{type(exc).__name__}: {exc}",
                                              max_attempts=self.max_attempts)
            logger.warning("fact extraction job %s failed (%s): %s", fragment_id, status, exc)


def _with_occurrences(fragments: Iterable[MemoryFragment]) -> list[tuple[MemoryFragment, int]]:
    """Pair each fragment with how many identical texts preceded it in the batch.

    Two identical lines in one batch (a status that went back and forth on
    one day) are distinct events; the same line re-ingested later is not.
    """
    seen: dict[str, int] = {}
    paired: list[tuple[MemoryFragment, int]] = []
    for fragment in fragments:
        text = " ".join(fragment.content.split())
        paired.append((fragment, seen.get(text, 0)))
        seen[text] = seen.get(text, 0) + 1
    return paired


def _aware(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
