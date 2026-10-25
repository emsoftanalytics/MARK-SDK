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

Lifecycle (background mode): jobs are claimed a few at a time
(``batch_size``, default 4) so a slow extractor never sits on a long batch;
the lease (``lease_seconds``) is renewed as each job starts; on start the
service requeues jobs orphaned by dead processes; :meth:`close` stops the
worker cooperatively, waits a bounded time for the job in flight, and gives
every claimed-but-unfinished job back to the queue.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

from mark.intelligence.fact_extraction.apply import candidates_for, record_candidate
from mark.intelligence.fact_extraction.base import FactExtractor, extractor_name, runs_in_background
from mark.intelligence.fact_extraction.patterns import PatternFactExtractor
from mark.intelligence.fact_extraction.service_queue import DEFAULT_WAIT_SECONDS, JobQueueMixin  # noqa: F401
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


class FactExtractionService(JobQueueMixin):
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
        batch_size: int = 4,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        self.store = store
        self.extractor = extractor
        self.name = extractor_name(extractor)
        self.background = runs_in_background(extractor)
        self.max_attempts = max_attempts
        self.lease_seconds = float(lease_seconds)
        self.batch_size = max(1, int(batch_size))
        self._worker = None
        self._stopping = threading.Event()
        self._held: set[str] = set()  # claimed by this service and not settled yet
        self._started: set[str] = set()  # the subset of _held already running
        self._held_lock = threading.Lock()
        if self.background:
            self._reclaim_orphans()
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
        candidates = candidates_for(self.extractor, fragment.content, occurred_at=event)
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
                fact = record_candidate(self.store, agent_id, candidate, key=key, valid_from=valid_from,
                                        source_fragment_id=fragment.id, metadata=metadata)
            except ValueError:
                logger.debug("rejected fact candidate %r", candidate)
                continue
            if fact is not None:
                written.append(fact)
        return written


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
