"""Background thread that drains the durable fact-extraction queue.

The worker owns no state of its own: jobs live in the store's ``fact_jobs``
table, so stopping the process (or the worker) loses nothing — the next
worker for the same database picks the jobs up. It wakes when a write
enqueues work and otherwise polls every ``poll_interval`` seconds, which
also retries failed jobs that are still below their attempt limit.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mark.intelligence.fact_extraction.service import FactExtractionService

logger = logging.getLogger(__name__)


class FactExtractionWorker:
    """Daemon thread calling ``service.run_pending()`` until stopped."""

    def __init__(self, service: "FactExtractionService", *, poll_interval: float = 5.0) -> None:
        self._service = service
        self._poll_interval = poll_interval
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"mark-facts-{service.name}", daemon=True)

    def start(self) -> None:
        self._wake.set()  # drain anything left over from an earlier run
        self._thread.start()

    def wake(self) -> None:
        """Signal that new jobs were enqueued."""
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to finish its current job and exit."""
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._poll_interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._service.run_pending()
            except Exception:  # noqa: BLE001 - e.g. the store was closed under us
                logger.warning("fact extraction worker pass failed", exc_info=True)
