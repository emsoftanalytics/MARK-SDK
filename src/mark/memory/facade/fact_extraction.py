# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Write-path hook that turns stored memories into facts.

Called by the store methods *after* a write has completed (and after the
middleware stack, governance, redaction and the consolidation gate ran), so
extraction sees exactly what was persisted. Fragments are re-read from the
store by id for that reason; quarantined or otherwise non-retrievable ones
are skipped by the service.

Choosing an extractor (``extract_facts`` argument of the store methods):
  - ``None``  (default) use the runtime's configured extractor, if any;
  - ``True``  use the configured extractor, else the built-in pattern packs;
  - ``False`` never extract for this call.

Extraction never makes a write fail: every error is logged and swallowed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mark.memory.facade.core import MemoryCore

if TYPE_CHECKING:
    from mark.intelligence.fact_extraction.service import FactExtractionService

logger = logging.getLogger(__name__)

# Built-in pattern extractors shared by every memory that asks for
# extract_facts=True without a configured extractor, one per user subject
# (who "I"/"my" is; the store's ``mark config user-subject``).
_DEFAULT_PATTERNS: dict[str, Any] = {}


class FactExtractionMixin(MemoryCore):
    """Resolve the fact extractor for a write and run it on the stored fragments."""

    #: Facts written inline by the most recent write on this memory object.
    last_extracted: int = 0

    def _fact_service(self, extract_facts: bool | None) -> "FactExtractionService | None":
        """The service to use for one write, or None when extraction is off."""
        if extract_facts is False:
            return None
        configured = getattr(self, "_fact_extraction", None)
        if configured is not None or not extract_facts:
            return configured
        from mark.intelligence.fact_extraction import FactExtractionService, PatternFactExtractor

        # The user subject applied to this runtime (stored with the store), so
        # "I live in Lisbon" becomes "Sam city = Lisbon", not "user city".
        subject = getattr(getattr(self, "_pipeline", None), "_user_subject", None) or "user"
        extractor = _DEFAULT_PATTERNS.get(subject)
        if extractor is None:
            extractor = _DEFAULT_PATTERNS.setdefault(subject, PatternFactExtractor(user_subject=subject))
        # Inline extractor: no worker thread is started, so this is cheap.
        return FactExtractionService(self._store, extractor, start_worker=False)

    def _extract_stored(self, fragment_ids: list[str], extract_facts: bool | None) -> int:
        """Extract facts from just-written fragments (by id, as stored); never raises.

        Returns how many facts were written inline (0 when extraction is off
        or queued for the background worker) and keeps it on
        ``last_extracted`` so callers that only get ids back can report it.
        """
        count = 0
        if fragment_ids:
            try:
                service = self._fact_service(extract_facts)
                if service is not None:
                    stored = self._store.get_many(fragment_ids)
                    count = service.submit(self.agent_id, [stored.get(fid) for fid in fragment_ids])
            except Exception:  # noqa: BLE001 - extraction must never fail a write
                logger.warning("fact extraction after write failed", exc_info=True)
        self.last_extracted = count
        return count
