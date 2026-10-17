# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
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

# Built-in pattern extractor shared by every memory that asks for
# extract_facts=True without a configured extractor (compiled once).
_DEFAULT_PATTERNS: Any = None


class FactExtractionMixin(MemoryCore):
    """Resolve the fact extractor for a write and run it on the stored fragments."""

    def _fact_service(self, extract_facts: bool | None) -> "FactExtractionService | None":
        """The service to use for one write, or None when extraction is off."""
        if extract_facts is False:
            return None
        configured = getattr(self, "_fact_extraction", None)
        if configured is not None or not extract_facts:
            return configured
        from mark.intelligence.fact_extraction import FactExtractionService, PatternFactExtractor

        global _DEFAULT_PATTERNS
        if _DEFAULT_PATTERNS is None:
            _DEFAULT_PATTERNS = PatternFactExtractor()
        # Inline extractor: no worker thread is started, so this is cheap.
        return FactExtractionService(self._store, _DEFAULT_PATTERNS, start_worker=False)

    def _extract_stored(self, fragment_ids: list[str], extract_facts: bool | None) -> None:
        """Extract facts from just-written fragments (by id, as stored); never raises."""
        if not fragment_ids:
            return
        try:
            service = self._fact_service(extract_facts)
            if service is None:
                return
            stored = self._store.get_many(fragment_ids)
            service.submit(self.agent_id, [stored.get(fid) for fid in fragment_ids])
        except Exception:  # noqa: BLE001 - extraction must never fail a write
            logger.warning("fact extraction after write failed", exc_info=True)
