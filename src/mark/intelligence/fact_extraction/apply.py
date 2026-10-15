"""Apply extracted candidates to the store: assertions, corrections and retractions.

The extraction service records every candidate through
:func:`record_candidate`, so the meaning of the flags lives in one place:

  - plain assertion -> ``fact_set_extracted`` (idempotent per ledger key);
  - ``correction=True`` -> the current value of the slot was *wrong*: it is
    retracted (kept for audit) before the new value is recorded, so it no
    longer shows up in history either;
  - ``retract=True`` -> the text denies the value ("no longer runs on port
    8080"): it is retracted only if it is the current value and was not
    asserted *after* the denial, so an old note can never undo a newer fact.

Every operation is safe to repeat: re-running a job finds nothing left to
withdraw and the ledger skips the assertion.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mark.intelligence.fact_extraction.base import FactCandidate


def candidates_for(extractor: Any, text: str, *, occurred_at: datetime | None) -> list[FactCandidate]:
    """``extract_changes`` when the extractor offers it (retractions included), else ``extract``."""
    changes = getattr(extractor, "extract_changes", None)
    if callable(changes) and not _extract_overridden(extractor):
        return list(changes(text, occurred_at=occurred_at))
    return list(extractor.extract(text, occurred_at=occurred_at))


def _extract_overridden(extractor: Any) -> bool:
    """True when a subclass overrides ``extract`` below the class that defines ``extract_changes``.

    A wrapper that customises ``extract`` (filtering, gating, failing on
    purpose) must keep being called; ``extract_changes`` would bypass it.
    """
    for cls in type(extractor).__mro__:
        if "extract_changes" in cls.__dict__:
            return False
        if "extract" in cls.__dict__:
            return True
    return False


def record_candidate(
    store: Any,
    agent_id: str,
    candidate: FactCandidate,
    *,
    key: str,
    valid_from: datetime,
    source_fragment_id: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Record one candidate; returns the fact written/withdrawn, or None when nothing changed."""
    metadata = dict(metadata)
    if candidate.tags:
        metadata["tags"] = list(candidate.tags)
    if candidate.retract:
        return _retract_current(store, agent_id, candidate, valid_from)
    if candidate.correction:
        metadata["correction"] = True
        _withdraw_replaced(store, agent_id, candidate, valid_from)
    return store.fact_set_extracted(
        agent_id, key, candidate.subject, candidate.attribute, candidate.value,
        valid_from=valid_from, source_fragment_id=source_fragment_id, metadata=metadata,
    )


def _same(a: str, b: str) -> bool:
    return " ".join(str(a).lower().split()) == " ".join(str(b).lower().split())


def _not_newer(fact: dict[str, Any], when: datetime) -> bool:
    """True when *fact* took effect at or before *when* (the statement that changes it)."""
    start = fact.get("valid_from")
    if not isinstance(start, datetime):
        return True
    start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return start <= when


def _retract_current(store: Any, agent_id: str, candidate: FactCandidate, when: datetime) -> dict[str, Any] | None:
    current = store.fact_current(agent_id, candidate.subject, candidate.attribute)
    match = next((f for f in current if _same(f["value"], candidate.value) and _not_newer(f, when)), None)
    if match is None:
        return None
    store.fact_retract(agent_id, candidate.subject, candidate.attribute, value=match["value"])
    return {**match, "retracted": True}


def _withdraw_replaced(store: Any, agent_id: str, candidate: FactCandidate, when: datetime) -> None:
    for fact in store.fact_current(agent_id, candidate.subject, candidate.attribute):
        if not _same(fact["value"], candidate.value) and _not_newer(fact, when):
            store.fact_retract(agent_id, candidate.subject, candidate.attribute, value=fact["value"])
