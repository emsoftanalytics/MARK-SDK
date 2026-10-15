"""Temporal ordering: answer "current" / "original" questions by event time.

Relevance ranking alone cannot tell which of several records about the same
subject is the latest. For temporal queries the records about the asked
subject and attribute (the "band") are reordered by when the remembered
event happened, while every other result keeps its rank.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from mark.intelligence.query_understanding import (
    TemporalIntent,
    _anchor_norm,
    _contains_anchor,
    _content_stems,
    _content_terms,
    detect_temporal_intent,
    query_anchors,
)
from mark.types import MemoryFragment

# Event time: only a date stamp at the very start of a memory counts
# ("[2026-03-05] ...", "2026-03-05T10:00Z: ..."). Dates elsewhere in the text
# ("drill scheduled for 2026-04-16") describe other events and are ignored.
_LEADING_DATE_RE = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?))?")


def declared_event_time(fragment: MemoryFragment) -> datetime | None:
    """The event time a memory declares, or None.

    Explicit ``metadata["occurred_at"]`` wins; otherwise a date stamp at the
    very start of the content.
    """
    raw = fragment.metadata.get("occurred_at") if fragment.metadata else None
    if raw:
        parsed = _parse_when(str(raw))
        if parsed is not None:
            return parsed
    match = _LEADING_DATE_RE.match(fragment.content)
    if match:
        return _parse_when(match.group(1) + (f"T{match.group(2)}" if match.group(2) else ""))
    return None


def event_time(fragment: MemoryFragment) -> datetime:
    """When the remembered event happened: declared time, else when it was stored."""
    declared = declared_event_time(fragment)
    if declared is not None:
        return declared
    stamp = fragment.created_at
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _parse_when(text: str) -> datetime | None:
    """Parse an ISO date/datetime; naive values are taken as UTC."""
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _order_band(fragments: list[MemoryFragment], band: list[int], *, newest_first: bool) -> list[int]:
    """Order records about one subject/attribute by when they happened.

    Declared event times (occurred_at / leading date stamps) and storage
    times are different clocks and are never compared directly. Dated
    records are ordered by their declared time. An undated record goes
    first for "current" queries only when it was *stored* after every dated
    record (a live update after an imported log); otherwise undated records
    follow the dated ones in relevance order.
    """
    dated = [i for i in band if declared_event_time(fragments[i]) is not None]
    undated = [i for i in band if declared_event_time(fragments[i]) is None]
    if not dated:
        return sorted(band, key=lambda i: event_time(fragments[i]), reverse=newest_first)
    # Same declared time (e.g. two changes logged on one day): fall back to
    # storage order, which preserves the sequence of an ingested log.
    dated.sort(key=lambda i: (declared_event_time(fragments[i]), _stored_at(fragments[i])),
               reverse=newest_first)
    if not newest_first or not undated:
        return dated + undated
    last_dated_store = max(_stored_at(fragments[i]) for i in dated)
    later = sorted((i for i in undated if _stored_at(fragments[i]) > last_dated_store),
                   key=lambda i: _stored_at(fragments[i]), reverse=True)
    earlier = [i for i in undated if i not in later]
    return later + dated + earlier


def _stored_at(fragment: MemoryFragment) -> datetime:
    """When the memory was written (the storage clock), timezone-aware."""
    stamp = fragment.created_at
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def temporal_order(
    query: str,
    fragments: list[MemoryFragment],
    scores: list[float],
    relevances: list[float],
) -> tuple[list[MemoryFragment], list[float], list[float], TemporalIntent]:
    """Reorder results about the query's subject by event time.

    For CURRENT queries the newest matching record comes first; for ORIGINAL
    queries the oldest. Only fragments that mention every query anchor (e.g.
    the service name) and cover the remaining query terms at least as well as
    the best such fragment are reordered, so a newer record about a different
    attribute never jumps ahead. Other results keep their rank. Scores are
    re-assigned so they stay non-increasing.
    """
    intent = detect_temporal_intent(query)
    anchors = query_anchors(query)
    if intent == TemporalIntent.NONE or not anchors or len(fragments) < 2:
        return fragments, scores, relevances, intent
    anchor_words = {word for anchor in anchors for word in anchor.split()}
    attribute_terms = [t for t in _content_terms(query) if t not in anchor_words]
    subject: list[int] = []
    stems: dict[int, set[str]] = {}
    for i, fragment in enumerate(fragments):
        content_norm = _anchor_norm(fragment.content)
        if all(_contains_anchor(content_norm, anchor) for anchor in anchors):
            subject.append(i)
            stems[i] = _content_stems(fragment.content)
    if len(subject) < 2:
        return fragments, scores, relevances, intent
    # The band is the subject's records about the *asked* attribute: those
    # containing every attribute term that most of the subject's records
    # share. Incidental words ("listen" in "which port does X listen on")
    # don't split the band, and a record about another attribute ("deploy
    # failed") can't join it. No common term → no reordering at all.
    if attribute_terms:
        common = [t for t in attribute_terms if sum(t in stems[i] for i in subject) * 2 > len(subject)]
        if not common:
            return fragments, scores, relevances, intent
        band = [i for i in subject if all(t in stems[i] for t in common)]
    else:
        band = list(subject)
    if len(band) < 2:
        return fragments, scores, relevances, intent
    band = _order_band(fragments, band, newest_first=intent == TemporalIntent.CURRENT)
    band_set = set(band)
    rest = [i for i in range(len(fragments)) if i not in band_set]
    order = band + rest
    # Scores are permuted, never invented: the band takes its own scores in
    # descending order, and nothing after it may score higher (monotonic).
    band_scores = sorted((scores[i] for i in band), reverse=True)
    new_scores = list(band_scores)
    floor = band_scores[-1]
    for i in rest:
        floor = min(floor, scores[i])
        new_scores.append(floor)
    return ([fragments[i] for i in order], new_scores, [relevances[i] for i in order], intent)
