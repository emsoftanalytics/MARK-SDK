"""Time in retrieval: ``since`` / ``until`` scoping and dates named in a query.

Two uses of event time (``occurred_at`` metadata, a leading date stamp, or
when the memory was stored):

- **Scope.** ``SessionFilter(since=..., until=...)`` keeps only memories whose
  event time falls in the window, before anything is scored. Bounds accept
  datetimes, dates, ISO text or natural expressions ("last week").
- **Evidence.** A query naming a date ("What did we discuss on 13 October?",
  "What did we talk about last Tuesday?", "Which invoices are due on
  5 March 2026?") counts the date as one more query term: a memory matches
  it when its event date, or a date written in its text ("due 05/03/2026",
  "next Tuesday" said on 2026-10-08), falls in the named range. Memories
  from that range are also added to the candidate pool when there are few
  of them, since nothing in their wording may match the question.

Expects the host class to provide ``_store`` and ``_get_many``.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from functools import lru_cache

from mark.intelligence.date_parsing import DateSpan, as_date as as_day, find_dates, parse_when
from mark.intelligence.temporal_order import declared_event_time, event_time
from mark.types import GapReport, GapSeverity, MemoryFragment

_ISO_DAY_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Scanning event times costs one decode per memory; above this many memories
# in scope, date-named candidates are not injected (scoring still uses dates).
_SCAN_LIMIT = 20_000
# Inject date-matched memories only when the range is selective.
_INJECT_LIMIT = 50
_CHUNK = 900
# Dates are looked for in the first part of very long memories only.
_DATE_SCAN_CHARS = 20_000

DateRange = tuple[date, date]


def query_date_ranges(query: str, now: datetime | None = None) -> tuple[list[DateRange], str]:
    """Calendar ranges named in *query* and the query with those expressions removed."""
    spans = find_dates(query, now or datetime.now(timezone.utc))
    stripped = query
    for span in reversed(spans):
        stripped = stripped[: span.pos[0]] + " " + stripped[span.pos[1]:]
    return [(span.start, span.end) for span in spans], stripped


def content_dates(fragment: MemoryFragment) -> tuple[DateSpan, ...]:
    """Dates written in a memory, relative ones resolved against the memory's own date.

    The reference is the declared event time, else the first ISO date in the
    text ("said Ana on 2026-10-08"), else when the memory was stored.
    """
    reference: date | datetime | None = declared_event_time(fragment)
    if reference is None:
        iso = _ISO_DAY_RE.search(fragment.content)
        if iso is not None:
            try:
                reference = date.fromisoformat(iso.group(1))
            except ValueError:
                reference = None
    if reference is None:
        reference = event_time(fragment)
    return _dates_in(fragment.content[:_DATE_SCAN_CHARS], as_day(reference))


@lru_cache(maxsize=2048)
def _dates_in(text: str, reference: date) -> tuple[DateSpan, ...]:
    """Cached: the same candidates are tested for every date-naming query."""
    return tuple(find_dates(text, reference))


def matches_dates(fragment: MemoryFragment, ranges: list[DateRange]) -> bool:
    """True when the memory's event date or a date in its text falls in any range."""
    when = event_time(fragment).date()
    if any(start <= when <= end for start, end in ranges):
        return True
    return any(span.overlaps(start, end) for span in content_dates(fragment) for start, end in ranges)


def window_gap(query: str, ranges: list[DateRange] | None, fragments: list[MemoryFragment], gap: GapReport):
    """A high gap naming the asked period when no result falls in it ("What did I sign in June?").

    Results stay (they may still help), but the report says plainly that
    they are from other dates instead of presenting them as the answer.
    """
    if not ranges or not fragments or any(matches_dates(f, ranges) for f in fragments):
        return gap
    named = ", ".join(span.text.strip() for span in find_dates(query)) or "the asked period"
    return GapReport(severity=GapSeverity.HIGH, suggested_action="none", missing_coverage=1.0,
                     confidence=min(gap.confidence, 0.25),
                     reason=f"Nothing in memory is dated {named}; the results are from other dates.")


def time_bounds(since: object, until: object) -> tuple[datetime | None, datetime | None]:
    """Parse ``since`` / ``until`` (inclusive; a date-only ``until`` covers its whole day)."""
    return parse_when(since), parse_when(until, end=True)  # type: ignore[arg-type]


class _TimeMixin:
    """since/until scoping and candidates for date-naming queries."""

    def _time_scope(self, agent_id: str, allowed_ids: set[str], since: object, until: object) -> set[str]:
        """Allowed ids whose event time lies within [since, until]."""
        lower, upper = time_bounds(since, until)
        if lower is None and upper is None:
            return allowed_ids
        # A store may answer this in SQL; otherwise decode and test each memory.
        between = getattr(self._store, "fragment_ids_between", None)
        if between is not None:
            return set(between(agent_id, lower, upper)) & allowed_ids
        kept: set[str] = set()
        for fid, fragment in self._iter_fragments(allowed_ids):
            when = event_time(fragment)
            if (lower is None or when >= lower) and (upper is None or when <= upper):
                kept.add(fid)
        return kept

    def _date_candidates(self, allowed_ids: set[str], ranges: list[DateRange]) -> dict[str, MemoryFragment]:
        """Memories whose event date falls in the query's ranges, when that set is small."""
        if not ranges or len(allowed_ids) > _SCAN_LIMIT:
            return {}
        found: dict[str, MemoryFragment] = {}
        for fid, fragment in self._iter_fragments(allowed_ids):
            when = event_time(fragment).date()
            if any(start <= when <= end for start, end in ranges):
                found[fid] = fragment
                if len(found) > _INJECT_LIMIT:
                    return {}
        return found

    def _iter_fragments(self, ids: set[str]):
        ordered = sorted(ids)
        for i in range(0, len(ordered), _CHUNK):
            yield from self._get_many(ordered[i:i + _CHUNK]).items()
