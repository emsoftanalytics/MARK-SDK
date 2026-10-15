"""Two-clock chronological ordering shared by timelines and chronological recall.

Records carry up to two clocks that must never be compared with each other:

  - the *declared* event time: when the remembered thing happened, from
    ``metadata["occurred_at"]``, a leading date stamp ("[2026-04-07] ..."),
    or a fact's ``valid_from``;
  - the *storage* time: when MARK wrote the record (``created_at`` /
    ``recorded_at``). An imported log can be stored months after the events
    it describes, so a storage time says nothing about where an undated
    record falls among dated ones.

The rule (the same one ``temporal_order._order_band`` applies to recall):

  1. dated records are ordered by declared time, then storage order (which
     preserves the logged sequence of several changes on one day);
  2. an undated record stored *after every dated record* is a live update
     that followed the imported history: it goes last, in storage order
     (basis ``"stored_after_dated"``);
  3. every other undated record cannot be placed: it follows the dated ones
     in storage order with basis ``"unknown"`` and never counts as latest;
  4. with no dated records at all, storage order is the only clock
     (basis ``"stored"``).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Sequence, TypeVar

from mark.intelligence.temporal_order import declared_event_time
from mark.types import MemoryFragment

if TYPE_CHECKING:
    from mark.intelligence.retrieval_result import RetrievalResult

T = TypeVar("T")

#: How an item's position was decided (see the module docstring).
BASIS_DECLARED = "declared"
BASIS_AFTER_DATED = "stored_after_dated"
BASIS_UNKNOWN = "unknown"
BASIS_STORED = "stored"

ORDERS = ("relevance", "chronological")


def utc(stamp: datetime) -> datetime:
    """Timezone-aware copy of *stamp* (naive values are taken as UTC)."""
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def two_clock_order(
    items: Sequence[T],
    declared: Callable[[T], datetime | None],
    stored: Callable[[T], datetime],
) -> list[tuple[T, str]]:
    """Order *items* oldest first under the two-clock rule; returns (item, basis).

    Sorting is stable, so items with identical keys keep their input order
    (callers pass storage/rowid order).
    """
    # Work on positions, so equal-but-distinct items never collide.
    when = [declared(item) for item in items]
    store = [utc(stored(item)) for item in items]
    dated = [i for i in range(len(items)) if when[i] is not None]
    undated = sorted((i for i in range(len(items)) if when[i] is None), key=lambda i: store[i])
    if not dated:
        return [(items[i], BASIS_STORED) for i in undated]
    dated.sort(key=lambda i: (utc(when[i]), store[i]))
    last_dated_store = max(store[i] for i in dated)
    later = [i for i in undated if store[i] > last_dated_store]
    unknown = [i for i in undated if store[i] <= last_dated_store]
    return (
        [(items[i], BASIS_DECLARED) for i in dated]
        + [(items[i], BASIS_UNKNOWN) for i in unknown]
        + [(items[i], BASIS_AFTER_DATED) for i in later]
    )


def fragment_order(fragments: Sequence[MemoryFragment]) -> list[tuple[int, str]]:
    """Indices of *fragments* in chronological order, with their basis."""
    return two_clock_order(
        list(range(len(fragments))),
        lambda i: declared_event_time(fragments[i]),
        lambda i: fragments[i].created_at,
    )


def chronological_result(result: RetrievalResult) -> RetrievalResult:
    """Re-sort a recall result's fragments oldest first (two-clock rule).

    Fragments, scores and relevances are permuted together, so each score
    still belongs to its fragment; scores are therefore no longer
    non-increasing. Which fragments were returned is unchanged.
    """
    if len(result.fragments) < 2:
        return result
    order = [i for i, _ in fragment_order(result.fragments)]

    def permute(values: list) -> list:
        return [values[i] for i in order] if len(values) == len(order) else list(values)

    return replace(
        result,
        fragments=[result.fragments[i] for i in order],
        scores=permute(result.scores),
        relevance=permute(result.relevance),
    )


def apply_order(result: RetrievalResult, order: str) -> RetrievalResult:
    """Apply a recall ``order`` option; ``"relevance"`` returns *result* untouched."""
    if order == "relevance":
        return result
    if order == "chronological":
        return chronological_result(result)
    raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
