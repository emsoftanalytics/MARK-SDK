"""Assemble a subject's timeline: collect, filter, deduplicate, order, label.

Collection (no top-k cap anywhere):
  - memories: every retrievable, unexpired, unquarantined fragment whose
    content contains the subject as a *whole anchor* (``_contains_anchor``
    on ``_anchor_norm`` text, so "titan" never matches "titanium" and
    "atlas core" never matches "atlas cored"). The store pages through FTS5
    phrase candidates (``mention_candidates``); the anchor check here is the
    authority, since FTS candidates are a stemmed superset;
  - facts: the subject's history for ``attribute``, or for every attribute
    that has facts.

With ``attribute`` a memory must also contain every content term of the
attribute (stemmed: "team" matches "teams"), mirroring how temporal recall
picks a subject's records about the asked attribute.

Deduplication: a fact whose ``source_fragment_id`` is one of the collected
memories is folded into that memory's event (its ``fact_id`` is set), and
memories repeating the same text at the same declared time appear once.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from mark.intelligence.query_understanding import _TOKEN_RE, _anchor_norm, _contains_anchor, _content_terms, _stem
from mark.intelligence.temporal_order import declared_event_time
from mark.memory.facts import _when
from mark.memory.timeline import Timeline, TimelineEvent
from mark.memory.timeline_order import BASIS_DECLARED, two_clock_order, utc
from mark.types import MemoryFragment


def build_timeline(
    store: Any,
    agent_id: str,
    subject: str,
    attribute: str | None = None,
    *,
    limit: int | None = None,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    page_size: int = 1000,
) -> Timeline:
    """Build the ordered timeline of *subject* (optionally one *attribute*).

    ``since``/``until`` (inclusive) bound each event's own time: the declared
    time of dated events, the storage time of undated ones. ``limit`` keeps
    the newest *limit* events; same-day labels are computed before trimming
    so "same day 2/3" stays true.
    """
    anchor = _anchor_norm(subject)
    if not anchor:
        raise ValueError("timeline subject must contain letters or digits")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    attribute = attribute.strip() if attribute and attribute.strip() else None
    attribute_terms = [t for t in _content_terms(attribute) if t not in anchor.split()] if attribute else []

    memories = _mentions(store, agent_id, subject, anchor, attribute_terms, page_size)
    facts = _facts(store, agent_id, subject, attribute)
    by_fragment = {fragment.id for fragment, _ in memories}
    folded: dict[str, str] = {}  # fragment id -> fact id extracted from it
    # Events are collected as keyword dicts and frozen once their order,
    # basis and same-day position are known.
    candidates: list[dict[str, Any]] = []
    for fact in facts:
        source = fact.get("source_fragment_id")
        if source and source in by_fragment and source not in folded:
            folded[source] = fact["id"]
            continue
        valid_from = utc(fact["valid_from"])
        candidates.append(dict(
            when=valid_from, kind="fact", dated=True, fact_id=fact["id"],
            text=f"{fact['subject']} {fact['attribute']} = {fact['value']}",
            stored_at=utc(fact["recorded_at"]) if fact.get("recorded_at") else valid_from,
        ))
    for fragment, declared in memories:
        stored = utc(fragment.created_at)
        candidates.append(dict(
            when=declared or stored, kind="memory", dated=declared is not None,
            text=fragment.content, fragment_id=fragment.id, stored_at=stored,
        ))
    for event in candidates:
        if event["kind"] == "memory" and event["fragment_id"] in folded:
            event["fact_id"] = folded[event["fragment_id"]]

    ordered = two_clock_order(
        candidates, lambda e: e["when"] if e["dated"] else None, lambda e: e["stored_at"],
    )
    for event, basis in ordered:
        event["basis"] = basis
    _label_same_day([event for event, _ in ordered])
    events = [TimelineEvent(**event) for event, _ in ordered]
    low, high = _when(since), _when(until)
    if low is not None or high is not None:
        events = [e for e in events if e.when is None
                  or ((low is None or e.when >= low) and (high is None or e.when <= high))]
    truncated = 0
    if limit is not None and len(events) > limit:
        truncated = len(events) - limit
        events = events[truncated:]
    return Timeline(subject=subject, attribute=attribute, events=events, truncated=truncated)


def _mentions(
    store: Any, agent_id: str, subject: str, anchor: str, attribute_terms: list[str], page_size: int,
) -> list[tuple[MemoryFragment, datetime | None]]:
    """Every retrievable memory naming *anchor* as a whole (and the attribute terms).

    Returns (fragment, declared event time) pairs in storage order;
    identical text at the same declared time (a re-ingested line) is kept
    once, the first-stored copy winning.
    """
    if not hasattr(store, "mention_candidates"):
        return []
    stem_cache: dict[str, str] = {}  # a subject's records repeat most words

    def accept(content: str) -> bool:
        # Runs on raw content before the store decodes the row, so FTS false
        # positives (stemmed / diacritic-folded matches) cost almost nothing.
        if not _contains_anchor(_anchor_norm(content), anchor):
            return False
        if not attribute_terms:
            return True
        stems = set()
        for token in _TOKEN_RE.findall(content.lower()):
            stem = stem_cache.get(token)
            if stem is None:
                stem = stem_cache[token] = _stem(token)
            stems.add(stem)
        return all(term in stems for term in attribute_terms)

    found: list[tuple[MemoryFragment, datetime | None]] = []
    seen: set[tuple[str, str]] = set()
    for page in store.mention_candidates(agent_id, subject, page_size=page_size, accept=accept):
        for fragment in page:
            declared = declared_event_time(fragment)
            key = (_anchor_norm(fragment.content), declared.isoformat() if declared else "")
            if key in seen:
                continue
            seen.add(key)
            found.append((fragment, declared))
    return found


def _facts(store: Any, agent_id: str, subject: str, attribute: str | None) -> list[dict[str, Any]]:
    """Fact history rows for the subject (one attribute, or all of them)."""
    if not hasattr(store, "fact_history"):
        return []
    if attribute is not None:
        return list(store.fact_history(agent_id, subject, attribute))
    rows: list[dict[str, Any]] = []
    # Every non-retracted (subject, attribute) timeline has exactly one
    # current row, so the current rows enumerate the subject's attributes.
    for current in store.fact_current(agent_id, subject):
        rows.extend(store.fact_history(agent_id, subject, current["attribute"]))
    return rows


def _label_same_day(events: list[dict[str, Any]]) -> None:
    """Set ``same_day=(k, m)`` on dated events whose date is shared (in place).

    Events are already in order, so k follows the timeline order (declared
    time, then storage order). Only events placed by declared time count.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event["basis"] == BASIS_DECLARED:
            groups.setdefault(event["when"].date().isoformat(), []).append(event)
    for members in groups.values():
        if len(members) > 1:
            for k, event in enumerate(members, 1):
                event["same_day"] = (k, len(members))
