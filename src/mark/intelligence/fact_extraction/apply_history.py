"""Timeline-aware recording: earlier values, past values, misdated values, multi-valued values.

:func:`~.apply.record_candidate` hands candidates with these flags here:

  - ``previous`` (the "A" of "from A to B" at time T): A must appear in
    history ending at T. If A is already known before T nothing changes. If
    A is only known from an *undated* statement that was stamped later than
    T (dev M-e: "runs on 9090" said today, then "moved from 9090 to 9191 on
    Oct 1"), that stamp was the recording time, not the event time: the row
    is withdrawn and A is recorded just before T, so B becomes current and
    the history reads A -> B. A *dated* later A is a genuine return to A and
    is left alone;
  - ``change`` without "from" ("I moved to Porto in September"): the same
    reasoning for the current value of the slot, when it is undated and
    stamped after the move;
  - ``retract`` + ``historical`` ("I used to work at X"): X held in the past.
    A current X is ended at the statement time (history keeps it); an
    unknown X is recorded as a past value placed before the slot's known
    history, never current;
  - ``multi``: ``facts.add`` instead of ``facts.set`` (values sit side by
    side); a preference also ends the opposite preference for the same
    thing ("hates jazz" ends "likes jazz").

Stores without the multi-valued API fall back to the plain single-value path.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from mark.intelligence.fact_extraction.base import FactCandidate
from mark.intelligence.temporal_order import declared_event_time

_TICK = timedelta(microseconds=1)
_OPPOSITES = {"likes": ("hates", "dislikes"), "loves": ("hates", "dislikes"), "prefers": ("hates", "dislikes"),
              "hates": ("likes", "loves", "prefers"), "dislikes": ("likes", "loves", "prefers")}


def same(a: Any, b: Any) -> bool:
    return " ".join(str(a).lower().split()) == " ".join(str(b).lower().split())


def aware(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def is_multi(fact: dict[str, Any]) -> bool:
    return bool((fact.get("metadata") or {}).get("multi_valued"))


def supports_multi(store: Any) -> bool:
    return all(callable(getattr(store, name, None)) for name in ("fact_add", "fact_remove", "fact_history"))


def undated(store: Any, fact: dict[str, Any]) -> bool:
    """True when *fact* came from a memory that gave no event time (its start is when it was stored)."""
    source = fact.get("source_fragment_id")
    getter = getattr(store, "get", None)
    if not source or not callable(getter):
        return False
    fragment = getter(source)
    if fragment is None or declared_event_time(fragment) is not None:
        return False
    return abs((aware(fact["valid_from"]) - aware(fragment.created_at)).total_seconds()) < 1.0


def add_multi(store: Any, agent_id: str, candidate: FactCandidate, when: datetime,
              source: str | None, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """``facts.add`` the value; None when it was already current (a re-run)."""
    for attribute in _OPPOSITES.get(candidate.attribute.lower(), ()):
        end_value(store, agent_id, candidate.subject, attribute, candidate.value, when)
    if candidate.attribute.lower() == "prefers":
        # "prefers coffee over tea" ends an earlier "prefers tea over coffee".
        for fact in store.fact_current(agent_id, candidate.subject, candidate.attribute):
            if _flipped(str(fact["value"]), candidate.value):
                end_value(store, agent_id, candidate.subject, candidate.attribute, fact["value"], when)
    started = datetime.now(timezone.utc)
    fact = store.fact_add(agent_id, candidate.subject, candidate.attribute, candidate.value, valid_from=when,
                          source_fragment_id=source, metadata=metadata)
    return fact if aware(fact["recorded_at"]) >= started else None


def _flipped(old: str, new: str) -> bool:
    """True when two comparisons name the same pair the other way round ("A over B" / "B over A")."""
    pairs = [re.split(r"\s+(?:over|to|than|instead\s+of|rather\s+than)\s+", v.lower(), maxsplit=1) for v in (old, new)]
    return all(len(p) == 2 for p in pairs) and pairs[0][0].strip() == pairs[1][1].strip() \
        and pairs[0][1].strip() == pairs[1][0].strip()


def end_value(store: Any, agent_id: str, subject: str, attribute: str, value: Any, when: datetime) -> bool:
    """End *value* at *when* if it is current and began by then; True when something ended.

    A multi-valued row is closed in place; a single value has no "end
    without successor" in the store, so it is withdrawn and re-added as a
    closed multi-valued row with its original start: history keeps it.
    """
    for fact in store.fact_current(agent_id, subject, attribute):
        if not same(fact["value"], value) or aware(fact["valid_from"]) > aware(when):
            continue
        if not is_multi(fact):
            store.fact_retract(agent_id, subject, attribute, value=fact["value"])
            meta = {**(fact.get("metadata") or {}), "ended_by": "statement"}
            store.fact_add(agent_id, subject, attribute, fact["value"], valid_from=aware(fact["valid_from"]),
                           source_fragment_id=fact.get("source_fragment_id"), metadata=meta)
        store.fact_remove(agent_id, subject, attribute, fact["value"], valid_to=aware(when))
        return True
    return False


def record_previous(store: Any, agent_id: str, candidate: FactCandidate, when: datetime,
                    source: str | None, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Make the "A" of "from A to B" history that ends at *when* (see module docstring)."""
    subject, attribute, value = candidate.subject, candidate.attribute, candidate.value
    history = store.fact_history(agent_id, subject, attribute)
    rows = [f for f in history if same(f["value"], value)]
    if any(aware(f["valid_from"]) < aware(when) for f in rows):
        # Known before the change. An added value stays current until removed: end it now.
        end_value(store, agent_id, subject, attribute, value, when) if any(is_multi(f) for f in rows) else None
        return None
    if rows and not all(undated(store, f) for f in rows):
        return None  # a dated later statement: the value really came back
    if rows:
        store.fact_retract(agent_id, subject, attribute, value=value)  # misdated: stamped when it was told
    # At the change time itself (recorded before B, so B is the later of the two),
    # unless another value already sits at that exact time.
    clash = any(aware(f["valid_from"]) == aware(when) for f in history if not same(f["value"], value))
    start = aware(when) - _TICK if clash else aware(when)
    meta = {**metadata, "previous_value": True}
    if candidate.multi:
        fact = store.fact_add(agent_id, subject, attribute, value, valid_from=start, source_fragment_id=source,
                              metadata=meta)
        store.fact_remove(agent_id, subject, attribute, value, valid_to=aware(when))
        return fact
    return store.fact_set(agent_id, subject, attribute, value, valid_from=start, source_fragment_id=source,
                          metadata=meta)


def record_past(store: Any, agent_id: str, candidate: FactCandidate, when: datetime,
                source: str | None, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """"Used to": end a current value at *when*, or record an unknown one as past history."""
    subject, attribute, value = candidate.subject, candidate.attribute, candidate.value
    if end_value(store, agent_id, subject, attribute, value, when):
        return {"subject": subject, "attribute": attribute, "value": value, "ended": True}
    history = store.fact_history(agent_id, subject, attribute)
    if any(same(f["value"], value) for f in history):
        return None  # already in history
    # Its start is unknown: place it just before everything known about the slot.
    end = min([aware(when), *(aware(f["valid_from"]) for f in history)])
    meta = {**metadata, "past_value": True, "valid_from_unknown": True}
    fact = store.fact_add(agent_id, subject, attribute, value, valid_from=end - _TICK, source_fragment_id=source,
                          metadata=meta)
    store.fact_remove(agent_id, subject, attribute, value, valid_to=end)
    return fact


def demote_misdated(store: Any, agent_id: str, candidate: FactCandidate, when: datetime) -> None:
    """Before recording a dated change, move undated values stamped after it to just before it."""
    for fact in store.fact_current(agent_id, candidate.subject, candidate.attribute):
        if is_multi(fact) or same(fact["value"], candidate.value) or aware(fact["valid_from"]) < aware(when):
            continue
        rows = [f for f in store.fact_history(agent_id, candidate.subject, candidate.attribute)
                if same(f["value"], fact["value"])]
        if not rows or not all(undated(store, f) for f in rows):
            continue
        store.fact_retract(agent_id, candidate.subject, candidate.attribute, value=fact["value"])
        meta = {**(fact.get("metadata") or {}), "redated_before": aware(when).isoformat()}
        store.fact_set(agent_id, candidate.subject, candidate.attribute, fact["value"],
                       valid_from=aware(when) - _TICK, source_fragment_id=fact.get("source_fragment_id"),
                       metadata=meta)
