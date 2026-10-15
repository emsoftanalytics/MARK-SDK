"""Memory kinds for the session brief: todo, decision, bug, fix, change, done.

Kinds can be recorded three ways, and all three are read so the brief works
with what agents actually write (usage test 3: "TODO:", "DECISION:", "BUG:",
"FIX:" prefixes) as well as with structured writes:

1. ``metadata["kind"]`` (e.g. ``store_sync(..., metadata={"kind": "todo"})``)
2. a tag ``kind:<name>`` / ``mark:kind:<name>`` or a bare tag such as ``todo``
3. a content prefix: ``TODO:``, ``DECISION:``/``ADR-7:``, ``BUG:``/``ERROR:``,
   ``FIX:``/``FIXED:``, ``CHANGE:``, ``DONE:``/``RESOLVED:`` (an optional
   leading ``[2026-03-01]`` date is skipped)

A TODO counts as closed when its own status/tags say so (``status`` done /
resolved / closed, tag ``done`` / ``resolved``), when another memory
``resolves`` / ``supersedes`` its id (``metadata``), or when a later
DONE/FIX/RESOLVED note names its id or repeats most of its words.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from mark.context.watch_lines import words

#: Canonical kind for each spelling found in metadata, tags or prefixes.
_KINDS = {
    "todo": "todo", "task": "todo",
    "decision": "decision", "adr": "decision",
    "bug": "bug", "error": "bug", "incident": "bug", "fix": "fix", "fixed": "fix",
    "change": "change", "changed": "change", "switched": "change", "migration": "change",
    "done": "done", "resolved": "done", "closed": "done",
}
_PREFIX = re.compile(r"^\s*(?:\[[^\]]{1,40}\]\s*)?(TODO|DECISION|ADR(?:[-\s]?\d+)?|BUG|ERROR|FIX(?:ED)?|"
                     r"CHANGE|DONE|RESOLVED|INCIDENT)\b\s*[:\-]", re.IGNORECASE)
_CLOSED_STATUS = {"done", "resolved", "closed", "fixed", "complete", "completed"}
_STOP = frozenset("a an and the to of for in on at is are be with from by it this that todo done "
                  "fix fixed resolved bug decision".split())
#: Share of a TODO's content words a DONE note must repeat to close it.
_CLOSE_OVERLAP = 0.6


def _meta(fragment: Any) -> dict[str, Any]:
    meta = getattr(fragment, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def kind_of(fragment: Any) -> str | None:
    """The fragment's canonical kind, or None."""
    raw = str(_meta(fragment).get("kind") or "").lower()
    if raw in _KINDS:
        return _KINDS[raw]
    for tag in getattr(fragment, "tags", None) or ():
        lowered = str(tag).lower()
        name = lowered.rsplit(":", 1)[-1]
        if name not in _KINDS:
            continue
        if lowered.startswith(("kind:", "mark:kind:")):
            return _KINDS[name]
        # A bare "done"/"resolved" tag is a status of the memory, not its kind.
        if ":" not in lowered and _KINDS[name] != "done":
            return _KINDS[name]
    match = _PREFIX.match(getattr(fragment, "content", "") or "")
    if match:
        word = re.match(r"[A-Za-z]+", match.group(1)).group(0).lower()
        return _KINDS.get(word)
    return None


def when_of(fragment: Any) -> datetime | None:
    """When the memory happened: its declared event time (``occurred_at`` or a
    leading date in the text, as retrieval reads it), else when it was stored."""
    try:
        from mark.intelligence.temporal_order import declared_event_time

        declared = declared_event_time(fragment)
    except Exception:   # duck-typed fragments without the fields retrieval reads
        declared = None
    return declared or getattr(fragment, "created_at", None)


def recent_first(fragments: list[Any]) -> list[Any]:
    """Newest first (undated last)."""
    def key(fragment: Any) -> tuple[int, float]:
        when = when_of(fragment)
        if when is None:
            return (0, 0.0)
        return (1, when.timestamp())
    return sorted(fragments, key=key, reverse=True)


def _content_words(fragment: Any) -> frozenset[str]:
    return frozenset(w for w in words(getattr(fragment, "content", "") or "") if w not in _STOP)


def closed_todos(fragments: list[Any]) -> set[str]:
    """Ids of TODOs that are done (see the module rules)."""
    todos = [f for f in fragments if kind_of(f) == "todo"]
    closers = [f for f in fragments if kind_of(f) in ("done", "fix")]
    resolved_ids: set[str] = set()
    for fragment in fragments:
        meta = _meta(fragment)
        for key in ("resolves", "supersedes", "closes"):
            value = meta.get(key)
            resolved_ids.update(value if isinstance(value, (list, tuple)) else [value] if value else [])
    closed: set[str] = set()
    for todo in todos:
        meta = _meta(todo)
        tags = {str(t).lower() for t in getattr(todo, "tags", None) or ()}
        if (str(meta.get("status") or "").lower() in _CLOSED_STATUS or meta.get("resolved")
                or tags & {"done", "resolved", "status:done", "status:resolved"} or todo.id in resolved_ids):
            closed.add(todo.id)
            continue
        target = _content_words(todo)
        for closer in closers:
            text = getattr(closer, "content", "") or ""
            if todo.id[:8] and todo.id[:8] in text:
                closed.add(todo.id)
                break
            if target and len(target & _content_words(closer)) / len(target) >= _CLOSE_OVERLAP:
                closed.add(todo.id)
                break
    return closed
