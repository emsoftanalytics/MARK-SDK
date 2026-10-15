"""Data types and text helpers shared by the context watch.

``_Anchor`` is the watch's record of one subject in the conversation;
``Briefing`` is what it hands to the model. The helpers normalise anchors
the same way retrieval does, so "atlas-core", "Atlas Core" and "atlas_core"
are one subject everywhere.
"""
from __future__ import annotations

import re
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

_ROLES = {"user", "assistant", "tool", "system"}
_TOKEN_RE = re.compile(r"\w+")


@dataclass
class _Anchor:
    """One subject the conversation is about, with its recency and prefetch status."""

    key: str                    # normalised ("atlas core")
    label: str                  # as first seen ("atlas-core")
    first_turn: int
    last_turn: int
    last_seq: int = 0           # event sequence of the latest mention (recency order)
    mentions: int = 1
    known: bool | None = None   # None until a prefetch has answered
    linked: bool = False        # anticipated via a link, not mentioned by anyone yet
    assistant_only: bool = False  # only the model has named it (may be its own wording)


@dataclass(frozen=True)
class Briefing:
    """What the watch would put in front of the model right now."""
    text: str
    anchors: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    pending: int = 0            # prefetches still running (briefing may be incomplete)
    latency_ms: float = 0.0
    usage: Any = None           # BudgetUsage when a token budget was applied, else None
    items: tuple = ()           # all candidate BudgetItems (budgeted briefings only)

    def __bool__(self) -> bool:
        return bool(self.text)


# The three wrappers below import lazily so that importing mark.context does
# not load the retrieval stack (and cannot create an import cycle with it).

def _anchor_norm(text: str) -> str:
    """Retrieval's anchor normalisation ('Atlas-Core' -> 'atlas core')."""
    from mark.intelligence.query_understanding import _anchor_norm as norm

    return norm(text)


def _contains_anchor(content_norm: str, anchor: str) -> bool:
    """Whole-token anchor match on normalised text (same rule as retrieval)."""
    from mark.intelligence.query_understanding import _contains_anchor as contains

    return contains(content_norm, anchor)


def _timeline_key(fragment: Any) -> tuple[int, str, str]:
    """Sort key by declared event time, then storage time for same-day events.

    Undated memories get the lowest key (last when sorted newest first).
    """
    from mark.intelligence.temporal_order import declared_event_time

    when = declared_event_time(fragment)
    stored = fragment.created_at.isoformat() if getattr(fragment, "created_at", None) else ""
    return (1, when.isoformat(), stored) if when is not None else (0, "", stored)


def _mentions_value(value: str, lowered: str, tokens: set[str]) -> bool:
    """Does the text state *value*? Single words must match a whole token ('44' is not in '4444')."""
    value = value.lower().strip()
    if not value:
        return False
    if re.fullmatch(r"\w+", value):
        return value in tokens
    return value in lowered


def _label_for(text: str, key: str) -> str:
    """The anchor as written in the text (e.g. 'atlas-core'), falling back to the key."""
    pattern = r"(?<!\w)" + r"[\W_]+".join(re.escape(part) for part in key.split()) + r"(?!\w)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else key


def _one_line(text: str, limit: int = 240) -> str:
    """Collapse whitespace and cap the length for a briefing bullet."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _wait(future: Future, timeout: float | None) -> None:
    """Block until *future* finishes or *timeout* passes; its outcome is ignored here."""
    try:
        future.exception(timeout=timeout)
    except Exception:
        pass
