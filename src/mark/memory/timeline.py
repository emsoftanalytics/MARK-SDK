"""Subject timelines: the whole, ordered history of one subject.

Recall answers "what is relevant?" with a capped top-k. History questions
("which team is Tomasz Nowak on now?", "how did atlas-core's port change?")
need *every* record about the subject, in the order things happened — the
right record can otherwise rank 9th of 15, or fall outside k altogether.

    tl = memory.timeline("Tomasz Nowak", "team")
    tl.latest()          # newest event whose position is known
    print(tl.as_text())  # one compact line per event

A timeline merges the subject's fact history with every retrievable memory
that mentions the subject as a whole anchor (no top-k cap; see
``mark.memory.timeline_build``), ordered by the two-clock rule in
``mark.memory.timeline_order``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from mark.memory.timeline_order import BASIS_UNKNOWN

# A leading date stamp duplicates the event's date column in as_text().
_LEADING_STAMP_RE = re.compile(r"^\s*\[\d{4}-\d{2}-\d{2}(?:[T ][\d:]+Z?)?\]\s*")


@dataclass(frozen=True)
class TimelineEvent:
    """One event in a subject's history.

    ``when`` is the declared event time for dated events (``dated=True``)
    and the storage time otherwise. ``same_day`` is ``(k, m)`` — the k-th of
    m dated events sharing that date — or None when the date is unshared.
    ``basis`` says how the position was decided (declared / stored /
    stored_after_dated / unknown; see ``timeline_order``). A memory that a
    fact was extracted from carries that fact's id in ``fact_id``.
    """
    when: datetime | None
    kind: str  # "fact" | "memory"
    text: str
    dated: bool
    fragment_id: str | None = None
    fact_id: str | None = None
    stored_at: datetime | None = None
    same_day: tuple[int, int] | None = None
    basis: str = "declared"

    @property
    def id(self) -> str:
        """The fragment id for memories, the fact id for facts."""
        return self.fragment_id or self.fact_id or ""

    def label(self) -> str:
        """Date label: the event date ('2026-04-07'), else 'stored 2026-05-01'.

        A memory without a declared time shows the day it was stored, never
        "undated" (list and recall show that date too). Same-day events keep
        their order without a "same day k/m" marker (it stays in ``to_dict``).
        Only an event with no time at all reads 'undated'.
        """
        if self.dated and self.when is not None:
            return self.when.date().isoformat()
        stored = self.stored_at or self.when
        return f"stored {stored.date().isoformat()}" if stored is not None else "undated"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready rendering (ISO times, 'k/m' same-day label)."""
        return {
            "when": self.when.isoformat() if self.when else None,
            "dated": self.dated,
            "kind": self.kind,
            "text": self.text,
            "fragment_id": self.fragment_id,
            "fact_id": self.fact_id,
            "stored_at": self.stored_at.isoformat() if self.stored_at else None,
            "same_day": f"{self.same_day[0]}/{self.same_day[1]}" if self.same_day else None,
            "basis": self.basis,
        }


@dataclass
class Timeline:
    """A subject's events, oldest first (see the module docstring)."""
    subject: str
    attribute: str | None = None
    events: list[TimelineEvent] = field(default_factory=list)
    #: Events dropped by ``limit`` (the oldest ones are dropped first).
    truncated: int = 0

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[TimelineEvent]:
        return iter(self.events)

    def latest(self) -> TimelineEvent | None:
        """The newest event whose place in time is known.

        Undated records that cannot be placed among dated ones (basis
        ``unknown``) are never "latest": claiming so would compare clocks.
        """
        for event in reversed(self.events):
            if event.basis != BASIS_UNKNOWN:
                return event
        return None

    def as_text(self, *, max_chars: int = 240) -> str:
        """Compact rendering: a header, then one line per event.

        Lines look like ``2026-04-07 · HR: ... Platform team.`` (``stored
        2026-05-01 · ...`` for undated memories); a fact line is prefixed ``fact:``. Long texts are cut at *max_chars*.
        """
        scope = f" · {self.attribute}" if self.attribute else ""
        dropped = f" (+{self.truncated} earlier)" if self.truncated else ""
        lines = [f"Timeline of {self.subject}{scope}: {len(self.events)} event{'' if len(self.events) == 1 else 's'}{dropped}"]
        for event in self.events:
            text = _LEADING_STAMP_RE.sub("", event.text) if event.kind == "memory" else f"fact: {event.text}"
            text = " ".join(text.split())
            if len(text) > max_chars:
                text = text[: max_chars - 1] + "…"
            lines.append(f"{event.label()} · {text}")
        return "\n".join(lines)

    def as_context(self, *, max_chars: int = 240) -> str:
        """LLM-ready block: the ``as_text`` lines inside an escaped <timeline> tag.

        Memory content is untrusted, so it is XML-escaped and can never close
        the boundary tag.
        """
        attrs = f"subject={_xml_quoteattr(self.subject)}"
        if self.attribute:
            attrs += f" attribute={_xml_quoteattr(self.attribute)}"
        body = self.as_text(max_chars=max_chars).splitlines()[1:]
        lines = [f"<timeline {attrs} order=\"oldest-first\" events=\"{len(self.events)}\">"]
        lines += [f"  {_xml_escape(line)}" for line in body]
        lines.append("</timeline>")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready rendering of the whole timeline."""
        latest = self.latest()
        return {
            "subject": self.subject,
            "attribute": self.attribute,
            "count": len(self.events),
            "truncated": self.truncated,
            "latest": latest.to_dict() if latest else None,
            "events": [event.to_dict() for event in self.events],
        }


def timeline_json(memory: Any, subject: str, attribute: str | None = None, limit: int | None = None) -> dict:
    """CLI-ready helper: ``memory.timeline(...)`` as a JSON-serialisable dict.

    Adds ``text`` (the compact rendering) so a CLI can print either form.
    """
    timeline = memory.timeline(subject, attribute, limit=limit)
    payload = timeline.to_dict()
    payload["text"] = timeline.as_text()
    return payload
