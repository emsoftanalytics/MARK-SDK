"""Bi-temporal facts: the current value of (subject, attribute), with full history.

Facts complement free-text memories. When an agent knows a value is a
*state* that changes over time (a service's port, a ticket's status, a
person's team) it can record it as a fact; later values supersede earlier ones
by their ``valid_from`` time, never by the order they were written, and
nothing is deleted.

    mem.facts.set("atlas-core", "port", 4444, valid_from="2026-03-30")
    mem.facts.get("atlas-core", "port")          # -> Fact(value=4444, ...)
    mem.facts.history("atlas-core", "port")      # oldest → newest
    mem.facts.as_of("atlas-core", "port", "2026-03-20")
    mem.facts.add("Dr. Bello", "phone", "+234 1")  # several current values
    mem.facts.get_all("Dr. Bello", "phone")      # every current value
    mem.facts.scheduled()                        # values that start in the future

Values keep their Python type (str, int, float, bool, None, list, dict);
rows written before typed values existed read back as strings. A value whose
``valid_from`` is in the future is *scheduled*: not current until then (its
``as_text()`` says "scheduled from <date>"). Times (``valid_from``, ``as_of``)
accept datetimes, ISO text or words such as "yesterday" or "3 May 2026".

Facts can also be extracted automatically from stored memories (see
:mod:`mark.intelligence.fact_extraction`); slow extractors queue their work,
which :meth:`FactsView.extract_pending` processes on demand.

Subjects and attributes are matched case- and punctuation-insensitively
("Atlas-Core" == "atlas core"). Values are encrypted at rest when the store
is; subjects and attributes are not (they are lookup keys).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mark.store._sqlite.fact_codec import value_text


@dataclass(frozen=True)
class Fact:
    """One value of (subject, attribute) and the interval it was valid for."""
    subject: str
    attribute: str
    value: Any
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime | None = None
    source_fragment_id: str | None = None
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        """True when the value holds now: started already and not ended."""
        now = datetime.now(timezone.utc)
        return self.valid_from <= now and (self.valid_to is None or self.valid_to > now)

    @property
    def is_scheduled(self) -> bool:
        """True when the value starts in the future (recorded ahead of time)."""
        return self.valid_from > datetime.now(timezone.utc)

    @property
    def text(self) -> str:
        """The value as text (str as-is, numbers plainly, other types as JSON)."""
        return value_text(self.value)

    def as_text(self) -> str:
        """One line: 'atlas-core port = 4444 (since 2026-03-30)', or '(scheduled from ...)'."""
        since = self.valid_from.date().isoformat()
        until = f" until {self.valid_to.date().isoformat()}" if self.valid_to else ""
        # A future value is not current: "since" would claim it already holds.
        label = "scheduled from" if self.is_scheduled else "since"
        return f"{self.subject} {self.attribute} = {self.text} ({label} {since}{until})"


class FactsView:
    """Agent-scoped access to the store's fact timeline."""

    def __init__(self, store: Any, agent_id: str, *, extraction: Any = None) -> None:
        if not hasattr(store, "fact_set"):
            raise TypeError("this memory store does not support facts (requires LocalMemoryStore v4+)")
        self._store = store
        self._agent_id = agent_id
        # The runtime's FactExtractionService (None when extraction is off).
        self._extraction = extraction

    def set(
        self,
        subject: str,
        attribute: str,
        value: Any,
        *,
        valid_from: datetime | str | None = None,
        source_fragment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        correction: bool = False,
    ) -> Fact:
        """Record the value of (subject, attribute); ``valid_from`` defaults to now.

        The value keeps its type (str, int, float, bool, None, list, dict;
        other objects are stored as text). It replaces the previous value —
        and any values added with :meth:`add` — from ``valid_from`` on; a
        future ``valid_from`` schedules it (see :meth:`scheduled`). Several
        values with the same ``valid_from`` are kept as a sequence (the last
        one is current). Pass ``correction=True`` to replace what was
        recorded for that time instead.
        """
        row = self._store.fact_set(
            self._agent_id, subject, attribute, value,
            valid_from=_when(valid_from) or datetime.now(timezone.utc),
            source_fragment_id=source_fragment_id, metadata=metadata, correction=correction,
        )
        return _fact(row)

    def add(
        self,
        subject: str,
        attribute: str,
        value: Any,
        *,
        valid_from: datetime | str | None = None,
        source_fragment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Add one more current value (``set`` replaces; ``add`` accumulates).

        For attributes with several values at once: phones, languages,
        members. ``current(subject, attribute)`` returns all of them; a later
        ``set`` replaces them all; :meth:`remove` ends one.
        """
        row = self._store.fact_add(
            self._agent_id, subject, attribute, value,
            valid_from=_when(valid_from) or datetime.now(timezone.utc),
            source_fragment_id=source_fragment_id, metadata=metadata,
        )
        return _fact(row)

    def remove(self, subject: str, attribute: str, value: Any, *, valid_to: datetime | str | None = None) -> int:
        """End an added value (from :meth:`add`) at ``valid_to`` (default now); returns how many."""
        return self._store.fact_remove(self._agent_id, subject, attribute, value,
                                       valid_to=_when(valid_to) or datetime.now(timezone.utc))

    def current(self, subject: str | None = None, attribute: str | None = None) -> list[Fact]:
        """Facts valid now, optionally filtered to one subject and/or attribute.

        Scheduled values (``valid_from`` in the future) are not included; see
        :meth:`scheduled`. A multi-valued attribute yields one Fact per value.
        """
        return [_fact(row) for row in self._store.fact_current(self._agent_id, subject, attribute)]

    def get(self, subject: str, attribute: str) -> Fact | None:
        """The current value of (subject, attribute), or None (latest one if several).

        For multi-valued attributes (see :meth:`add`) use :meth:`get_all`.
        """
        found = self.current(subject, attribute)
        return found[-1] if found else None

    def get_all(self, subject: str, attribute: str) -> list[Fact]:
        """Every current value of (subject, attribute): one for ``set``, several after ``add``."""
        return self.current(subject, attribute)

    def scheduled(self, subject: str | None = None, attribute: str | None = None) -> list[Fact]:
        """Values recorded to take effect in the future, soonest first."""
        return [_fact(row) for row in self._store.fact_scheduled(self._agent_id, subject, attribute)]

    def history(self, subject: str, attribute: str) -> list[Fact]:
        """Every distinct value (subject, attribute) has had, oldest first."""
        return [_fact(row) for row in self._store.fact_history(self._agent_id, subject, attribute)]

    def as_of(self, subject: str, attribute: str, when: datetime | str) -> Fact | None:
        """The value (subject, attribute) had at ``when``."""
        moment = _when(when)
        if moment is None:
            raise ValueError("as_of requires a time")
        row = self._store.fact_as_of(self._agent_id, subject, attribute, moment)
        return None if row is None else _fact(row)

    def retract(self, subject: str, attribute: str | None = None, *, value: Any = None) -> int:
        """Retract facts recorded by mistake (kept for audit); returns how many."""
        return self._store.fact_retract(self._agent_id, subject, attribute, value=value)

    def delete(self, subject: str, attribute: str | None = None, *, value: Any = None) -> int:
        """Hard-delete facts (nothing kept for audit, unlike retract); returns how many."""
        return self._store.fact_delete(
            self._agent_id, subject, attribute, value=None if value is None else str(value),
        )

    def extract_pending(self, *, limit: int | None = None, wait: float | bool = True) -> int:
        """Finish this agent's background extraction jobs now; returns how many finished.

        Queued jobs are processed in the calling thread. Jobs the background
        worker has already claimed are not re-run; with ``wait`` (default
        True: up to 30 seconds; a float: that many seconds; False: no
        waiting) the call blocks until they are done, so their facts are
        readable when it returns. The wait is bounded so a hung extractor
        can never hang the caller; on timeout, unfinished jobs stay queued. The count includes jobs the
        worker finished while waiting. Returns 0 when no background extractor
        is configured (inline extractors have no queue).
        """
        service = self._extraction
        if service is None or not getattr(service, "background", False):
            return 0
        return service.drain(agent_id=self._agent_id, limit=limit, wait=wait)

    def subjects(self) -> list[str]:
        """Normalised subjects that have facts."""
        return self._store.fact_subjects(self._agent_id)


def _fact(row: dict[str, Any]) -> Fact:
    return Fact(**{key: row[key] for key in (
        "subject", "attribute", "value", "valid_from", "valid_to", "recorded_at",
        "source_fragment_id", "id", "metadata",
    )})


def _when(value: datetime | str | None) -> datetime | None:
    """A UTC-aware time from a datetime, ISO text or words ("yesterday", "3 May 2026")."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return parse_time(value)


def parse_time(value: Any, what: str = "date") -> datetime:
    """Parse *value* (ISO text or natural words) into a UTC-aware datetime.

    Shared by facts (``valid_from``, ``as_of``) and writes (``occurred_at``);
    bad input raises :class:`mark.errors.InvalidInputError` (a ValueError).
    """
    from mark.errors import InvalidInputError

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None:
        from mark.intelligence.date_parsing import parse_when

        try:
            parsed = parse_when(text) if text else None
        except ValueError:
            parsed = None
    if parsed is None:
        raise InvalidInputError(
            f"could not read {'an' if str(what)[:1].lower() in 'aeiou' else 'a'} {what} from {value!r}; use an ISO date such as 2026-10-01 "
            "(or 2026-10-01T14:30) or words such as 'yesterday', '3 May 2026', 'last tuesday'")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
