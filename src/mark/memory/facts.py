"""Bi-temporal facts: the current value of (subject, attribute), with full history.

Facts complement free-text memories. When an agent knows a value is a
*state* that changes over time (a service's port, a ticket's status, a
person's team) it can record it as a fact; later values supersede earlier ones
by their ``valid_from`` time, never by the order they were written, and
nothing is deleted.

    mem.facts.set("atlas-core", "port", "4444", valid_from="2026-03-30")
    mem.facts.current("atlas-core", "port")      # -> Fact(value="4444", ...)
    mem.facts.history("atlas-core", "port")      # oldest → newest
    mem.facts.as_of("atlas-core", "port", "2026-03-20")

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


@dataclass(frozen=True)
class Fact:
    """One value of (subject, attribute) and the interval it was valid for."""
    subject: str
    attribute: str
    value: str
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime | None = None
    source_fragment_id: str | None = None
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        """True when no later value has been recorded."""
        return self.valid_to is None

    def as_text(self) -> str:
        """One-line rendering, e.g. 'atlas-core port = 4444 (since 2026-03-30)'."""
        since = self.valid_from.date().isoformat()
        until = f" until {self.valid_to.date().isoformat()}" if self.valid_to else ""
        return f"{self.subject} {self.attribute} = {self.value} (since {since}{until})"


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
        """Record a value; ``valid_from`` defaults to now.

        Several values with the same ``valid_from`` are kept as a sequence (the
        last one is current). Pass ``correction=True`` to replace what was
        recorded for that time instead.
        """
        row = self._store.fact_set(
            self._agent_id, subject, attribute, str(value),
            valid_from=_when(valid_from) or datetime.now(timezone.utc),
            source_fragment_id=source_fragment_id, metadata=metadata, correction=correction,
        )
        return _fact(row)

    def current(self, subject: str | None = None, attribute: str | None = None) -> list[Fact]:
        """Facts valid now, optionally filtered to one subject and/or attribute."""
        return [_fact(row) for row in self._store.fact_current(self._agent_id, subject, attribute)]

    def get(self, subject: str, attribute: str) -> Fact | None:
        """The current value of (subject, attribute), or None."""
        found = self.current(subject, attribute)
        return found[0] if found else None

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
        return self._store.fact_retract(
            self._agent_id, subject, attribute, value=None if value is None else str(value),
        )

    def extract_pending(self, *, limit: int | None = None, wait: float | bool = True) -> int:
        """Finish this agent's background extraction jobs now; returns how many finished.

        Queued jobs are processed in the calling thread. Jobs the background
        worker has already claimed are not re-run; with ``wait`` (default
        True: up to the job lease, 300 s by default; a float: that many
        seconds; False: no waiting) the call blocks until they are done, so
        their facts are readable when it returns. The count includes jobs the
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
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"expected an ISO-8601 time, got {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
