# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Write-time guards for MarkMemory: fragment building, gating, quarantine.

Every write path (single store, batch store, observe, gap-healing evidence)
builds its fragment through :meth:`GuardMixin._guarded_fragment`, which
decides whether the fragment may be *indexed* (become retrievable) or must
only be *stored* for audit as QUARANTINED. Also home to the ``occurred_at``
normaliser shared by the store paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mark.memory.facade.core import MemoryCore
from mark.types import MemoryFragment, MemoryScope, MemoryState


@dataclass(frozen=True)
class MemoryWriteRejection:
    """Details for the most recent guarded memory write."""

    fragment_id: str
    reason: str
    content: str


def _with_occurred_at(metadata: dict[str, Any] | None, occurred_at: Any) -> dict[str, Any] | None:
    """Return metadata with ``occurred_at`` normalised to an ISO-8601 string.

    Naive datetimes are assumed to be UTC. Strings are validated (a trailing
    ``Z`` is accepted) but stored as given, so callers keep their precision.
    The input dict is never mutated; ``None`` passes metadata through as-is.
    """
    if occurred_at is None:
        return metadata
    if isinstance(occurred_at, datetime):
        stamp = occurred_at if occurred_at.tzinfo else occurred_at.replace(tzinfo=timezone.utc)
        value = stamp.isoformat()
    else:
        value = str(occurred_at).strip()
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"occurred_at must be ISO-8601, got {occurred_at!r}") from exc
    return {**(metadata or {}), "occurred_at": value}


class GuardMixin(MemoryCore):
    """Fragment construction, consolidation-gate checks, and quarantine."""

    def quarantine_sync(
        self,
        content: str,
        *,
        reason: str,
        scope: MemoryScope = MemoryScope.AGENT,
        tags: list[str] | None = None,
        source: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a diagnostic artifact outside normal retrieval.

        Quarantined fragments remain inspectable through the store but are not
        indexed into retrieval and cannot become continuity/context by accident.
        """
        meta = {
            **(metadata or {}),
            "mark_guard": {
                "passed": False,
                "reason": reason,
                "quarantined_by": "quarantine_sync",
            },
        }
        fragment = self._fragment(
            content,
            importance=0.0,
            scope=scope,
            tags=[*(tags or []), "mark:quarantined"],
            source=source,
            state=MemoryState.QUARANTINED,
            confidence=0.0,
            session_id=session_id,
            ttl_seconds=None,
            metadata=meta,
        )
        # Written straight to the store: bypasses the index on purpose.
        return self._store.store(fragment)

    @property
    def last_rejection(self) -> MemoryWriteRejection | None:
        """Return details for the most recent automatic quarantine."""
        return self._last_rejection

    def quarantined(self, *, limit: int = 100) -> list[MemoryFragment]:
        """List quarantined fragments for this agent."""
        return self._store.list_by_agent(
            self.agent_id,
            states=[MemoryState.QUARANTINED],
            limit=limit,
        )

    def _fragment(
        self,
        content: str,
        *,
        importance: float,
        scope: MemoryScope,
        tags: list[str] | None,
        source: str | None,
        state: MemoryState,
        confidence: float,
        session_id: str | None,
        ttl_seconds: int | None,
        metadata: dict[str, Any] | None,
    ) -> MemoryFragment:
        """Build an (unpersisted) fragment owned by this agent."""
        return MemoryFragment(
            content=content,
            agent_id=self.agent_id,
            importance=importance,
            scope=scope,
            tags=tags or [],
            source=source,
            state=state,
            confidence=confidence,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )

    def _guarded_fragment(
        self,
        content: str,
        *,
        importance: float,
        scope: MemoryScope,
        tags: list[str] | None,
        source: str | None,
        state: MemoryState,
        confidence: float,
        session_id: str | None,
        ttl_seconds: int | None,
        metadata: dict[str, Any] | None,
    ) -> tuple[MemoryFragment, bool]:
        """Build a fragment and decide whether it may be indexed.

        Returns ``(fragment, accepted)``. ``accepted=False`` means the caller
        must persist the fragment for audit only and never index it. Rules:

        * no gate: accept everything except an explicitly QUARANTINED state
          (e.g. set by governance middleware);
        * gate + QUARANTINED request: zero importance/confidence and tag it,
          without consulting the gate;
        * gate passes: keep the (possibly truncated) content, recording the
          truncation in ``metadata["mark_guard"]``;
        * gate rejects: store as QUARANTINED with the reason, and remember it
          as :attr:`last_rejection`.
        """
        if self._consolidation_gate is None:
            self._last_rejection = None
            # A QUARANTINED write (e.g. from governance middleware) is stored
            # for audit but must never be indexed into retrieval.
            return self._fragment(
                content,
                importance=importance,
                scope=scope,
                tags=tags,
                source=source,
                state=state,
                confidence=confidence,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            ), state != MemoryState.QUARANTINED

        if state == MemoryState.QUARANTINED:
            # Note: _last_rejection is deliberately left untouched here; it
            # only reports rejections made by the gate itself.
            fragment = self._fragment(
                content,
                importance=0.0,
                scope=scope,
                tags=[*(tags or []), "mark:quarantined"],
                source=source,
                state=state,
                confidence=0.0,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
            return fragment, False

        gate_result = self._consolidation_gate.check(
            content,
            confidence=confidence,
        )
        if gate_result.passed:
            self._last_rejection = None
            passed_meta = metadata
            if getattr(gate_result, "truncated", False):
                passed_meta = {
                    **(metadata or {}),
                    "mark_guard": {
                        "passed": True,
                        "truncated": True,
                        "original_length": gate_result.original_length,
                    },
                }
            return self._fragment(
                gate_result.content,
                importance=importance,
                scope=scope,
                tags=tags,
                source=source,
                state=state,
                confidence=confidence,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=passed_meta,
            ), True

        # Rejected by the gate: keep an auditable, non-retrievable record that
        # remembers what the caller originally asked for.
        meta = {
            **(metadata or {}),
            "mark_guard": {
                "passed": False,
                "reason": gate_result.reason,
                "original_importance": importance,
                "requested_state": state.value,
                "truncated": bool(getattr(gate_result, "truncated", False)),
            },
        }
        fragment = self._fragment(
            gate_result.content,
            importance=0.0,
            scope=scope,
            tags=[*(tags or []), "mark:quarantined", "mark:failure"],
            source=source,
            state=MemoryState.QUARANTINED,
            confidence=0.0,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=meta,
        )
        self._last_rejection = MemoryWriteRejection(
            fragment_id=fragment.id,
            reason=gate_result.reason,
            content=gate_result.content,
        )
        return fragment, False
