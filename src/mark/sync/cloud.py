# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Redacted sync envelope preparation; transport is always caller-supplied."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SyncMode(str, Enum):
    """
    When to sync local memory to cloud.

    MANUAL: sync only when explicitly called.
    WRITE_THROUGH: sync on every store call when a cloud client is configured.
    """

    MANUAL = "manual"
    WRITE_THROUGH = "write_through"


@dataclass
class SyncOptions:
    """Options that control what gets prepared for sync."""

    include_blocks: list[str] | None = None
    redact_secrets: bool = True
    include_events: bool = False
    mode: SyncMode = SyncMode.MANUAL


@dataclass
class SyncStats:
    """Counts describing a prepared sync delta."""
    synced: int = 0
    skipped: int = 0
    errors: int = 0
    mode: str = SyncMode.MANUAL.value

    @property
    def total(self) -> int:
        """Return the total count."""
        return self.synced + self.skipped + self.errors


@dataclass
class SyncDelta:
    """
    Inspectable local sync envelope.

    This is safe MIT client-side packaging only. Upload transport, encryption,
    tenancy, conflict resolution, and retry policy belong to mark-cloud.
    """

    fragments: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    stats: SyncStats = field(default_factory=SyncStats)


class CloudSync:
    """
    Local sync preparer and optional cloud-client forwarder.

    `prepare_delta()` builds a redacted payload envelope from local memory and
    does not perform network calls. `sync()` forwards that envelope only when a
    caller supplies a cloud client with `upload_delta(delta)`.
    """

    def prepare_delta(
        self,
        runtime: Any,
        *,
        options: SyncOptions | None = None,
    ) -> SyncDelta:
        """Build a redacted sync envelope from local memory."""
        opts = options or SyncOptions()
        stats = SyncStats(mode=opts.mode.value)
        delta = SyncDelta(
            options={
                "include_blocks": opts.include_blocks,
                "redact_secrets": opts.redact_secrets,
                "include_events": opts.include_events,
                "mode": opts.mode.value,
            },
            stats=stats,
        )

        store = getattr(runtime, "store", None)
        if store is None:
            return delta

        try:
            fragments = store.list_by_agent("__mark__") + (
                store.list_all() if hasattr(store, "list_all") else []
            )
        except Exception:
            stats.errors += 1
            return delta

        from mark.security import redact_secrets, redact_sync_value

        selected_blocks = set(opts.include_blocks or [])
        seen: set[str] = set()

        for frag in fragments:
            if frag.id in seen:
                continue
            seen.add(frag.id)

            if selected_blocks:
                block_tags = {tag[6:] for tag in frag.tags if tag.startswith("block:")}
                if not (block_tags & selected_blocks):
                    stats.skipped += 1
                    continue

            content = redact_secrets(frag.content) if opts.redact_secrets else frag.content
            if not content.strip():
                stats.skipped += 1
                continue

            delta.fragments.append(
                {
                    "id": frag.id,
                    "agent_id": frag.agent_id,
                    "session_id": frag.session_id,
                    "scope": getattr(frag.scope, "value", frag.scope),
                    "tier": getattr(frag.tier, "value", frag.tier),
                    "state": getattr(frag.state, "value", frag.state),
                    "importance": frag.importance,
                    "confidence": frag.confidence,
                    "tags": redact_sync_value(list(frag.tags)) if opts.redact_secrets else list(frag.tags),
                    "source": redact_secrets(frag.source) if opts.redact_secrets and frag.source else frag.source,
                    "metadata": redact_sync_value(dict(frag.metadata)) if opts.redact_secrets else dict(frag.metadata),
                    "content": content,
                    "created_at": frag.created_at.isoformat(),
                    "updated_at": frag.updated_at.isoformat(),
                }
            )
            stats.synced += 1

        if opts.include_events and hasattr(runtime, "tracer"):
            try:
                tracer = runtime.tracer()
                delta.events = [
                    {
                        "type": event.type,
                        "payload": redact_sync_value(event.payload) if opts.redact_secrets else event.payload,
                        "created_at": event.created_at.isoformat(),
                    }
                    for event in tracer.events()
                ]
            except Exception:
                stats.errors += 1

        return delta

    def prepare(
        self,
        runtime: Any,
        *,
        options: SyncOptions | None = None,
    ) -> SyncStats:
        """Prepare a local delta and return only its statistics."""

        return self.prepare_delta(runtime, options=options).stats

    def sync(
        self,
        runtime: Any,
        cloud: Any = None,
        *,
        options: SyncOptions | None = None,
    ) -> SyncStats:
        """
        Prepare local data and optionally forward it to a mark-cloud client.

        Without a cloud client, this is local-only and returns statistics.
        """

        delta = self.prepare_delta(runtime, options=options)
        stats = delta.stats
        if cloud is None:
            return stats

        if hasattr(cloud, "upload_delta"):
            try:
                cloud.upload_delta(delta)
            except Exception:
                stats.errors += 1

        return stats
