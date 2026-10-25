# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Redacted sync envelope preparation; transport is always caller-supplied."""
from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Local integrity metadata that must never leave the machine: the stored
# ``content_hash`` is an unsalted digest of the *unredacted* content and would
# let a sync receiver confirm guesses about redacted text.
_LOCAL_ONLY_METADATA_KEYS = frozenset({"content_hash"})


def _governance_failed(frag: Any) -> bool:
    if getattr(frag.state, "value", frag.state) == "quarantined":
        return True
    metadata = frag.metadata or {}
    for key in ("mark_governance", "mark_guard"):
        info = metadata.get(key)
        if isinstance(info, dict) and info.get("passed") is False:
            return True
    return False


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
    error_messages: list[str] = field(default_factory=list)
    high_water_mark: datetime | None = None

    @property
    def total(self) -> int:
        """Return the total count."""
        return self.synced + self.skipped + self.errors


@dataclass
class SyncDelta:
    """
    Inspectable local sync envelope.

    This is safe Apache-2.0 client-side packaging only. Upload transport, encryption,
    tenancy, conflict resolution, and retry policy belong to mark-cloud.
    """

    fragments: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    stats: SyncStats = field(default_factory=SyncStats)
    high_water_ids: set[str] = field(default_factory=set)


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
        since: datetime | None = None,
        exclude_ids: Collection[str] = (),
    ) -> SyncDelta:
        """Build a redacted sync envelope from local memory.

        since        only include fragments updated at or after this instant
                     (a high-water mark from a previous delta)
        exclude_ids  fragment ids already synced *at* ``since`` (ties)

        Quarantined / governance-rejected fragments are never included.
        ``delta.stats.high_water_mark`` is the newest ``updated_at`` seen.
        """
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
        except Exception as exc:
            logger.warning("sync: listing fragments failed: %s: %s", type(exc).__name__, exc)
            stats.errors += 1
            stats.error_messages.append(f"list: {type(exc).__name__}: {exc}")
            return delta

        from mark.security import redact_secrets, redact_sync_value

        selected_blocks = set(opts.include_blocks or [])
        seen: set[str] = set()

        for frag in fragments:
            if frag.id in seen:
                continue
            seen.add(frag.id)

            if since is not None:
                if frag.updated_at < since or (frag.updated_at == since and frag.id in exclude_ids):
                    continue
            if stats.high_water_mark is None or frag.updated_at > stats.high_water_mark:
                stats.high_water_mark = frag.updated_at
                delta.high_water_ids = {frag.id}
            elif frag.updated_at == stats.high_water_mark:
                delta.high_water_ids.add(frag.id)

            if _governance_failed(frag):
                stats.skipped += 1
                continue

            if selected_blocks:
                block_tags = {tag[6:] for tag in frag.tags if tag.startswith("block:")}
                if not (block_tags & selected_blocks):
                    stats.skipped += 1
                    continue

            metadata = {
                key: value for key, value in frag.metadata.items()
                if key not in _LOCAL_ONLY_METADATA_KEYS
            }
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
                    "metadata": redact_sync_value(metadata) if opts.redact_secrets else metadata,
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
            except Exception as exc:
                logger.warning("sync: collecting events failed: %s: %s", type(exc).__name__, exc)
                stats.errors += 1
                stats.error_messages.append(f"events: {type(exc).__name__}: {exc}")

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
        raise_on_error: bool = False,
    ) -> SyncStats:
        """
        Prepare local data and optionally forward it to a mark-cloud client.

        Without a cloud client, this is local-only and returns statistics.
        """

        delta = self.prepare_delta(runtime, options=options)
        self.upload(delta, cloud, raise_on_error=raise_on_error)
        return delta.stats

    def upload(self, delta: SyncDelta, cloud: Any, *, raise_on_error: bool = False) -> bool:
        """Forward a prepared delta to ``cloud.upload_delta``.

        Failures are logged and recorded in ``delta.stats`` (``errors`` and
        ``error_messages``); pass ``raise_on_error=True`` to re-raise.
        Returns True when the upload succeeded (or there was no cloud).
        """
        if cloud is None:
            return True
        stats = delta.stats
        upload_delta = getattr(cloud, "upload_delta", None)
        if upload_delta is None:
            message = f"cloud client {type(cloud).__name__} has no upload_delta()"
            logger.warning("sync: %s", message)
            stats.errors += 1
            stats.error_messages.append(message)
            if raise_on_error:
                raise TypeError(message)
            return False
        try:
            upload_delta(delta)
        except Exception as exc:
            logger.warning("sync: upload failed: %s: %s", type(exc).__name__, exc)
            stats.errors += 1
            stats.error_messages.append(f"upload: {type(exc).__name__}: {exc}")
            if raise_on_error:
                raise
            return False
        return True
