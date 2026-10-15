# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Trust-aware global memory bus with subscriptions and snapshots.

:class:`TrustAwareGlobalMemoryBus` layers per-message trust on top of
:class:`~mark.memory.global_bus.GlobalMemoryBus`: it stamps the authoritative
trust record on publish, rebuilds (and re-verifies) it on load, and filters
retrieval by trust. Value types live in :mod:`.messages`.
"""
from __future__ import annotations

import hmac
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from mark.intelligence.retrieval import RetrievalPolicy, RetrievalResult
from mark.memory.global_bus import GlobalMemoryBus

# Value types are re-exported here, their historical import path.
from mark.middlewares.trust_bus.messages import (
    _TRUST_RANK,
    _TRUST_TAG_PREFIX,
    TRUST_METADATA_KEY,
    BusMessage,
    BusSnapshot,
    BusSubscription,
    PublisherTrust,
    _is_trust_tag,
    _lowest,
    _trust_signature,
)
from mark.types import MemoryFragment

logger = logging.getLogger(__name__)


class TrustAwareGlobalMemoryBus:
    """
    Multi-agent memory bus with per-message trust levels.

    Wraps GlobalMemoryBus and adds trust filtering on publish and retrieve.

    The authoritative trust level is stored in fragment metadata under
    ``mark_trust`` (set only by this class). Caller-supplied ``trust:*`` tags
    are stripped on publish and ignored on load, so a publisher cannot
    promote itself by tagging. Legacy fragments without the metadata record
    fall back to their ``trust:`` tags, taking the *lowest* level when tags
    disagree.

    Pass ``signing_key`` to HMAC-sign (publisher, trust, content); messages
    whose signature does not verify on load are downgraded to UNKNOWN.
    """

    def __init__(
        self,
        *,
        store: Any,
        pipeline: Any,
        embedder: Any,
        executor: Any,
        signing_key: bytes | None = None,
    ) -> None:
        self._signing_key = signing_key
        self._bus = GlobalMemoryBus(
            store    = store,
            pipeline = pipeline,
            embedder = embedder,
            executor = executor,
        )
        self._messages: list[BusMessage]                  = []
        self._subs:     dict[str, BusSubscription]        = {}
        self._load_existing_messages()

    def _load_existing_messages(self) -> None:
        """Rebuild message metadata from persisted global-bus fragments."""
        seen: set[str] = set()
        for frag in self._bus.all_facts(limit=None):
            msg = self._message_from_fragment(frag, signing_key=self._signing_key)
            if msg is None or msg.id in seen:
                continue
            self._messages.append(msg)
            seen.add(msg.id)

    @staticmethod
    def _message_from_fragment(
        frag: MemoryFragment,
        *,
        signing_key: bytes | None = None,
    ) -> BusMessage | None:
        """Rebuild a :class:`BusMessage` from a persisted global-bus fragment.

        Trust authority: only the ``mark_trust`` metadata record (written by
        this class) is believed. ``trust:*`` tags are never copied into the
        message, and are consulted only for legacy fragments without a
        record. With a signing key, a bad signature or a missing record
        downgrades the message to UNKNOWN.
        """
        if "global_bus" not in frag.tags:
            return None

        tag_levels: list[PublisherTrust] = []
        tags: list[str] = []
        for tag in frag.tags:
            if tag == "global_bus":
                continue
            if _is_trust_tag(tag):
                raw = tag.split(":", 1)[1].strip().lower()
                try:
                    tag_levels.append(PublisherTrust(raw))
                except ValueError:
                    tag_levels.append(PublisherTrust.UNKNOWN)
                continue
            tags.append(tag)

        publisher = frag.source or str(frag.metadata.get("publisher", ""))
        record = frag.metadata.get(TRUST_METADATA_KEY)
        if isinstance(record, dict) and "level" in record:
            try:
                trust = PublisherTrust(str(record["level"]))
            except ValueError:
                trust = PublisherTrust.UNKNOWN
            if signing_key is not None:
                expected = _trust_signature(signing_key, publisher, trust, frag.content)
                if not hmac.compare_digest(str(record.get("sig", "")), expected):
                    logger.warning("trust bus: signature mismatch for %s; downgrading to unknown", frag.id)
                    trust = PublisherTrust.UNKNOWN
        else:
            # Legacy fragment: only tags are available. Ambiguity resolves low.
            trust = _lowest(tag_levels)
            if signing_key is not None:
                trust = PublisherTrust.UNKNOWN

        metadata = {k: v for k, v in frag.metadata.items() if k != TRUST_METADATA_KEY}
        return BusMessage(
            id        = frag.id,
            content   = frag.content,
            publisher = publisher,
            trust     = trust,
            tags      = tags,
            timestamp = frag.created_at.isoformat(),
            metadata  = metadata,
        )

    # ── publish ───────────────────────────────────────────────────────────────

    def _prepare(
        self,
        content: str,
        publisher: str,
        trust: PublisherTrust,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
    ) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any]]:
        """Return ``(stored_tags, public_tags, stored_meta, public_meta)``.

        Caller ``trust:*`` tags and ``mark_trust`` metadata are dropped: the only
        trust record is the one stamped (and optionally signed) here."""
        trust = PublisherTrust(trust)
        clean_tags = [t for t in (tags or []) if not _is_trust_tag(t)]
        if len(clean_tags) != len(tags or []):
            logger.warning("trust bus: stripped caller-supplied trust:* tags from %r", publisher)
        clean_meta = {k: v for k, v in (metadata or {}).items() if k != TRUST_METADATA_KEY}
        record: dict[str, Any] = {"level": trust.value, "publisher": publisher}
        if self._signing_key is not None:
            record["sig"] = _trust_signature(self._signing_key, publisher, trust, content.strip())
        stored_meta = {**clean_meta, TRUST_METADATA_KEY: record}
        all_tags = [f"{_TRUST_TAG_PREFIX}{trust.value}", *clean_tags]
        return all_tags, clean_tags, stored_meta, clean_meta

    def publish_sync(
        self,
        content: str,
        *,
        publisher:  str,
        trust:      PublisherTrust = PublisherTrust.UNKNOWN,
        importance: float          = 0.7,
        tags:       list[str] | None = None,
        metadata:   dict[str, Any] | None = None,
    ) -> BusMessage:
        """Synchronous variant of publish()."""
        trust = PublisherTrust(trust)
        all_tags, clean_tags, stored_meta, clean_meta = self._prepare(
            content, publisher, trust, tags, metadata,
        )
        frag_id = self._bus.publish_sync(
            content,
            publisher  = publisher,
            importance = importance,
            tags       = all_tags,
            metadata   = stored_meta,
        )
        msg = BusMessage(
            id        = frag_id,
            content   = content,
            publisher = publisher,
            trust     = trust,
            tags      = clean_tags,
            timestamp = datetime.now(timezone.utc).isoformat(),
            metadata  = clean_meta,
        )
        self._messages.append(msg)
        return msg

    async def publish(
        self,
        content: str,
        *,
        publisher:  str,
        trust:      PublisherTrust = PublisherTrust.UNKNOWN,
        importance: float          = 0.7,
        tags:       list[str] | None = None,
        metadata:   dict[str, Any] | None = None,
    ) -> BusMessage:
        """Publish content onto the bus."""
        trust = PublisherTrust(trust)
        all_tags, clean_tags, stored_meta, clean_meta = self._prepare(
            content, publisher, trust, tags, metadata,
        )
        frag_id = await self._bus.publish(
            content,
            publisher  = publisher,
            importance = importance,
            tags       = all_tags,
            metadata   = stored_meta,
        )
        msg = BusMessage(
            id        = frag_id,
            content   = content,
            publisher = publisher,
            trust     = trust,
            tags      = clean_tags,
            timestamp = datetime.now(timezone.utc).isoformat(),
            metadata  = clean_meta,
        )
        self._messages.append(msg)
        return msg

    # ── subscriptions ─────────────────────────────────────────────────────────

    def subscribe(self, subscription: BusSubscription) -> str:
        """Register a subscription and return its id."""
        sub_id = str(uuid.uuid4())
        self._subs[sub_id] = subscription
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription by id."""
        self._subs.pop(sub_id, None)

    def messages_for(self, sub_id: str) -> list[BusMessage]:
        """Return messages matching a subscription."""
        sub = self._subs.get(sub_id)
        if sub is None:
            return []
        return [m for m in self._messages if sub.matches(m)]

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve_sync(
        self,
        query:     str,
        policy:    RetrievalPolicy = RetrievalPolicy.BALANCED,
        min_trust: PublisherTrust  = PublisherTrust.UNKNOWN,
    ) -> RetrievalResult:
        """Trust-filtered retrieval over bus messages."""
        result = self._bus.retrieve_sync(query, policy=policy)
        if min_trust == PublisherTrust.UNKNOWN:
            return result
        min_rank = _TRUST_RANK[min_trust]
        keep_ids: set[str] = {
            m.id for m in self._messages
            if _TRUST_RANK[m.trust] >= min_rank
        }
        frags  = [f for f in result.fragments if f.id in keep_ids]
        scores = [s for f, s in zip(result.fragments, result.scores) if f.id in keep_ids]
        from dataclasses import replace
        return replace(result, fragments=frags, scores=scores)

    # ── snapshot ──────────────────────────────────────────────────────────────

    def snapshot(
        self,
        subscription: BusSubscription | None = None,
    ) -> BusSnapshot:
        """Return a BusSnapshot, optionally filtered by subscription."""
        msgs = (
            [m for m in self._messages if subscription.matches(m)]
            if subscription is not None
            else list(self._messages)
        )
        return BusSnapshot(
            messages    = msgs,
            captured_at = datetime.now(timezone.utc).isoformat(),
        )

    def all_messages(self) -> list[BusMessage]:
        """Return every message on the bus."""
        return list(self._messages)
