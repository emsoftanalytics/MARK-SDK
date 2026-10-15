# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Trust-aware global memory bus with subscriptions and snapshots."""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from mark.intelligence.retrieval import RetrievalPolicy, RetrievalResult
from mark.memory.global_bus import GlobalMemoryBus
from mark.types import MemoryFragment

logger = logging.getLogger(__name__)

#: Reserved fragment-metadata key holding the authoritative trust record.
TRUST_METADATA_KEY = "mark_trust"
_TRUST_TAG_PREFIX = "trust:"


class PublisherTrust(str, Enum):
    """Trust levels assigned to bus publishers."""
    SYSTEM         = "system"
    VERIFIED_AGENT = "verified_agent"
    RAW_AGENT      = "raw_agent"
    EXTERNAL       = "external"
    UNKNOWN        = "unknown"


_TRUST_RANK: dict[PublisherTrust, int] = {
    PublisherTrust.SYSTEM:         4,
    PublisherTrust.VERIFIED_AGENT: 3,
    PublisherTrust.RAW_AGENT:      2,
    PublisherTrust.EXTERNAL:       1,
    PublisherTrust.UNKNOWN:        0,
}


def _lowest(levels: list[PublisherTrust]) -> PublisherTrust:
    if not levels:
        return PublisherTrust.UNKNOWN
    return min(levels, key=lambda level: _TRUST_RANK[level])


def _is_trust_tag(tag: str) -> bool:
    return tag.strip().lower().startswith(_TRUST_TAG_PREFIX)


def _trust_signature(key: bytes, publisher: str, trust: PublisherTrust, content: str) -> str:
    message = json.dumps([publisher, trust.value, content], ensure_ascii=False).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


@dataclass
class BusMessage:
    """Envelope for one trust-tagged bus message."""
    id:        str
    content:   str
    publisher: str
    trust:     PublisherTrust
    tags:      list[str]
    timestamp: str
    metadata:  dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id":        self.id,
            "content":   self.content,
            "publisher": self.publisher,
            "trust":     self.trust.value,
            "tags":      self.tags,
            "timestamp": self.timestamp,
            "metadata":  self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BusMessage":
        """Construct an instance from a plain dictionary."""
        return cls(
            id        = d["id"],
            content   = d["content"],
            publisher = d["publisher"],
            trust     = PublisherTrust(d["trust"]),
            tags      = d.get("tags", []),
            timestamp = d["timestamp"],
            metadata  = d.get("metadata", {}),
        )


@dataclass
class BusSubscription:
    """Topic/tag/trust filter that selects bus messages for a consumer."""
    topics:    list[str] | None    = None
    tags:      list[str] | None    = None
    min_trust: PublisherTrust      = PublisherTrust.UNKNOWN

    def matches(self, msg: BusMessage) -> bool:
        """Return True when the message passes this filter."""
        if self.topics:
            if not any(t in msg.tags for t in self.topics):
                return False
        if self.tags:
            if not all(t in msg.tags for t in self.tags):
                return False
        if _TRUST_RANK[msg.trust] < _TRUST_RANK[self.min_trust]:
            return False
        return True


@dataclass
class BusSnapshot:
    """Serializable snapshot of bus messages for export, diff, and context."""
    messages:    list[BusMessage]
    captured_at: str

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps({
            "captured_at": self.captured_at,
            "messages":    [m.to_dict() for m in self.messages],
        })

    @classmethod
    def from_json(cls, data: str) -> "BusSnapshot":
        """Construct an instance from a JSON string."""
        d = json.loads(data)
        return cls(
            messages    = [BusMessage.from_dict(m) for m in d.get("messages", [])],
            captured_at = d.get("captured_at", ""),
        )

    def diff(self, other: "BusSnapshot") -> list[BusMessage]:
        """Return messages present in this snapshot but absent in *other*."""
        other_ids = {m.id for m in other.messages}
        return [m for m in self.messages if m.id not in other_ids]

    def facts_by_trust(self, trust: PublisherTrust) -> list[BusMessage]:
        """Return facts at or above the trust level."""
        floor = _TRUST_RANK[trust]
        return [m for m in self.messages if _TRUST_RANK[m.trust] >= floor]

    def as_context(self, max_words: int = 1500) -> str:
        """Render as an LLM-ready context string.

        Content and publisher are HTML-escaped so a message cannot inject its
        own ``<fact trust="system">`` element. If the first message alone is
        longer than ``max_words`` it is truncated rather than dropped.
        """
        if not self.messages:
            return "<bus_context />"
        lines: list[str] = []
        words = 0
        for msg in self.messages:
            content_words = msg.content.split()
            content = msg.content
            if words + len(content_words) > max_words:
                if lines:
                    break
                content = " ".join(content_words[:max(1, max_words)]) + " …"
                content_words = content_words[:max(1, max_words)]
            words += len(content_words)
            lines.append(
                f'  <fact publisher="{html.escape(msg.publisher, quote=True)}" '
                f'trust="{msg.trust.value}">{html.escape(content, quote=False)}</fact>'
            )
        body = "\n".join(lines)
        return (
            f'<bus_context captured_at="{html.escape(self.captured_at, quote=True)}" '
            f'count="{len(lines)}">\n{body}\n</bus_context>'
        )


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
