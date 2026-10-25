# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Trust-bus value types: trust levels, message envelopes, filters, snapshots.

This module is standard-library only so the bus data model can be used (and
serialized) without loading the memory runtime. It also owns the trust
primitives shared with :mod:`mark.middlewares.trust_bus.bus`: the rank order
used for every "at least this trusted" comparison, the reserved metadata key
that carries the authoritative trust record, and the HMAC signature over
(publisher, trust, content).
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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


#: Ordering of trust levels; higher ranks are more trusted.  Every trust
#: comparison goes through this table, never through the enum's string values.
_TRUST_RANK: dict[PublisherTrust, int] = {
    PublisherTrust.SYSTEM:         4,
    PublisherTrust.VERIFIED_AGENT: 3,
    PublisherTrust.RAW_AGENT:      2,
    PublisherTrust.EXTERNAL:       1,
    PublisherTrust.UNKNOWN:        0,
}


def _lowest(levels: list[PublisherTrust]) -> PublisherTrust:
    """Least-trusted level of *levels*: conflicting legacy tags resolve low."""
    if not levels:
        return PublisherTrust.UNKNOWN
    return min(levels, key=lambda level: _TRUST_RANK[level])


def _is_trust_tag(tag: str) -> bool:
    """True for ``trust:*`` tags, which callers must never be able to set."""
    return tag.strip().lower().startswith(_TRUST_TAG_PREFIX)


def _trust_signature(key: bytes, publisher: str, trust: PublisherTrust, content: str) -> str:
    """HMAC binding a trust level to its publisher and exact content."""
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
