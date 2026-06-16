"""TrustBus middleware for explicit trust-aware publication."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class TrustBusMiddleware(BaseMiddleware):
    """Publish selected write/observe outcomes onto the local trust-aware bus."""

    publisher: str | None = None
    trust: Any = "raw_agent"
    tags: list[str] = field(default_factory=list)
    publish_observations: bool = True
    publish_writes: bool = False
    name: str = "trust_bus"
    last_message: Any = None

    def after(self, context: MiddlewareContext) -> None:
        should_publish = (
            context.operation == "observe" and self.publish_observations
        ) or (
            context.operation == "store" and self.publish_writes
        )
        if not should_publish or not hasattr(context.runtime, "trust_bus"):
            return

        key = "text" if context.operation == "observe" else "content"
        content = str(context.payload.get(key) or "")
        if not content.strip():
            return

        publisher = self.publisher or context.agent_id or "mark"
        payload_tags = list(context.payload.get("tags") or [])
        metadata = {
            "source_operation": context.operation,
            "source_result": getattr(context.result, "fragment_id", context.result),
        }
        from mark.middlewares.trust_bus import PublisherTrust

        trust = self.trust if isinstance(self.trust, PublisherTrust) else PublisherTrust(str(self.trust))
        self.last_message = context.runtime.trust_bus().publish_sync(
            content,
            publisher=publisher,
            trust=trust,
            tags=[*self.tags, *payload_tags],
            metadata=metadata,
        )
        context.metadata["trust_bus_message"] = self.last_message


__all__ = ["TrustBusMiddleware"]
