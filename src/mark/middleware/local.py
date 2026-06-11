"""Local middleware wrappers for existing MARK runtime capabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.governance import ConsolidationGate
from mark.middleware.base import BaseMiddleware, MiddlewareContext
from mark.types import MemoryState


@dataclass
class RecallMiddleware(BaseMiddleware):
    """Set default retrieval options for downstream memory calls."""

    compress: bool | None = None
    expand: bool | None = None
    heal_gaps: bool | None = None
    escalate_on_gap: bool | None = None
    name: str = "recall"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation != "retrieve":
            return
        defaults = {
            "compress": self.compress,
            "expand": self.expand,
            "heal_gaps": self.heal_gaps,
            "escalate_on_gap": self.escalate_on_gap,
        }
        for key, value in defaults.items():
            if value is not None:
                context.payload[key] = value


@dataclass
class ObserveMiddleware(BaseMiddleware):
    """Apply default observation metadata and tags."""

    source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = "observe"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation not in {"observe", "store"}:
            return
        if self.source is not None:
            if context.payload.get("source") is None:
                context.payload["source"] = self.source
        if self.tags:
            incoming = list(context.payload.get("tags") or [])
            context.payload["tags"] = [*incoming, *self.tags]
        if self.metadata:
            incoming_meta = dict(context.payload.get("metadata") or {})
            context.payload["metadata"] = {**self.metadata, **incoming_meta}


@dataclass
class CompressionMiddleware(BaseMiddleware):
    """Configure a runtime compressor and optionally enable retrieval compression."""

    compressor: Any
    enabled_by_default: bool = True
    name: str = "compression"

    def bind(self, runtime: Any) -> None:
        if hasattr(runtime, "configure_compressor"):
            runtime.configure_compressor(self.compressor)

    def before(self, context: MiddlewareContext) -> None:
        if self.enabled_by_default and context.operation == "retrieve":
            context.payload["compress"] = True


@dataclass
class QueryExpansionMiddleware(BaseMiddleware):
    """Configure a runtime query expander and optionally enable expansion."""

    expander: Any
    enabled_by_default: bool = True
    name: str = "query_expansion"

    def bind(self, runtime: Any) -> None:
        if hasattr(runtime, "configure_query_expander"):
            runtime.configure_query_expander(self.expander)

    def before(self, context: MiddlewareContext) -> None:
        if self.enabled_by_default and context.operation == "retrieve":
            context.payload["expand"] = True


@dataclass
class GapHealingMiddleware(BaseMiddleware):
    """Opt in to MARK's explicit local gap-healing path for retrieval."""

    heal_gaps: bool = True
    escalate_on_gap: bool = True
    web_skill: Any = None
    name: str = "gap_healing"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation != "retrieve":
            return
        context.payload["heal_gaps"] = self.heal_gaps
        context.payload["escalate_on_gap"] = self.escalate_on_gap
        if self.web_skill is not None:
            context.payload.setdefault("web_skill", self.web_skill)


@dataclass
class LifecycleMiddleware(BaseMiddleware):
    """Run local maintenance cycles after write-like operations."""

    every_n_writes: int = 0
    promote_threshold: float = 0.7
    name: str = "lifecycle"
    _write_count: int = 0

    def after(self, context: MiddlewareContext) -> None:
        if self.every_n_writes <= 0:
            return
        if context.operation not in {"store", "observe"}:
            return
        if context.agent_id is None or not hasattr(context.runtime, "run_cycle"):
            return
        self._write_count += 1
        if self._write_count % self.every_n_writes == 0:
            context.metadata["lifecycle"] = context.runtime.run_cycle(
                context.agent_id,
                promote_threshold=self.promote_threshold,
            )


@dataclass
class ObservabilityMiddleware(BaseMiddleware):
    """Emit local tracer events around middleware operations."""

    event_prefix: str = "middleware"
    name: str = "observability"

    def before(self, context: MiddlewareContext) -> None:
        tracer = context.runtime.tracer() if hasattr(context.runtime, "tracer") else None
        if tracer is not None:
            tracer.emit(
                f"{self.event_prefix}.{context.operation}.start",
                agent_id=context.agent_id,
                middleware=True,
            )

    def after(self, context: MiddlewareContext) -> None:
        tracer = context.runtime.tracer() if hasattr(context.runtime, "tracer") else None
        if tracer is not None:
            tracer.emit(
                f"{self.event_prefix}.{context.operation}.finish",
                agent_id=context.agent_id,
                middleware=True,
                stopped=context.stopped,
            )


@dataclass
class SyncMiddleware(BaseMiddleware):
    """Prepare redacted sync deltas after write-like operations."""

    options: Any = None
    cloud: Any = None
    upload: bool = False
    name: str = "sync"
    last_delta: Any = None
    last_stats: Any = None

    def after(self, context: MiddlewareContext) -> None:
        if context.operation not in {"store", "observe"}:
            return
        from mark.sync import CloudSync

        sync = CloudSync()
        if self.upload and self.cloud is not None:
            self.last_stats = sync.sync(context.runtime, self.cloud, options=self.options)
            context.metadata["sync_stats"] = self.last_stats
            return
        self.last_delta = sync.prepare_delta(context.runtime, options=self.options)
        self.last_stats = self.last_delta.stats
        context.metadata["sync_delta"] = self.last_delta


@dataclass
class GovernanceMiddleware(BaseMiddleware):
    """Apply an explicit local governance gate before memory writes."""

    gate: ConsolidationGate = field(default_factory=ConsolidationGate)
    quarantine_on_reject: bool = True
    name: str = "governance"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation not in {"store", "observe"}:
            return
        key = "text" if context.operation == "observe" else "content"
        content = str(context.payload.get(key) or "")
        confidence = float(context.payload.get("confidence", 1.0))
        result = self.gate.check(content, confidence=confidence)
        context.payload[key] = result.content

        metadata = dict(context.payload.get("metadata") or {})
        metadata["mark_governance"] = {
            "middleware": self.name,
            "passed": result.passed,
            "reason": result.reason,
        }
        context.payload["metadata"] = metadata

        if not result.passed:
            tags = list(context.payload.get("tags") or [])
            context.payload["tags"] = [*tags, "mark:governance", "mark:quarantined"]
            if self.quarantine_on_reject and context.operation == "store":
                context.payload["state"] = MemoryState.QUARANTINED
                context.payload["importance"] = 0.0
                context.payload["confidence"] = 0.0


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
        from mark.memory.trust_bus import PublisherTrust

        trust = self.trust if isinstance(self.trust, PublisherTrust) else PublisherTrust(str(self.trust))
        self.last_message = context.runtime.trust_bus().publish_sync(
            content,
            publisher=publisher,
            trust=trust,
            tags=[*self.tags, *payload_tags],
            metadata=metadata,
        )
        context.metadata["trust_bus_message"] = self.last_message


@dataclass
class MediaContinuityMiddleware(BaseMiddleware):
    """Apply local media/creative-continuity defaults."""

    memory_type: str = "scene"
    tags: list[str] = field(default_factory=lambda: ["continuity"])
    retrieve_with_expansion: bool = True
    name: str = "media_continuity"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation == "observe":
            context.payload.setdefault("memory_type", self.memory_type)
            incoming = list(context.payload.get("tags") or [])
            continuity_tags = ["continuity", *self.tags]
            context.payload["tags"] = [*incoming, *continuity_tags]
            metadata = dict(context.payload.get("metadata") or {})
            metadata.setdefault("mark_continuity", {"memory_type": self.memory_type})
            context.payload["metadata"] = metadata
            return

        if context.operation == "retrieve" and self.retrieve_with_expansion:
            context.payload["expand"] = True


@dataclass
class SandboxMiddleware(BaseMiddleware):
    """Local sandbox execution wrapper for explicit sandbox operations."""

    sandbox: Any = None
    enabled: bool = False
    name: str = "sandbox"
    last_result: Any = None

    def bind(self, runtime: Any) -> None:
        if self.sandbox is None:
            try:
                from mark.sandbox import DockerSandbox
                self.sandbox = DockerSandbox()
            except Exception:
                self.sandbox = None

    def before(self, context: MiddlewareContext) -> None:
        if context.operation != "sandbox_execute":
            return
        if not self.enabled or self.sandbox is None:
            context.stop(
                {
                    "ok": False,
                    "sandbox_mode": "blocked/local-disabled",
                    "stderr": "Local sandbox middleware is disabled.",
                }
            )
            return

        code = str(context.payload.get("code") or "")
        timeout = context.payload.get("timeout")
        workspace = context.payload.get("workspace")
        env = context.payload.get("env")
        self.last_result = self.sandbox.execute(
            code,
            timeout=timeout,
            workspace=workspace,
            env=env,
        )
        context.stop(self.last_result)


__all__ = [
    "CompressionMiddleware",
    "GapHealingMiddleware",
    "GovernanceMiddleware",
    "LifecycleMiddleware",
    "MediaContinuityMiddleware",
    "ObservabilityMiddleware",
    "ObserveMiddleware",
    "QueryExpansionMiddleware",
    "RecallMiddleware",
    "SandboxMiddleware",
    "SyncMiddleware",
    "TrustBusMiddleware",
]
