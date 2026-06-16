"""Framework-neutral backend protocol for MARK adapters."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendResult:
    """Small framework-neutral result envelope used by adapter helpers."""

    ok: bool
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class MarkBackend(Protocol):
    """Public backend protocol implemented by MARK adapters."""

    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        """Retrieve memory relevant to the query."""
        ...
    async def write(self, content: str, **options: Any) -> BackendResult:
        """Write content into memory."""
        ...
    async def observe(self, content: str, **options: Any) -> BackendResult:
        """Record an observation through the backend."""
        ...
    async def compress(self, query: str, candidates: list[Any], **options: Any) -> BackendResult:
        """Reduce candidate fragments to query-relevant evidence."""
        ...
    async def heal_gap(self, query: str, **options: Any) -> BackendResult:
        """Attempt explicit gap healing for the query."""
        ...
    async def list_sessions(self, agent_id: str, **options: Any) -> BackendResult:
        """Return known session ids for the agent."""
        ...
    async def feedback(self, **payload: Any) -> BackendResult:
        """Submit a feedback signal."""
        ...


class LocalMarkBackend:
    """Adapter backend over local ``mark.Mark`` or ``mark.MarkRuntime`` objects."""

    def __init__(self, mark: Any, *, default_agent_id: str = "__mark__") -> None:
        self.mark = mark
        self.default_agent_id = default_agent_id

    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        """Retrieve memory relevant to the query."""
        try:
            memory = self._memory(options.pop("agent_id", self.default_agent_id))
            blocks = options.pop("blocks", None)
            if blocks and "block_ids" not in options and "block_id" not in options:
                options["block_ids"] = list(blocks)
            if hasattr(memory, "retrieve_sync"):
                result = memory.retrieve_sync(query, **options)
                return BackendResult(True, result)
            result = self.mark.memory.retrieve(query, **options)
            return BackendResult(True, result)
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    async def write(self, content: str, **options: Any) -> BackendResult:
        """Write content into memory."""
        try:
            memory = self._memory(options.pop("agent_id", self.default_agent_id))
            if hasattr(memory, "store_sync"):
                fragment_id = memory.store_sync(content, **options)
                return BackendResult(True, fragment_id, {"fragment_id": fragment_id})
            block = self.mark.memory.block(options.pop("block", "adapter"))
            record = block.write(content, **options)
            return BackendResult(True, record, {"record_id": getattr(record, "id", "")})
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    async def observe(self, content: str, **options: Any) -> BackendResult:
        """Record an observation through the backend."""
        try:
            agent_id = options.pop("agent_id", self.default_agent_id)
            if hasattr(self.mark, "observe"):
                result = self.mark.observe(content, agent_id=agent_id, **options)
            else:
                result = self._memory(agent_id).observe(content, **options)
            return BackendResult(True, result, {"fragment_id": getattr(result, "fragment_id", "")})
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    async def compress(self, query: str, candidates: list[Any], **options: Any) -> BackendResult:
        # Local compression is performed during retrieve(compress=True). This method
        # exists so framework adapters can share one backend protocol.
        """Reduce candidate fragments to query-relevant evidence."""
        return BackendResult(True, candidates, {"local_passthrough": True, "query": query, **options})

    async def heal_gap(self, query: str, **options: Any) -> BackendResult:
        """Attempt explicit gap healing for the query."""
        options["heal_gaps"] = True
        return await self.retrieve(query, **options)

    async def list_sessions(self, agent_id: str, **options: Any) -> BackendResult:
        """Return known session ids for the agent."""
        try:
            memory = self._memory(agent_id)
            if hasattr(memory, "list_sessions"):
                return BackendResult(True, memory.list_sessions())
            return BackendResult(True, [])
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    async def feedback(self, **payload: Any) -> BackendResult:
        """Submit a feedback signal."""
        try:
            if hasattr(self.mark, "memory") and hasattr(self.mark.memory, "feedback"):
                feedback = self.mark.memory.feedback
                params = inspect.signature(feedback).parameters
                if all(key in params for key in payload):
                    feedback(**payload)
                    return BackendResult(True, payload)
                return BackendResult(True, payload, {"skipped": "incompatible_feedback_signature"})
            return BackendResult(True, payload, {"skipped": "feedback_unavailable"})
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    def _memory(self, agent_id: str) -> Any:
        if hasattr(self.mark, "runtime") and hasattr(self.mark.runtime, "memory"):
            return self.mark.runtime.memory(agent_id)
        if hasattr(self.mark, "memory") and callable(self.mark.memory):
            return self.mark.memory(agent_id)
        return self.mark.memory
