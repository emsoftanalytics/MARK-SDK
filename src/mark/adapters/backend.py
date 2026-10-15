"""Framework-neutral backend protocol for MARK adapters.

Blocking memory calls from async adapters run on one lazily created,
process-wide offload pool (threads named ``mark-backend_*``).
:func:`shutdown_offload_pool` retires it — ``MarkRuntime.shutdown()`` calls
it — and the next async call simply creates a fresh pool.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from mark.adapters._feedback import _feedback_fragment_ids, _feedback_score


_OFFLOAD_EXECUTOR: ThreadPoolExecutor | None = None
_OFFLOAD_LOCK = threading.Lock()


def _offload_executor() -> ThreadPoolExecutor:
    global _OFFLOAD_EXECUTOR
    if _OFFLOAD_EXECUTOR is None:
        with _OFFLOAD_LOCK:
            if _OFFLOAD_EXECUTOR is None:
                _OFFLOAD_EXECUTOR = ThreadPoolExecutor(
                    max_workers=min(32, (os.cpu_count() or 1) + 4),
                    thread_name_prefix="mark-backend",
                )
    return _OFFLOAD_EXECUTOR


def shutdown_offload_pool(wait: bool = False) -> None:
    """Retire the shared offload pool so its threads exit once idle.

    Calls already queued still run (other runtimes in the process may own
    them), but no thread outlives its last task. ``wait=True`` also blocks
    until they finish.
    """
    global _OFFLOAD_EXECUTOR
    with _OFFLOAD_LOCK:
        executor, _OFFLOAD_EXECUTOR = _OFFLOAD_EXECUTOR, None
    if executor is not None:
        executor.shutdown(wait=wait)


async def run_off_loop(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run blocking *fn* in MARK's shared worker pool and await the result.

    Like :func:`asyncio.to_thread` (context variables are propagated), but on a
    process-wide executor instead of the loop's default one. A short-lived
    loop (``asyncio.run`` in a sync bridge) therefore never blocks at shutdown
    waiting for a call that was abandoned by an ``asyncio.wait_for`` timeout.
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    return await loop.run_in_executor(_offload_executor(), call)


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
    """Adapter backend over local ``mark.Mark`` or ``mark.MarkRuntime`` objects.

    The local memory engine is synchronous (SQLite + embeddings). Every async
    endpoint therefore runs its blocking work in a worker thread via
    :func:`run_off_loop`, so the caller's event loop keeps running,
    ``asyncio.wait_for`` timeouts are honoured and ``asyncio.gather`` gives
    real parallelism. Each endpoint also has a blocking ``*_sync`` variant.
    """

    def __init__(self, mark: Any, *, default_agent_id: str = "__mark__") -> None:
        self.mark = mark
        self.default_agent_id = default_agent_id

    def memory_for(self, agent_id: str | None = None) -> Any:
        """The ``MarkMemory`` for an agent namespace (used for anticipatory context)."""
        return self._memory(agent_id or self.default_agent_id)

    # ── async protocol (off-loop) ─────────────────────────────────────────

    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        """Retrieve memory relevant to the query."""
        return await run_off_loop(self.retrieve_sync, query, **options)

    async def write(self, content: str, **options: Any) -> BackendResult:
        """Write content into memory."""
        return await run_off_loop(self.write_sync, content, **options)

    async def observe(self, content: str, **options: Any) -> BackendResult:
        """Record an observation through the backend."""
        return await run_off_loop(self.observe_sync, content, **options)

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
        return await run_off_loop(self.list_sessions_sync, agent_id, **options)

    async def feedback(self, **payload: Any) -> BackendResult:
        """Submit a retrieval-quality feedback signal.

        Recognised payload keys:

        * ``fragment_ids`` (or a single ``fragment_id`` / ``record_id``) — the
          memory fragments the signal is about.
        * ``score`` — float in ``[-1.0, 1.0]``. When omitted it is derived from
          ``outcome`` (``"completed"`` → ``1.0``, ``"failed"`` → ``-1.0``, ...).

        Each fragment is forwarded to ``mark.memory.feedback(record_id, score)``.
        Other keys (``agent_id``, ``summary``, ...) are informational and are
        returned unchanged as the result value.
        """
        return await run_off_loop(self.feedback_sync, **payload)

    # ── blocking implementations ──────────────────────────────────────────

    def retrieve_sync(self, query: str, **options: Any) -> BackendResult:
        """Blocking variant of :meth:`retrieve`."""
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

    def write_sync(self, content: str, **options: Any) -> BackendResult:
        """Blocking variant of :meth:`write`."""
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

    def observe_sync(self, content: str, **options: Any) -> BackendResult:
        """Blocking variant of :meth:`observe`."""
        try:
            agent_id = options.pop("agent_id", self.default_agent_id)
            if hasattr(self.mark, "observe"):
                result = self.mark.observe(content, agent_id=agent_id, **options)
            else:
                result = self._memory(agent_id).observe(content, **options)
            return BackendResult(True, result, {"fragment_id": getattr(result, "fragment_id", "")})
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    def list_sessions_sync(self, agent_id: str, **options: Any) -> BackendResult:
        """Blocking variant of :meth:`list_sessions`."""
        try:
            memory = self._memory(agent_id)
            if hasattr(memory, "list_sessions"):
                return BackendResult(True, memory.list_sessions())
            return BackendResult(True, [])
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    def feedback_sync(self, **payload: Any) -> BackendResult:
        """Blocking variant of :meth:`feedback`."""
        try:
            fragment_ids = _feedback_fragment_ids(payload)
            if not fragment_ids:
                return BackendResult(True, payload, {"skipped": "no_fragment_ids"})
            score = _feedback_score(payload)
            if score is None:
                return BackendResult(True, payload, {"skipped": "no_score"})
            sink = self._feedback_sink()
            if sink is None:
                return BackendResult(True, payload, {"skipped": "feedback_unavailable"})
            for fragment_id in fragment_ids:
                sink(fragment_id, score)
            return BackendResult(True, payload, {"applied": len(fragment_ids), "score": score})
        except Exception as exc:
            return BackendResult(False, error=str(exc))

    # ── helpers ───────────────────────────────────────────────────────────

    def _feedback_sink(self) -> Any:
        """Return ``feedback(record_id, score)`` from the simple memory API, if any."""
        memory = getattr(self.mark, "memory", None)
        if memory is None or callable(memory):
            return None
        feedback = getattr(memory, "feedback", None)
        if feedback is None:
            return None
        try:
            params = inspect.signature(feedback).parameters
        except (TypeError, ValueError):
            return feedback
        return feedback if len(params) >= 2 else None

    def _memory(self, agent_id: str) -> Any:
        if hasattr(self.mark, "runtime") and hasattr(self.mark.runtime, "memory"):
            return self.mark.runtime.memory(agent_id)
        if hasattr(self.mark, "memory") and callable(self.mark.memory):
            return self.mark.memory(agent_id)
        return self.mark.memory


async def call_backend_off_loop(backend: Any, endpoint: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a backend endpoint without blocking the running event loop.

    Backends that expose a blocking ``<endpoint>_sync`` variant (such as
    :class:`LocalMarkBackend`) run it in a worker thread. Otherwise the async
    endpoint is awaited; a plain synchronous endpoint is also moved off-loop.
    """
    sync_fn = getattr(backend, f"{endpoint}_sync", None)
    if callable(sync_fn):
        return await run_off_loop(sync_fn, *args, **kwargs)
    fn = getattr(backend, endpoint)
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await run_off_loop(fn, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
