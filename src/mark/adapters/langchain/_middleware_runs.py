"""Run bookkeeping mixin for :class:`MarkAgentMiddleware`.

Diagnostics and failure logging, the per-run state registry, tracked
background tasks, the never-raising backend call wrapper, and the
de-duplicating observation queue.

Invariants:

* **Per-run isolation.** Each agent run owns one :class:`_RunState`, keyed by
  the ``mark_run_id`` stored in graph state, so one middleware instance can
  serve concurrent runs. The registry is a bounded LRU: runs that never reach
  ``after_agent`` are evicted instead of leaking.
* **Background task tracking.** Every spawned task is strongly referenced
  (instance-wide and per run) until done, so it cannot be garbage-collected
  mid-flight; ``after_agent`` drains the run's tasks with a bounded wait.
"""

from __future__ import annotations

import asyncio
import hashlib
from time import perf_counter
from typing import Any

from mark.adapters.backend import BackendResult, call_backend_off_loop
from mark.adapters.langchain._middleware_support import (
    _BG_DRAIN_TIMEOUT_S,
    _DEFAULT_RUN_ID,
    _IN_SYNC_BRIDGE,
    _MAX_RUNS,
    _MAX_SEEN_OBSERVATIONS,
    _GraphBubbleUp,
    _message_content,
    _RunState,
    _state_get,
    content_text,
    logger,
)


class _RunsMixin:
    """Per-run state, background tasks, diagnostics and observation queueing.

    Expects the host class to set ``_backend``, ``_agent_id``, ``_runs``,
    ``_runs_lock``, ``_seen_observation_hashes``, ``_seen_lock``,
    ``_bg_tasks`` and ``_diagnostics`` in ``__init__``.
    """

    # ── diagnostics / error handling ──────────────────────────────────────

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        """Return lightweight middleware telemetry for tracing and benchmarks.

        Bounded to the most recent ``1000`` events.
        """
        return list(self._diagnostics)

    def _record(self, kind: str, **data: Any) -> None:
        self._diagnostics.append({"kind": kind, **data})

    def _hook_failed(self, hook: str, exc: BaseException) -> None:
        """Record and log a MARK failure; the agent continues without MARK."""
        self._record("hook_error", hook=hook, error=repr(exc)[:200])
        logger.warning(
            "MARK %s failed; continuing without MARK memory: %r",
            hook, exc, exc_info=(type(exc), exc, exc.__traceback__),
        )

    # ── per-run state ─────────────────────────────────────────────────────

    def _run_id(self, state: Any) -> str:
        return str(_state_get(state, "mark_run_id", None) or _DEFAULT_RUN_ID)

    def _run(self, state: Any) -> _RunState:
        """Return (creating if needed) the per-run record for *state*."""
        return self._run_by_id(self._run_id(state))

    def _run_by_id(self, run_id: str) -> _RunState:
        with self._runs_lock:
            run = self._runs.get(run_id)
            if run is None:
                run = _RunState()
                self._runs[run_id] = run
                while len(self._runs) > _MAX_RUNS:
                    evicted, _ = self._runs.popitem(last=False)
                    self._record("run_evicted", run_id=evicted)
            else:
                self._runs.move_to_end(run_id)
            return run

    def _pop_run(self, run_id: str) -> _RunState | None:
        with self._runs_lock:
            return self._runs.pop(run_id, None)

    def _on_run_error(self, state: Any, exc: BaseException) -> None:
        """Release per-run state when a model/tool call raises.

        LangGraph interrupts (human-in-the-loop) are control flow, not errors:
        the run resumes later, so its state is kept.
        """
        if isinstance(exc, _GraphBubbleUp):
            return
        run_id = self._run_id(state)
        if self._pop_run(run_id) is not None:
            self._record("run_aborted", run_id=run_id, error=repr(exc)[:200])

    @property
    def active_runs(self) -> int:
        """Number of runs currently holding per-run state."""
        with self._runs_lock:
            return len(self._runs)

    # ── background tasks ──────────────────────────────────────────────────

    async def _spawn(self, run: _RunState | None, coro: Any) -> asyncio.Task[Any] | None:
        """Run *coro* in the background, tracked and with errors logged.

        On the sync hook path (inside :func:`_run_sync`) the temporary event
        loop is closed when the hook returns, which would cancel pending
        tasks — the coroutine is therefore awaited inline instead.
        """
        if _IN_SYNC_BRIDGE.get():
            try:
                await coro
            except Exception as exc:
                self._hook_failed("background", exc)
            return None
        task = asyncio.get_running_loop().create_task(coro)
        self._bg_tasks.add(task)
        if run is not None:
            run.tasks.add(task)
        task.add_done_callback(lambda t: self._task_done(run, t))
        return task

    def _task_done(self, run: _RunState | None, task: asyncio.Task[Any]) -> None:
        """Drop the task's strong references and report any failure."""
        self._bg_tasks.discard(task)
        if run is not None:
            run.tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._hook_failed("background", exc)

    async def _drain(self, run: _RunState) -> None:
        """Wait (bounded) for the run's background tasks on this loop."""
        loop = asyncio.get_running_loop()
        pending = [t for t in run.tasks if not t.done() and t.get_loop() is loop]
        if not pending:
            return
        _, still_pending = await asyncio.wait(pending, timeout=_BG_DRAIN_TIMEOUT_S)
        if still_pending:
            self._record("bg_drain_timeout", pending=len(still_pending))

    # ── backend access ────────────────────────────────────────────────────

    async def _backend_call(self, endpoint: str, *args: Any, **kwargs: Any) -> BackendResult:
        """Call a backend endpoint off-loop; never raises (except cancellation)."""
        started = perf_counter()
        try:
            result = await call_backend_off_loop(self._backend, endpoint, *args, **kwargs)
        except Exception as exc:
            self._record(
                "endpoint",
                endpoint=endpoint,
                ok=False,
                latency_ms=round((perf_counter() - started) * 1000, 1),
                error=str(exc)[:200],
            )
            logger.warning("MARK backend %s failed: %r", endpoint, exc)
            return BackendResult(False, error=str(exc))
        self._record(
            "endpoint",
            endpoint=endpoint,
            ok=bool(getattr(result, "ok", False)),
            latency_ms=round((perf_counter() - started) * 1000, 1),
            error=getattr(result, "error", None),
        )
        return result

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]

    def _message_text(self, msg: Any) -> str:
        return content_text(_message_content(msg))

    def _queue_observation(self, run: _RunState, content: str) -> None:
        """Queue *content* for run-end archival unless trivial or already seen.

        De-duplication is cross-run (bounded LRU of content hashes), so the
        same tool output observed by many runs is archived once.
        """
        if not content:
            return
        normalized = " ".join(content.split())
        if len(normalized) < 40:
            return
        key = self._hash(normalized)
        with self._seen_lock:
            if key in self._seen_observation_hashes:
                self._seen_observation_hashes.move_to_end(key)
                return
            self._seen_observation_hashes[key] = None
            while len(self._seen_observation_hashes) > _MAX_SEEN_OBSERVATIONS:
                self._seen_observation_hashes.popitem(last=False)
        run.pending_observations.append(normalized[:1200])

    async def _flush_observations_async(self, run: _RunState) -> int:
        """Write the run's queued observations; return how many succeeded."""
        pending = run.pending_observations
        run.pending_observations = []
        count = 0
        for content in pending:
            result = await self._backend_call("observe", content, agent_id=self._agent_id)
            if result.ok:
                count += 1
        if count:
            self._record("observe_flush", count=count)
        return count
