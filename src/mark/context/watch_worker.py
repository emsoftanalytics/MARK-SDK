"""The context watch's single background worker: ordered, lazy and self-stopping.

Why not ``ThreadPoolExecutor(max_workers=1)`` (used up to 0.2.0a10): its
thread lives until ``shutdown()``. A watch the host forgot to close kept a
thread alive after its store was closed (integration test N8), and an MCP
server shutting down logged "prefetch failed: store ... is closed" for
prefetches still queued (agent test N5).

This worker keeps the same contract (one task at a time, in submission
order, a :class:`~concurrent.futures.Future` per task) but

* starts its daemon thread only when there is work,
* lets the thread exit after ``idle_seconds`` without work (a later submit
  starts a new one), and
* drops queued work once :meth:`stop` was called.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable

from mark.context.watch_support import _wait


class WatchWorker:
    """Runs submitted callables one at a time on a short-lived daemon thread."""

    def __init__(self, name: str = "mark-watch", idle_seconds: float = 2.0) -> None:
        self.name = name
        self.idle_seconds = idle_seconds
        self._queue: queue.Queue[tuple[Future, Callable[..., Any], tuple[Any, ...]]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = False

    @property
    def alive(self) -> bool:
        """True while the worker thread is running."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        """Queue ``fn(*args)``; the returned future is cancelled if the worker is stopped."""
        future: Future = Future()
        with self._lock:
            if self._stopped:
                future.cancel()
                return future
            self._queue.put((future, fn, args))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
                self._thread.start()
        return future

    def _run(self) -> None:
        while True:
            try:
                future, fn, args = self._queue.get(timeout=self.idle_seconds)
            except queue.Empty:
                with self._lock:   # exit only if nothing arrived meanwhile
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            if self._stopped or not future.set_running_or_notify_cancel():
                future.cancel()
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001 - reported through the future
                future.set_exception(exc)

    def stop(self, wait: bool = True, timeout: float | None = 10.0) -> None:
        """Cancel queued work and (optionally) wait for the running task to finish."""
        with self._lock:
            self._stopped = True
            thread = self._thread
        while True:
            try:
                future, _, _ = self._queue.get_nowait()
            except queue.Empty:
                break
            future.cancel()
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout)


class _WatchLifecycleMixin:
    """flush / reset / close for ContextWatch (expects ``_lock``, ``_pending``,
    ``_executor``, ``_closed``, ``_task``, ``_task_results`` and the state dicts)."""

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for running prefetches. Returns True when none are left."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return True
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return False
            _wait(pending[0], remaining)

    def reset(self) -> None:
        """Forget the conversation (keep memory)."""
        self.flush(timeout=5)
        with self._lock:
            self._anchors.clear()
            self._results.clear()
            self._task_results.clear()
            self._task_changes.clear()
            self._corrections.clear()
            getattr(self, "_previous", {}).clear()
            getattr(self, "_task_pool", {}).clear()
            self._task = ""
            self._turn = 0

    def _cold(self) -> bool:
        """True when the current request has no prefetched result yet (first call for it)."""
        with self._lock:
            return bool(self._task and self._pending and self._task not in self._task_results)

    def close(self) -> None:
        """Stop the background worker (queued prefetches are dropped). Idempotent."""
        self._closed = True
        if self._executor is not None:
            self._executor.stop(wait=True)
            self._executor = None

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
