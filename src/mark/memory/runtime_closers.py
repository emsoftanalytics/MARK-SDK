# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Shutdown hooks of :class:`mark.memory.runtime.MarkRuntime`: ``register_closer``.

Objects that run background work against a runtime's store (a
``ContextWatch`` worker prefetching memories, for example) must stop before
the store closes: a worker still querying while the connections close
crashed the interpreter in a release re-test. Such an object registers a
closer::

    unregister = runtime.register_closer(watch.close)
    ...
    unregister()          # when it was closed on its own first

``shutdown()`` (and ``Mark.close()``) calls every registered closer first,
newest first, before stopping the fact worker and closing the store. A
closer that raises is logged and skipped; the others still run. Each closer
runs once at most.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger("mark.runtime")


class ClosersMixin:
    """``register_closer`` and the shutdown step that runs the closers."""

    def register_closer(self, closer: Callable[[], object]) -> Callable[[], None]:
        """Run *closer* at the start of :meth:`shutdown` (newest first); returns an unregister function.

        If the runtime is already shut down, *closer* runs immediately.
        """
        if not callable(closer):
            raise TypeError("register_closer() needs a callable")
        lock, closers = self._closer_state()
        with lock:
            if not getattr(self, "_closed", False) and not self.__dict__.get("_closers_done"):
                closers.append(closer)
                token = closer

                def unregister() -> None:
                    with lock:
                        if any(c is token for c in closers):
                            closers[:] = [c for c in closers if c is not token]
                return unregister
        _run_one(closer)
        return lambda: None

    def _run_closers(self) -> None:
        """Call and forget every registered closer, newest first (shutdown step 1)."""
        lock, closers = self._closer_state()
        while True:  # a closer may register another (or a thread may, meanwhile)
            with lock:
                pending, closers[:] = list(closers), []
                if not pending:
                    self.__dict__["_closers_done"] = True  # later registrations run at once
                    return
            for closer in reversed(pending):
                _run_one(closer)

    def _closer_state(self) -> tuple[threading.Lock, list[Callable[[], object]]]:
        state = self.__dict__.get("_closers_state")
        if state is None:
            state = self.__dict__.setdefault("_closers_state", (threading.Lock(), []))
        return state


def _run_one(closer: Callable[[], object]) -> None:
    try:
        closer()
    except Exception:  # noqa: BLE001 - one failing closer must not keep the store open
        logger.warning("closer %r failed during shutdown", closer, exc_info=True)
