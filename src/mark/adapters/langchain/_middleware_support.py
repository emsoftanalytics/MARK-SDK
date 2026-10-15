"""Shared plumbing for :mod:`mark.adapters.langchain.middleware`.

Holds the optional LangChain/LangGraph imports (with fallbacks so the module
imports without them), the tuning constants, the custom agent state schema,
the sync-to-async bridge used by the synchronous hooks, message/state access
helpers that accept both dicts and LangChain objects, and :class:`_RunState`,
the per-run record that isolates concurrent agent runs.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Any

try:
    from typing_extensions import NotRequired
except ImportError:
    from typing import Optional as NotRequired  # type: ignore[assignment]

try:
    from langchain.agents.middleware import AgentMiddleware as _AgentMiddlewareBase
    from langchain.agents.middleware import AgentState
    from langchain.agents.middleware.types import PrivateStateAttr
    from langchain.messages import SystemMessage as _SystemMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AgentMiddlewareBase = object  # type: ignore[assignment,misc]
    AgentState = dict  # type: ignore[assignment,misc]
    PrivateStateAttr = None  # type: ignore[assignment]
    _SystemMessage = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False

try:
    from langgraph.errors import GraphBubbleUp as _GraphBubbleUp
except ImportError:  # pragma: no cover
    class _GraphBubbleUp(Exception):  # type: ignore[no-redef]
        """Placeholder when LangGraph is unavailable."""


logger = logging.getLogger("mark.adapters.langchain")

#: Fallback run key for direct hook calls made outside a LangGraph run
#: (no ``mark_run_id`` in state).
_DEFAULT_RUN_ID = "__default__"
#: Upper bound on concurrently tracked runs (LRU; protects against runs that
#: never reach ``after_agent``).
_MAX_RUNS = 256
#: Upper bound on cached recall queries per run.
_MAX_RECALL_CACHE = 64
#: Upper bound on fragment ids remembered per run for feedback.
_MAX_FRAGMENT_IDS = 256
#: Upper bound on observation hashes remembered for de-duplication.
_MAX_SEEN_OBSERVATIONS = 4096
#: Upper bound on retained diagnostics events.
_MAX_DIAGNOSTICS = 1000
#: How long ``aafter_agent`` waits for the run's background tasks.
_BG_DRAIN_TIMEOUT_S = 5.0

#: True while a coroutine runs inside :func:`_run_sync` (sync hook path).
#: Background work is then executed inline, because the temporary event loop
#: is closed — cancelling any pending tasks — as soon as the hook returns.
_IN_SYNC_BRIDGE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mark_in_sync_bridge", default=False
)


# ---------------------------------------------------------------------------
# Custom agent state — carries MARK context between abefore_model and
# wrap_model_call without any blocking I/O in the wrap layer.
# ---------------------------------------------------------------------------

class _MarkAgentState(AgentState):  # type: ignore[misc,valid-type]
    """Extends AgentState with MARK's retrieved context and run identity."""
    mark_context: NotRequired[str]
    mark_context_meta: NotRequired[dict[str, Any]]
    mark_run_id: NotRequired[Annotated[str, PrivateStateAttr]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Lazily created; one worker is enough because the caller blocks on the result.
_SYNC_EXECUTOR: ThreadPoolExecutor | None = None
_SYNC_EXECUTOR_LOCK = threading.Lock()


async def _bridged(coro: Any) -> Any:
    """Mark the coroutine's context as running on the sync bridge, then run it."""
    _IN_SYNC_BRIDGE.set(True)
    return await coro


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Outside an event loop this is plain ``asyncio.run()``. When a loop is
    already running in the current thread — Jupyter kernels, GUI frameworks,
    or servers driving the sync agent API — the coroutine executes on a
    dedicated worker thread with its own loop instead: the caller blocks for
    the result and the outer loop is never touched. MARK's middleware
    coroutines only talk to the (thread-safe) local backend, so they do not
    depend on the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_bridged(coro))
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        with _SYNC_EXECUTOR_LOCK:
            if _SYNC_EXECUTOR is None:
                _SYNC_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mark-sync-bridge"
                )
    return _SYNC_EXECUTOR.submit(asyncio.run, _bridged(coro)).result()


def content_text(content: Any) -> str:
    """Return the plain text of a LangChain message ``content`` value.

    Handles string content and list content made of text blocks (``{"type":
    "text", "text": ...}``), objects with a ``text`` attribute, or bare
    strings.  Non-text blocks (images, tool-use payloads) are skipped instead
    of being rendered as a Python ``repr``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text: Any = block
            elif isinstance(block, dict):
                if block.get("type") not in (None, "text", "text_delta", "output_text"):
                    continue
                text = block.get("text") or ""
            else:
                text = getattr(block, "text", None) or ""
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("type") or msg.get("role") or "")
    return str(getattr(msg, "type", None) or getattr(msg, "role", "") or "")


def _message_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _state_messages(state: Any) -> list[Any]:
    messages = _state_get(state, "messages", None)
    return list(messages or [])


@dataclass
class _RunState:
    """Everything MARK tracks for one agent run."""

    recall_cache: OrderedDict[str, str] = field(default_factory=OrderedDict)
    pending_observations: list[str] = field(default_factory=list)
    fragment_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    last_injected_key: str = ""
    last_context: str = ""
    last_tool_trigger_hash: str = ""
    thoughts: str | None = None
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    prewarm: asyncio.Task[Any] | None = None
    # Queries whose prewarm retrieval timed out (or is still running): skipped
    # once so a step never waits two full timeouts, retried on the next step.
    timed_out: set[str] = field(default_factory=set)
    # Anticipatory context (ContextWatch) for this run and how many messages it has seen.
    watch: Any = None
    watch_seen: int = 0

    def close(self) -> None:
        """Release resources owned by the run (the watch's worker thread)."""
        if self.watch is not None:
            self.watch.close()
            self.watch = None
