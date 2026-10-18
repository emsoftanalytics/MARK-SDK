"""``mark_briefing`` for the MARK MCP server: push context for MCP clients.

Small models are bad at deciding when to call a search tool. ``mark_briefing``
lets the client (or a thin host loop) hand MARK the latest conversation text
and get back a ready-made ``<mark_briefing>`` block — facts about the subjects
in play, linked subjects, corrections of outdated values and "Not in memory"
notices — computed by a :class:`~mark.context.watch.ContextWatch`.

Invariants:

* **Per-session isolation.** One server may serve many clients. Every
  ``(agent_id, session_id)`` pair gets its own watch; nothing observed in one
  session can appear in another session's briefing.
* **Bounded memory.** Sessions live in an LRU of ``max_sessions`` entries;
  the least recently used one is evicted and its watch closed (its worker
  thread stopped) outside the LRU lock. Session ids and input text are
  length-capped.
* **Never blocks the event loop, never waits long.** All watch work runs on
  MARK's shared worker pool (``run_off_loop``); ``wait`` is 0-``_MAX_WAIT_S``
  seconds. A session's first call waits at least ``_FIRST_WAIT_S`` so a cold
  call does not answer "Nothing relevant" before its lookups ran.
* **Clear failures.** A denied namespace, a bad role, a backend without
  local memory or an internal error comes back as an ``isError=True``
  result with one sentence (never a traceback, never a "success" text).
* ``allowed_agent_ids`` is enforced exactly as for the other tools.

The optional resource ``mark://briefing/{session_id}`` returns the current
briefing of an existing session in the default agent namespace without
observing anything (for clients that poll). It never creates a session, so
polling unknown ids cannot grow server state.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, Any

from mark.adapters.backend import run_off_loop
from mark.adapters.mcp._common import AgentId, bounded, fail, tool_error_class

logger = logging.getLogger("mark.adapters.mcp")

_MAX_WAIT_S = 10.0           # upper bound on how long one call waits for prefetches
_FIRST_WAIT_S = 2.0          # a new session's first call waits at least this long
_BUDGET = (50, 4000)         # budget_tokens bounds (0 = the preset's own size)
_MAX_TEXT = 8000             # only the tail of very long conversation text is scanned
_MAX_SESSION_ID = 200
_UNAVAILABLE = "MARK briefing unavailable: this backend has no local memory."


class BriefingSessions:
    """Thread-safe LRU of per-session :class:`ContextWatch` instances.

    ``factory(agent_id)`` builds a watch for a new session (or returns None
    when the backend cannot). ``close_all()`` stops every watch.
    """

    def __init__(self, factory: Callable[[str], Any], max_sessions: int = 64) -> None:
        self._factory = factory
        self.max_sessions = max(1, int(max_sessions))
        self._lock = threading.Lock()
        self._watches: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._last_text: dict[tuple[str, str], str] = {}   # skip re-observing identical polls

    def __len__(self) -> int:
        with self._lock:
            return len(self._watches)

    def keys(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._watches)

    def get(self, agent_id: str, session_id: str, *, create: bool = True) -> Any:
        """The session's watch (most recently used now), creating it when allowed."""
        key = (agent_id, session_id)
        with self._lock:
            watch = self._watches.get(key)
            if watch is not None:
                self._watches.move_to_end(key)
                return watch
        if not create:
            return None
        watch = self._factory(agent_id)   # outside the lock: may touch the store
        if watch is None:
            return None
        evicted: list[Any] = []
        with self._lock:
            existing = self._watches.get(key)
            if existing is not None:      # another caller created it meanwhile
                evicted.append(watch)
                watch = existing
            else:
                self._watches[key] = watch
            self._watches.move_to_end(key)
            while len(self._watches) > self.max_sessions:
                old_key, old = self._watches.popitem(last=False)
                self._last_text.pop(old_key, None)
                evicted.append(old)
        for old in evicted:               # closing joins a worker thread: never under the lock
            _close(old)
        return watch

    def is_new_text(self, agent_id: str, session_id: str, text: str) -> bool:
        """False when *text* repeats the session's previous input (a client re-polling)."""
        key = (agent_id, session_id)
        with self._lock:
            if self._last_text.get(key) == text:
                return False
            self._last_text[key] = text
            return True

    def close_all(self) -> None:
        with self._lock:
            watches = list(self._watches.values())
            self._watches.clear()
            self._last_text.clear()
        for watch in watches:
            _close(watch)


def _close(watch: Any) -> None:
    try:
        watch.close()
    except Exception as exc:  # pragma: no cover - close is best-effort
        logger.warning("closing a briefing watch failed: %s", exc)


def _watch_factory(backend: Any, preset: str) -> Callable[[str], Any]:
    """Build watches over ``backend.memory_for(agent_id)`` with the chosen preset."""

    def factory(agent_id: str) -> Any:
        memory_for = getattr(backend, "memory_for", None)
        if memory_for is None:
            return None
        memory = memory_for(agent_id)
        if memory is None:
            return None
        from mark.context.watch import ContextWatch

        return ContextWatch.for_preset(memory, preset)

    return factory


def _briefing_sync(
    sessions: BriefingSessions, agent_id: str, session_id: str,
    text: str, role: str, messages: list[dict[str, Any]] | None, wait: float,
    budget_tokens: int | None = None,
) -> str:
    """Observe the new input and render the session's briefing (blocking; run off-loop)."""
    fresh = (agent_id, session_id) not in sessions.keys()
    watch = sessions.get(agent_id, session_id)
    if watch is None:
        fail(_UNAVAILABLE)
    if fresh:
        wait = max(wait, _FIRST_WAIT_S)  # first call: let the session's lookups finish
    if messages:
        watch.observe_messages(messages)
    text = text.strip()[-_MAX_TEXT:]
    if text and sessions.is_new_text(agent_id, session_id, f"{role}\x00{text}"):
        watch.observe(text, role=role)
    brief = watch.briefing(wait=wait, budget_tokens=budget_tokens)
    return brief.text or "<mark_briefing>\nNothing relevant in memory yet.\n</mark_briefing>"


def register_briefing(
    server: Any,
    backend: Any,
    resolve_agent: Callable[[str], tuple[str, str | None]],
    *,
    default_agent_id: str,
    max_sessions: int = 64,
    preset: str = "default",
) -> BriefingSessions:
    """Add the ``mark_briefing`` tool (and the briefing resource) to *server*."""
    if preset not in ("default", "small", "tiny"):
        raise ValueError("briefing preset must be 'default', 'small' or 'tiny'")
    sessions = BriefingSessions(_watch_factory(backend, preset), max_sessions)

    @server.tool()
    async def mark_briefing(
        text: Annotated[str, bounded("The latest message (user, assistant or tool text).")] = "",
        session_id: Annotated[str, bounded("Stable id of this conversation; one briefing state per id.",
                                           maxLength=_MAX_SESSION_ID)] = "default",
        role: Annotated[str, bounded("Who wrote text: user, assistant or tool.",
                                     enum=["user", "assistant", "tool"])] = "user",
        messages: Annotated[list[dict[str, Any]] | None,
                            bounded("Recent chat as [{role, content}] instead of text.")] = None,
        agent_id: AgentId = "",
        wait: Annotated[float, bounded("Seconds (0-10) to let running lookups finish; a session's "
                                       "first call waits at least 2.", minimum=0, maximum=_MAX_WAIT_S)] = 0.5,
        budget_tokens: Annotated[int, bounded("Size cap in tokens (chars/4), 50-4000; 0 = the preset's "
                                              "size.", minimum=0, maximum=_BUDGET[1])] = 0,
    ) -> str:
        """Memory briefing for an ongoing conversation, meant for the context window.

        Pass the latest message as ``text`` (``role`` user, assistant or tool),
        or the recent chat as ``messages`` (role/content dicts), and a stable
        ``session_id`` per conversation. Returns current facts about the
        subjects being discussed, related notes, corrections of outdated
        values and the subjects memory knows nothing about. ``budget_tokens``
        > 0 caps the briefing (tokens estimated as characters / 4), keeping
        corrections and unknowns first. ``wait`` (0-10 s) lets running
        lookups finish.
        """
        if isinstance(wait, bool) or not 0 <= wait <= _MAX_WAIT_S:
            fail(f"wait must be between 0 and {_MAX_WAIT_S:g} seconds (got {wait!r})")
        if budget_tokens and not _BUDGET[0] <= budget_tokens <= _BUDGET[1]:
            fail(f"budget_tokens must be 0 (preset size) or between {_BUDGET[0]} and {_BUDGET[1]}")
        resolved, denied = resolve_agent(agent_id)
        if denied:
            fail(f"MARK briefing failed: {denied}")
        if role not in ("user", "assistant", "tool"):
            fail("MARK briefing failed: role must be user, assistant or tool")
        session = (session_id or "default")[:_MAX_SESSION_ID]
        try:
            return await run_off_loop(
                _briefing_sync, sessions, resolved, session, text or "", role, messages,
                float(wait), int(budget_tokens) or None,
            )
        except tool_error_class():
            raise
        except Exception as exc:  # reported as a tool error, never as a traceback
            logger.warning("mark_briefing failed: %s", exc)
            fail(f"MARK briefing failed: {type(exc).__name__}")

    _register_resource(server, sessions, default_agent_id)
    return sessions


def _register_resource(server: Any, sessions: BriefingSessions, default_agent_id: str) -> None:
    """``mark://briefing/{session_id}``: read-only view for polling clients.

    Registered only when the server exposes the ``resource`` decorator with
    URI templates (MCPServer in mcp 2.x, FastMCP in mcp 1.x both do).
    """
    resource = getattr(server, "resource", None)
    if resource is None:  # pragma: no cover - every supported mcp version has it
        return

    def _read(session_id: str) -> str:
        watch = sessions.get(default_agent_id, session_id[:_MAX_SESSION_ID], create=False)
        if watch is None:
            return ""
        return watch.briefing().text

    try:
        @resource("mark://briefing/{session_id}", name="mark_briefing",
                  description="Current MARK briefing of a session (default agent namespace).",
                  mime_type="text/plain")
        async def briefing_resource(session_id: str) -> str:
            try:
                return await run_off_loop(_read, session_id)
            except Exception as exc:
                logger.warning("briefing resource failed: %s", exc)
                return ""
    except Exception as exc:  # pragma: no cover - resource API differences: tool still works
        logger.info("mark://briefing resource not registered: %s", exc)
