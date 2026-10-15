"""Anticipatory context for ``MarkAgentMiddleware`` via :class:`ContextWatch`.

Every agent run gets its own watch (stored in its ``_RunState``). Before each
model call the middleware feeds the watch the messages it has not seen yet —
user turns, the model's own text and tool calls, tool results — and appends
the watch's briefing to the injected MARK context. The agent therefore gets
facts about the subjects in play, corrections of outdated values it just
stated, and "not in memory" notices, without calling any memory tool.

Anticipation needs direct memory access, so it is enabled only when the
backend exposes ``memory_for(agent_id)`` (``LocalMarkBackend`` does). Custom
backends keep the plain retrieval behaviour.
"""
from __future__ import annotations

import json
from typing import Any

from mark.adapters.backend import run_off_loop
from mark.adapters.langchain._middleware_support import _RunState, _state_get, content_text

#: On the first model call of a run, at most this many trailing messages are
#: fed to the watch, so long checkpointed histories don't flood it.
_FIRST_FEED_LIMIT = 20
#: How long a model call may wait for prefetches that the latest messages started.
_BRIEFING_WAIT_S = 0.25


class _AnticipationMixin:
    """Mixin for ``MarkAgentMiddleware``; expects ``_anticipate``, ``_backend``,
    ``_agent_id``, ``_max_chars`` and ``_record`` on the host class."""

    def _watch_for(self, run: _RunState) -> Any:
        """The run's ContextWatch, created on first use (None when unavailable)."""
        if not getattr(self, "_anticipate", False):
            return None
        if run.watch is None:
            memory_for = getattr(self._backend, "memory_for", None)
            if memory_for is None:
                return None
            from mark.context.watch import ContextWatch

            memory = memory_for(self._agent_id)
            if memory is None:
                return None
            # Half the context budget: the watch complements, not replaces,
            # the query-driven retrieval block.
            run.watch = ContextWatch(memory, budget_chars=max(400, self._max_chars // 2))
        return run.watch

    def _feed_watch(self, watch: Any, run: _RunState, state: Any) -> None:
        """Observe every message the watch has not seen yet (blocking; run off-loop)."""
        messages = list(_state_get(state, "messages", []) or [])
        start = run.watch_seen
        if start == 0 and len(messages) > _FIRST_FEED_LIMIT:
            start = len(messages) - _FIRST_FEED_LIMIT
        for message in messages[start:]:
            kind = getattr(message, "type", "")
            text = content_text(getattr(message, "content", ""))
            if kind == "human":
                watch.observe(text, role="user")
            elif kind == "ai":
                if text:
                    watch.observe(text, role="assistant")
                for call in getattr(message, "tool_calls", None) or []:
                    watch.observe_tool_call(call.get("name", ""), call.get("args"))
            elif kind == "tool":
                watch.observe(text[:4000], role="tool")
        run.watch_seen = len(messages)

    async def _anticipated_briefing(self, run: _RunState, state: Any) -> str:
        """Feed new messages and return the current briefing text ("" when none)."""
        run.watch_items = ()
        watch = self._watch_for(run)
        if watch is None:
            return ""
        try:
            await run_off_loop(self._feed_watch, watch, run, state)
            # With a middleware budget the watch returns all candidate lines
            # (brief.items); the final fit happens once, with retrieval included.
            brief = await run_off_loop(watch.briefing, wait=_BRIEFING_WAIT_S,
                                       budget_tokens=getattr(self, "_budget", None))
            run.watch_items = brief.items
        except Exception as exc:  # anticipation is best-effort, never fatal
            self._record("anticipate_error", error=f"{type(exc).__name__}: {exc}")
            return ""
        if brief.text:
            self._record(
                "anticipate",
                chars=len(brief.text),
                anchors=json.dumps(brief.anchors),
                corrections=len(brief.corrections),
                unknown=len(brief.unknown),
                pending=brief.pending,
            )
        return brief.text
