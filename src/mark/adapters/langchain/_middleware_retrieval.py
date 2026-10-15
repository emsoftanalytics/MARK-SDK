"""Retrieval mixin for :class:`MarkAgentMiddleware`.

Turns agent state into retrieval queries (the user goal, a planning excerpt
of the latest AI message, and the latest search-like tool result), runs them
against the backend under a timeout, renders results into a bounded context
block, and records retrieved fragment ids for run-end feedback.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mark.adapters.backend import BackendResult
from mark.adapters.langchain._middleware_support import (
    _MAX_FRAGMENT_IDS,
    _MAX_RECALL_CACHE,
    _message_role,
    _RunState,
    _state_messages,
)


class _RetrievalMixin:
    """Query extraction, timed recall with a per-run cache, and result merging."""

    def _tool_name(self, msg: Any) -> str:
        return str(
            getattr(msg, "name", "")
            or getattr(msg, "tool_name", "")
            or (getattr(msg, "additional_kwargs", None) or {}).get("name", "")
        )

    def _latest_tool_trigger(self, state: Any) -> tuple[str, str]:
        """Return a retrieval query from the latest useful tool/subagent result.

        Tool output is valuable context, but write/delete results often reflect
        unfinished intermediate work. Those are queued for observation at session
        end instead of being made immediately retrievable during the same run.
        Ordinary filesystem reads are also already present in the live context,
        so their right MARK action is observe-later, not retrieve-now.
        """
        for msg in reversed(_state_messages(state)):
            role = _message_role(msg)
            if role in ("human", "user"):
                # Tool results from earlier turns (checkpointed history) are
                # not triggers for the current request.
                return "", ""
            if role not in ("tool", "function"):
                continue
            name = self._tool_name(msg)
            lowered = name.lower()
            if any(skip in lowered for skip in ("write", "delete", "remove", "mark_")):
                return "", ""
            if not any(
                trigger in lowered
                for trigger in ("subagent", "search", "grep", "retrieve", "lookup", "web")
            ):
                return "", ""
            content = self._message_text(msg).strip()
            if len(content) < 40:
                return "", ""
            query = f"Context from {name or 'tool'} result:\n{content[:700]}"
            return query, self._hash(f"{name}:{content[:700]}")
        return "", ""

    def _truncate_context(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        keep = max(self._max_chars - 80, 0)
        return text[:keep].rstrip() + "\n\n[MARK context truncated]"

    # ── query extraction helpers ──────────────────────────────────────────

    def _build_queries(self, state: Any) -> tuple[str, str]:
        """Extract retrieval queries from agent state.

        Returns ``(human_query, ai_query)`` where:

        * ``human_query`` — the user's original goal (stable across all model
          calls in one run; cached so only one backend round-trip per run).
        * ``ai_query``    — the latest AI reasoning excerpt (first 300 chars of
          the most recent non-tool-call AI message).
        """
        human_query = ""
        ai_query = ""
        for msg in reversed(_state_messages(state)):
            role = _message_role(msg)
            if not human_query and role in ("human", "user"):
                human_query = self._message_text(msg).strip()
            if not ai_query and role in ("ai", "assistant"):
                if getattr(msg, "tool_calls", None):
                    continue
                content = self._message_text(msg).strip()
                if len(content) > 20:
                    ai_query = content[:300]
            if human_query and ai_query:
                break
        return human_query, ai_query

    def _get_human_query_from_messages(self, messages: Any) -> str:
        """Extract last human message text from a ModelRequest message list."""
        for msg in reversed(messages or []):
            if _message_role(msg) in ("human", "user"):
                return self._message_text(msg).strip()
        return ""

    # ── core retrieval ────────────────────────────────────────────────────

    @staticmethod
    def _render(value: Any) -> tuple[str, list[str]]:
        """Render a retrieval value as ``(context_text, fragment_ids)``.

        Uses the structured result: a result whose ``fragments`` list is empty
        has no memory, so ``""`` is returned (nothing gets injected).
        """
        fragments = getattr(value, "fragments", None)
        if fragments is not None and not fragments:
            return "", []
        text = (value.as_context() if hasattr(value, "as_context")
                else value.as_text() if hasattr(value, "as_text")
                else str(value))
        ids = [str(getattr(f, "id", "") or "") for f in (fragments or [])]
        return (text or "").strip(), [i for i in ids if i]

    @staticmethod
    def _needs_heal(value: Any, has_memory: bool) -> bool:
        if not has_memory:
            return True
        from mark.types.gap import GapSeverity

        severity = getattr(getattr(value, "gap_report", None), "severity", None)
        try:
            return GapSeverity(severity) in (GapSeverity.HIGH, GapSeverity.CRITICAL)
        except ValueError:
            return False

    async def _timed_backend(self, endpoint: str, query: str) -> BackendResult | None:
        """Call a retrieval endpoint under ``retrieval_timeout_ms``.

        Returns ``None`` on timeout.
        """
        call = self._backend_call(endpoint, query, agent_id=self._agent_id, compress=self._compress)
        if not self._retrieval_timeout_s:
            return await call
        try:
            return await asyncio.wait_for(call, timeout=self._retrieval_timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            self._record(
                "endpoint",
                endpoint=endpoint,
                ok=False,
                latency_ms=round(self._retrieval_timeout_s * 1000, 1),
                error="timeout",
            )
            return None

    async def _recall(self, query: str, run: _RunState, *, cache: bool = True) -> str:
        """Retrieve relevant MARK memory for *query* (bounded, ``""`` if none).

        Args:
            query: Retrieval query string.
            run:   Per-run record owning the cache.
            cache: When ``True`` (default), result is read from / written to the
                   run's recall cache.  Pass ``cache=False`` for step-specific
                   queries (e.g., AI reasoning excerpt) that change each model call.
        """
        if not query:
            return ""
        if cache and query in run.recall_cache:
            run.recall_cache.move_to_end(query)
            return run.recall_cache[query]
        if query in run.timed_out:
            run.timed_out.discard(query)
            return ""

        text, ids = "", []
        result = await self._timed_backend("retrieve", query)
        if result is not None and result.ok:
            text, ids = self._render(result.value)
            used = result.value
            if self._auto_heal and self._needs_heal(result.value, bool(text)):
                healed = await self._timed_backend("heal_gap", query)
                if healed is not None and healed.ok:
                    healed_text, healed_ids = self._render(healed.value)
                    if healed_text:
                        text, ids, used = healed_text, healed_ids, healed.value
                        self._record("heal", chars=len(healed_text))

        if getattr(self, "_budget", None) is not None and result is not None and result.ok:
            run.recall_items[query] = self._retrieval_items(used, query) if text else []
            while len(run.recall_items) > _MAX_RECALL_CACHE:
                run.recall_items.pop(next(iter(run.recall_items)))
        text = self._truncate_context(text) if text else ""
        for fragment_id in ids:
            run.fragment_ids[fragment_id] = None
        while len(run.fragment_ids) > _MAX_FRAGMENT_IDS:
            run.fragment_ids.popitem(last=False)
        if cache and result is not None:
            run.recall_cache[query] = text
            while len(run.recall_cache) > _MAX_RECALL_CACHE:
                run.recall_cache.popitem(last=False)
        return text

    async def _retrieve_and_combine(
        self,
        run: _RunState,
        human_query: str,
        ai_query: str = "",
        tool_query: str = "",
    ) -> str:
        """Run triggered retrieval queries and return a bounded combined block.

        Human query is cached (stable). AI/tool queries are only used when they
        represent a meaningful trigger point, such as a planning response or a
        newly observed read/search/subagent result.

        Returns empty string when no relevant memory is found.
        """
        run.step_queries = [q for q in (human_query, ai_query, tool_query) if q]
        tasks: list[Any] = [self._recall(human_query, run, cache=True)]
        if ai_query:
            tasks.append(self._recall(ai_query, run, cache=False))
        if tool_query:
            tasks.append(self._recall(tool_query, run, cache=False))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        blocks: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                if not isinstance(r, Exception):
                    raise r
                self._hook_failed("retrieve", r)
                continue
            if r and r not in blocks:
                blocks.append(r)

        if not blocks:
            return ""

        combined = self._truncate_context("\n\n---\n\n".join(blocks))
        self._record(
            "retrieve",
            queries=1 + bool(ai_query) + bool(tool_query),
            chars=len(combined),
            context_hash=self._hash(combined),
        )
        return combined
