"""Agent/model lifecycle hooks mixin for :class:`MarkAgentMiddleware`.

Implements ``(a)before_agent`` (run start + retrieval prewarm),
``(a)before_model`` (triggered retrieval, written to ``mark_context``),
``(a)after_model`` (queue reasoning) and ``(a)after_agent`` (flush, feedback,
release), plus the optional thinking-model helper. Each sync hook runs its
async twin through :func:`_run_sync`; every hook degrades to "no MARK" on
failure instead of breaking the agent.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from mark.adapters.backend import run_off_loop
from mark.adapters.langchain._middleware_support import (
    _IN_SYNC_BRIDGE,
    _message_role,
    _run_sync,
    _RunState,
    _state_messages,
    content_text,
    logger,
)


class _LifecycleHooksMixin:
    """before/after agent and model hooks (sync and async)."""

    # ── thinking engine ───────────────────────────────────────────────────

    def _should_think(self, query: str) -> bool:
        """Return True when *query* contains planning/reasoning trigger keywords."""
        lowered = query.lower()
        return any(t in lowered for t in self.THINKING_TRIGGERS)

    async def _think(self, query: str, context: str) -> str:
        """Call the thinking model to generate a structured reasoning plan.

        On the sync hook path the model's synchronous ``invoke`` is used (in a
        worker thread), so providers that cache event-loop-bound async clients
        are never driven from a throwaway loop.  On the async path ``ainvoke``
        is awaited.  Failures are logged and recorded; ``""`` is returned.
        """
        model = self._thinking_model
        if model is None:
            return ""

        thinking_prompt = (
            "You are a focused reasoning assistant. Given the user's request and "
            "relevant project context, think through the best approach step by step.\n\n"
            f"User request: {query}\n\n"
            f"Project context:\n{context}\n\n"
            "Think through:\n"
            "## SESSION INTENT\nWhat exactly needs to be done?\n\n"
            "## APPROACH\nWhat is the best approach given the project context?\n\n"
            "## STEPS\nWhat concrete steps should be taken? List them.\n\n"
            "## RISKS\nWhat could go wrong? What should be avoided?\n\n"
            "Be concise and actionable. Focus on what matters for this specific request."
        )
        try:
            if (_IN_SYNC_BRIDGE.get() or not hasattr(model, "ainvoke")) and hasattr(model, "invoke"):
                response = await run_off_loop(model.invoke, thinking_prompt)
            else:
                response = await model.ainvoke(thinking_prompt)
            thought = content_text(getattr(response, "content", response)).strip()
            return thought[:self._thinking_budget * 4]  # approx chars to tokens ratio
        except Exception as exc:
            self._record("thinking", ok=False, error=repr(exc)[:200])
            logger.warning("MARK thinking_model call failed; continuing without it: %r", exc)
            return ""

    # ── abefore_agent: run entry intercept ────────────────────────────────

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: start a run — assign its id, capture the query, warm retrieval.

        Returns ``{"mark_run_id": ..., "mark_context": ""}`` so per-run data is
        isolated and context checkpointed by a previous turn is never reused.
        """
        try:
            # A fresh id per run, carried in (private) graph state, keys this
            # run's _RunState so concurrent ainvoke() calls never share data.
            run_id = uuid.uuid4().hex
            run = self._run_by_id(run_id)
            human_query, _ = self._build_queries(state)
            if human_query:
                run.prewarm = await self._spawn(run, self._recall(human_query, run, cache=True))
            return {"mark_run_id": run_id, "mark_context": "", "mark_context_meta": {}}
        except Exception as exc:
            self._hook_failed("before_agent", exc)
            return None

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: start a run (see :meth:`abefore_agent`)."""
        try:
            return _run_sync(self.abefore_agent(state, runtime))
        except Exception as exc:
            self._hook_failed("before_agent", exc)
            return None

    # ── abefore_model: per-step multi-query retrieval ─────────────────────

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: triggered retrieval before every model call.

        Always returns a ``mark_context`` update — the combined retrieved
        memory, or ``""`` when nothing relevant was found — so stale context
        from earlier turns or steps never leaks into this model call.
        """
        try:
            run = self._run(state)
            human_query, ai_query = self._build_queries(state)
            if not human_query:
                return {"mark_context": "", "mark_context_meta": {"chars": 0, "source": "none"}}

            # Consume the prewarm started in before_agent (first step only).
            # Wait for it at most one retrieval timeout, and only on the loop
            # that owns it.  If it has not filled the cache by then, mark the
            # query timed out so _recall skips it once instead of waiting a
            # second full timeout; the next step retries normally.
            prewarm = run.prewarm
            if prewarm is not None:
                run.prewarm = None
                if not prewarm.done() and prewarm.get_loop() is asyncio.get_running_loop():
                    await asyncio.wait({prewarm}, timeout=self._retrieval_timeout_s or None)
                if human_query not in run.recall_cache:
                    run.timed_out.add(human_query)

            tool_query, tool_hash = self._latest_tool_trigger(state)
            # Anticipatory briefing (subjects in play, corrections, unknowns).
            # It is part of the cache key: a correction triggered by the model's
            # own last message must reach the next call even if nothing else changed.
            briefing = await self._anticipated_briefing(run, state)

            # Skip re-retrieval when neither the user goal nor the triggers have
            # changed since the last model call of this run.
            ai_trigger = ai_query if self._should_think(ai_query) else ""
            inject_key = f"{human_query}||{ai_trigger}||{tool_hash}||{self._hash(briefing)}"
            if inject_key == run.last_injected_key:
                cached = run.last_context
                return {
                    "mark_context": cached,
                    "mark_context_meta": {
                        "chars": len(cached),
                        "context_hash": self._hash(cached),
                        "source": "cache",
                    },
                }

            refresh_tool = bool(tool_hash) and tool_hash != run.last_tool_trigger_hash
            combined = await self._retrieve_and_combine(
                run,
                human_query,
                ai_trigger,
                tool_query if refresh_tool else "",
            )
            if refresh_tool:
                run.last_tool_trigger_hash = tool_hash
            if briefing:
                combined = f"{briefing}\n\n{combined}" if combined else briefing

            # Thinking mode — secondary LLM when the query has planning triggers.
            # Computed once per run, injected alongside MARK memory and archived.
            if self._thinking_model and self._should_think(human_query):
                if run.thoughts is None:
                    run.thoughts = await self._think(human_query, combined or "")
                    if run.thoughts:
                        await self._spawn(run, self._backend_call(
                            "observe",
                            f"MARK reasoning plan for query '{human_query[:100]}':\n"
                            f"{run.thoughts[:400]}",
                            agent_id=self._agent_id,
                        ))
                if run.thoughts:
                    thought_block = f"[MARK reasoning]\n{run.thoughts}\n[/MARK reasoning]"
                    combined = (combined + "\n\n---\n\n" + thought_block) if combined else thought_block

            run.last_injected_key = inject_key
            run.last_context = combined
            return {
                "mark_context": combined,
                "mark_context_meta": {
                    "chars": len(combined),
                    "context_hash": self._hash(combined),
                    "source": "retrieve",
                    "tool_trigger": refresh_tool,
                },
            }
        except Exception as exc:
            self._hook_failed("before_model", exc)
            return {"mark_context": "", "mark_context_meta": {"chars": 0, "source": "error"}}

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: triggered MARK retrieval before every model call."""
        try:
            return _run_sync(self.abefore_model(state, runtime))
        except Exception as exc:
            self._hook_failed("before_model", exc)
            return {"mark_context": "", "mark_context_meta": {"chars": 0, "source": "error"}}

    # ── aafter_model: queue AI reasoning step ─────────────────────────────

    def _queue_latest_reasoning(self, state: Any) -> None:
        for msg in reversed(_state_messages(state)):
            if _message_role(msg) in ("ai", "assistant"):
                content = self._message_text(msg).strip()
                has_tool_calls = bool(getattr(msg, "tool_calls", None))
                if content and len(content) > 100 and not has_tool_calls:
                    self._queue_observation(self._run(state), f"Agent reasoning: {content[:600]}")
                break

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: queue each substantive AI reasoning step for archival.

        Only substantive turns (>100 chars, no tool-dispatch payloads) are
        queued, to avoid noise.  Returns ``None`` (no state update needed).
        """
        if not self._write_outcomes:
            return None
        try:
            self._queue_latest_reasoning(state)
        except Exception as exc:
            self._hook_failed("after_model", exc)
        return None

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: queue substantive AI reasoning for run-end archival."""
        if not self._write_outcomes:
            return None
        try:
            self._queue_latest_reasoning(state)
        except Exception as exc:
            self._hook_failed("after_model", exc)
        return None

    # ── aafter_agent: run end — flush, feedback, cleanup ──────────────────

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: flush observations, emit feedback, release the run.

        Fires once when the agent graph reaches its terminal node.  Queued
        observations are written, a ``feedback()`` signal is sent for the
        fragments retrieved during the run, pending background tasks are
        drained (bounded wait), and ``mark_context`` is cleared so the next
        turn on the same thread never sees stale memory.
        """
        run_id = self._run_id(state)
        run = self._pop_run(run_id) or _RunState()
        try:
            if self._write_outcomes:
                for msg in reversed(_state_messages(state)):
                    if _message_role(msg) in ("ai", "assistant"):
                        content = self._message_text(msg)[:200]
                        self._queue_observation(run, f"Agent completed task. Response: {content}")
                        await self._spawn(run, self._backend_call(
                            "feedback",
                            agent_id=self._agent_id,
                            outcome="completed",
                            summary=content,
                            fragment_ids=list(run.fragment_ids),
                        ))
                        break
                await self._flush_observations_async(run)
        except Exception as exc:
            self._hook_failed("after_agent", exc)
        finally:
            try:
                await self._drain(run)
            except Exception as exc:
                self._hook_failed("after_agent", exc)
        return {"mark_context": "", "mark_context_meta": {}}

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: flush queued observations and release the run."""
        try:
            return _run_sync(self.aafter_agent(state, runtime))
        except Exception as exc:
            self._hook_failed("after_agent", exc)
            self._pop_run(self._run_id(state))
            return {"mark_context": "", "mark_context_meta": {}}
