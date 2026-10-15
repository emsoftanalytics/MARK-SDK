# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Individual steps of a MarkAgent run.

These are the pieces ``MarkAgent._execute`` orchestrates: rendering the
final prompt, calling the developer's LLM without blocking the event loop,
running requested skills, and persisting the output back into memory.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from mark._agent.run_types import AgentRun
from mark.context import ContextBundle
from mark.middlewares.skills import SkillRegistry, SkillResult


class AgentStepsMixin:
    """Run steps shared by MarkAgent; relies on attributes set in its __init__."""

    _llm: Callable[..., object]
    _memory: Any
    _skills: SkillRegistry
    _block_labels: list[str]
    _skill_prompt: str
    _agent_id: str

    def _render_prompt(
        self,
        prompt: str,
        context: ContextBundle,
        skill_results: dict[str, SkillResult] | None = None,
    ) -> str:
        """Compose skill prompt, memory context, and skill output around the request."""
        parts: list[str] = []
        if self._skill_prompt:
            parts.append(self._skill_prompt.strip())
        if context.memories:
            parts.append(context.as_text())
        rendered_skills = [
            result.as_context()
            for result in (skill_results or {}).values()
            if result.success and result.content
        ]
        if rendered_skills:
            parts.append("MARK skill results:\n" + "\n\n".join(rendered_skills))
        parts.append(f"User request:\n{prompt}" if parts else prompt)
        return "\n\n".join(parts)

    async def _call_llm(self, prompt: str) -> object:
        """Call the developer LLM, awaiting async callables directly."""
        if inspect.iscoroutinefunction(self._llm) or inspect.iscoroutinefunction(
            getattr(self._llm, "__call__", None)
        ):
            return await self._llm(prompt)
        # Synchronous LLM callables block (network / local inference); run them
        # in a worker thread so the caller's event loop keeps running.
        result = await asyncio.to_thread(self._llm, prompt)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_skills(
        self,
        request: AgentRun,
        skill_names: list[str],
        events: list[dict[str, Any]],
    ) -> dict[str, SkillResult]:
        """Run each named skill sequentially, recording a ``skill.run`` event."""
        results: dict[str, SkillResult] = {}
        for name in skill_names:
            skill_input = dict(request.skill_inputs.get(name) or {"prompt": request.prompt})
            skill_context = {
                "agent_id": self._agent_id,
                "prompt": request.prompt,
                "blocks": request.blocks if request.blocks is not None else self._block_labels,
                "metadata": request.metadata,
                "session_id": request.session_id,
            }
            result = await self._skills.run_skill(name, skill_input, skill_context)
            results[name] = result
            events.append(
                {
                    "type": "skill.run",
                    "skill": name,
                    "success": result.success,
                    "error": result.error,
                }
            )
        return results

    def _store_output(
        self,
        request: AgentRun,
        output: str,
        events: list[dict[str, Any]],
    ) -> list[str]:
        """Persist the LLM output and return the new record ids.

        Prefers ``memory.observe``; memory objects without it (AttributeError)
        fall back to writing into ``request.output_block``.
        """
        tags = ["agent-output", f"agent:{self._agent_id}"]
        metadata = {"prompt": request.prompt, **request.metadata}
        try:
            observed = self._memory.observe(
                output,
                session_id=request.session_id,
                importance=0.5,
                tags=tags,
                source="mark-agent",
                metadata=metadata,
            )
            record_id = getattr(observed, "fragment_id", None) or getattr(observed, "id", "")
            events.append({"type": "memory.store", "record_id": record_id})
            return [record_id] if record_id else []
        except AttributeError:
            record = self._memory.block(request.output_block).write(
                output,
                importance=0.5,
                metadata=metadata,
                source="mark-agent",
            )
            events.append({"type": "memory.store", "record_id": record.id})
            return [record.id]

    def _empty_context(self, prompt: str) -> ContextBundle:
        """Placeholder context used when retrieval is skipped or the run stops."""
        text = "MARK context: retrieval skipped."
        return ContextBundle(query=prompt, memories=[], text=text, token_estimate=max(1, len(text) // 4))
