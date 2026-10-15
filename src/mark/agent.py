"""Reusable MARK agent wrapper for memory, policies, skills, and middleware.

The public class :class:`MarkAgent` orchestrates one run in ``_execute``;
the run value objects, individual steps, and middleware handling live in
:mod:`mark._agent` and are re-exported here.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Callable

from mark._agent.middleware import AgentMiddlewareMixin
from mark._agent.run_types import AgentResult, AgentRun
from mark._agent.steps import AgentStepsMixin
from mark.context import ContextBundle  # noqa: F401  (re-export)
from mark.middlewares.base import MarkMiddleware, MiddlewareContext, MiddlewareStack  # noqa: F401
from mark.middlewares.skills import SkillRegistry, SkillResult  # noqa: F401
from mark.memory import MemoryManager
from mark.policies import AgentState, PolicyAction, PolicyDecision, PolicyRegistry  # noqa: F401


class MarkAgent(AgentMiddlewareMixin, AgentStepsMixin):
    """Production-oriented wrapper for a developer-provided LLM callable.

    The wrapper keeps the simple API intact while adding the reusable pieces a
    real application needs: policy-directed actions, memory retrieval, optional
    skill execution, output observation, and operation-level middleware.
    """

    def __init__(
        self,
        *,
        llm: Callable[..., object],
        memory: MemoryManager,
        policies: PolicyRegistry,
        skills: SkillRegistry,
        block_labels: list[str],
        policy_name: str,
        skill_names: list[str],
        skill_prompt: str = "",
        runtime: object | None = None,
        middleware: Iterable[MarkMiddleware] | None = None,
        agent_id: str = "mark-agent",
        default_max_context_chars: int = 6000,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._policies = policies
        self._skills = skills
        self._block_labels = list(block_labels)
        self._policy_name = policy_name
        self._skill_names = list(skill_names)
        self._skill_prompt = skill_prompt
        self._runtime = runtime or self
        self._agent_id = agent_id
        self._default_max_context_chars = default_max_context_chars
        self._middleware = self._build_middleware_stack(runtime, middleware)
        # True while ``self._middleware`` is the runtime-wide stack shared with
        # every other agent; ``use()`` copies it before extending.
        self._shares_runtime_stack = (
            self._middleware is not None
            and self._middleware is getattr(runtime, "middleware", None)
        )

    async def run(
        self,
        prompt: str | AgentRun,
        *,
        blocks: list[str] | None = None,
        policy: str | None = None,
        skills: list[str] | None = None,
        skill_inputs: dict[str, dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        retrieve: bool | None = None,
        store_output: bool | None = None,
        top_k: int | None = None,
        max_context_chars: int | None = None,
        session_id: str | None = None,
        output_block: str | None = None,
    ) -> AgentResult:
        """Run the wrapped agent with MARK memory, policy, skills, and middleware."""
        request = self._build_request(
            prompt,
            blocks=blocks,
            policy=policy,
            skills=skills,
            skill_inputs=skill_inputs,
            metadata=metadata,
            retrieve=retrieve,
            store_output=store_output,
            top_k=top_k,
            max_context_chars=max_context_chars,
            session_id=session_id,
            output_block=output_block,
        )
        return await self._run_with_middleware(request)

    def available_skills(self) -> list[str]:
        """Return registered skill names."""
        return self._skills.names()

    def available_policies(self) -> list[str]:
        """Return registered policy names."""
        return self._policies.names()

    async def _execute(self, request: AgentRun) -> AgentResult:
        """Run one request: policy -> retrieval -> skills -> LLM -> store.

        Explicit request flags (``retrieve``/``store_output``) override the
        policy decision. Blocking memory calls run in worker threads so the
        caller's event loop is never blocked.
        """
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        policy = self._policies.get(request.policy or self._policy_name)
        blocks = list(request.blocks if request.blocks is not None else self._block_labels)
        skill_names = list(request.skills if request.skills is not None else self._skill_names)
        state = AgentState(prompt=request.prompt, block_labels=blocks, skill_names=skill_names)
        decision = policy.choose(state)
        actions = list(decision.actions)
        action_values = [a.value for a in actions]
        events.append(
            {
                "type": "policy.decision",
                "policy": policy.name,
                "actions": action_values,
                "reason": decision.reason,
            }
        )

        context = self._empty_context(request.prompt)
        should_retrieve = request.retrieve
        if should_retrieve is None:
            should_retrieve = PolicyAction.RETRIEVE in actions
        if should_retrieve:
            context = await asyncio.to_thread(
                self._memory.retrieve,
                request.prompt,
                blocks=blocks or None,
                top_k=request.top_k or decision.retrieve_top_k,
                max_chars=request.max_context_chars,
                session_id=request.session_id,
            )
            events.append({"type": "memory.retrieve", "records": len(context.memories)})

        skill_results = await self._run_skills(request, skill_names, events)
        rendered_prompt = self._render_prompt(request.prompt, context, skill_results)

        output = ""
        if PolicyAction.STOP in actions:
            events.append({"type": "agent.stop"})
        elif PolicyAction.CALL_LLM in actions:
            raw_output = await self._call_llm(rendered_prompt)
            output = str(raw_output)
            events.append({"type": "llm.call"})

        stored_record_ids: list[str] = []
        should_store = request.store_output
        if should_store is None:
            should_store = decision.store_output or PolicyAction.STORE in actions
        if should_store and output:
            stored_record_ids = await asyncio.to_thread(
                self._store_output, request, output, events
            )

        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        metadata = {
            **request.metadata,
            "agent_id": self._agent_id,
            "duration_ms": duration_ms,
        }
        event = {
            "type": "agent.run",
            "prompt": request.prompt,
            "output": output,
            "policy": policy.name,
            "actions": action_values,
            "duration_ms": duration_ms,
        }
        policy.observe(event)
        events.append({"type": "agent.finish", "duration_ms": duration_ms})
        return AgentResult(
            output=output,
            context=context,
            events=events,
            rendered_prompt=rendered_prompt,
            decision=decision,
            skill_results=skill_results,
            stored_record_ids=stored_record_ids,
            metadata=metadata,
        )

    def _build_request(self, prompt: str | AgentRun, **overrides: Any) -> AgentRun:
        """Merge per-call overrides onto agent defaults (None = use default)."""
        if isinstance(prompt, AgentRun):
            # Never mutate the caller's (possibly reused) AgentRun in place.
            request = replace(
                prompt,
                blocks=list(prompt.blocks) if prompt.blocks is not None else None,
                skills=list(prompt.skills) if prompt.skills is not None else None,
                skill_inputs=dict(prompt.skill_inputs),
                metadata=dict(prompt.metadata),
            )
        else:
            request = AgentRun(
                prompt=prompt,
                blocks=list(self._block_labels),
                policy=self._policy_name,
                skills=list(self._skill_names),
                max_context_chars=self._default_max_context_chars,
            )

        for key, value in overrides.items():
            if value is not None:
                setattr(request, key, value)
        if request.blocks is None:
            request.blocks = list(self._block_labels)
        if request.policy is None:
            request.policy = self._policy_name
        if request.skills is None:
            request.skills = list(self._skill_names)
        if request.max_context_chars <= 0:
            request.max_context_chars = self._default_max_context_chars
        return request
