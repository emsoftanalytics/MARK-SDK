"""Reusable MARK agent wrapper for memory, policies, skills, and middleware."""
from __future__ import annotations

import inspect
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

from mark.context import ContextBundle
from mark.middlewares.base import MarkMiddleware, MiddlewareContext, MiddlewareStack
from mark.middlewares.skills import SkillRegistry, SkillResult
from mark.memory import MemoryManager
from mark.policies import AgentState, PolicyAction, PolicyDecision, PolicyRegistry


@dataclass
class AgentRun:
    """Per-run configuration for a MARK-wrapped agent call."""

    prompt: str
    blocks: list[str] | None = None
    policy: str | None = None
    skills: list[str] | None = None
    skill_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieve: bool | None = None
    store_output: bool | None = None
    top_k: int | None = None
    max_context_chars: int = 6000
    session_id: str | None = None
    output_block: str = "agent-output"


@dataclass(frozen=True)
class AgentResult:
    """Result of a wrapped agent run."""

    output: str
    context: ContextBundle
    events: list[dict[str, Any]] = field(default_factory=list)
    rendered_prompt: str = ""
    decision: PolicyDecision | None = None
    skill_results: dict[str, SkillResult] = field(default_factory=dict)
    stored_record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MarkAgent:
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

    def use(self, *middleware: MarkMiddleware) -> "MarkAgent":
        """Append agent-level middleware and return this agent for chaining."""
        if self._middleware is None:
            self._middleware = MiddlewareStack()
            self._middleware.bind(self._runtime)
        self._middleware.extend(middleware)
        return self

    def available_skills(self) -> list[str]:
        """Return registered skill names."""
        return self._skills.names()

    def available_policies(self) -> list[str]:
        """Return registered policy names."""
        return self._policies.names()

    async def _execute(self, request: AgentRun) -> AgentResult:
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
            context = self._memory.retrieve(
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
            stored_record_ids = self._store_output(request, output, events)

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

    def _render_prompt(
        self,
        prompt: str,
        context: ContextBundle,
        skill_results: dict[str, SkillResult] | None = None,
    ) -> str:
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
        result = self._llm(prompt)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_skills(
        self,
        request: AgentRun,
        skill_names: list[str],
        events: list[dict[str, Any]],
    ) -> dict[str, SkillResult]:
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

    async def _run_with_middleware(self, request: AgentRun) -> AgentResult:
        if self._middleware is None:
            return await self._execute(request)

        context = MiddlewareContext(
            operation="agent.run",
            runtime=self._runtime,
            agent_id=self._agent_id,
            payload={"request": request, "prompt": request.prompt, "metadata": dict(request.metadata)},
        )
        executed: list[MarkMiddleware] = []
        try:
            for middleware in self._middleware.list():
                middleware.before(context)
                executed.append(middleware)
                if context.stopped:
                    break

            if context.stopped:
                result = context.result
                if isinstance(result, AgentResult):
                    context.result = result
                else:
                    context.result = AgentResult(
                        output="" if result is None else str(result),
                        context=self._empty_context(request.prompt),
                        events=[{"type": "middleware.stop"}],
                        metadata={"agent_id": self._agent_id},
                    )
            else:
                current = context.payload.get("request", request)
                if not isinstance(current, AgentRun):
                    raise TypeError("agent.run middleware payload['request'] must be an AgentRun")
                context.result = await self._execute(current)
        except Exception as exc:
            context.metadata["error"] = repr(exc)
            raise
        finally:
            for middleware in reversed(executed):
                middleware.after(context)

        if not isinstance(context.result, AgentResult):
            raise TypeError("agent.run middleware must return an AgentResult or a string-like stop result")
        return context.result

    def _store_output(
        self,
        request: AgentRun,
        output: str,
        events: list[dict[str, Any]],
    ) -> list[str]:
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

    def _build_request(self, prompt: str | AgentRun, **overrides: Any) -> AgentRun:
        if isinstance(prompt, AgentRun):
            request = prompt
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

    def _empty_context(self, prompt: str) -> ContextBundle:
        text = "MARK context: retrieval skipped."
        return ContextBundle(query=prompt, memories=[], text=text, token_estimate=max(1, len(text) // 4))

    def _build_middleware_stack(
        self,
        runtime: object | None,
        middleware: Iterable[MarkMiddleware] | None,
    ) -> MiddlewareStack | None:
        runtime_stack = getattr(runtime, "middleware", None)
        if middleware is None and isinstance(runtime_stack, MiddlewareStack):
            return runtime_stack
        if middleware is None:
            return None

        items: list[MarkMiddleware] = []
        if isinstance(runtime_stack, MiddlewareStack):
            items.extend(runtime_stack.list())
        items.extend(list(middleware))
        stack = MiddlewareStack(items)
        stack.bind(runtime or self)
        return stack
