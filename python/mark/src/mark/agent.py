"""Agent wrapper that injects MARK memory context into plain callables."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from mark.context import ContextBundle
from mark.memory import MemoryManager
from mark.policies import AgentState, PolicyRegistry
from mark.skills import SkillRegistry


@dataclass(frozen=True)
class AgentResult:
    """Result of a wrapped agent run: output plus the memory context used."""
    output: str
    context: ContextBundle
    events: list[dict[str, Any]] = field(default_factory=list)


class MarkAgent:
    """Small wrapper that injects MARK memory into a developer-provided LLM callable."""

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
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._policies = policies
        self._skills = skills
        self._block_labels = block_labels
        self._policy_name = policy_name
        self._skill_names = skill_names

    async def run(self, prompt: str) -> AgentResult:
        """Run the wrapped agent with MARK context injected into the prompt."""
        policy = self._policies.get(self._policy_name)
        state = AgentState(prompt=prompt, block_labels=self._block_labels, skill_names=self._skill_names)
        decision = policy.choose(state)

        context = self._memory.retrieve(
            prompt,
            blocks=self._block_labels or None,
            top_k=decision.retrieve_top_k,
        )
        rendered_prompt = self._render_prompt(prompt, context)
        output = await self._call_llm(rendered_prompt)

        events = [
            {"type": "policy.decision", "policy": policy.name, "actions": [a.value for a in decision.actions]},
            {"type": "memory.retrieve", "records": len(context.memories)},
            {"type": "llm.call"},
        ]
        policy.observe({"type": "agent.run", "prompt": prompt, "output": output})
        return AgentResult(output=str(output), context=context, events=events)

    @staticmethod
    def _render_prompt(prompt: str, context: ContextBundle) -> str:
        if not context.memories:
            return prompt
        return f"{context.as_text()}\n\nUser request:\n{prompt}"

    async def _call_llm(self, prompt: str) -> object:
        result = self._llm(prompt)
        if inspect.isawaitable(result):
            return await result
        return result
