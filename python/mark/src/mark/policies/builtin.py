"""Built-in agent policies."""
from __future__ import annotations

from mark.policies.base import AgentState, PolicyAction, PolicyDecision, RuntimeEvent


class DefaultPolicy:
    """Baseline policy: always proceeds."""
    name = "default"

    def choose(self, state: AgentState) -> PolicyDecision:
        """Choose the next action for the given state."""
        return PolicyDecision(
            actions=[PolicyAction.RETRIEVE, PolicyAction.CALL_LLM],
            retrieve_top_k=5,
            store_output=False,
            reason="Retrieve local context and call the model.",
        )

    def observe(self, event: RuntimeEvent) -> None:
        """Receive an event after each agent step."""
        return None


class CodingAgentPolicy(DefaultPolicy):
    """Policy tuned for coding agents: asks for context on low confidence."""
    name = "coding-agent"

    def choose(self, state: AgentState) -> PolicyDecision:
        """Choose the next action for the given state."""
        return PolicyDecision(
            actions=[PolicyAction.RETRIEVE, PolicyAction.CALL_LLM],
            retrieve_top_k=8,
            store_output=False,
            reason="Prefer broader project/procedure context for coding tasks.",
        )


class ResearchPolicy(DefaultPolicy):
    """Policy tuned for research agents: prefers retrieval before answering."""
    name = "research"

    def choose(self, state: AgentState) -> PolicyDecision:
        """Choose the next action for the given state."""
        return PolicyDecision(
            actions=[PolicyAction.RETRIEVE, PolicyAction.CALL_LLM],
            retrieve_top_k=10,
            store_output=False,
            reason="Retrieve more candidate memories for source-heavy research tasks.",
        )
