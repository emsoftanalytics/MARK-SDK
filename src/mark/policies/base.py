"""Policy protocol and decision types governing agent behavior."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

RuntimeEvent = dict[str, Any]


class PolicyAction(str, Enum):
    """Actions a policy may choose for the next agent step."""
    RETRIEVE = "retrieve"
    CALL_LLM = "call_llm"
    STORE = "store"
    STOP = "stop"


@dataclass(frozen=True)
class AgentState:
    """Inputs a policy inspects when choosing an action."""
    prompt: str
    block_labels: list[str]
    skill_names: list[str]


@dataclass(frozen=True)
class PolicyDecision:
    """A policy's chosen action with its reasoning."""
    actions: list[PolicyAction]
    retrieve_top_k: int = 5
    store_output: bool = False
    reason: str = ""


class Policy(Protocol):
    """Protocol for pluggable agent policies."""
    name: str

    def choose(self, state: AgentState) -> PolicyDecision:
        """Choose the next action for the given state."""
        ...

    def observe(self, event: RuntimeEvent) -> None:
        """Receive an event after each agent step."""
        ...
