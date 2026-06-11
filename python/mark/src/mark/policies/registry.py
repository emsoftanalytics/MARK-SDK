"""Registry of named agent policies."""
from __future__ import annotations

from mark.policies.base import Policy
from mark.policies.builtin import CodingAgentPolicy, DefaultPolicy, ResearchPolicy


class PolicyRegistry:
    """Registry of named policies."""
    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}

    @classmethod
    def with_builtins(cls) -> "PolicyRegistry":
        """Construct a registry preloaded with built-ins."""
        registry = cls()
        registry.register(DefaultPolicy())
        registry.register(CodingAgentPolicy())
        registry.register(ResearchPolicy())
        return registry

    def register(self, policy: Policy) -> None:
        """Register an entry."""
        self._policies[policy.name] = policy

    def get(self, name: str) -> Policy:
        """Return a registered policy by name, or raise KeyError."""
        try:
            return self._policies[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._policies))
            raise KeyError(f"Unknown MARK policy '{name}'. Available policies: {available}") from exc

    def names(self) -> list[str]:
        """Return the registered names."""
        return sorted(self._policies)
