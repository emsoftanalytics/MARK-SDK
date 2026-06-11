"""Skill registry preloaded with built-ins."""
from __future__ import annotations

from typing import Any

from mark.skills.base import Skill
from mark.skills.builtin import EchoSkill, MarkMemorySkill


class SkillRegistry:
    """Registry of named skills."""
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    @classmethod
    def with_builtins(cls, *, memory: Any | None = None) -> "SkillRegistry":
        """Construct a registry preloaded with built-ins."""
        registry = cls()
        registry.register(EchoSkill())
        if memory is not None:
            registry.register(MarkMemorySkill(memory))
        return registry

    def register(self, skill: Skill) -> None:
        """Register an entry."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        """Return a registered skill by name, or raise KeyError."""
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills))
            raise KeyError(f"Unknown MARK skill '{name}'. Available skills: {available}") from exc

    def names(self) -> list[str]:
        """Return the registered names."""
        return sorted(self._skills)
