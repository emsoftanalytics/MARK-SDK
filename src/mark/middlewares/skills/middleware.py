# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Skills middleware — make packaged agent skills visible automatically."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext
from mark.middlewares.skills.loader import list_packaged_skills, load_skill_text


@dataclass
class SkillMiddleware(BaseMiddleware):
    """Expose MARK's packaged agent skills (SKILL.md) to agents automatically.

    Add it to the runtime middleware stack like any other battery::

        mark = Mark.local(".", middleware=[SkillMiddleware()])

        # one line to teach any agent how to use MARK:
        agent = create_agent(model, tools,
                             system_prompt=PROMPT + mark.runtime.skill_prompt())

    What it does:

    * **bind** — loads the packaged ``SKILL.md`` texts once, registers them
      into the runtime's skill registry when one is available, and exposes
      ``runtime.skill_prompt()`` so agent builders can pull the combined
      skill text without knowing where skills live on disk.
    * **retrieve** — stamps ``skills_available`` metadata onto the operation
      so downstream middleware and observability can see which skills the
      agent was offered.

    skills  names of packaged skills to expose; defaults to every skill
            bundled with the installation (``mark-usage`` and
            ``mark-memory-middleware``).
    """

    skills: list[str] = field(default_factory=list_packaged_skills)
    name: str = "skills"
    _texts: dict[str, str] = field(default_factory=dict, repr=False)

    def bind(self, runtime: Any) -> None:
        """Load skill texts, register them, and attach skill_prompt() to the runtime."""
        self._texts = {
            skill_name: text
            for skill_name in self.skills
            if (text := load_skill_text(skill_name))
        }
        registry = getattr(runtime, "skills", None)
        if registry is not None and hasattr(registry, "register"):
            try:
                from mark.middlewares.skills.base import MarkdownSkill
                for skill_name, text in self._texts.items():
                    if hasattr(registry, "find") and registry.find(skill_name):
                        continue
                    registry.register(MarkdownSkill.from_text(skill_name, text))
            except Exception:
                pass  # registry wiring is best-effort; prompt() always works
        if not hasattr(runtime, "skill_prompt"):
            try:
                runtime.skill_prompt = self.prompt
            except Exception:
                pass

    def prompt(self) -> str:
        """Combined skill text, ready to append to an agent system prompt."""
        return "".join(f"\n\n{text}" for text in self._texts.values())

    def texts(self) -> dict[str, str]:
        """Mapping of skill name to its SKILL.md text."""
        return dict(self._texts)

    def before(self, context: MiddlewareContext) -> None:
        """Stamp which skills were available onto retrieval operations."""
        if context.operation != "retrieve" or not self._texts:
            return
        context.metadata["skills_available"] = sorted(self._texts)


__all__ = ["SkillMiddleware"]
