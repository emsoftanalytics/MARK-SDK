# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Skills are the unit of capability an agent can invoke outside the LLM —
# web search, code search, planning, data retrieval, external APIs, etc.
#
# Third-party skills are distributed as Python packages and auto-discovered
# by SkillRegistry.discover() via the "mark.middlewares.skills" entry-point group.
#
# Publishing a skill:
#   1. Create a Python package, e.g. mark-skill-webscraper
#   2. Implement a Skill subclass
#   3. Register the entry point in pyproject.toml:
#        [project.entry-points."mark.middlewares.skills"]
#        webscraper = "mark_skill_webscraper:WebScraperSkill"
#   4. pip install mark-skill-webscraper
#   5. registry.discover() loads it automatically
#
# Built-in Apache-2.0 skills: EchoSkill, MarkMemorySkill, WebSearchSkill
# MARK Core skills (MSAL): AdvancedWebSearchSkill, ConsolidationSkill
"""Skill registry, plus re-exports of the skill contracts.

:class:`SkillRegistry` owns registration, entry-point discovery, SKILL.md
directory loading (which never shadows protected skills) and permission-gated
dispatch. The contracts live in :mod:`.contracts` and :mod:`.markdown_skill`;
they are re-exported here because this module is their historical import path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mark.middlewares.skills.contracts import (
    Skill,
    SkillContext,
    SkillInput,
    SkillManifest,
    SkillResult,
)
from mark.middlewares.skills.markdown_skill import MarkdownSkill, _parse_frontmatter

logger = logging.getLogger(__name__)


# ── SkillRegistry ──────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Registry that manages, discovers, and dispatches skills.

    Permission gating:
        registry.grant("web")   — allow web skills
        registry.grant("*")     — grant all (dev/test only)

    Discovery:
        registry.discover()     — auto-load installed middleware skill packages
        registry.from_skills_dir(path)  — load SKILL.md files from a directory
    """

    def __init__(self) -> None:
        self._skills:              Dict[str, Skill] = {}
        self._granted_permissions: Set[str]         = set()

    @classmethod
    def with_builtins(cls, *, memory: Any = None) -> "SkillRegistry":
        """Create a registry pre-loaded with built-in skills."""
        from mark.middlewares.skills.builtin import EchoSkill, MarkMemorySkill
        registry = cls()
        registry.register(EchoSkill())
        if memory is not None:
            registry.register(MarkMemorySkill(memory))
        return registry

    def register(self, skill: Skill, *, overwrite: bool = False) -> None:
        """Register an entry."""
        if skill.name in self._skills and not overwrite:
            raise ValueError(
                f"Skill {skill.name!r} already registered. Pass overwrite=True to replace."
            )
        self._skills[skill.name] = skill

    def discover(self, group: str = "mark.middlewares.skills") -> List[str]:
        """Auto-discover and register skills installed as Python packages."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group=group)
        except Exception:
            return []
        loaded: List[str] = []
        for ep in eps:
            try:
                skill_cls = ep.load()
                skill     = skill_cls()
                if skill.name not in self._skills:
                    self._skills[skill.name] = skill
                    loaded.append(skill.name)
            except Exception:
                pass
        return loaded

    def load_from(self, module_path: str, class_name: str) -> Skill:
        """Import and register a skill class from a module path."""
        import importlib
        skill_cls = getattr(importlib.import_module(module_path), class_name)
        skill     = skill_cls()
        self.register(skill)
        return skill

    def from_skills_dir(self, path: "str | Path", filename: str = "SKILL.md",
                        overwrite: bool = True,
                        allow_builtin_override: bool = False) -> List[str]:
        """Load skills from a directory of skill files.

        ``overwrite`` lets a SKILL.md replace another skill previously loaded
        from a skills directory. Built-in / packaged / Python skills are never
        shadowed unless ``allow_builtin_override=True`` is passed explicitly.
        """
        skills_path = Path(path)
        if not skills_path.is_dir():
            return []
        loaded: List[str] = []
        for entry in skills_path.iterdir():
            skill_file = entry / filename
            if not (entry.is_dir() and skill_file.exists()):
                continue
            try:
                skill = MarkdownSkill.from_file(skill_file)
            except Exception as exc:
                logger.warning("skipping skill file %s: %s: %s", skill_file, type(exc).__name__, exc)
                continue
            existing = self._skills.get(skill.name)
            if existing is not None and not allow_builtin_override and self._is_protected(existing):
                logger.warning(
                    "skills dir %s tried to shadow protected skill %r; ignored "
                    "(pass allow_builtin_override=True to permit)", skills_path, skill.name,
                )
                continue
            try:
                self.register(skill, overwrite=overwrite)
            except ValueError as exc:
                logger.warning("skipping skill %r: %s", skill.name, exc)
                continue
            loaded.append(skill.name)
        return loaded

    @staticmethod
    def _is_protected(skill: Skill) -> bool:
        """Built-in, packaged, and Python skills; only dir-loaded SKILL.md skills are replaceable."""
        return not (isinstance(skill, MarkdownSkill) and skill._source_path is not None)

    def grant(self, *permissions: str) -> None:
        """Grant the permission."""
        self._granted_permissions.update(permissions)

    def revoke(self, *permissions: str) -> None:
        """Revoke the permission."""
        self._granted_permissions.difference_update(permissions)

    def get(self, name: str) -> Skill:
        """Return a registered skill by name, or raise KeyError."""
        if name not in self._skills:
            available = sorted(self._skills)
            raise KeyError(
                f"Unknown MARK skill: {name!r}. "
                f"Registered: {available}. Run registry.discover() to load packages."
            )
        return self._skills[name]

    def find(self, name: str) -> Optional[Skill]:
        """Find an entry by name, or None."""
        return self._skills.get(name)

    def names(self) -> List[str]:
        """Return the registered names."""
        return sorted(self._skills)

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return manifests for all registered skills."""
        results = []
        for s in self._skills.values():
            try:
                results.append(s.manifest().to_dict())
            except Exception:
                results.append({"name": s.name, "description": s.description})
        return results

    async def run_skill(self, name: str, input: SkillInput, context: SkillContext) -> SkillResult:
        """Run a registered skill by name."""
        try:
            skill = self.get(name)
        except KeyError as e:
            return SkillResult(content="", success=False, error=str(e))
        if "*" not in self._granted_permissions:
            missing = skill.permissions - self._granted_permissions
            if missing:
                return SkillResult(
                    content = "",
                    success = False,
                    error   = (
                        f"Skill {name!r} requires permissions: {missing}. "
                        f"Call registry.grant({', '.join(repr(p) for p in missing)}) first."
                    ),
                )
        return await skill.run(input, context)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
