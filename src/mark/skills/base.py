# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Skills are the unit of capability an agent can invoke outside the LLM —
# web search, code search, planning, data retrieval, external APIs, etc.
#
# Third-party skills are distributed as Python packages and auto-discovered
# by SkillRegistry.discover() via the "mark.skills" entry-point group.
#
# Publishing a skill:
#   1. Create a Python package, e.g. mark-skill-webscraper
#   2. Implement a Skill subclass
#   3. Register the entry point in pyproject.toml:
#        [project.entry-points."mark.skills"]
#        webscraper = "mark_skill_webscraper:WebScraperSkill"
#   4. pip install mark-skill-webscraper
#   5. registry.discover() loads it automatically
#
# Built-in MIT skills: EchoSkill, MarkMemorySkill, WebSearchSkill
# MARK Core skills (MSAL): AdvancedWebSearchSkill, ConsolidationSkill
"""Skill contracts: manifests, results, base classes, and the registry."""
from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Legacy type aliases kept for backward compat
SkillInput   = Dict[str, Any]
SkillContext = Dict[str, Any]


# ── Frontmatter parser ────────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    front = text[3:end].strip()
    body  = text[end + 4:].strip()
    try:
        import yaml  # type: ignore[import]
        meta = yaml.safe_load(front) or {}
    except Exception:
        meta: Dict[str, Any] = {}
        for line in front.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip(); v = v.strip().strip("'\"")
                if v.startswith("[") and v.endswith("]"):
                    meta[k] = [i.strip().strip("'\"") for i in v[1:-1].split(",") if i.strip()]
                elif v.lower() in ("true", "false"):
                    meta[k] = v.lower() == "true"
                else:
                    meta[k] = v
    return meta, body


# ── SkillManifest ──────────────────────────────────────────────────────────────

@dataclass
class SkillManifest:
    """Metadata descriptor for a skill — equivalent to SKILL.md frontmatter."""
    name:         str
    description:  str
    version:      str       = "0.1.0"
    author:       str       = ""
    permissions:  List[str] = field(default_factory=list)
    tags:         List[str] = field(default_factory=list)
    install_hint: str       = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "name":         self.name,
            "description":  self.description,
            "version":      self.version,
            "author":       self.author,
            "permissions":  self.permissions,
            "tags":         self.tags,
            "install_hint": self.install_hint,
        }


# ── SkillResult ────────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    """
    Structured output from a skill execution.

    content      : primary text result
    attributions : source citations — REQUIRED for any web/external content
    fragments    : pre-built MemoryFragments ready to store (with attribution)
    metadata     : skill-specific extra data (query, latency, result_count, etc.)
    success      : False if the skill failed gracefully
    error        : error message when success=False
    """
    content:      str
    attributions: List[Any]      = field(default_factory=list)
    fragments:    List[Any]      = field(default_factory=list)
    metadata:     Dict[str, Any] = field(default_factory=dict)
    success:      bool           = True
    error:        Optional[str]  = None

    def has_attribution(self) -> bool:
        """Return True when source attribution is present."""
        return len(self.attributions) > 0

    def as_context(self) -> str:
        """Format for injection into an LLM prompt, with citations."""
        lines = [f"<skill_result>\n{self.content}"]
        if self.attributions:
            lines.append("\nSources:")
            for attr in self.attributions:
                cite = attr.cite() if hasattr(attr, "cite") else str(attr)
                lines.append(f"  - {cite}")
        lines.append("</skill_result>")
        return "\n".join(lines)


# ── Skill ABC ──────────────────────────────────────────────────────────────────

class Skill(ABC):
    """
    Abstract base class for all MARK skills.

    Built-in MIT skills:
        EchoSkill         — smoke test / passthrough
        MarkMemorySkill   — read/write local memory blocks
        WebSearchSkill    — DuckDuckGo + Wikipedia + basic fetch

    MARK Core skills (MSAL):
        AdvancedWebSearchSkill  — LLM-guided browser lookup + credibility scoring
        ConsolidationSkill      — LLM-backed memory consolidation

    Third-party skills:
        pip install mark-skill-<name>  →  registry.discover()
    """

    name:        str      = "skill"
    description: str      = ""
    permissions: Set[str] = set()

    @abstractmethod
    async def run(self, input: SkillInput, context: SkillContext) -> SkillResult:
        """Execute the skill with the given input and context."""
        ...

    def run_sync(self, input: SkillInput, context: SkillContext) -> SkillResult:
        """Synchronous wrapper — safe to call whether or not an event loop is running."""
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.run(input, context)).result()
        except RuntimeError:
            return asyncio.run(self.run(input, context))

    @classmethod
    def manifest(cls) -> SkillManifest:
        """Return this skill's manifest."""
        return SkillManifest(
            name        = cls.name if isinstance(cls.name, str) else "skill",
            description = cls.description if isinstance(cls.description, str) else "",
            permissions = list(cls.permissions) if isinstance(cls.permissions, set) else [],
        )


# ── MarkdownSkill ─────────────────────────────────────────────────────────────

class MarkdownSkill(Skill):
    """
    A skill loaded from a SKILL.md file — compatible with local agent skills
    directory structure. Returns the Markdown instructions as content.
    """

    def __init__(self, manifest: SkillManifest, instructions: str,
                 source_path: Optional[Path] = None) -> None:
        self.name        = manifest.name
        self.description = manifest.description
        self.permissions = set(manifest.permissions)
        self._manifest_data = manifest
        self._instructions  = instructions
        self._source_path   = source_path

    @classmethod
    def from_file(cls, path: Path) -> "MarkdownSkill":
        """Load a skill from a Python file."""
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        name = meta.get("name") or path.parent.name
        manifest = SkillManifest(
            name         = name,
            description  = str(meta.get("description", "")),
            version      = str(meta.get("version", "0.1.0")),
            author       = str(meta.get("author", "")),
            permissions  = list(meta.get("permissions") or []),
            tags         = list(meta.get("tags") or []),
            install_hint = str(meta.get("install_hint", "")),
        )
        return cls(manifest=manifest, instructions=body, source_path=path)

    async def run(self, input: SkillInput, context: SkillContext) -> SkillResult:
        """Execute the skill with the given input and context."""
        rendered = self._instructions
        for k, v in input.items():
            rendered = rendered.replace(f"{{{{{k}}}}}", str(v))
        return SkillResult(
            content  = rendered,
            metadata = {"skill_type": "markdown", "skill_name": self.name,
                        "source_path": str(self._source_path) if self._source_path else ""},
        )

    def manifest(self) -> SkillManifest:  # type: ignore[override]
        """Return this skill's manifest."""
        return self._manifest_data


# ── SkillRegistry ──────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Registry that manages, discovers, and dispatches skills.

    Permission gating:
        registry.grant("web")   — allow web skills
        registry.grant("*")     — grant all (dev/test only)

    Discovery:
        registry.discover()     — auto-load installed mark.skills packages
        registry.from_skills_dir(path)  — load SKILL.md files from a directory
    """

    def __init__(self) -> None:
        self._skills:              Dict[str, Skill] = {}
        self._granted_permissions: Set[str]         = set()

    @classmethod
    def with_builtins(cls, *, memory: Any = None) -> "SkillRegistry":
        """Create a registry pre-loaded with built-in skills."""
        from mark.skills.builtin import EchoSkill, MarkMemorySkill
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

    def discover(self, group: str = "mark.skills") -> List[str]:
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
                        overwrite: bool = True) -> List[str]:
        """Load skills from a directory of skill files."""
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
                self.register(skill, overwrite=overwrite)
                loaded.append(skill.name)
            except Exception:
                pass
        return loaded

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
