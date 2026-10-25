# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Skill contracts: the :class:`Skill` ABC, its manifest and its result type.

Every skill (built-in, packaged, third-party, or SKILL.md-backed) implements
:class:`Skill` and returns a :class:`SkillResult`. Kept free of MARK runtime
imports so skill packages can depend on it cheaply. Re-exported from
:mod:`mark.middlewares.skills.base`, the historical import path.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Legacy type aliases kept for backward compat
SkillInput   = Dict[str, Any]
SkillContext = Dict[str, Any]


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

    Built-in Apache-2.0 skills:
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
        """Synchronous wrapper — safe to call whether or not an event loop is running.

        The running-loop check happens *before* the skill runs, so a
        RuntimeError raised by the skill itself propagates unchanged instead
        of being mistaken for "no running loop" and re-executed.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(input, context))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(input, context)).result()

    @classmethod
    def manifest(cls) -> SkillManifest:
        """Return this skill's manifest."""
        return SkillManifest(
            name        = cls.name if isinstance(cls.name, str) else "skill",
            description = cls.description if isinstance(cls.description, str) else "",
            permissions = list(cls.permissions) if isinstance(cls.permissions, set) else [],
        )
