# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""SKILL.md support: frontmatter parsing and the :class:`MarkdownSkill` adapter.

A SKILL.md file is YAML-ish frontmatter (name, description, permissions, ...)
followed by Markdown instructions. :class:`MarkdownSkill` exposes it as a
regular :class:`~mark.middlewares.skills.contracts.Skill` whose result is the
instructions with ``{{placeholders}}`` filled from the input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mark.middlewares.skills.contracts import Skill, SkillContext, SkillInput, SkillManifest, SkillResult


# ── Frontmatter parser ────────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split SKILL.md *text* into ``(frontmatter, body)``.

    Uses PyYAML when installed; otherwise a minimal ``key: value`` parser that
    understands ``[a, b]`` lists and ``true``/``false``. Text without a
    frontmatter block is returned unchanged as the body.
    """
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
        """Load a skill from a SKILL.md file (name defaults to its directory)."""
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

    @classmethod
    def from_text(cls, name: str, text: str) -> "MarkdownSkill":
        """Build a skill from raw SKILL.md text (e.g. a packaged skill)."""
        meta, body = _parse_frontmatter(text)
        manifest = SkillManifest(
            name         = str(meta.get("name") or name),
            description  = str(meta.get("description", "")),
            version      = str(meta.get("version", "0.1.0")),
            author       = str(meta.get("author", "")),
            permissions  = list(meta.get("permissions") or []),
            tags         = list(meta.get("tags") or []),
            install_hint = str(meta.get("install_hint", "")),
        )
        return cls(manifest=manifest, instructions=body)

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
