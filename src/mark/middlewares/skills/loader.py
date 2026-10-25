# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Loaders for the agent skills packaged with mark-sdk.

MARK ships portable Agent Skills as ``SKILL.md`` files inside the package
(e.g. ``mark/middlewares/skills/mark-usage/SKILL.md``). These teach an agent *how* to use
MARK's tools and middleware. Append one to any agent's system prompt:

    from mark.middlewares.skills import agent_skill

    system_prompt = BASE_PROMPT + agent_skill("mark-usage")

Packaged skills:

    mark-usage              how to use mark_retrieve / mark_observe /
                            mark_write tools effectively
    mark-memory-middleware  how to work with MARK middleware-injected context
    mark-memory-compact     ~170-token variant of mark-memory-middleware for
                            small models, opt-in only (it lowered llama3.2:1b
                            accuracy in the a10 measurements); not in
                            PACKAGED_SKILLS, so SkillMiddleware does not add
                            it next to the long one
"""

from __future__ import annotations

from importlib import resources

PACKAGED_SKILLS = ("mark-usage", "mark-memory-middleware")
#: Size variants, loaded by name only (never added to prompts by default).
COMPACT_SKILLS = ("mark-memory-compact",)


def load_skill_text(name: str = "mark-usage") -> str:
    """Return the raw SKILL.md text for a packaged skill, or "" when missing.

    Works both from a source checkout and from an installed wheel.
    """
    try:
        path = resources.files("mark.middlewares.skills") / name / "SKILL.md"
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def agent_skill(name: str = "mark-usage") -> str:
    """Return a packaged skill formatted for appending to a system prompt.

    The text is prefixed with two newlines so callers can simply concatenate::

        create_agent(model, tools, system_prompt=PROMPT + agent_skill())

    Returns "" when the skill is not found, so concatenation is always safe.
    """
    text = load_skill_text(name)
    return f"\n\n{text}" if text else ""


def prompt_text(skill_text: str) -> str:
    """*skill_text* without its YAML frontmatter, for a model's system prompt.

    The frontmatter is metadata for skill loaders. Small models read its
    ``name: mark-memory-middleware`` line as a tool and "call" it (usage
    test M7, llama3.2:1b), so hosts that inject a skill for a model use this.
    """
    text = (skill_text or "").strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text


def list_packaged_skills() -> list[str]:
    """Return the names of skills bundled with this installation."""
    available = []
    for name in PACKAGED_SKILLS:
        if load_skill_text(name):
            available.append(name)
    return available
