# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""The compact small-model skill: short, frontmatter-free in prompts, counted in budgets."""
from __future__ import annotations

from mark.context.budget import estimate_tokens
from mark.middlewares.skills.loader import (
    COMPACT_SKILLS,
    PACKAGED_SKILLS,
    list_packaged_skills,
    load_skill_text,
    prompt_text,
)


def test_compact_skill_is_short_and_explains_the_briefing_headers() -> None:
    raw = load_skill_text("mark-memory-compact")
    assert raw.startswith("---") and "name: mark-memory-compact" in raw
    body = prompt_text(raw)
    assert 120 <= estimate_tokens(body) <= 250
    for header in ("Known facts (current):", "Corrections:", "Related notes:", "Not in memory:"):
        assert header in body
    # The long skill is unchanged for large models and much bigger.
    assert estimate_tokens(prompt_text(load_skill_text("mark-memory-middleware"))) > 2 * estimate_tokens(body)


def test_prompt_text_drops_frontmatter_only() -> None:
    assert prompt_text("---\nname: x\ndescription: y\n---\n\n# Body\ntext") == "# Body\ntext"
    assert prompt_text("# No frontmatter") == "# No frontmatter"
    assert "name:" not in prompt_text(load_skill_text("mark-memory-compact"))


def test_compact_skill_is_not_added_by_default() -> None:
    assert "mark-memory-compact" in COMPACT_SKILLS and "mark-memory-compact" not in PACKAGED_SKILLS
    assert "mark-memory-compact" not in list_packaged_skills()
