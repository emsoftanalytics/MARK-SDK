"""Skills middleware battery and packaged agent skills."""

from mark.middlewares.skills.base import (
    MarkdownSkill,
    Skill,
    SkillContext,
    SkillInput,
    SkillManifest,
    SkillRegistry,
    SkillResult,
)
from mark.middlewares.skills.builtin import EchoSkill, MarkMemorySkill
from mark.middlewares.skills.loader import agent_skill, list_packaged_skills, load_skill_text
from mark.middlewares.skills.middleware import SkillMiddleware
from mark.middlewares.skills.persona import AgentPersona

__all__ = [
    "AgentPersona",
    "EchoSkill",
    "MarkdownSkill",
    "MarkMemorySkill",
    "Skill",
    "SkillContext",
    "SkillInput",
    "SkillManifest",
    "SkillMiddleware",
    "SkillRegistry",
    "SkillResult",
    "agent_skill",
    "list_packaged_skills",
    "load_skill_text",
]
