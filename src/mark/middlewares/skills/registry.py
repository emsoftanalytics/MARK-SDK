"""Backward-compatible import path for the skill registry.

The canonical implementation lives in :mod:`mark.middlewares.skills.base`.
"""
from __future__ import annotations

from mark.middlewares.skills.base import SkillRegistry

__all__ = ["SkillRegistry"]
