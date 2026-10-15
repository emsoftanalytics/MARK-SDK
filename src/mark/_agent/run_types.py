# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Value objects for a MARK-wrapped agent call: AgentRun in, AgentResult out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.context import ContextBundle
from mark.middlewares.skills import SkillResult
from mark.policies import PolicyDecision


@dataclass
class AgentRun:
    """Per-run configuration for a MARK-wrapped agent call."""

    prompt: str
    blocks: list[str] | None = None
    policy: str | None = None
    skills: list[str] | None = None
    skill_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieve: bool | None = None
    store_output: bool | None = None
    top_k: int | None = None
    max_context_chars: int = 6000
    session_id: str | None = None
    output_block: str = "agent-output"


@dataclass(frozen=True)
class AgentResult:
    """Result of a wrapped agent run."""

    output: str
    context: ContextBundle
    events: list[dict[str, Any]] = field(default_factory=list)
    rendered_prompt: str = ""
    decision: PolicyDecision | None = None
    skill_results: dict[str, SkillResult] = field(default_factory=dict)
    stored_record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
