# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Built-in skills bundled with the SDK."""
from __future__ import annotations

from typing import Any

from mark.middlewares.skills.base import Skill, SkillContext, SkillInput, SkillManifest, SkillResult


class EchoSkill(Skill):
    """Returns the provided input — useful for SDK smoke tests and examples."""
    name        = "echo"
    description = "Returns the provided input for SDK smoke tests and examples."
    permissions: set[str] = set()

    async def run(self, input: SkillInput, context: SkillContext) -> SkillResult:
        """Execute with the given input and context."""
        return SkillResult(
            content  = str(input.get("message", str(input))),
            metadata = {"input": input, "context": context},
        )

    @classmethod
    def manifest(cls) -> SkillManifest:
        """Return this skill's manifest."""
        return SkillManifest(name=cls.name, description=cls.description, tags=["utility", "debug"])


class MarkMemorySkill(Skill):
    """Read and write local MARK memory blocks for coding agents."""
    name        = "mark_memory"
    description = "Read and write local MARK memory for coding agents."
    permissions = {"memory:read", "memory:write"}

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    async def run(self, input: SkillInput, context: SkillContext) -> SkillResult:
        """Execute with the given input and context."""
        action = str(input.get("action", "retrieve"))

        if action == "retrieve":
            query  = str(input.get("query", ""))
            top_k  = int(input.get("top_k", 5))
            blocks = input.get("blocks")
            # recall() is the one read verb (SimpleMemory.retrieve is deprecated in 0.2.0a10).
            result = self._memory.recall(query, blocks=blocks, k=top_k)
            return SkillResult(
                content  = result.as_context(),
                metadata = {
                    "action":  "retrieve",
                    "records": [{"id": f.id, "content": f.content, "tags": list(f.tags or [])}
                                for f in result.fragments],
                    "scores":  list(result.scores),
                },
            )

        if action == "write":
            block_label = str(input.get("block", "project"))
            content     = str(input["content"])
            record = self._memory.block(block_label).write(
                content,
                importance = float(input.get("importance", 0.5)),
                confidence = float(input.get("confidence", 0.5)),
                metadata   = dict(input.get("metadata") or {}),
            )
            return SkillResult(content=content, metadata={"action": "write", "record": record.to_dict()})

        if action == "list_blocks":
            blocks = self._memory.list_blocks()
            return SkillResult(
                content  = f"{len(blocks)} memory blocks",
                metadata = {"action": "list_blocks", "blocks": [b.to_dict() for b in blocks]},
            )

        return SkillResult(content="", success=False,
                           error=f"Unsupported mark_memory action: {action!r}")

    @classmethod
    def manifest(cls) -> SkillManifest:
        """Return this skill's manifest."""
        return SkillManifest(
            name        = cls.name,
            description = cls.description,
            permissions = list(cls.permissions),
            tags        = ["memory", "storage", "retrieval"],
        )
