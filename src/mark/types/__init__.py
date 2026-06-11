# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from .attribution import SourceAttribution, SourceCredibility
from .gap import GapReport, GapSeverity
from .llm import LLMProvider
from .memory import BlockStatus, MemoryBlock, MemoryDocument, MemoryFragment, MemoryScope, MemoryState, MemoryTier
from .graph import BlockLink, EdgeRelation, MemoryEdge, MemoryNode, NodeType, chunk_document

# Backward-compatibility alias — MemoryBlockModel was renamed to MemoryBlock
MemoryBlockModel = MemoryBlock

__all__ = [
    "BlockLink",
    "BlockStatus",
    "EdgeRelation",
    "GapReport",
    "GapSeverity",
    "LLMProvider",
    "MemoryBlock",
    "MemoryBlockModel",  # deprecated alias
    "MemoryDocument",
    "MemoryEdge",
    "MemoryFragment",
    "MemoryNode",
    "MemoryScope",
    "MemoryState",
    "MemoryTier",
    "NodeType",
    "SourceAttribution",
    "SourceCredibility",
    "chunk_document",
]
