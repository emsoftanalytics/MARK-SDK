# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from mark.memory.block import MemoryBlock
from mark.memory.block_chain import BlockChain, BlockVerification, ChainVerification
from mark.memory.block_graph import BlockGraph
from mark.memory.contradiction import ContradictionDetector, ContradictionReport
from mark.memory.conversation import ConversationMemory, ConversationRecall
from mark.memory.deduplication import ClusterResult, DeduplicationConsolidator, DeduplicationResult
from mark.memory.global_bus import GlobalMemoryBus
from mark.memory.graph import GraphNeighborhood
from mark.memory.mark_memory import MarkMemory, MemoryWriteRejection
from mark.memory.observe import ObserveEvent, ObserveResult
from mark.memory.record import MemoryRecord, MemoryState
from mark.memory.retrieval import LocalRetriever, ScoredRecord
from mark.memory.runtime import MarkRuntime
from mark.memory.simple import SimpleBlock, SimpleMemory
from mark.memory.store import MemoryManager
from mark.memory.working import WorkingMemoryManager

_LAZY_EXPORTS = {
    "BusMessage": "mark.middlewares.trust_bus",
    "BusSnapshot": "mark.middlewares.trust_bus",
    "BusSubscription": "mark.middlewares.trust_bus",
    "ConsolidationManager": "mark.memory.consolidation",
    "ConsolidationResult": "mark.memory.consolidation",
    "PublisherTrust": "mark.middlewares.trust_bus",
    "TrustAwareGlobalMemoryBus": "mark.middlewares.trust_bus",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'mark.memory' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "BlockChain",
    "BlockGraph",
    "BlockVerification",
    "ChainVerification",
    "BusMessage",
    "ClusterResult",
    "BusSnapshot",
    "BusSubscription",
    "ConsolidationManager",
    "ConsolidationResult",
    "ContradictionDetector",
    "ContradictionReport",
    "ConversationMemory",
    "ConversationRecall",
    "DeduplicationConsolidator",
    "DeduplicationResult",
    "GlobalMemoryBus",
    "PublisherTrust",
    "TrustAwareGlobalMemoryBus",
    "GraphNeighborhood",
    "LocalRetriever",
    "MarkMemory",
    "MarkRuntime",
    "MemoryBlock",
    "MemoryManager",
    "MemoryRecord",
    "MemoryState",
    "MemoryWriteRejection",
    "ObserveEvent",
    "ObserveResult",
    "ScoredRecord",
    "SimpleBlock",
    "SimpleMemory",
    "WorkingMemoryManager",
]
