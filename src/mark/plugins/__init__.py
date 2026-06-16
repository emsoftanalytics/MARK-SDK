# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from mark.plugins.registry import (
    # Core retrieval hooks
    HOOK_SCORING,
    HOOK_POLICY_ENGINE,
    HOOK_CRITIC,
    HOOK_DECOMPOSER,
    HOOK_SELF_HEALING,
    HOOK_CONTRADICTION_RESOLVER,
    HOOK_REASONING_CHAIN,
    HOOK_PROOF_ANCHOR,
    # Plasticity and learning hooks
    HOOK_PLASTICITY,
    HOOK_CONSOLIDATION,
    HOOK_LEARNING_ROUTER,
    HOOK_RL_POLICY,
    HOOK_FEEDBACK,
    # Governance and sync hooks
    HOOK_GOVERNANCE,
    HOOK_SYNC,
    HOOK_EMBEDDING,
    # Media / long-running agent hooks
    HOOK_NARRATOR,
    HOOK_CONTINUITY,
    HOOK_WORLD_STATE,
    # Observability hooks
    HOOK_OBSERVE_EVENT,
    # Plugin infrastructure
    MarkCorePlugin,
    PluginRegistry,
)

__all__ = [
    "HOOK_CONSOLIDATION",
    "HOOK_CONTINUITY",
    "HOOK_CONTRADICTION_RESOLVER",
    "HOOK_CRITIC",
    "HOOK_DECOMPOSER",
    "HOOK_EMBEDDING",
    "HOOK_FEEDBACK",
    "HOOK_GOVERNANCE",
    "HOOK_LEARNING_ROUTER",
    "HOOK_NARRATOR",
    "HOOK_OBSERVE_EVENT",
    "HOOK_PLASTICITY",
    "HOOK_POLICY_ENGINE",
    "HOOK_PROOF_ANCHOR",
    "HOOK_REASONING_CHAIN",
    "HOOK_RL_POLICY",
    "HOOK_SCORING",
    "HOOK_SELF_HEALING",
    "HOOK_SYNC",
    "HOOK_WORLD_STATE",
    "MarkCorePlugin",
    "PluginRegistry",
]
