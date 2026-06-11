# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Plugin hook constants — the cloud integration points.
#
# Each HOOK_* constant names a slot in the PluginRegistry. The local SDK
# defines the slot and its fallback; cloud plugins register implementations
# that activate transparently — no developer code changes required.
"""Plugin registry and the public HOOK_* extension slots."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

# ── Retrieval and ranking ──────────────────────────────────────────────────────

HOOK_SCORING              = "scoring"
# Cloud: advanced memory ranking plugin.
# Local: simple similarity + importance reranker.

HOOK_POLICY_ENGINE        = "policy_engine"
# Cloud: cloud routing plugin (adaptive, learned from usage).
# Local: QueryClassifier heuristic.

HOOK_CRITIC               = "critic"
# Cloud: cloud memory quality critic.
# Local: not active.

HOOK_DECOMPOSER           = "decomposer"
# Cloud: cloud query decomposition plugin.
# Local: single-pass retrieval.

HOOK_SELF_HEALING         = "self_healing"
# Cloud: cloud gap-repair plugin (automatic search + fill).
# Local: GapReport surfaced to developer; no automatic action.

HOOK_CONTRADICTION_RESOLVER = "contradiction_resolver"
# Cloud: cloud contradiction arbitration plugin.
# Local: ContradictionDetector flags conflicts for developer review.

HOOK_REASONING_CHAIN      = "reasoning_chain"
# Cloud: cloud reasoning record plugin.
# Local: not active.

HOOK_PROOF_ANCHOR         = "proof_anchor"
# Cloud: cloud proof anchoring plugin.
# Local: not active.

# ── Memory learning and plasticity ────────────────────────────────────────────

HOOK_PLASTICITY           = "plasticity"
# Cloud: cloud plasticity plugin.
# Local: ExponentialDecay + HebbianReinforcement (plasticity/).

HOOK_CONSOLIDATION        = "consolidation"
# Cloud: cloud consolidation plugin (intelligent working→LTM promotion).
# Local: ConsolidationManager importance threshold (memory/).

HOOK_LEARNING_ROUTER      = "learning_router"
# Cloud: cloud adaptive routing plugin.
# Local: StaticRouter wrapping QueryClassifier.

HOOK_RL_POLICY            = "rl_policy"
# Cloud: cloud RL memory policy plugin (server-side, trained model).
# Local: not active (falls back to HOOK_LEARNING_ROUTER or QueryClassifier).

HOOK_FEEDBACK             = "feedback"
# Cloud: cloud reward collection and training pipeline plugin.
# Local: RewardSignal schema in rl/ (local capture only, not uploaded).

# ── Governance and data ───────────────────────────────────────────────────────

HOOK_GOVERNANCE           = "governance"
# Cloud: cloud governance validation plugin.
# Local: content hash integrity check.

HOOK_SYNC                 = "sync"
# Cloud: cloud sync plugin (encrypted block sync).
# Local: not active.

HOOK_EMBEDDING            = "embedding"
# Cloud: cloud embedding plugin (hosted model with caching).
# Local: sentence-transformers / Ollama / hash embedder.

# ── Media and long-running agents ─────────────────────────────────────────────

HOOK_NARRATOR             = "narrator"
# Cloud: cloud narrative consistency plugin (for media agents).
# Local: not active (graph expansion provides context; no active validation).

HOOK_CONTINUITY           = "continuity"
# Cloud: cloud continuity guard plugin (character/object consistency).
# Local: ContradictionDetector flags conflicts post-hoc.

HOOK_WORLD_STATE          = "world_state"
# Cloud: cloud world-state management plugin (long-running creative agents).
# Local: working memory blocks managed by developer.

# ── Observability ──────────────────────────────────────────────────────────────

HOOK_OBSERVE_EVENT        = "observe_event"
# Cloud: Memory Observatory event collector.
#   Receives an ObserveEvent on every mark.observe() call and emits structured
#   telemetry to the cloud dashboard (memory timeline, graph activity,
#   session explorer, brain-firing replay).
# Local: no-op (event is constructed and fired, but no handler is registered).


# ── Plugin interface ───────────────────────────────────────────────────────────

class MarkCorePlugin(ABC):
    """MIT interface boundary for separately licensed MARK extensions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the plugin version."""
        ...

    @property
    @abstractmethod
    def provides(self) -> set[str]:
        """Return the hook names this plugin provides."""
        ...

    @abstractmethod
    def get_hook(self, hook_name: str) -> Callable[..., Any]:
        """Return the callable for a hook name, or None."""
        ...

    def on_register(self, runtime: Any) -> None:
        """Hook invoked when the plugin is registered."""
        return None


class PluginRegistry:
    """
    Runtime hook table for MARK Core, MARK Cloud, and third-party plugins.

    Local SDK defines hook constants and fallback behaviour.
    Cloud plugins register implementations transparently at startup.
    """

    def __init__(self) -> None:
        self._plugins:    dict[str, MarkCorePlugin]                  = {}
        self._hook_index: dict[str, tuple[str, Callable[..., Any]]]  = {}

    def register(self, plugin: MarkCorePlugin, runtime: Any = None) -> None:
        """Register an entry."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name!r} is already registered")
        hooks = {h: plugin.get_hook(h) for h in plugin.provides}
        for hook_name in hooks:
            if hook_name in self._hook_index:
                owner, _ = self._hook_index[hook_name]
                raise ValueError(f"Hook {hook_name!r} is already claimed by {owner!r}")
        self._plugins[plugin.name] = plugin
        for hook_name, hook in hooks.items():
            self._hook_index[hook_name] = (plugin.name, hook)
        if runtime is not None:
            plugin.on_register(runtime)

    def get(self, hook_name: str) -> Callable[..., Any] | None:
        """Look up an entry by key, or None when missing."""
        item = self._hook_index.get(hook_name)
        return None if item is None else item[1]

    def has(self, hook_name: str) -> bool:
        """Return True when the hook is registered."""
        return hook_name in self._hook_index

    def list_plugins(self) -> list[dict[str, object]]:
        """Return registered plugin names."""
        return [
            {"name": p.name, "version": p.version, "provides": sorted(p.provides)}
            for p in self._plugins.values()
        ]

    def list_hooks(self) -> list[str]:
        """Return registered hook names."""
        return sorted(self._hook_index)

    def available_cloud_hooks(self) -> list[str]:
        """Return hook names with no registered plugin (available cloud upgrade slots)."""
        all_hooks = [
            HOOK_SCORING, HOOK_POLICY_ENGINE, HOOK_CRITIC, HOOK_DECOMPOSER,
            HOOK_SELF_HEALING, HOOK_CONTRADICTION_RESOLVER, HOOK_REASONING_CHAIN,
            HOOK_PROOF_ANCHOR, HOOK_PLASTICITY, HOOK_CONSOLIDATION, HOOK_GOVERNANCE,
            HOOK_SYNC, HOOK_LEARNING_ROUTER, HOOK_RL_POLICY, HOOK_EMBEDDING,
            HOOK_FEEDBACK, HOOK_NARRATOR, HOOK_CONTINUITY, HOOK_WORLD_STATE,
            HOOK_OBSERVE_EVENT,
        ]
        return [h for h in all_hooks if h not in self._hook_index]
