# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# mark.deep_rl — MIT contract/safety layer for the cloud RL policy system.
#
# Open-contract note:
#   This package intentionally ships only auditable request/response shapes and
#   safety filters. It does not ship a policy network, training loop, model
#   registry, PPO/finetuning code, or any proprietary cloud inference logic.
#
# MIT (open SDK):
#   RLAction              — structured action type for cloud policy plugins
#   SafetyConstraintLayer — governance-invariant filter; auditable by developers
#
# Cloud-only (BSL) — registered via HOOK_RL_POLICY:
#   Cloud RL memory policy plugin (server-side only).
#   Cloud model training and versioning infrastructure.
from .safety import RLAction, SafetyConstraintLayer

__all__ = ["RLAction", "SafetyConstraintLayer"]
