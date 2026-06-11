# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors — v1.4 RL data pipeline
#
# MIT (open SDK): open-contract data structures for collecting training signals.
# Developers can inspect exactly what is being recorded about their agents.
#
# Open-contract note:
#   This package does not include MARK Cloud training, hosted datasets, model
#   promotion, adaptive policy inference, or proprietary reward calibration.
#
# Cloud: HOOK_FEEDBACK uploads collected signals to MARK Cloud's training
# pipeline (encrypted, tenanted, with consent and audit hooks).
# The full multi-factor reward formula and training pipeline are MSAL/BSL.
from .buffer     import ExperienceTuple, ReplayBuffer
from .collectors import GovernanceSignal, GovernanceSignalCollector, RoutingDecisionLogger
from .encoder    import StateEncoder
from .reward     import RewardComputer, RewardSignal

__all__ = [
    "ExperienceTuple",
    "GovernanceSignal",
    "GovernanceSignalCollector",
    "ReplayBuffer",
    "RewardComputer",
    "RewardSignal",
    "RoutingDecisionLogger",
    "StateEncoder",
]
