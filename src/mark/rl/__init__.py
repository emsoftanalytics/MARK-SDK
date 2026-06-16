# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.4 RL data pipeline
#
# Apache-2.0 (open SDK): open-contract data structures for collecting training signals.
# Developers can inspect exactly what is being recorded about their agents.
#
# Open-contract note:
#   This package does not include training pipelines, managed datasets, model
#   promotion, adaptive policy inference, or proprietary reward calibration.
#
# HOOK_FEEDBACK lets callers register their own signal handling pipeline.
from .buffer     import ExperienceTuple, ReplayBuffer
from .collectors import GovernanceSignal, GovernanceSignalCollector, RoutingDecisionLogger
from .encoder    import StateEncoder
from .reward     import RewardComputer, RewardSignal
from .safety     import RLAction, SafetyConstraintLayer

__all__ = [
    "ExperienceTuple",
    "GovernanceSignal",
    "GovernanceSignalCollector",
    "RLAction",
    "ReplayBuffer",
    "RewardComputer",
    "RewardSignal",
    "RoutingDecisionLogger",
    "SafetyConstraintLayer",
    "StateEncoder",
]
