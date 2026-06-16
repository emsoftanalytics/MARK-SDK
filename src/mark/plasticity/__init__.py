# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# mark.plasticity — local memory plasticity for agent applications.
#
# Apache-2.0 components (ship in open SDK):
#   ExponentialDecay, TemporalScope — local decay math
#   HebbianReinforcement, EdgeCoActivation — local reinforcement math
#   MemoryPruner — local maintenance culling
#   RoutingFeedback, RouterDecision, StaticRouter — routing data structures
#
# Extension components can be registered via HOOK_PLASTICITY and
# HOOK_LEARNING_ROUTER.
from .decay   import DECAY_FLOOR, REINFORCE_DELTA, ExponentialDecay, TemporalScope
from .hebbian import COACTIVATION_DELTA, EDGE_FLOOR, EdgeCoActivation, HebbianReinforcement
from .pruner  import MemoryPruner, PrunerStats
from .router  import RouterDecision, RoutingFeedback, StaticRouter

__all__ = [
    "COACTIVATION_DELTA",
    "DECAY_FLOOR",
    "EDGE_FLOOR",
    "REINFORCE_DELTA",
    "EdgeCoActivation",
    "ExponentialDecay",
    "HebbianReinforcement",
    "MemoryPruner",
    "PrunerStats",
    "RouterDecision",
    "RoutingFeedback",
    "StaticRouter",
    "TemporalScope",
]
