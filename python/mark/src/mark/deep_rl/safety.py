# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# SafetyConstraintLayer — governance-invariant filter for cloud policy actions.
# Ships in the open SDK so developers can inspect and audit safety guarantees.
"""Safety constraints applied to learned memory-policy actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RLAction:
    """
    Structured action returned by a cloud policy plugin (HOOK_RL_POLICY).

    Open-contract note:
    This MIT SDK type is only the public action envelope. The policy model,
    training pipeline, model registry, hosted inference, and adaptation logic
    remain MARK Cloud / MARK Core implementations outside this package.

    All fields are suggestions — SafetyConstraintLayer may modify them
    before execution to enforce governance invariants.

    policy                — retrieval policy: "FAST" | "BALANCED" | "DEEP"
    consolidation_decision — "APPROVE" | "HOLD" | "QUARANTINE"
    gov_override_attempt  — True if RL tried to bypass a governance gate
    edge_weight_delta     — suggested delta for co-activated edges [0, 1]
    hook_mask             — dict of hook names to bool (True = invoke)
    """
    policy:                 str               = "BALANCED"
    consolidation_decision: str               = "APPROVE"
    gov_override_attempt:   bool              = False
    edge_weight_delta:      float             = 0.0
    hook_mask:              Dict[str, bool]   = field(default_factory=dict)


class SafetyConstraintLayer:
    """
    Governance-invariant filter applied to every RL action before execution.

    Open-contract note:
    This MIT SDK class is an auditable safety boundary for actions produced by
    a registered cloud policy plugin. It does not implement the cloud policy,
    reinforcement learning model, online finetuner, or model-version registry.

    Guarantees:
    1. RL cannot promote fragments that failed governance gates
       (gov_override_attempt=True → consolidation demoted to QUARANTINE)
    2. Edge weight delta is clamped to [0, 1]
    3. Mandatory governance hooks (ValidationGate, FailureFilter) cannot
       be disabled by the RL hook_mask

    This class never raises — it always returns a safe RLAction.
    If the action is already safe it returns the same object unchanged
    (identity check usable in tests).

    The cloud RL policy plugin (HOOK_RL_POLICY) is constrained by this
    layer before actions are transmitted to the developer's runtime.
    """

    _MANDATORY_HOOKS = {"validation_gate", "failure_filter"}

    def filter(self, action: RLAction) -> RLAction:
        """Apply safety constraints to the action."""
        consolidated = action.consolidation_decision
        edge_delta   = action.edge_weight_delta
        hook_mask    = dict(action.hook_mask)

        # Constraint 1: RL cannot promote governance-rejected fragments
        if action.gov_override_attempt and consolidated in ("APPROVE", "PROMOTE"):
            consolidated = "QUARANTINE"

        # Constraint 2: edge weight delta clamped to [0, 1]
        edge_delta = max(0.0, min(1.0, edge_delta))

        # Constraint 3: mandatory governance hooks cannot be disabled
        for h in self._MANDATORY_HOOKS:
            if h in hook_mask:
                hook_mask[h] = True

        # Return original object if unchanged (for identity-based test assertions)
        if (consolidated == action.consolidation_decision
                and edge_delta == action.edge_weight_delta
                and hook_mask == action.hook_mask):
            return action

        return RLAction(
            policy                 = action.policy,
            consolidation_decision = consolidated,
            gov_override_attempt   = action.gov_override_attempt,
            edge_weight_delta      = edge_delta,
            hook_mask              = hook_mask,
        )
