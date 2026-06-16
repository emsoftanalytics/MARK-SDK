# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import hashlib

import pytest

from mark.rl import (
    ExperienceTuple,
    GovernanceSignal,
    GovernanceSignalCollector,
    RLAction,
    ReplayBuffer,
    RewardComputer,
    RewardSignal,
    RoutingDecisionLogger,
    SafetyConstraintLayer,
    StateEncoder,
)


# ---------------------------------------------------------------------------
# ExperienceTuple
# ---------------------------------------------------------------------------

def _make_tuple(action: str = "BALANCED", reward: float = 0.8) -> ExperienceTuple:
    state = [0.1, 0.2, 0.0] + [0.0] * 61
    return ExperienceTuple(
        state=state, action=action, reward=reward,
        next_state=state, done=False, agent_id="agent",
    )


def test_experience_tuple_valid() -> None:
    assert _make_tuple().is_valid()


def test_safety_constraint_layer_lives_in_rl_package() -> None:
    layer = SafetyConstraintLayer()
    action = RLAction(
        consolidation_decision="APPROVE",
        gov_override_attempt=True,
        edge_weight_delta=2.0,
        hook_mask={"validation_gate": False},
    )

    filtered = layer.filter(action)

    assert filtered.consolidation_decision == "QUARANTINE"
    assert filtered.edge_weight_delta == 1.0
    assert filtered.hook_mask["validation_gate"] is True


def test_experience_tuple_invalid_action() -> None:
    assert not _make_tuple(action="UNKNOWN").is_valid()


def test_experience_tuple_reward_out_of_range_invalid() -> None:
    assert not _make_tuple(reward=1.5).is_valid()
    assert not _make_tuple(reward=-0.1).is_valid()


def test_experience_tuple_reward_boundary_valid() -> None:
    assert _make_tuple(reward=0.0).is_valid()
    assert _make_tuple(reward=1.0).is_valid()


def test_experience_tuple_all_actions_valid() -> None:
    for action in ("FAST", "BALANCED", "DEEP"):
        assert _make_tuple(action=action).is_valid()


def test_experience_tuple_has_timestamp() -> None:
    t = _make_tuple()
    assert t.timestamp > 0


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

def test_replay_buffer_push_and_size() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.push(_make_tuple())
    assert buf.size() == 1


def test_replay_buffer_evicts_fifo_at_capacity() -> None:
    buf = ReplayBuffer(capacity=3)
    for _ in range(5):
        buf.push(_make_tuple())
    assert buf.size() == 3


def test_replay_buffer_sample_within_size() -> None:
    buf = ReplayBuffer(capacity=10)
    for _ in range(5):
        buf.push(_make_tuple())
    sample = buf.sample(3)
    assert len(sample) == 3


def test_replay_buffer_sample_capped_at_size() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.push(_make_tuple())
    sample = buf.sample(100)
    assert len(sample) == 1


def test_replay_buffer_all_returns_all_items() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.push(_make_tuple(action="FAST"))
    buf.push(_make_tuple(action="DEEP"))
    items = buf.all()
    assert len(items) == 2


def test_replay_buffer_latest_class_reference() -> None:
    buf = ReplayBuffer(capacity=5)
    assert ReplayBuffer.load_latest() is buf


# ---------------------------------------------------------------------------
# StateEncoder
# ---------------------------------------------------------------------------

def test_state_encoder_returns_vector_of_expected_dim() -> None:
    enc = StateEncoder(state_dim=64)
    vec = enc.encode({"query_complexity": 0.7, "result_count": 8})
    assert len(vec) == 64


def test_state_encoder_returns_normalized_vector() -> None:
    enc  = StateEncoder(state_dim=16)
    vec  = enc.encode({"query_complexity": 0.5, "avg_importance": 0.8})
    norm = sum(x ** 2 for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_state_encoder_zero_features_returns_zero_vector() -> None:
    enc = StateEncoder(state_dim=16)
    vec = enc.encode({})
    assert sum(v ** 2 for v in vec) < 1e-12


def test_state_encoder_dim_32() -> None:
    enc = StateEncoder(state_dim=32)
    vec = enc.encode({"result_count": 5})
    assert len(vec) == 32


# ---------------------------------------------------------------------------
# RewardSignal + RewardComputer
# ---------------------------------------------------------------------------

def test_reward_computer_clamps_above_one() -> None:
    rc     = RewardComputer()
    signal = RewardSignal(
        decision_id="d1", agent_id="a", policy_used="BALANCED",
        result_count=5, latency_ms=50.0, evaluator_score=1.5,
    )
    assert rc.compute(signal) == 1.0


def test_reward_computer_clamps_below_zero() -> None:
    rc     = RewardComputer()
    signal = RewardSignal(
        decision_id="d2", agent_id="a", policy_used="FAST",
        result_count=0, latency_ms=10.0, evaluator_score=-0.3,
    )
    assert rc.compute(signal) == 0.0


def test_reward_computer_passthrough_mid_range() -> None:
    rc     = RewardComputer()
    signal = RewardSignal(
        decision_id="d3", agent_id="a", policy_used="DEEP",
        result_count=8, latency_ms=200.0, evaluator_score=0.85,
    )
    assert abs(rc.compute(signal) - 0.85) < 1e-9


def test_reward_signal_fields() -> None:
    s = RewardSignal(
        decision_id="d4", agent_id="a", policy_used="BALANCED",
        result_count=3, latency_ms=40.0, evaluator_score=0.7,
        hallucination_flagged=True,
    )
    assert s.hallucination_flagged is True
    assert s.governance_rejected   is False


# ---------------------------------------------------------------------------
# GovernanceSignalCollector
# ---------------------------------------------------------------------------

def test_governance_collector_hashes_content_by_default() -> None:
    coll    = GovernanceSignalCollector()
    content = "Elena is left-handed."
    coll.collect(content, label="approve", confidence=0.9)
    sig  = coll.all()[0]
    # Should be SHA-256, not raw text
    assert sig.content_hash != content
    expected = hashlib.sha256(content.encode()).hexdigest()
    assert sig.content_hash == expected


def test_governance_collector_raw_mode_stores_content() -> None:
    coll = GovernanceSignalCollector(raw=True)
    coll.collect("raw content", label="reject", confidence=0.5)
    assert coll.all()[0].content_hash == "raw content"


def test_governance_collector_count() -> None:
    coll = GovernanceSignalCollector()
    coll.collect("a", label="approve", confidence=0.9)
    coll.collect("b", label="reject",  confidence=0.2)
    assert coll.count() == 2


def test_governance_signal_fields() -> None:
    coll = GovernanceSignalCollector()
    coll.collect("text", label="quarantine", confidence=0.55,
                 context_hint="scope:agent")
    sig = coll.all()[0]
    assert sig.label        == "quarantine"
    assert sig.confidence   == 0.55
    assert sig.context_hint == "scope:agent"


# ---------------------------------------------------------------------------
# RoutingDecisionLogger
# ---------------------------------------------------------------------------

def test_routing_logger_stores_decision() -> None:
    logger = RoutingDecisionLogger()
    exp    = _make_tuple(action="DEEP", reward=0.9)
    logger.log(exp, query_hash="abc123", policy="DEEP")
    recent = logger.recent(limit=10)
    assert len(recent) == 1
    assert recent[0]["policy"] == "DEEP"


def test_routing_logger_stores_hash_not_raw_query() -> None:
    logger = RoutingDecisionLogger()
    exp    = _make_tuple()
    raw_q  = "What is Elena wearing?"
    hashed = hashlib.sha256(raw_q.encode()).hexdigest()
    logger.log(exp, query_hash=hashed, policy="BALANCED")
    assert logger.recent()[0]["query_hash"] == hashed
    assert raw_q not in str(logger.recent())


def test_routing_logger_recent_respects_limit() -> None:
    logger = RoutingDecisionLogger()
    exp    = _make_tuple()
    for _ in range(10):
        logger.log(exp, query_hash="h", policy="FAST")
    assert len(logger.recent(limit=5)) == 5
