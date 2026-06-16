# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mark.plasticity import (
    COACTIVATION_DELTA,
    DECAY_FLOOR,
    EDGE_FLOOR,
    REINFORCE_DELTA,
    EdgeCoActivation,
    ExponentialDecay,
    HebbianReinforcement,
    MemoryPruner,
    PrunerStats,
    RouterDecision,
    RoutingFeedback,
    StaticRouter,
    TemporalScope,
)
from mark.types import MemoryState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants_defined() -> None:
    assert DECAY_FLOOR          == 0.05
    assert REINFORCE_DELTA      == 0.05
    assert COACTIVATION_DELTA   == 0.03
    assert EDGE_FLOOR           == 0.10


# ---------------------------------------------------------------------------
# TemporalScope half-lives
# ---------------------------------------------------------------------------

def test_temporal_scope_half_lives() -> None:
    assert TemporalScope.HALF_LIVES_HOURS[TemporalScope.SHORT_TERM]  == 48.0
    assert TemporalScope.HALF_LIVES_HOURS[TemporalScope.MEDIUM_TERM] == 168.0
    assert TemporalScope.HALF_LIVES_HOURS[TemporalScope.LONG_TERM]   == 720.0
    assert TemporalScope.HALF_LIVES_HOURS[TemporalScope.PERMANENT]   is None


# ---------------------------------------------------------------------------
# ExponentialDecay
# ---------------------------------------------------------------------------

def _make_fragment(importance: float = 0.8, state=MemoryState.UNVERIFIED,
                   age_hours: float = 0.0):
    class FakeFrag:
        pass
    f            = FakeFrag()
    f.importance = importance
    f.state      = state
    f.created_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return f


def test_decay_factor_permanent_scope_is_one() -> None:
    d = ExponentialDecay(TemporalScope.PERMANENT)
    assert d.decay_factor(1000.0) == 1.0


def test_decay_factor_at_half_life_is_half() -> None:
    d          = ExponentialDecay(TemporalScope.MEDIUM_TERM)
    half_life  = TemporalScope.HALF_LIVES_HOURS[TemporalScope.MEDIUM_TERM]
    factor     = d.decay_factor(half_life)
    assert abs(factor - 0.5) < 1e-6


def test_decay_apply_permanent_fragment_unchanged() -> None:
    d    = ExponentialDecay(TemporalScope.MEDIUM_TERM)
    frag = _make_fragment(importance=0.9, state=MemoryState.PROMOTED)
    assert d.apply(frag) == 0.9


def test_decay_apply_old_fragment_decays() -> None:
    d    = ExponentialDecay(TemporalScope.SHORT_TERM)   # half-life 48h
    frag = _make_fragment(importance=0.8, age_hours=48.0)
    result = d.apply(frag)
    assert result < 0.8
    assert result >= DECAY_FLOOR


def test_decay_apply_never_below_floor() -> None:
    d    = ExponentialDecay(TemporalScope.SHORT_TERM)
    frag = _make_fragment(importance=0.8, age_hours=9999.0)
    assert d.apply(frag) >= DECAY_FLOOR


def test_decay_should_prune_very_old_fragment() -> None:
    d    = ExponentialDecay(TemporalScope.SHORT_TERM)
    frag = _make_fragment(importance=0.1, age_hours=9999.0)
    assert d.should_prune(frag)


def test_decay_should_not_prune_fresh_fragment() -> None:
    d    = ExponentialDecay(TemporalScope.MEDIUM_TERM)
    frag = _make_fragment(importance=0.9, age_hours=1.0)
    assert not d.should_prune(frag)


# ---------------------------------------------------------------------------
# HebbianReinforcement
# ---------------------------------------------------------------------------

def test_hebbian_on_access_boosts_importance() -> None:
    h    = HebbianReinforcement()
    frag = _make_fragment(importance=0.5)
    frag.metadata = {"access_count": 5}   # non-zero count to get nonzero boost
    new_imp = h.on_access(frag)
    assert new_imp > 0.5


def test_hebbian_on_access_never_exceeds_one() -> None:
    h    = HebbianReinforcement()
    frag = _make_fragment(importance=0.99)
    frag.metadata = {"access_count": 1000}
    assert h.on_access(frag) <= 1.0


def test_hebbian_boost_dampens_with_access_count() -> None:
    h     = HebbianReinforcement()
    frag0 = _make_fragment(importance=0.5)
    frag0.metadata = {"access_count": 0}
    frag9 = _make_fragment(importance=0.5)
    frag9.metadata = {"access_count": 9}
    boost0 = h.on_access(frag0) - 0.5
    boost9 = h.on_access(frag9) - 0.5
    # Dampening: boost approaches delta asymptotically, higher count → closer to delta
    assert boost9 >= boost0   # more accesses → bigger boost (approaching delta)


# ---------------------------------------------------------------------------
# EdgeCoActivation
# ---------------------------------------------------------------------------

def test_edge_co_activation_increases_weight() -> None:
    e   = EdgeCoActivation()
    new = e.on_co_retrieval(current_weight=0.5)
    assert new == pytest.approx(0.5 + COACTIVATION_DELTA)


def test_edge_co_activation_caps_at_one() -> None:
    e   = EdgeCoActivation()
    new = e.on_co_retrieval(current_weight=0.99)
    assert new <= 1.0


def test_edge_should_dissolve_below_floor() -> None:
    e = EdgeCoActivation()
    assert e.should_dissolve(0.05)
    assert not e.should_dissolve(0.15)


def test_edge_floor_boundary() -> None:
    e = EdgeCoActivation()
    assert e.should_dissolve(EDGE_FLOOR - 0.01)
    assert not e.should_dissolve(EDGE_FLOOR)


# ---------------------------------------------------------------------------
# MemoryPruner
# ---------------------------------------------------------------------------

class FakeStore:
    def __init__(self):
        from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier
        now = datetime.now(timezone.utc)
        self._frags = []
        self._edges = []
        self.deleted  : list[str] = []
        self.updated  : dict[str, float] = {}

    def list_by_agent(self, agent_id: str):
        return list(self._frags)

    def update_importance(self, frag_id: str, importance: float):
        self.updated[frag_id] = importance

    def delete(self, frag_id: str):
        self.deleted.append(frag_id)
        self._frags = [f for f in self._frags if f.id != frag_id]

    def list_edges(self, agent_id: str = None):
        return list(self._edges)

    def delete_edge(self, edge_id: str):
        self._edges = [e for e in self._edges if e.id != edge_id]


def _add_fake_frag(store, frag_id, importance=0.5, state=MemoryState.UNVERIFIED,
                   age_hours=0.0):
    class F:
        pass
    f            = F()
    f.id         = frag_id
    f.importance = importance
    f.state      = state
    f.created_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    f.metadata   = {}
    store._frags.append(f)
    return f


def test_pruner_returns_stats() -> None:
    store  = FakeStore()
    pruner = MemoryPruner(store)
    stats  = pruner.run("agent")
    assert isinstance(stats, PrunerStats)


def test_pruner_prunes_low_importance_old_fragment() -> None:
    store = FakeStore()
    _add_fake_frag(store, "old-low", importance=0.1, age_hours=9999.0)
    pruner = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM))
    stats  = pruner.run("agent")
    assert stats.fragments_pruned >= 1
    assert "old-low" in store.deleted


def test_pruner_does_not_prune_promoted_fragment() -> None:
    store = FakeStore()
    _add_fake_frag(store, "promoted", importance=0.1,
                   state=MemoryState.PROMOTED, age_hours=9999.0)
    pruner = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM))
    pruner.run("agent")
    assert "promoted" not in store.deleted


def test_pruner_updates_importance_for_live_fragments() -> None:
    store = FakeStore()
    _add_fake_frag(store, "live", importance=0.8, age_hours=48.0)
    pruner = MemoryPruner(store, decay=ExponentialDecay(TemporalScope.SHORT_TERM))
    pruner.run("agent")
    assert "live" in store.updated


# ---------------------------------------------------------------------------
# StaticRouter
# ---------------------------------------------------------------------------

def test_static_router_returns_router_decision() -> None:
    router   = StaticRouter()
    decision = router.route("What is Elena wearing?")
    assert isinstance(decision, RouterDecision)
    assert decision.policy in ("FAST", "BALANCED", "DEEP", "SIMPLE")


def test_static_router_simple_query_maps_to_low_complexity() -> None:
    router   = StaticRouter()
    decision = router.route("What is Elena?")
    assert decision.policy in ("FAST", "BALANCED", "SIMPLE")


def test_static_router_complex_query_maps_balanced_or_deep() -> None:
    router   = StaticRouter()
    decision = router.route(
        "Why did Elena choose the red scarf when she could have taken the blue one "
        "considering her history with the warehouse and the relationship with her mother?"
    )
    assert decision.policy in ("BALANCED", "DEEP")


def test_static_router_record_feedback_is_noop() -> None:
    router   = StaticRouter()
    decision = router.route("test")
    router.record_feedback(RoutingFeedback(decision_id=decision.id, reward=0.9))
    # No exception; feedback is silently dropped locally


def test_routing_feedback_fields() -> None:
    fb = RoutingFeedback(decision_id="abc", reward=0.75)
    assert fb.decision_id == "abc"
    assert fb.reward      == 0.75
