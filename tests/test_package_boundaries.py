"""Package boundary regression tests for consolidated SDK modules."""

from __future__ import annotations

import importlib.util


def test_json_store_lives_under_store_package():
    from mark.store import JsonMemoryStore

    assert JsonMemoryStore.__module__ == "mark.store.json_store"
    assert importlib.util.find_spec("mark.storage") is None


def test_deep_rl_is_consolidated_into_rl_package():
    from mark.rl import RLAction, SafetyConstraintLayer

    assert RLAction.__module__ == "mark.rl.safety"
    assert SafetyConstraintLayer.__module__ == "mark.rl.safety"
    assert importlib.util.find_spec("mark.deep_rl") is None
