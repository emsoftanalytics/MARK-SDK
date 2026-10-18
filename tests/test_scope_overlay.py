"""ScopeIds derives in O(delta): a shared base set plus small added/removed overlays.

At a million memories a recall right after a write used to copy the whole
retrievable-id set (~0.4 s under memory pressure). These tests pin the
overlay behaviour: derived sets share the base, answer exactly like the
equivalent plain set, compact occasionally, and stay immutable.
"""
from __future__ import annotations

import pickle
import random

import pytest

from mark.store._sqlite import scope_ids as scope_module
from mark.store._sqlite.scope_ids import ScopeIds


def test_derive_shares_the_base_instead_of_copying_it() -> None:
    big = ScopeIds(str(i) for i in range(50_000))
    derived = ScopeIds.derive(big, {"new"}, {"7"})
    assert derived.frozen_base() is big.frozen_base()  # no O(N) copy per write
    again = ScopeIds.derive(derived, {"newer"}, {"new"})
    assert again.frozen_base() is big.frozen_base()
    assert "newer" in again and "new" not in again and "7" not in again and "8" in again
    assert len(again) == 50_000 and again.added == {"newer"} and again.removed == {"new"}
    assert again.parent == derived.token


def test_overlays_match_a_plain_set_through_random_derive_chains(monkeypatch) -> None:
    monkeypatch.setattr(scope_module, "COMPACT_MIN", 8)  # exercise compaction too
    rnd = random.Random(5)
    reference = {f"id{i}" for i in range(300)}
    scope = ScopeIds(reference)
    universe = [f"id{i}" for i in range(400)]
    for _ in range(300):
        added = set(rnd.sample(universe, rnd.randint(0, 4)))
        removed = set(rnd.sample(universe, rnd.randint(0, 4))) - added
        new = ScopeIds.derive(scope, added, removed)
        assert new.added == frozenset(added - reference)
        assert new.removed == frozenset(removed & reference)
        reference = (reference - removed) | added
        scope = new
        assert scope == reference and len(scope) == len(reference)
        assert set(scope) == reference and sorted(scope) == sorted(reference)
        probe = rnd.choice(universe)
        assert (probe in scope) == (probe in reference)
    assert scope.difference({"id1", "id2"}) == reference - {"id1", "id2"}
    assert scope.difference({"id1": 1}) == reference - {"id1"}  # dict operand (index partitions)
    assert scope.intersection({"id1", "nope"}) == reference & {"id1", "nope"}
    assert scope.union({"x"}) == reference | {"x"}
    assert {"id1", "nope"} & scope == reference & {"id1", "nope"}
    assert {"id1", "nope"} - scope == {"id1", "nope"} - reference
    assert scope.issubset(reference | {"x"}) and scope.issuperset(set(list(reference)[:3]))
    assert scope.frozen() == reference and isinstance(scope.frozen(), frozenset)


def test_immutable_hashable_and_picklable() -> None:
    base = ScopeIds({"a", "b"})
    for mutate in ("add", "discard", "remove", "pop", "clear", "update"):
        with pytest.raises((TypeError, AttributeError)):
            getattr(base, mutate)("c") if mutate not in ("pop", "clear") else getattr(base, mutate)()
    alias = base
    alias |= {"c"}
    assert base == {"a", "b"} and alias == {"a", "b", "c"} and isinstance(alias, frozenset)
    derived = ScopeIds.derive(base, {"c", "a"}, {"b", "z"})
    assert hash(derived) == hash(frozenset({"a", "c"}))
    assert pickle.loads(pickle.dumps(derived)) == frozenset({"a", "c"})
    assert derived != {"a"} and not (derived == {"a", "b"})


def test_equality_of_siblings_compares_overlays_not_the_base() -> None:
    base = ScopeIds(str(i) for i in range(1000))
    one = ScopeIds.derive(base, {"x"}, {"1"})
    two = ScopeIds.derive(base, {"x"}, {"1"})
    assert one == two and one != base
