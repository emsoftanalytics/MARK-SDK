"""``ScopeIds``: the retrievable-id set handed from the store to the retrieval pipeline.

Two properties make recall cheap at a million memories:

- equality short-circuits on identity: the pipeline checks
  ``indexed_ids == allowed_ids`` on every recall, and a real comparison of
  two million-element sets costs ~80 ms even when they are the same object;
- lineage: a set derived from a previous one by a small change records
  ``parent`` (the previous set's ``token``) and the ``added`` / ``removed``
  ids, so a consumer that kept state for the parent (the vector-index sync)
  can apply just the difference instead of comparing whole sets.

It subclasses ``set`` rather than ``frozenset`` only so that a derived set
costs one C-level copy plus O(delta) updates (a ``frozenset`` cannot be
updated after construction, which would cost a second full copy, ~60 ms at
1M ids). Once built it is treated as immutable: the mutating methods raise,
and the augmented operators return a new plain ``frozenset`` (as they would
for a ``frozenset``), so a cached scope can never be changed by a caller.
"""
from __future__ import annotations

import itertools
from typing import Iterable

_tokens = itertools.count(1)


def _immutable(*_args: object, **_kwargs: object) -> None:
    raise TypeError("ScopeIds is immutable; copy it with set(...) first")


class ScopeIds(set):
    """Immutable id set with O(1) identity equality and optional delta lineage."""

    __slots__ = ("token", "parent", "added", "removed")

    def __init__(self, ids: Iterable[str] = ()) -> None:
        set.__init__(self, ids)
        self.token: int = next(_tokens)
        self.parent: int | None = None  # token of the set this one was derived from
        self.added: frozenset[str] = frozenset()
        self.removed: frozenset[str] = frozenset()

    @classmethod
    def derive(cls, base: "ScopeIds", added: Iterable[str], removed: Iterable[str]) -> "ScopeIds":
        """A copy of *base* plus *added* minus *removed*, remembering the delta.

        ``added`` / ``removed`` are normalised to the ids that really changed,
        so consumers can trust them as an exact difference from *base*.
        """
        added = frozenset(added).difference(base)
        removed = frozenset(removed).intersection(base)
        new = cls(base)  # one C-level copy (no rehashing: str hashes are cached)
        set.difference_update(new, removed)
        set.update(new, added)
        new.parent = getattr(base, "token", None)
        new.added, new.removed = added, removed
        return new

    def __eq__(self, other: object) -> bool:
        return self is other or set.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:  # type: ignore[override]
        # Hashable like the frozenset it replaces (equal sets hash equal).
        return hash(frozenset(self))

    def __reduce__(self):  # pickles as a plain frozenset; lineage is process-local
        return (frozenset, (frozenset(self),))

    add = discard = remove = pop = clear = update = _immutable  # type: ignore[assignment]
    difference_update = intersection_update = symmetric_difference_update = _immutable  # type: ignore[assignment]

    def __ior__(self, other):  # type: ignore[override]
        return frozenset(self) | other

    def __iand__(self, other):  # type: ignore[override]
        return frozenset(self) & other

    def __isub__(self, other):  # type: ignore[override]
        return frozenset(self) - other

    def __ixor__(self, other):  # type: ignore[override]
        return frozenset(self) ^ other
