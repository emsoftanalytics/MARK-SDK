"""``ScopeIds``: the retrievable-id set handed from the store to the retrieval pipeline.

Three properties make recall cheap at a million memories:

- equality short-circuits on identity: the pipeline checks
  ``indexed_ids == allowed_ids`` on every recall, and a real comparison of
  two million-element sets costs ~80 ms even when they are the same object;
- lineage: a set derived from a previous one by a small change records
  ``parent`` (the previous set's ``token``) and the ``added`` / ``removed``
  ids, so a consumer that kept state for the parent (the vector-index sync)
  can apply just the difference instead of comparing whole sets;
- O(delta) derivation: a derived set shares its parent's *base* frozenset
  and only carries two small overlays (ids added to / removed from the
  base). Copying a million-id set on every write cost ~0.4 s under memory
  pressure; the overlays are folded into a new base only once they grow past
  ``max(COMPACT_MIN, len / 64)`` ids, so that O(N) step is amortised over
  thousands of writes.

It is an immutable :class:`collections.abc.Set` (not a ``set`` subclass: a
C-level set operation on a subclass would read the base's hash table and
silently ignore the overlays). Set operators return plain ``frozenset``
objects, and the named ``set`` methods the codebase uses (``difference``,
``intersection``, ``union``, ``issubset``, ...) are provided, with fast
paths that work on the base with C-level set operations.
"""
from __future__ import annotations

import itertools
from collections.abc import Set as AbstractSet
from typing import Any, Iterable, Iterator

_tokens = itertools.count(1)
# Overlays larger than max(COMPACT_MIN, len // 64) are folded into a new base.
COMPACT_MIN = 4096
_EMPTY: frozenset[str] = frozenset()


def _immutable(*_args: object, **_kwargs: object) -> None:
    raise TypeError("ScopeIds is immutable; copy it with set(...) first")


class ScopeIds(AbstractSet):
    """Immutable id set: shared base + small overlays, identity equality, delta lineage."""

    __slots__ = ("_base", "_plus", "_minus", "_len", "_frozen", "token", "parent", "added", "removed")

    def __init__(self, ids: Iterable[str] = ()) -> None:
        if isinstance(ids, ScopeIds):
            base = ids.frozen()
        else:
            base = ids if type(ids) is frozenset else frozenset(ids)
        self._init(base, _EMPTY, _EMPTY)
        self.parent: int | None = None  # token of the set this one was derived from
        self.added: frozenset[str] = _EMPTY
        self.removed: frozenset[str] = _EMPTY

    def _init(self, base: frozenset[str], plus: frozenset[str], minus: frozenset[str]) -> None:
        # Invariants: plus ∩ base = ∅ and minus ⊆ base, so the size is exact.
        self._base, self._plus, self._minus = base, plus, minus
        self._len = len(base) + len(plus) - len(minus)
        self._frozen: frozenset[str] | None = base if not plus and not minus else None
        self.token: int = next(_tokens)

    @classmethod
    def derive(cls, base: "ScopeIds | Iterable[str]", added: Iterable[str], removed: Iterable[str]) -> "ScopeIds":
        """*base* plus *added* minus *removed*, remembering the exact delta; O(delta + overlays).

        ``added`` / ``removed`` are normalised to the ids that really changed,
        so consumers can trust them as an exact difference from *base*.
        """
        parent = base if isinstance(base, ScopeIds) else cls(base)
        added = frozenset(itertools.filterfalse(parent.__contains__, added))
        removed = frozenset(filter(parent.__contains__, removed))
        core = parent._base
        # An id added back that the base already holds only leaves the minus overlay.
        plus = parent._plus.difference(removed).union(added.difference(core))
        minus = parent._minus.difference(added).union(removed.intersection(core))
        new = cls.__new__(cls)
        if len(plus) + len(minus) > max(COMPACT_MIN, parent._len >> 6):
            # Rare O(N) step: fold the overlays into a fresh base.
            new._init(core.difference(minus).union(plus), _EMPTY, _EMPTY)
        else:
            new._init(core, plus, minus)
        new.parent = parent.token
        new.added, new.removed = added, removed
        return new

    # ── Set protocol ──────────────────────────────────────────────────────────

    def __contains__(self, item: object) -> bool:
        if item in self._plus:
            return True
        return item in self._base and item not in self._minus

    def __iter__(self) -> Iterator[str]:
        if self._frozen is not None:
            return iter(self._frozen)
        # C-level iterators: no Python frame per id.
        return itertools.chain(itertools.filterfalse(self._minus.__contains__, self._base), self._plus)

    def __len__(self) -> int:
        return self._len

    def __bool__(self) -> bool:
        return self._len > 0

    @classmethod
    def _from_iterable(cls, it: Iterable[str]) -> frozenset[str]:
        return frozenset(it)  # operators (|, &, -, ^) return plain frozensets

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, ScopeIds):
            if other._len != self._len:
                return False
            if other._base is self._base:  # siblings: compare the overlays only
                return other._plus == self._plus and other._minus == self._minus
            return self.frozen() == other.frozen()
        if isinstance(other, (set, frozenset)):
            return len(other) == self._len and all(map(other.__contains__, self))
        if isinstance(other, AbstractSet):
            return len(other) == self._len and all(map(self.__contains__, other))
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        # Hashable like the frozenset it replaces (equal sets hash equal).
        return hash(self.frozen())

    def __reduce__(self) -> Any:  # pickles as a plain frozenset; lineage is process-local
        return (frozenset, (self.frozen(),))

    def __repr__(self) -> str:
        return f"ScopeIds({self._len} ids, +{len(self._plus)}/-{len(self._minus)})"

    # ── set-compatible helpers (fast paths on the base) ───────────────────────

    def frozen(self) -> frozenset[str]:
        """An equal plain frozenset (the base itself when there are no overlays; cached)."""
        if self._frozen is None:
            self._frozen = self._base.difference(self._minus).union(self._plus)
        return self._frozen

    def frozen_base(self) -> frozenset[str]:
        """The shared base frozenset (for tests and diagnostics)."""
        return self._base

    def difference(self, *others: Iterable[str]) -> frozenset[str]:
        # frozenset.difference takes sets and dicts at C speed (index partitions are dicts).
        return self._base.difference(self._minus, *others).union(self._plus.difference(*others))

    def intersection(self, *others: Iterable[str]) -> frozenset[str]:
        result = self.frozen() if not others else frozenset(filter(self.__contains__, others[0]))
        return result.intersection(*others[1:])

    def union(self, *others: Iterable[str]) -> frozenset[str]:
        return self.frozen().union(*others)

    def issubset(self, other: Iterable[str]) -> bool:
        container = other if isinstance(other, (set, frozenset, dict, AbstractSet)) else set(other)
        return len(self) <= len(container) and all(map(container.__contains__, self))

    def issuperset(self, other: Iterable[str]) -> bool:
        return all(map(self.__contains__, other))

    def copy(self) -> frozenset[str]:
        return self.frozen()

    add = discard = remove = pop = clear = update = _immutable  # type: ignore[assignment]
    difference_update = intersection_update = symmetric_difference_update = _immutable  # type: ignore[assignment]
