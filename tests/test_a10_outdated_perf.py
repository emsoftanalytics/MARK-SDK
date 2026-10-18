"""a10 re-test issue 4: outdated-line detection is cached, windowed and not O(n^2).

12 x 16 KB memories took 1-6 s per briefing: every pair was compared and
both full texts were stemmed each time.
"""
from __future__ import annotations

import random
import time
from types import SimpleNamespace

from mark.context.outdated import replaced_by, replaces

_WORDS = ("owner team region moved from now changed new not Porto Braga Ana Ines number "
          "phone vet October instead SQLite Postgres no longer at").split()


def _pairwise(fragments, key):
    """The pre-a10 definition: every older line against every newer one, newest first."""
    timed = sorted(fragments, key=key)
    out = {}
    for index, older in enumerate(timed):
        for newer in reversed(timed[index + 1:]):
            if key(newer) != key(older) and replaces(newer, older):
                out[older.id] = newer.id
                break
    return out


def test_bucketed_comparison_matches_every_pair() -> None:
    rng = random.Random(7)
    for _ in range(5):
        pool = [SimpleNamespace(id=str(i), content=" ".join(rng.choice(_WORDS) for _ in range(9)))
                for i in range(150)]
        key = lambda f: int(f.id) // 3   # noqa: E731 - some equal keys on purpose
        assert replaced_by(pool, key) == _pairwise(pool, key)


def test_known_replacements_still_found() -> None:
    old = SimpleNamespace(id="a", content="Joao's phone number is 555 1234.")
    new = SimpleNamespace(id="b", content="Joao changed his phone number to 555 9999.")
    moved = SimpleNamespace(id="c", content="We moved from Porto to Braga.")
    home = SimpleNamespace(id="d", content="Home is in Porto.")
    order = {"a": 1, "d": 2, "b": 3, "c": 4}
    assert replaced_by([old, new, moved, home], lambda f: order[f.id]) == {"a": "b", "d": "c"}


def test_long_memories_are_cheap() -> None:
    rng = random.Random(1)
    def text(i: int) -> str:
        head = f"Update {i}: the Atlas-Core rollout moved from Porto{i % 3} and the owner changed. "
        body = " ".join(rng.choice(_WORDS) + rng.choice(("s", "ing", "ed", "")) for _ in range(2600))
        return head + body
    pool = [SimpleNamespace(id=str(i), content=text(i)) for i in range(12)]
    started = time.process_time()
    for _ in range(5):
        replaced_by(pool, lambda f: int(f.id))
    assert time.process_time() - started < 0.5   # was seconds per call


def test_large_pools_are_not_quadratic() -> None:
    pool = [SimpleNamespace(id=str(i), content=f"svc{i} owner is team{i % 97} in region{i % 31}")
            for i in range(3000)]
    started = time.process_time()
    assert replaced_by(pool, lambda f: int(f.id)) == {}
    assert time.process_time() - started < 1.0
