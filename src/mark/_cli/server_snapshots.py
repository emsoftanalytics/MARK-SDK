"""``mark serve`` start and stop work on the vector index: warm-up and snapshots.

Why the stop is selective: a graceful stop used to re-sync the index and
rewrite every large agent's vector snapshot after *any* write. At a million
memories that is ~1 GB and 38-60 s, longer than service managers wait
before killing the process (release scale check). A snapshot does not have
to be fresh to be correct: every write since it was saved is listed in
``vector_dirty`` and the next cold start reads exactly those rows (see
:mod:`mark.store._sqlite.vector_snapshot`). So the stop rewrites a snapshot
only when its delta is large enough to slow that cold start noticeably
(``REFRESH_MIN_ROWS`` or ``1 / REFRESH_FRACTION`` of the agent), or when a
large agent has no usable snapshot yet.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("mark.cli")

#: A delta below this many rows (and below 1/REFRESH_FRACTION of the agent)
#: is left for the next cold start: reading it costs well under a second.
REFRESH_MIN_ROWS = 10_000
REFRESH_FRACTION = 20


def _warm(engine: Any) -> None:
    """Preload the index for every agent; failures only cost the speed-up."""
    warm_up = getattr(engine.runtime, "warm_up", None)
    if warm_up is None:
        return
    started = time.monotonic()
    try:
        count = warm_up(sketches=True)  # encrypted stores: timelines and id recalls never pay it
    except Exception:  # noqa: BLE001 - e.g. shut down while warming
        logger.debug("mark serve: index warm-up skipped", exc_info=True)
        return
    print(f"MARK index warm: {count} memories in {time.monotonic() - started:.1f}s", flush=True)


def snapshots_need_refresh(runtime: Any) -> bool:
    """True when some large agent has no usable snapshot or a large delta since it.

    Unknown (a store or embedder without the needed hooks) counts as True, so
    such hosts keep the previous behaviour.
    """
    from mark.store._sqlite.vector_snapshot import MIN_ROWS

    store, index = getattr(runtime, "store", None), getattr(runtime, "index", None)
    delta_of = getattr(store, "vector_snapshot_delta", None)
    fingerprint = getattr(getattr(runtime, "embedder", None), "fingerprint", None)
    if delta_of is None or index is None or not fingerprint:
        return True
    for agent_id in [a for a in list(getattr(index, "_partitions", {})) if a is not None]:
        count = index.count(agent_id)
        if count < MIN_ROWS:
            continue   # small agents never get a snapshot
        delta = delta_of(agent_id, fingerprint)
        if delta is None or delta >= max(REFRESH_MIN_ROWS, count // REFRESH_FRACTION):
            return True
    return False


def _save_snapshots(engine: Any) -> None:
    """Refresh vector snapshots on a graceful stop when worth it; failures only cost the next cold start."""
    runtime = engine.runtime
    save = getattr(runtime, "save_vector_snapshots", None)
    if save is None:
        return
    try:
        if not snapshots_need_refresh(runtime):
            logger.debug("mark serve: vector snapshots are current enough; not rewritten")
            return
        save()
    except Exception:  # noqa: BLE001 - e.g. disk full: never block the shutdown
        logger.warning("mark serve: could not save the vector snapshot", exc_info=True)
