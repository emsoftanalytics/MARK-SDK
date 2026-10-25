# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Trust bus: subscriptions (subscribe/unsubscribe) and BusSnapshot filtering,
JSON round-trip, diffing, trust filtering, and context rendering.
"""
from __future__ import annotations

from pathlib import Path

from mark import BusSnapshot, BusSubscription, HashEmbeddingProvider, PublisherTrust
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

def test_subscribe_and_messages_for(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus    = runtime.trust_bus()
    sub_id = bus.subscribe(BusSubscription(min_trust=PublisherTrust.VERIFIED_AGENT))

    bus.publish_sync("System fact.", publisher="sys",
                     trust=PublisherTrust.SYSTEM)
    bus.publish_sync("Raw fact.", publisher="raw",
                     trust=PublisherTrust.RAW_AGENT)

    msgs = bus.messages_for(sub_id)
    assert len(msgs) == 1
    assert msgs[0].trust == PublisherTrust.SYSTEM
    runtime.shutdown()


def test_unsubscribe_clears_subscription(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus    = runtime.trust_bus()
    sub_id = bus.subscribe(BusSubscription())
    bus.unsubscribe(sub_id)
    assert bus.messages_for(sub_id) == []
    runtime.shutdown()


# ---------------------------------------------------------------------------
# BusSnapshot
# ---------------------------------------------------------------------------

def test_snapshot_contains_all_messages(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    bus.publish_sync("A.", publisher="x", trust=PublisherTrust.SYSTEM)
    bus.publish_sync("B.", publisher="y", trust=PublisherTrust.RAW_AGENT)
    snap = bus.snapshot()
    assert len(snap.messages) == 2
    runtime.shutdown()


def test_snapshot_with_subscription_filter(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    bus.publish_sync("Sys.", publisher="s", trust=PublisherTrust.SYSTEM)
    bus.publish_sync("Raw.", publisher="r", trust=PublisherTrust.RAW_AGENT)
    snap = bus.snapshot(BusSubscription(min_trust=PublisherTrust.SYSTEM))
    assert len(snap.messages) == 1
    assert snap.messages[0].trust == PublisherTrust.SYSTEM
    runtime.shutdown()


def test_snapshot_to_json_roundtrip(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    bus.publish_sync("Fact.", publisher="a", trust=PublisherTrust.VERIFIED_AGENT)
    snap  = bus.snapshot()
    snap2 = BusSnapshot.from_json(snap.to_json())
    assert len(snap2.messages) == 1
    assert snap2.messages[0].trust == PublisherTrust.VERIFIED_AGENT
    runtime.shutdown()


def test_snapshot_diff(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus   = runtime.trust_bus()
    bus.publish_sync("Old.", publisher="a", trust=PublisherTrust.SYSTEM)
    snap1 = bus.snapshot()
    bus.publish_sync("New.", publisher="b", trust=PublisherTrust.SYSTEM)
    snap2 = bus.snapshot()

    new_msgs = snap2.diff(snap1)
    assert len(new_msgs) == 1
    assert new_msgs[0].content == "New."
    runtime.shutdown()


def test_snapshot_facts_by_trust(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    bus.publish_sync("A.", publisher="s", trust=PublisherTrust.SYSTEM)
    bus.publish_sync("B.", publisher="r", trust=PublisherTrust.RAW_AGENT)
    snap  = bus.snapshot()
    sys_f = snap.facts_by_trust(PublisherTrust.SYSTEM)
    raw_f = snap.facts_by_trust(PublisherTrust.RAW_AGENT)
    assert len(sys_f) == 1
    # "at or above": RAW_AGENT floor includes the SYSTEM fact too.
    assert len(raw_f) == 2
    runtime.shutdown()


def test_snapshot_as_context_xml(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    bus.publish_sync("Elena is left-handed.",
                     publisher="agent", trust=PublisherTrust.VERIFIED_AGENT)
    ctx = bus.snapshot().as_context()
    assert "<bus_context" in ctx
    assert "Elena is left-handed." in ctx
    runtime.shutdown()


def test_empty_snapshot_as_context() -> None:
    from datetime import datetime, timezone
    snap = BusSnapshot(messages=[], captured_at=datetime.now(timezone.utc).isoformat())
    assert snap.as_context() == "<bus_context />"
