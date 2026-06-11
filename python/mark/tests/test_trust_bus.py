# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mark import (
    BusMessage,
    BusSnapshot,
    BusSubscription,
    HashEmbeddingProvider,
    PublisherTrust,
    TrustAwareGlobalMemoryBus,
)
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# Import / instantiation
# ---------------------------------------------------------------------------

def test_publisher_trust_importable() -> None:
    assert PublisherTrust.SYSTEM is not None


def test_bus_message_importable() -> None:
    assert BusMessage is not None


def test_bus_subscription_importable() -> None:
    assert BusSubscription is not None


def test_bus_snapshot_importable() -> None:
    assert BusSnapshot is not None


def test_trust_aware_bus_importable() -> None:
    assert TrustAwareGlobalMemoryBus is not None


def test_runtime_trust_bus_returns_instance(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    assert isinstance(bus, TrustAwareGlobalMemoryBus)
    runtime.shutdown()


def test_runtime_trust_bus_accessor_is_stable(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus1 = runtime.trust_bus()
    bus2 = runtime.trust_bus()
    assert bus1 is bus2

    bus1.publish_sync("Stable bus fact.", publisher="agent",
                      trust=PublisherTrust.VERIFIED_AGENT)
    assert len(bus2.all_messages()) == 1
    runtime.shutdown()


# ---------------------------------------------------------------------------
# PublisherTrust ordering
# ---------------------------------------------------------------------------

def test_publisher_trust_values() -> None:
    assert PublisherTrust.SYSTEM         == "system"
    assert PublisherTrust.VERIFIED_AGENT == "verified_agent"
    assert PublisherTrust.RAW_AGENT      == "raw_agent"
    assert PublisherTrust.EXTERNAL       == "external"
    assert PublisherTrust.UNKNOWN        == "unknown"


# ---------------------------------------------------------------------------
# BusMessage
# ---------------------------------------------------------------------------

def test_bus_message_to_dict() -> None:
    from datetime import datetime, timezone
    msg = BusMessage(
        id="m1", content="Elena is left-handed.", publisher="agent-A",
        trust=PublisherTrust.VERIFIED_AGENT, tags=["character:elena"],
        timestamp=datetime.now(timezone.utc).isoformat(), metadata={"note": "test"},
    )
    d = msg.to_dict()
    assert d["trust"]     == "verified_agent"
    assert d["publisher"] == "agent-A"
    assert "character:elena" in d["tags"]


def test_bus_message_roundtrip() -> None:
    from datetime import datetime, timezone
    msg = BusMessage(
        id="m2", content="Red scarf.", publisher="agent-B",
        trust=PublisherTrust.RAW_AGENT, tags=["object:scarf"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    assert BusMessage.from_dict(msg.to_dict()).trust == PublisherTrust.RAW_AGENT


# ---------------------------------------------------------------------------
# BusSubscription.matches
# ---------------------------------------------------------------------------

def test_subscription_matches_min_trust() -> None:
    from datetime import datetime, timezone
    sub = BusSubscription(min_trust=PublisherTrust.VERIFIED_AGENT)
    ts  = datetime.now(timezone.utc).isoformat()

    raw_msg = BusMessage(id="x", content=".", publisher="a",
                         trust=PublisherTrust.RAW_AGENT, tags=[], timestamp=ts)
    sys_msg = BusMessage(id="y", content=".", publisher="b",
                         trust=PublisherTrust.SYSTEM, tags=[], timestamp=ts)

    assert not sub.matches(raw_msg)
    assert sub.matches(sys_msg)


def test_subscription_matches_tags_and_filter() -> None:
    from datetime import datetime, timezone
    sub = BusSubscription(tags=["character:elena"])
    ts  = datetime.now(timezone.utc).isoformat()

    no_tag  = BusMessage(id="a", content=".", publisher="x",
                         trust=PublisherTrust.UNKNOWN, tags=[], timestamp=ts)
    has_tag = BusMessage(id="b", content=".", publisher="x",
                         trust=PublisherTrust.UNKNOWN,
                         tags=["character:elena", "extra"], timestamp=ts)

    assert not sub.matches(no_tag)
    assert sub.matches(has_tag)


def test_subscription_topic_filter() -> None:
    from datetime import datetime, timezone
    sub = BusSubscription(topics=["character"])
    ts  = datetime.now(timezone.utc).isoformat()

    has_topic = BusMessage(id="a", content=".", publisher="x",
                           trust=PublisherTrust.UNKNOWN, tags=["character"],
                           timestamp=ts)
    no_topic  = BusMessage(id="b", content=".", publisher="x",
                           trust=PublisherTrust.UNKNOWN, tags=["object"],
                           timestamp=ts)

    assert sub.matches(has_topic)
    assert not sub.matches(no_topic)


# ---------------------------------------------------------------------------
# TrustAwareGlobalMemoryBus: publish_sync
# ---------------------------------------------------------------------------

def test_publish_sync_returns_bus_message(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    msg = bus.publish_sync(
        "Elena is left-handed.",
        publisher="agent-A",
        trust=PublisherTrust.VERIFIED_AGENT,
    )
    assert isinstance(msg, BusMessage)
    assert msg.trust == PublisherTrust.VERIFIED_AGENT
    assert msg.publisher == "agent-A"
    runtime.shutdown()


def test_publish_sync_message_appears_in_all_messages(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    bus = runtime.trust_bus()
    bus.publish_sync("Fact one.", publisher="a", trust=PublisherTrust.SYSTEM)
    bus.publish_sync("Fact two.", publisher="b", trust=PublisherTrust.RAW_AGENT)
    assert len(bus.all_messages()) == 2
    runtime.shutdown()


def test_trust_bus_rehydrates_persisted_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "m.db"

    runtime = MarkRuntime.local(store_path=db_path,
                                embedder=HashEmbeddingProvider(dim=32))
    runtime.trust_bus().publish_sync(
        "Persisted verified fact.",
        publisher="agent-A",
        trust=PublisherTrust.VERIFIED_AGENT,
        tags=["character:elena"],
    )
    runtime.shutdown()

    reopened = MarkRuntime.local(store_path=db_path,
                                 embedder=HashEmbeddingProvider(dim=32))
    bus = reopened.trust_bus()
    messages = bus.all_messages()

    assert len(messages) == 1
    assert messages[0].trust == PublisherTrust.VERIFIED_AGENT
    assert messages[0].publisher == "agent-A"
    assert "character:elena" in messages[0].tags

    result = bus.retrieve_sync(
        "Persisted verified",
        min_trust=PublisherTrust.VERIFIED_AGENT,
    )
    assert any(f.content == "Persisted verified fact." for f in result.fragments)
    reopened.shutdown()


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
    assert len(raw_f) == 1
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
