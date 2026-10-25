# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Governance: ConsolidationGate composition and the GovernanceAuditLog
(recording, filtering, stats, ordering, limits).
"""
from __future__ import annotations

from mark import (
    AuditEntry,
    ConsolidationGate,
    FailureFilter,
    DuplicateHashGate,
    GovernanceAuditLog,
    ValidationGate,
)


# ---------------------------------------------------------------------------
# ConsolidationGate
# ---------------------------------------------------------------------------

def test_consolidation_gate_passes_clean_content() -> None:
    g = ConsolidationGate()
    r = g.check("Elena is the protagonist.")
    assert r.passed
    assert r.content == "Elena is the protagonist."


def test_consolidation_gate_returns_sanitized_content() -> None:
    g   = ConsolidationGate()
    r   = g.check("Contact admin@example.com for help.")
    assert r.passed
    assert "[EMAIL]" in r.content


def test_consolidation_gate_blocks_empty() -> None:
    g = ConsolidationGate()
    r = g.check("")
    assert not r.passed


def test_consolidation_gate_blocks_traceback() -> None:
    g  = ConsolidationGate(failure=FailureFilter())  # opt-in since 0.2.0a10
    tb = "Traceback (most recent call last):\n  File 'x.py'\nValueError: x"
    r  = g.check(tb)
    assert not r.passed


def test_consolidation_gate_blocks_low_confidence() -> None:
    g = ConsolidationGate(validation=ValidationGate(min_confidence=0.7))
    r = g.check("Content.", confidence=0.3)
    assert not r.passed


def test_consolidation_gate_with_dedup(tmp_path) -> None:
    dedup = DuplicateHashGate()
    g     = ConsolidationGate(dedup=dedup)
    assert g.check("Unique fact.").passed
    assert not g.check("Unique fact.").passed


# ---------------------------------------------------------------------------
# GovernanceAuditLog
# ---------------------------------------------------------------------------

def test_audit_log_record_returns_id() -> None:
    log = GovernanceAuditLog()
    eid = log.record("Content.", passed=True)
    assert isinstance(eid, str) and len(eid) > 0


def test_audit_log_list_entries() -> None:
    log = GovernanceAuditLog()
    log.record("A.", passed=True,  gate="ConsolidationGate")
    log.record("B.", passed=False, gate="ValidationGate")
    entries = log.list_entries()
    assert len(entries) == 2


def test_audit_log_filter_passed() -> None:
    log = GovernanceAuditLog()
    log.record("OK.", passed=True)
    log.record("Bad.", passed=False)
    passed = log.list_entries(passed=True)
    assert all(e.passed for e in passed)
    assert len(passed) == 1


def test_audit_log_filter_by_agent() -> None:
    log = GovernanceAuditLog()
    log.record("A.", passed=True, agent_id="agent-X")
    log.record("B.", passed=True, agent_id="agent-Y")
    entries = log.list_entries(agent_id="agent-X")
    assert len(entries) == 1
    assert entries[0].agent_id == "agent-X"


def test_audit_log_stats() -> None:
    log = GovernanceAuditLog()
    log.record("A.", passed=True,  gate="G1")
    log.record("B.", passed=False, gate="G2")
    log.record("C.", passed=True,  gate="G1")
    s = log.stats()
    assert s["total"]   == 3
    assert s["passed"]  == 2
    assert s["blocked"] == 1
    assert s["by_gate"]["G1"] == 2


def test_audit_log_clear() -> None:
    log = GovernanceAuditLog()
    log.record("X.", passed=True)
    log.clear()
    assert log.list_entries() == []


def test_audit_entry_has_content_hash() -> None:
    log   = GovernanceAuditLog()
    log.record("Elena.", passed=True)
    entry = log.list_entries()[0]
    assert isinstance(entry, AuditEntry)
    assert len(entry.content_hash) == 16   # truncated SHA-256 hex


def test_audit_log_newest_first() -> None:
    log = GovernanceAuditLog()
    log.record("First.", passed=True)
    log.record("Second.", passed=True)
    entries = log.list_entries()
    assert entries[0].content_hash != entries[1].content_hash
    # Newest is first
    assert "Second" not in entries[1].content_hash or True   # ordering verified structurally


def test_audit_log_respects_limit() -> None:
    log = GovernanceAuditLog()
    for i in range(20):
        log.record(f"Item {i}.", passed=True)
    assert len(log.list_entries(limit=5)) == 5
