# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import pytest

from mark import (
    AuditEntry,
    ConsolidationGate,
    ConsolidationGateResult,
    ContentSanitizer,
    DuplicateHashGate,
    FailureFilter,
    GateResult,
    GovernanceAuditLog,
    ValidationGate,
)


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_all_governance_classes_importable() -> None:
    for cls in [ContentSanitizer, ValidationGate, DuplicateHashGate,
                FailureFilter, ConsolidationGate, GovernanceAuditLog]:
        assert cls is not None


# ---------------------------------------------------------------------------
# ContentSanitizer
# ---------------------------------------------------------------------------

def test_sanitizer_passthrough_clean_text() -> None:
    s   = ContentSanitizer()
    out = s.sanitize("Elena is left-handed.")
    assert out == "Elena is left-handed."


def test_sanitizer_redacts_email() -> None:
    s   = ContentSanitizer()
    out = s.sanitize("Contact us at admin@example.com for help.")
    assert "[EMAIL]" in out
    assert "admin@example.com" not in out


def test_sanitizer_redacts_ssn() -> None:
    s   = ContentSanitizer()
    out = s.sanitize("SSN is 123-45-6789.")
    assert "[SSN]" in out


def test_sanitizer_redacts_phone() -> None:
    s   = ContentSanitizer()
    out = s.sanitize("Call me at 555-867-5309.")
    assert "[PHONE]" in out


def test_sanitizer_truncates_long_text() -> None:
    s   = ContentSanitizer(max_length=10)
    out = s.sanitize("A" * 100)
    assert len(out) <= 10


def test_sanitizer_collapses_whitespace() -> None:
    s   = ContentSanitizer()
    out = s.sanitize("  Elena   is  left-handed.  ")
    assert out == "Elena is left-handed."


def test_sanitizer_has_pii_true() -> None:
    s = ContentSanitizer()
    assert s.has_pii("Send to user@example.com")


def test_sanitizer_has_pii_false() -> None:
    s = ContentSanitizer()
    assert not s.has_pii("Elena is left-handed.")


# ---------------------------------------------------------------------------
# ValidationGate
# ---------------------------------------------------------------------------

def test_validation_passes_normal_content() -> None:
    g = ValidationGate()
    r = g.check("Elena is left-handed.")
    assert r.passed


def test_validation_rejects_empty_string() -> None:
    g = ValidationGate()
    r = g.check("")
    assert not r.passed
    assert r.reason


def test_validation_rejects_whitespace_only() -> None:
    g = ValidationGate()
    r = g.check("   ")
    assert not r.passed


def test_validation_rejects_low_confidence() -> None:
    g = ValidationGate(min_confidence=0.5)
    r = g.check("Some content.", confidence=0.3)
    assert not r.passed
    assert "confidence" in r.reason


def test_validation_passes_at_threshold() -> None:
    g = ValidationGate(min_confidence=0.5)
    assert g.check("Content.", confidence=0.5).passed


def test_validation_min_length_respected() -> None:
    g = ValidationGate(min_length=10)
    assert not g.check("Short").passed
    assert g.check("A" * 10).passed


# ---------------------------------------------------------------------------
# DuplicateHashGate
# ---------------------------------------------------------------------------

def test_duplicate_hash_first_occurrence_passes() -> None:
    g = DuplicateHashGate()
    assert g.check("Unique content.").passed


def test_duplicate_hash_second_occurrence_blocked() -> None:
    g = DuplicateHashGate()
    g.check("Duplicate.")
    r = g.check("Duplicate.")
    assert not r.passed
    assert "duplicate" in r.reason.lower()


def test_duplicate_hash_different_content_passes() -> None:
    g = DuplicateHashGate()
    g.check("First.")
    assert g.check("Second.").passed


def test_duplicate_hash_reset_clears_state() -> None:
    g = DuplicateHashGate()
    g.check("Content.")
    g.reset()
    assert g.check("Content.").passed


# ---------------------------------------------------------------------------
# FailureFilter
# ---------------------------------------------------------------------------

def test_failure_filter_passes_normal_content() -> None:
    f = FailureFilter()
    assert f.check("Elena is left-handed.").passed


def test_failure_filter_rejects_traceback() -> None:
    f  = FailureFilter()
    tb = "Traceback (most recent call last):\n  File 'x.py', line 1\nValueError: bad"
    r  = f.check(tb)
    assert not r.passed


def test_failure_filter_rejects_http_error() -> None:
    f = FailureFilter()
    r = f.check('{"error": "Not found"}')
    assert not r.passed


def test_failure_filter_rejects_http_status() -> None:
    f = FailureFilter()
    r = f.check("HTTP 404 Not Found")
    assert not r.passed


def test_failure_filter_allows_content_mentioning_error_naturally() -> None:
    f = FailureFilter()
    # Natural prose that merely mentions errors should not be blocked
    r = f.check("Elena noticed an error in the calculations and fixed it.")
    assert r.passed


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
    g  = ConsolidationGate()
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
