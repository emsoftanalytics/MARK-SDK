# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Write-path security defaults: redaction on by default with opt-out,
redaction of quarantined and metadata values, no silent truncation, and
error memories (tracebacks, HTTP 5xx) kept retrievable unless the
FailureFilter is explicitly enabled.
"""
from __future__ import annotations

from pathlib import Path

from mark import Mark
from mark.memory.runtime import MarkRuntime
from mark.middlewares.governance import (
    ConsolidationGate,
    ContentSanitizer,
    FailureFilter,
    GovernanceMiddleware,
)
from mark.security import RedactionPolicy, WritePolicy, resolve_write_policy
from mark.types import MemoryState

_STRIPE = "sk_" + "test_51HxYzAbCdEfGhIjKlMnOp"


def test_mark_local_redacts_secrets_by_default(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        mem = mark.runtime.memory("a")
        fid = mem.store_sync(f"The staging Stripe key is {_STRIPE} and DB password=Hunter2!xyz&3")
        frag = mark.runtime.store.get(fid)
        assert _STRIPE not in frag.content and "Hunter2" not in frag.content
        assert "[SECRET:stripe_key]" in frag.content
        assert sorted(frag.metadata["mark_security"]["redacted"]) == ["password", "stripe_key"]
        # Recall still knows a key existed.
        assert "[SECRET:stripe_key]" in mem.retrieve_sync("staging Stripe key").top().content


def test_batch_and_observe_paths_are_redacted(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    mem = runtime.memory("a")
    ids = mem.store_many_sync([f"key {_STRIPE}", "plain"])
    assert _STRIPE not in runtime.store.get(ids[0]).content
    assert all(_STRIPE not in f.content for f in runtime.store.list_all())
    runtime.shutdown()


def test_metadata_strings_and_quarantine_are_redacted(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    mem = runtime.memory("a")
    fid = mem.store_sync("note", metadata={"cmd": f"curl -u x:{_STRIPE}", "n": 3})
    assert _STRIPE not in str(runtime.store.get(fid).metadata)
    qid = mem.quarantine_sync(f"raw dump {_STRIPE}", reason="debug")
    assert _STRIPE not in runtime.store.get(qid).content
    runtime.shutdown()


def test_redact_false_opts_out(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    runtime.write_policy = resolve_write_policy(redact=False)  # W6 wiring stand-in
    fid = runtime.memory("a").store_sync(f"key {_STRIPE}")
    assert _STRIPE in runtime.store.get(fid).content
    runtime.shutdown()


def test_resolve_write_policy_accepts_policies() -> None:
    assert resolve_write_policy().redaction == RedactionPolicy()
    assert resolve_write_policy(redact=True).redaction == RedactionPolicy()
    assert resolve_write_policy(redact=False).redaction is None
    custom = RedactionPolicy(pii=True)
    assert resolve_write_policy(redact=custom).redaction is custom
    whole = WritePolicy(redaction=None, injection="off")
    assert resolve_write_policy(redact=whole) is whole


def test_pii_policy_redacts_phone_at_write(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    runtime.write_policy = resolve_write_policy(redact=RedactionPolicy(pii=True))
    fid = runtime.memory("a").store_sync("Dr. Bello's clinic phone is +234 803 555 0192")
    assert runtime.store.get(fid).content == "Dr. Bello's clinic phone is [PHONE]"
    runtime.shutdown()


def test_errors_are_remembered_with_default_governance(tmp_path: Path) -> None:
    samples = [
        "Traceback (most recent call last):\n  File \"w.py\", line 88\nJSONDecodeError: Expecting value",
        "Error: ECONNREFUSED 10.0.3.17:6379 - redis sidecar not ready",
        "Lesson learned: the vendor API returns HTTP 503 during their Sunday maintenance.",
    ]
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware()]) as mark:
        mem = mark.runtime.memory("gov")
        for text in samples:
            frag = mark.runtime.store.get(mem.store_sync(text))
            assert frag.state != MemoryState.QUARANTINED, text
            assert frag.metadata["mark_governance"]["passed"] is True


def test_failure_filter_is_explicit_opt_in(tmp_path: Path) -> None:
    gate = ConsolidationGate(failure=FailureFilter())
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware(gate=gate)]) as mark:
        frag = mark.runtime.store.get(mark.runtime.memory("g").store_sync("HTTP 503 from vendor"))
        assert frag.state == MemoryState.QUARANTINED


def test_sanitizer_never_truncates_silently(tmp_path: Path) -> None:
    long_text = "word " * 5_000  # 25k chars, beyond the old 10k cap
    report = ContentSanitizer().sanitize_report(long_text)
    assert not report.truncated and report.text == long_text.strip()
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware()]) as mark:
        frag = mark.runtime.store.get(mark.runtime.memory("g").store_sync(long_text))
        assert len(frag.content) == len(long_text.strip())
        assert "truncated" not in frag.metadata["mark_governance"]


def test_governance_middleware_redacts_whole_values(tmp_path: Path) -> None:
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware()]) as mark:
        mem = mark.runtime.memory("g")
        frag = mark.runtime.store.get(mem.store_sync("Call (415) 555-0123 or +234 803 555 0192, pwd=a&b"))
        assert "(415" not in frag.content and "+234" not in frag.content and "&b" not in frag.content
