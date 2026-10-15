# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Security hardening: secret/PII redaction patterns, sanitizer behaviour, and
redaction of sync output and sensitive mapping values.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark import Mark
from mark.middlewares import GovernanceMiddleware

from _helpers_security import _runtime


# ── 5. redaction ──────────────────────────────────────────────────────────────

_FAKE_SECRETS = {
    "aws": "AKIAABCDEFGHIJKLMNOP",
    "aws_sts": "ASIAABCDEFGHIJKLMNOP",
    "github": "ghp_" + "a" * 36,
    "github_pat": "github_pat_" + "b" * 40,
    "slack": "xoxb-123456789012-abcdefghij",
    "openai": "sk-proj-" + "c" * 30,
    "anthropic": "sk-ant-api03-" + "d" * 30,
    "google": "AIza" + "e" * 35,
    "hf": "hf_" + "f" * 34,
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.abcdefghijklmnop",
}


@pytest.mark.parametrize("name", sorted(_FAKE_SECRETS))
def test_shared_patterns_redact_known_tokens(name: str) -> None:
    from mark.middlewares.governance import ContentSanitizer
    from mark.security import redact_secrets

    secret = _FAKE_SECRETS[name]
    text = f"here is the value {secret} ok"
    assert secret not in redact_secrets(text)
    assert secret not in ContentSanitizer().sanitize(text)


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("Authorization: Bearer abc.def-ghi_jkl", "abc.def-ghi_jkl"),
        ("db = postgres://admin:hunter2@db.internal/app", "hunter2"),
        ('{"password": "correct horse"}', "correct horse"),
        ("client_secret='s3cr3t-value'", "s3cr3t-value"),
        ("session cookie: abcdef123", "abcdef123"),
        ("PRIVATE_KEY=zzzz", "zzzz"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\nBBBB\n-----END OPENSSH PRIVATE KEY-----", "AAAA"),
        ("mail elena@example.com", "elena@example.com"),
    ],
)
def test_structured_secrets_redacted(text: str, leak: str) -> None:
    from mark.security import redact_secrets

    assert leak not in redact_secrets(text)


def test_benign_text_is_untouched() -> None:
    from mark.security import redact_secrets

    text = "author: Elena\nmax_tokens=512\nThe weather: sunny\nhttps://example.com/docs"
    assert redact_secrets(text) == text


def test_mapping_values_under_sensitive_keys_are_redacted() -> None:
    from mark.security import redact_sync_value

    payload = {"api_key": "plain", "nested": [{"Password": 12345}, {"note": "hi", "auth": {"x": "y"}}],
               "max_tokens": 100, "enabled": True}
    out = redact_sync_value(payload)
    assert out["api_key"] == "[REDACTED]"
    assert out["nested"][0]["Password"] == "[REDACTED]"
    assert out["nested"][1]["auth"] == {"x": "[REDACTED]"}
    assert out["nested"][1]["note"] == "hi"
    assert out["max_tokens"] == 100 and out["enabled"] is True


def test_sanitizer_preserves_newlines_and_records_truncation(tmp_path: Path) -> None:
    from mark.middlewares.governance import ConsolidationGate, ContentSanitizer

    code = "def f():\n    x  =  1\n\n    return x"
    assert ContentSanitizer().sanitize(code) == "def f():\n    x = 1\n\n    return x"

    report = ContentSanitizer(max_length=5).sanitize_report("abcdefghij")
    assert report.truncated and report.original_length == 10 and report.text == "abcde"

    gate = ConsolidationGate(sanitizer=ContentSanitizer(max_length=20))
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware(gate=gate)]) as mark:
        frag_id = mark.runtime.memory("t").store_sync("a long fact " * 10)
        stored = mark.runtime.store.get(frag_id)
    assert stored.metadata["mark_governance"]["truncated"] is True
    assert stored.metadata["mark_governance"]["original_length"] == len(("a long fact " * 10))


def test_sync_output_has_no_plain_content_hash(tmp_path: Path) -> None:
    from mark.middlewares.sync import CloudSync

    runtime = _runtime(tmp_path)
    runtime.memory("a").store_sync("my password=hunter2")
    assert "content_hash" in runtime.store.list_all()[0].metadata
    delta = CloudSync().prepare_delta(runtime)
    assert delta.fragments
    assert all("content_hash" not in f["metadata"] for f in delta.fragments)
    assert "hunter2" not in str(delta.fragments)
    runtime.shutdown()


def test_authorization_mapping_values_are_redacted() -> None:
    from mark.security.redaction import is_sensitive_key, redact_sync_value as redact_value

    assert is_sensitive_key("Authorization") and is_sensitive_key("x-auth-token")
    assert not is_sensitive_key("author") and not is_sensitive_key("authors")
    redacted = redact_value({"headers": {"Authorization": "hunter2-opaque-session"}, "author": "Ada"})
    assert "hunter2" not in str(redacted)
    assert redacted["author"] == "Ada"
