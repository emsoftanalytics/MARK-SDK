# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Security hardening: Fernet key sourcing, passphrase key-derivation caching,
keyed content digests, and a light `import mark`.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from mark.types import MemoryFragment


# ── 6. encryption ─────────────────────────────────────────────────────────────

def test_fernet_requires_key_source() -> None:
    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    with pytest.raises(ValueError):
        FernetEncryptionProvider()
    with pytest.raises(ValueError):
        FernetEncryptionProvider(passphrase="x", ephemeral=True)
    provider = FernetEncryptionProvider(ephemeral=True)
    assert provider.decrypt(provider.encrypt(b"hi")) == b"hi"


def test_passphrase_key_derivation_is_cached(monkeypatch) -> None:
    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    provider = FernetEncryptionProvider(passphrase="correct horse")
    calls: list[bytes] = []
    real = provider._derive_bytes
    monkeypatch.setattr(provider, "_derive_bytes", lambda salt: calls.append(salt) or real(salt))
    tokens = [provider.encrypt(f"fragment {i}".encode()) for i in range(5)]
    assert [provider.decrypt(t) for t in tokens] == [f"fragment {i}".encode() for i in range(5)]
    assert len(calls) == 1

    # A fresh instance (new process) still decrypts, deriving once per salt.
    other = FernetEncryptionProvider(passphrase="correct horse")
    assert other.decrypt(tokens[0]) == b"fragment 0"


def test_content_digest_is_keyed_when_encrypted() -> None:
    pytest.importorskip("cryptography")
    import hashlib

    from mark.security import ContentHasher, FernetEncryptionProvider, NoOpEncryptionProvider

    plain = NoOpEncryptionProvider().content_digest("secret fact")
    assert plain == hashlib.sha256(b"secret fact").hexdigest()

    key = FernetEncryptionProvider(ephemeral=True)
    digest = key.content_digest("secret fact")
    assert digest.startswith("hmac-sha256:") and plain not in digest
    assert digest == key.content_digest("secret fact")
    assert FernetEncryptionProvider(ephemeral=True).content_digest("secret fact") != digest

    p1 = FernetEncryptionProvider(passphrase="pw").content_digest("x")
    assert p1 == FernetEncryptionProvider(passphrase="pw").content_digest("x")

    frag = MemoryFragment(content="secret fact", agent_id="a")
    attached = ContentHasher.attach(frag, key)
    assert ContentHasher.check(attached, key)
    assert not ContentHasher.check(attached.model_copy(update={"content": "changed"}), key)
    assert not ContentHasher.check(attached)  # keyed digest cannot verify without the key
    assert ContentHasher.check(ContentHasher.attach(frag), key)  # legacy sha256 still verifies


@pytest.mark.parametrize("module", ["cryptography", "numpy", "pyarrow", "langchain_core", "requests",
                                    "bs4", "duckduckgo_search", "sentence_transformers", "torch", "mcp"])
def test_import_mark_is_light(module: str) -> None:
    code = f"import sys, mark; sys.exit(1 if {module!r} in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
