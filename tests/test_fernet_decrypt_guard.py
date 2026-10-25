# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Passphrase decryption rejects non-token input without deriving a key."""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from cryptography.fernet import InvalidToken  # noqa: E402

from mark.security import FernetEncryptionProvider  # noqa: E402


def test_plaintext_fails_fast_without_key_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FernetEncryptionProvider(passphrase="correct horse battery staple")
    token = provider.encrypt(b"secret note")
    assert provider.decrypt(token) == b"secret note"

    calls = []
    monkeypatch.setattr(provider, "_derive_bytes", lambda salt: calls.append(salt) or b"x" * 32)
    with pytest.raises(InvalidToken):
        provider.decrypt(b"plaintext written by an older MARK without the key")
    assert calls == []  # no PBKDF2 for input that cannot be a token
