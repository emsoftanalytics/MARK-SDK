"""Swappable at-rest encryption providers (NoOp default, optional Fernet).

``cryptography`` is imported lazily, the first time a Fernet provider is
constructed, so ``import mark`` never pays for (or requires) it.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

#: Prefix marking a keyed content digest (vs. a legacy bare SHA-256 hex digest).
HMAC_DIGEST_PREFIX = "hmac-sha256:"

_CRYPTO: tuple[Any, Any, Any] | None = None
_CRYPTO_LOCK = threading.Lock()


def _load_cryptography() -> tuple[Any, Any, Any]:
    """Import and cache (Fernet, hashes, PBKDF2HMAC) on first use."""
    global _CRYPTO
    if _CRYPTO is not None:
        return _CRYPTO
    with _CRYPTO_LOCK:
        if _CRYPTO is None:
            try:
                from cryptography.fernet import Fernet
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            except Exception as exc:
                raise ImportError(
                    "FernetEncryptionProvider requires the optional dependency "
                    '`cryptography`. Install "mark-sdk[crypto]" or use '
                    "NoOpEncryptionProvider for local development."
                ) from exc
            _CRYPTO = (Fernet, hashes, PBKDF2HMAC)
    return _CRYPTO


def _require_cryptography() -> None:
    _load_cryptography()


class EncryptionProvider(ABC):
    """Swappable symmetric encryption boundary for local stores."""

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        ...

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        ...

    def encrypt_str(self, text: str) -> bytes:
        """Encrypt a UTF-8 string to bytes."""
        return self.encrypt(text.encode("utf-8"))

    def decrypt_str(self, data: bytes) -> str:
        """Decrypt bytes back to a UTF-8 string."""
        return self.decrypt(data).decode("utf-8")

    def content_digest(self, plaintext: str) -> str:
        """Integrity digest of plaintext content, safe to store next to ciphertext.

        Unencrypted providers return a plain SHA-256 hex digest. Encrypting
        providers override this with a keyed HMAC so the stored digest does
        not let anyone confirm guesses about the encrypted content.
        """
        return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class NoOpEncryptionProvider(EncryptionProvider):
    """Default local mode: no encryption, no optional dependencies."""

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        return data

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        return data


class FernetEncryptionProvider(EncryptionProvider):
    """Optional Fernet encryption provider for local memory at rest.

    Exactly one key source is required:

    key         a Fernet key (urlsafe base64, 32 bytes)
    passphrase  a passphrase stretched with PBKDF2-SHA256
    ephemeral   ``True`` to generate a throwaway in-memory key (tests/demos);
                data encrypted with it is unreadable after the process exits

    Passphrase mode derives one key per salt and caches it, and reuses a
    single per-instance salt for encryption, so PBKDF2 runs once per
    provider instead of once per fragment.
    """

    _SALT_SIZE = 16
    _ITERATIONS = 390_000
    _CACHE_SIZE = 64
    _DIGEST_CONTEXT = b"mark-content-digest-v1"

    def __init__(
        self,
        key: str | bytes | None = None,
        passphrase: str | None = None,
        *,
        ephemeral: bool = False,
    ) -> None:
        Fernet, _hashes, _kdf = _load_cryptography()
        sources = sum(x is not None for x in (key, passphrase)) + (1 if ephemeral else 0)
        if sources == 0:
            raise ValueError(
                "FernetEncryptionProvider needs a key or passphrase "
                "(or ephemeral=True for a throwaway in-memory key)."
            )
        if sources > 1:
            raise ValueError("Pass only one of key, passphrase, or ephemeral=True.")

        self._lock = threading.Lock()
        self._derived: OrderedDict[bytes, Any] = OrderedDict()
        self._encrypt_salt: bytes | None = None
        self._digest_key: bytes | None = None

        if key is not None:
            if isinstance(key, str):
                key = key.encode("utf-8")
            self._fernet = Fernet(key)
            self._key_material: bytes | None = base64.urlsafe_b64decode(key)
            self._passphrase: bytes | None = None
        elif passphrase is not None:
            self._fernet = None
            self._key_material = None
            self._passphrase = passphrase.encode("utf-8")
        else:
            generated = Fernet.generate_key()
            self._fernet = Fernet(generated)
            self._key_material = base64.urlsafe_b64decode(generated)
            self._passphrase = None

    @classmethod
    def from_env(cls, env_var: str = "MARK_ENCRYPTION_KEY") -> "FernetEncryptionProvider":
        """Construct from an environment variable."""
        key = os.environ.get(env_var)
        if not key:
            raise EnvironmentError(f"Encryption key env var {env_var!r} is not set.")
        return cls(key=key)

    def _derive_bytes(self, salt: bytes) -> bytes:
        _Fernet, hashes, PBKDF2HMAC = _load_cryptography()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._ITERATIONS,
        )
        return kdf.derive(self._passphrase)

    def _derive(self, salt: bytes) -> Any:
        with self._lock:
            cached = self._derived.get(salt)
            if cached is not None:
                self._derived.move_to_end(salt)
                return cached
        Fernet, _hashes, _kdf = _load_cryptography()
        fernet = Fernet(base64.urlsafe_b64encode(self._derive_bytes(salt)))
        with self._lock:
            self._derived[salt] = fernet
            self._derived.move_to_end(salt)
            while len(self._derived) > self._CACHE_SIZE:
                self._derived.popitem(last=False)
        return fernet

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        if self._passphrase is not None:
            if self._encrypt_salt is None:
                with self._lock:
                    if self._encrypt_salt is None:
                        self._encrypt_salt = os.urandom(self._SALT_SIZE)
            salt = self._encrypt_salt
            return salt + self._derive(salt).encrypt(data)
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        if self._passphrase is not None:
            salt = data[: self._SALT_SIZE]
            token = data[self._SALT_SIZE :]
            return self._derive(salt).decrypt(token)
        return self._fernet.decrypt(data)

    def _content_digest_key(self) -> bytes:
        if self._digest_key is None:
            if self._key_material is not None:
                material = self._key_material
            else:
                # Stable across instances/processes for the same passphrase.
                material = self._derive_bytes(self._DIGEST_CONTEXT)
            self._digest_key = hmac.new(material, self._DIGEST_CONTEXT, hashlib.sha256).digest()
        return self._digest_key

    def content_digest(self, plaintext: str) -> str:
        """Keyed HMAC-SHA256 of the plaintext, prefixed ``hmac-sha256:``."""
        mac = hmac.new(self._content_digest_key(), plaintext.encode("utf-8"), hashlib.sha256)
        return HMAC_DIGEST_PREFIX + mac.hexdigest()
