"""Swappable at-rest encryption providers (NoOp default, optional Fernet)."""
from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod

_CRYPTO_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - exercised only when optional crypto is installed.
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except Exception as exc:  # pragma: no cover
    Fernet = None  # type: ignore[assignment]
    hashes = None  # type: ignore[assignment]
    PBKDF2HMAC = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc


def _require_cryptography() -> None:
    if _CRYPTO_IMPORT_ERROR is not None:
        raise ImportError(
            "FernetEncryptionProvider requires the optional dependency "
            "`cryptography`. Install `mark[crypto]` or use "
            "NoOpEncryptionProvider for local development."
        ) from _CRYPTO_IMPORT_ERROR


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


class NoOpEncryptionProvider(EncryptionProvider):
    """Default local mode: no encryption, no optional dependencies."""

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        return data

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        return data


class FernetEncryptionProvider(EncryptionProvider):
    """Optional Fernet encryption provider for local memory at rest."""

    _SALT_SIZE = 16
    _ITERATIONS = 390_000

    def __init__(self, key: str | bytes | None = None, passphrase: str | None = None) -> None:
        _require_cryptography()
        if key is not None:
            if isinstance(key, str):
                key = key.encode("utf-8")
            self._fernet = Fernet(key)  # type: ignore[operator]
            self._passphrase = None
        elif passphrase is not None:
            self._fernet = None
            self._passphrase = passphrase.encode("utf-8")
        else:
            self._fernet = Fernet(Fernet.generate_key())  # type: ignore[operator,union-attr]
            self._passphrase = None

    @classmethod
    def from_env(cls, env_var: str = "MARK_ENCRYPTION_KEY") -> "FernetEncryptionProvider":
        """Construct from an environment variable."""
        key = os.environ.get(env_var)
        if not key:
            raise EnvironmentError(f"Encryption key env var {env_var!r} is not set.")
        return cls(key=key)

    def _derive(self, salt: bytes) -> "Fernet":
        _require_cryptography()
        kdf = PBKDF2HMAC(  # type: ignore[operator]
            algorithm=hashes.SHA256(),  # type: ignore[union-attr]
            length=32,
            salt=salt,
            iterations=self._ITERATIONS,
        )
        return Fernet(base64.urlsafe_b64encode(kdf.derive(self._passphrase)))  # type: ignore[operator]

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        if self._passphrase is not None:
            salt = os.urandom(self._SALT_SIZE)
            return salt + self._derive(salt).encrypt(data)
        return self._fernet.encrypt(data)  # type: ignore[union-attr]

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        if self._passphrase is not None:
            salt = data[: self._SALT_SIZE]
            token = data[self._SALT_SIZE :]
            return self._derive(salt).decrypt(token)
        return self._fernet.decrypt(data)  # type: ignore[union-attr]
