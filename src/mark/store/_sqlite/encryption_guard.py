"""Refuse to open a store with the wrong (or missing) encryption provider.

Without this guard an encrypted store opened without its key decoded
ciphertext as UTF-8 (UnicodeDecodeError on recall), rebuilt a junk FTS index,
and — worst — accepted new writes in plaintext, producing a mixed store.

State lives in two ``schema_info`` keys (no schema-version bump needed):

``encrypted_at_rest``  "1" encrypted, "0" plaintext. Set when the store is
                       created; for older stores without it, decided once by
                       sniffing stored content (Fernet tokens / non-UTF-8).
``encryption_check``   base64 ciphertext of a fixed canary, written with the
                       store's provider, so a wrong key fails fast and clearly.

Rules: an encrypted store needs a provider that decrypts the canary; a
plaintext store that already holds data refuses an encryption provider (no
mixed stores); an empty plaintext store may be switched to encrypted.
"""
from __future__ import annotations

import base64
import binascii
import sqlite3

from mark.errors import EncryptionKeyError, EncryptionRequiredError, PlaintextStoreError, StoreEncryptionError
from mark.security import EncryptionProvider, NoOpEncryptionProvider
from mark.store._sqlite.codec import _as_bytes

_MARKER_KEY = "encrypted_at_rest"
_CHECK_KEY = "encryption_check"
_CANARY = "mark-encryption-check-v1"
# Rows sampled when sniffing a legacy store; enough to catch a mixed store.
_SNIFF_LIMIT = 50
# Fernet tokens: version byte 0x80, 8-byte timestamp, 16-byte IV, >=1 block, 32-byte HMAC.
_FERNET_MIN_LEN = 1 + 8 + 16 + 16 + 32


def _is_fernet_token(data: bytes) -> bool:
    if not data.startswith(b"gAAAAA"):
        return False
    try:
        raw = base64.urlsafe_b64decode(data)
    except (binascii.Error, ValueError):
        return False
    return len(raw) >= _FERNET_MIN_LEN and raw[0] == 0x80


def _looks_encrypted(data: bytes) -> bool:
    """Fernet token, passphrase-mode (16-byte salt + token), or non-UTF-8 bytes."""
    if _is_fernet_token(data) or _is_fernet_token(data[16:]):
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True  # custom providers: plaintext rows are always valid UTF-8
    return False


class EncryptionGuardMixin:
    """Validates the encryption provider against the store when it opens."""

    db_path: str
    encryption: EncryptionProvider

    @property
    def _encrypting(self) -> bool:
        return not isinstance(self.encryption, NoOpEncryptionProvider)

    def _init_encryption(self, conn: sqlite3.Connection) -> StoreEncryptionError | None:
        """Check (and persist) the store's encryption state inside ``_init_schema``.

        Returns the error to raise instead of raising it, so markers decided
        by sniffing are still committed before the open is refused.
        """
        row = conn.execute("SELECT value FROM schema_info WHERE key = ?", (_MARKER_KEY,)).fetchone()
        samples = self._encrypted_samples(conn)
        if row is not None:
            encrypted = str(row[0]) == "1"
        elif samples is None:
            encrypted = self._encrypting  # brand-new (or empty) store: adopt the provider
        else:
            encrypted = bool(samples)  # pre-marker store with data: sniff it
        if not encrypted and self._encrypting:
            if samples is not None:
                self._set(conn, _MARKER_KEY, "0")
                return PlaintextStoreError(self.db_path)
            encrypted = True  # empty plaintext store may switch to encrypted
        self._set(conn, _MARKER_KEY, "1" if encrypted else "0")
        if not encrypted:
            return None
        if not self._encrypting:
            return EncryptionRequiredError(self.db_path)
        return self._verify_key(conn, samples or [])

    def _verify_key(self, conn: sqlite3.Connection, samples: list[bytes]) -> StoreEncryptionError | None:
        check = conn.execute("SELECT value FROM schema_info WHERE key = ?", (_CHECK_KEY,)).fetchone()
        if check is not None:
            try:
                ok = self.encryption.decrypt_str(base64.b64decode(str(check[0]))) == _CANARY
            except Exception:  # noqa: BLE001 - any decrypt failure means a wrong key
                ok = False
            return None if ok else EncryptionKeyError(self.db_path)
        # Legacy encrypted store (no canary yet): its own rows must decrypt.
        for data in samples:
            try:
                self.encryption.decrypt(data)
            except Exception:  # noqa: BLE001
                return EncryptionKeyError(self.db_path)
        canary = base64.b64encode(self.encryption.encrypt_str(_CANARY)).decode("ascii")
        self._set(conn, _CHECK_KEY, canary)
        return None

    @staticmethod
    def _encrypted_samples(conn: sqlite3.Connection) -> list[bytes] | None:
        """None when the store holds no encrypted-column data; else the rows that look encrypted."""
        rows = conn.execute(
            "SELECT content FROM fragments LIMIT ?", (_SNIFF_LIMIT,)
        ).fetchall() + conn.execute(
            "SELECT value FROM facts LIMIT ?", (_SNIFF_LIMIT,)
        ).fetchall()
        if not rows:
            return None
        return [data for data in (_as_bytes(row[0]) for row in rows) if _looks_encrypted(data)]

    @staticmethod
    def _set(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
