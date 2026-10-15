"""MARK exception hierarchy.

Standard library only, so any module (including the CLI) can import it
cheaply. Store-encryption errors also subclass :class:`RuntimeError`, so code
that already catches ``RuntimeError`` (such as the ``mark`` command line)
reports them cleanly instead of crashing.
"""
from __future__ import annotations

__all__ = [
    "EncryptionKeyError",
    "EncryptionRequiredError",
    "InvalidMemoryError",
    "MarkError",
    "MemoryTooLargeError",
    "PlaintextStoreError",
    "StoreClosedError",
    "StoreEncryptionError",
    "StoreVersionError",
]

# How to supply a key, repeated in every encryption error message.
_KEY_HINT = (
    "Pass the provider it was created with, e.g. "
    "Mark.local(folder, encryption=FernetEncryptionProvider(passphrase=...)) or "
    "MarkRuntime.local(store_path=..., encryption=FernetEncryptionProvider.from_env()). "
    "The `mark` command line cannot open encrypted stores."
)


class MarkError(Exception):
    """Base class for errors raised by MARK itself."""


class StoreEncryptionError(MarkError, RuntimeError):
    """The store's at-rest encryption does not match the provider supplied."""


class EncryptionRequiredError(StoreEncryptionError):
    """The store holds encrypted content but was opened without a key."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"MARK store {path!r} is encrypted and was opened without an encryption "
            f"provider; refusing to read or write it. {_KEY_HINT}"
        )


class EncryptionKeyError(StoreEncryptionError):
    """The supplied key or passphrase cannot decrypt the store."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"MARK store {path!r} is encrypted with a different key or passphrase; "
            f"the supplied encryption provider cannot decrypt it. {_KEY_HINT}"
        )


class PlaintextStoreError(StoreEncryptionError):
    """An encryption provider was supplied for a store that already holds plaintext."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"MARK store {path!r} already holds unencrypted memories; refusing to open it "
            "with an encryption provider because that would mix plaintext and ciphertext. "
            "Open it without encryption=, or start a new (empty) store for encrypted memory."
        )


class StoreVersionError(MarkError, RuntimeError):
    """The store was written by a newer MARK whose schema this version cannot read safely."""

    def __init__(self, path: str, stored: int, supported: int) -> None:
        super().__init__(
            f"MARK store {path!r} uses schema version {stored}, but this mark-sdk supports "
            f"up to version {supported}; refusing to open it so it is not damaged. "
            "Upgrade mark-sdk (pip install -U mark-sdk) to use this store."
        )


# ── API input and lookup errors (W6: API, lifecycle and CLI) ─────────────────
# Each also subclasses the builtin callers historically caught (ValueError,
# KeyError, RuntimeError), so existing ``except ValueError`` code keeps working.

__all__ += [
    "EmbedderUnavailableError",
    "InvalidInputError",
    "MemoryNotFoundError",
    "UnknownBlockError",
]


class InvalidInputError(MarkError, ValueError):
    """A value passed to a MARK API is invalid (wraps pydantic validation errors)."""

    @classmethod
    def from_validation(cls, exc: Exception, what: str = "memory") -> "InvalidInputError":
        """Turn a pydantic ``ValidationError`` into a one-line, field-named MARK error."""
        errors = getattr(exc, "errors", None)
        parts: list[str] = []
        if callable(errors):
            for item in errors():
                field = ".".join(str(loc) for loc in item.get("loc", ())) or what
                parts.append(f"{field}: {item.get('msg', 'invalid value')}")
        return cls(f"invalid {what}: " + ("; ".join(parts) or str(exc)))


class UnknownBlockError(InvalidInputError):
    """``blocks=[...]`` named a block that has never been created or written."""

    def __init__(self, unknown: list[str], known: list[str]) -> None:
        listed = ", ".join(sorted(known)) or "(none yet)"
        super().__init__(
            f"unknown block(s) {', '.join(repr(name) for name in unknown)}; known blocks: {listed}. "
            "Create one with memory.block(name) or write to it first."
        )


class MemoryNotFoundError(MarkError, KeyError):
    """No memory with the given id exists for this agent."""

    def __init__(self, memory_id: str) -> None:
        super().__init__(memory_id)
        self.memory_id = memory_id

    def __str__(self) -> str:
        return f"no memory with id {self.memory_id!r}"


class EmbedderUnavailableError(MarkError, RuntimeError):
    """An embedding model could not be loaded or downloaded."""


# ── Store lifecycle and write-path errors (W1: durability) ──────────────────

class StoreClosedError(MarkError, RuntimeError):
    """The store (or the runtime that owns it) was closed and cannot be used again."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"MARK store {path!r} is closed; open a new one "
            "(e.g. MarkRuntime.local(store_path=...)) instead of reusing it after shutdown()/close()."
        )


class InvalidMemoryError(InvalidInputError):
    """A value passed to a memory write is not a storable memory.

    An :class:`InvalidInputError` (and so a :class:`ValueError`), so callers that already catch ``ValueError``
    keep working. ``reason`` is a short machine-friendly code such as
    ``"empty"``, ``"not_text"``, ``"invalid_unicode"``, ``"too_large"`` or
    ``"metadata"``.
    """

    def __init__(self, message: str, *, reason: str = "invalid") -> None:
        super().__init__(message)
        self.reason = reason


class MemoryTooLargeError(InvalidMemoryError):
    """A single memory exceeds the configured maximum write size."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(
            f"memory is {size:,} bytes, over the {limit:,}-byte limit for one memory; "
            "split it or use store_document(...) for long text "
            "(or raise max_memory_bytes if you really need one huge memory)",
            reason="too_large",
        )
        self.size = size
        self.limit = limit
