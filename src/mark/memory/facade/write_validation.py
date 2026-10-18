# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Up-front validation for memory writes, and the bulk-write report.

Every write checks its input *before* anything is embedded or persisted, and
reports problems as :class:`mark.errors.InvalidMemoryError` (a ``MarkError``
and a ``ValueError``) instead of raw pydantic, JSON or Unicode errors from deep
inside the stack:

* content must be ``str`` — ``bytes`` are refused (never silently coerced to
  ``"b'...'"``), as are ``None``, numbers, dicts and paths;
* content must not be blank and must be valid Unicode text — a lone
  surrogate (e.g. from a broken decode) cannot be stored in SQLite or hashed;
* content must fit ``max_memory_bytes`` (UTF-8, default 1 MB);
* metadata must be a JSON-serialisable dict, tags a list of strings.

:class:`StoreReport` is what ``store_many_sync`` returns: a ``list`` of ids
in input order (``None`` where an item was skipped) plus the reasons.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Iterable

from mark.errors import InvalidMemoryError, MemoryTooLargeError

#: Default maximum size of one memory (UTF-8 bytes). Longer text belongs in
#: store_document(), which chunks it into retrievable passages.
DEFAULT_MAX_MEMORY_BYTES = 1_000_000


def check_content(content: Any, max_bytes: int | None = DEFAULT_MAX_MEMORY_BYTES) -> str:
    """Return *content* if it is storable text, else raise InvalidMemoryError."""
    if isinstance(content, (bytes, bytearray, memoryview)):
        raise InvalidMemoryError(
            "memory content must be text (str), got bytes; decode it first, "
            "e.g. data.decode('utf-8')", reason="not_text",
        )
    if isinstance(content, PurePath):
        raise InvalidMemoryError(
            f"memory content must be text (str), got a path ({content}); to remember a file use "
            "memory.remember_file(path) (text, Markdown, code, HTML, PDF, Office, images, audio)",
            reason="not_text",
        )
    if not isinstance(content, str):
        kind = "None" if content is None else type(content).__name__
        raise InvalidMemoryError(f"memory content must be text (str), got {kind}", reason="not_text")
    if not content.strip():
        raise InvalidMemoryError("memory content must not be empty or blank", reason="empty")
    if content.isascii():  # fast path: ASCII is always valid and 1 byte per char
        size = len(content)
    else:
        try:
            size = len(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvalidMemoryError(
                f"memory content is not valid Unicode text (lone surrogate {content[exc.start]!r} "
                f"at position {exc.start}); it usually comes from a broken decode — "
                "re-decode the source with errors='replace'", reason="invalid_unicode",
            ) from None
    if max_bytes is not None and size > max_bytes:
        raise MemoryTooLargeError(size, max_bytes)
    return content


def check_metadata(metadata: Any) -> dict[str, Any] | None:
    """Return *metadata* if it is a JSON-serialisable dict (or None), else raise."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise InvalidMemoryError(
            f"metadata must be a dict, got {type(metadata).__name__}", reason="metadata",
        )
    try:
        # Same encoder the store uses, so anything accepted here persists later.
        json.dumps(metadata, allow_nan=True)
    except (TypeError, ValueError) as exc:
        raise InvalidMemoryError(
            f"metadata must be JSON-serialisable (str, int, float, bool, None, list, dict): {exc}",
            reason="metadata",
        ) from None
    return metadata


def check_tags(tags: Any) -> list[str] | None:
    """Return *tags* if it is a list/tuple of strings (or None), else raise."""
    if tags is None:
        return None
    if isinstance(tags, str) or not isinstance(tags, (list, tuple)) or \
            not all(isinstance(tag, str) for tag in tags):
        raise InvalidMemoryError("tags must be a list of strings", reason="tags")
    return list(tags)


def check_number(name: str, value: Any) -> None:
    """Importance/confidence must be a finite number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InvalidMemoryError(f"{name} must be a number between 0 and 1, got {value!r}", reason=name)


@dataclass(frozen=True)
class SkippedItem:
    """One input of a bulk write that was not stored, and why."""

    index: int
    reason: str
    message: str


class StoreReport(list):
    """Result of ``store_many_sync``: ids in input order, ``None`` where skipped.

    A plain ``list`` subclass, so code that treats the result as a list of
    ids keeps working when nothing is skipped. Extra attributes:

    * ``skipped`` — :class:`SkippedItem` per refused input (index, reason, message);
    * ``stored`` — how many inputs are in the store (including ones an
      idempotent retry found already stored);
    * ``existing`` — how many of those were already stored by an earlier call
      with the same ``idempotency_key``;
    * ``ids`` — only the stored ids, in input order.
    """

    def __init__(self, ids: Iterable[str | None] = (), skipped: Iterable[SkippedItem] = (),
                 existing: int = 0) -> None:
        super().__init__(ids)
        self.skipped: list[SkippedItem] = list(skipped)
        self.existing = existing

    @property
    def stored(self) -> int:
        return sum(1 for fragment_id in self if fragment_id is not None)

    @property
    def ids(self) -> list[str]:
        return [fragment_id for fragment_id in self if fragment_id is not None]

    def __repr__(self) -> str:
        return (f"StoreReport({list.__repr__(self)}, stored={self.stored}, "
                f"skipped={len(self.skipped)}, existing={self.existing})")
