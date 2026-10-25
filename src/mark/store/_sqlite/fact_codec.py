"""Typed fact values: what is stored in ``facts.value`` and how it is read back.

Text stays text, so rows written by earlier versions (always plain strings)
read back unchanged and older readers still see strings. Every other JSON
type — int, float, bool, None, list (tuples become lists), dict — is stored
as a marker followed by canonical JSON (sorted keys), and comes back as the
original Python type: ``facts.set("svc", "port", 4444)`` returns ``4444``,
not ``"4444"``. Other objects (datetime, Decimal, Path, ...) are stored as
``str(value)``, exactly as before typed values existed.

Encoded strings are compared to decide "same value" (re-asserting a value is
a no-op), so ``4444`` and ``"4444"`` are different values.
"""
from __future__ import annotations

import json
from typing import Any

from mark.errors import InvalidMemoryError

# Starts with a control character no real text value begins with.
_TYPED = "\x1fmark-json:"
_JSON_TYPES = (bool, int, float, list, tuple, dict)


def encode_value(value: Any) -> str:
    """Encode a fact value for storage (before encryption)."""
    if isinstance(value, str) and not value.startswith(_TYPED):
        text = value
    elif value is None or isinstance(value, (str, *_JSON_TYPES)):
        try:
            text = _TYPED + json.dumps(value, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise InvalidMemoryError(f"fact value must be JSON-serialisable: {exc}", reason="fact_value") from None
    else:
        text = str(value)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidMemoryError("fact value is not valid Unicode text (lone surrogate)",
                                 reason="invalid_unicode") from None
    return text


def decode_value(text: str) -> Any:
    """Inverse of :func:`encode_value`; legacy plain strings are returned as-is."""
    if text.startswith(_TYPED):
        try:
            return json.loads(text[len(_TYPED):])
        except ValueError:  # never written by MARK; keep the raw text rather than fail a read
            return text
    return text


def value_text(value: Any) -> str:
    """Human/LLM rendering: text as-is, numbers plainly, the rest as JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
