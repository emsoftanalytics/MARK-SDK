"""Character-budget helpers for fitting context into a window."""
from __future__ import annotations


def within_char_budget(text: str, *, max_chars: int) -> bool:
    """Return the text truncated to the character budget."""
    return len(text) <= max(0, max_chars)
