"""Regular expressions compiled on first use, for a cheap cold import.

Modules on the CLI's import path declared dozens of patterns at module
level; compiling them all (the date grammar's month/weekday alternations
in particular) cost noticeable start-up time for commands that never parse
a date. ``lazy(pattern, flags)`` keeps the module-level name and its call
sites unchanged (``_RE.search(text)``) and compiles once, on the first call.
"""
from __future__ import annotations

import re
from typing import Any


class LazyPattern:
    """A stand-in for ``re.compile(pattern, flags)`` that compiles on first attribute access."""

    __slots__ = ("_source", "_flags", "_compiled")

    def __init__(self, pattern: str, flags: int = 0) -> None:
        self._source = pattern
        self._flags = flags
        self._compiled: re.Pattern[str] | None = None

    def compiled(self) -> re.Pattern[str]:
        """The compiled pattern (compiled now if it was not yet)."""
        if self._compiled is None:
            self._compiled = re.compile(self._source, self._flags)
        return self._compiled

    def __getattr__(self, name: str) -> Any:
        # search / match / fullmatch / findall / finditer / sub / split / pattern / flags ...
        return getattr(self.compiled(), name)

    def __repr__(self) -> str:
        return f"LazyPattern({self._source!r})"


def lazy(pattern: str, flags: int = 0) -> LazyPattern:
    """Declare a module-level regex without paying for its compilation at import."""
    return LazyPattern(pattern, flags)
