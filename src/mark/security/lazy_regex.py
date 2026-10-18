# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Regexes compiled on first use instead of at import time.

MARK's redaction, injection screening and fact extraction hold a few
hundred module-level patterns. Compiling them all when a module is
imported cost ~0.1 s on every CLI start, even for commands that never
redact or extract. :func:`lazy_compile` returns a stand-in that compiles
the pattern the first time any attribute (``match``, ``sub``,
``groupindex``, ...) is used and then behaves exactly like the compiled
``re.Pattern``.

Not a ``re.Pattern`` subclass: pass ``.compiled`` to an API that insists
on a real pattern object (``re.sub(lazy, ...)`` is not supported; call
``lazy.sub(...)`` instead).
"""
from __future__ import annotations

import re
from typing import Any


class LazyPattern:
    """A regex compiled on first use (see module docstring)."""

    __slots__ = ("_source", "_flags", "_compiled")

    def __init__(self, pattern: str, flags: int = 0) -> None:
        self._source = pattern
        self._flags = flags
        self._compiled: re.Pattern[str] | None = None

    @property
    def compiled(self) -> "re.Pattern[str]":
        """The real compiled pattern (compiled now if needed; a race only compiles twice)."""
        if self._compiled is None:
            self._compiled = re.compile(self._source, self._flags)
        return self._compiled

    def __getattr__(self, name: str) -> Any:
        return getattr(self.compiled, name)

    def __repr__(self) -> str:
        return f"LazyPattern({self._source[:40]!r})"


def lazy_compile(pattern: str, flags: int = 0) -> LazyPattern:
    """Like :func:`re.compile`, but the work happens on first use."""
    return LazyPattern(pattern, flags)


__all__ = ["LazyPattern", "lazy_compile"]
