# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Redaction, injection screening and fact extraction compile no regex at import time.

Their few hundred patterns are compiled on first use (mark.security.lazy_regex),
so CLI start-up does not pay for them.
"""
from __future__ import annotations

import subprocess
import sys

_PROBE = r"""
import re, sys
seen = []
original = re._compile
def counting(pattern, flags):
    frame = sys._getframe(1)
    while frame and frame.f_code.co_filename.endswith("re/__init__.py"):
        frame = frame.f_back
    name = frame.f_code.co_filename if frame else ""
    if isinstance(pattern, str) and any(part in name for part in (
            "mark/security/", "mark/intelligence/fact_extraction/", "mark/documents/", "mark/media/",
            "mark/middlewares/governance/")):
        seen.append(name)
    return original(pattern, flags)
re._compile = counting
import mark.security, mark.intelligence.fact_extraction, mark.documents, mark.middlewares.governance
print(len(seen), sorted(set(seen))[:5])
"""


def test_no_regex_is_compiled_when_the_modules_are_imported() -> None:
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    assert out.startswith("0 "), out


def test_lazy_patterns_behave_like_compiled_ones() -> None:
    from mark.security import redact
    from mark.security.lazy_regex import lazy_compile

    pattern = lazy_compile(r"(?P<n>\d+)")
    assert pattern.sub("#", "a1b22") == "a#b#" and pattern.groupindex == {"n": 1}
    assert "[SECRET:" in redact("password: Tr0ub4dor&3").text
