# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The date grammar, query anchors and extraction patterns compile no regex at import time."""
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
    if isinstance(pattern, str) and name.endswith(("date_parsing.py", "query_anchors.py", "extraction_patterns.py")):
        seen.append(name)
    return original(pattern, flags)
re._compile = counting
assert "mark" not in sys.modules
import mark.intelligence.date_parsing, mark.intelligence.query_anchors, mark.intelligence.extraction_patterns
print(len(seen), sorted(set(seen))[:3])
"""


def test_no_regex_is_compiled_when_the_modules_are_imported() -> None:
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    assert out.startswith("0 "), out


def test_lazy_patterns_still_parse() -> None:
    from mark.intelligence.date_parsing import find_dates
    from mark.intelligence.query_anchors import query_anchors

    assert [span.text for span in find_dates("on 5 March 2026 and last week")] == ["5 March 2026", "last week"]
    assert query_anchors("Is atlas-core in eu-west-1?") == ["atlas core", "eu west 1"]
