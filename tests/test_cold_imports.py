# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Every module must import on its own in a fresh interpreter.

`import mark` loads public names lazily, so nothing hides circular imports
behind a lucky eager import order any more. This test imports each module
first-thing in its own process to catch cycles before users do.
"""
from __future__ import annotations

import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import mark

# Optional-dependency adapters may legitimately fail without their extras.
_OPTIONAL = ("mark.adapters.langchain", "mark.adapters.mcp", "mark.adapters.langgraph")


def _cold_import(module: str) -> tuple[str, str]:
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    if result.returncode == 0:
        return module, ""
    last = (result.stderr.strip().splitlines() or ["?"])[-1]
    if module.startswith(_OPTIONAL) and ("ModuleNotFoundError" in last or "ImportError" in last) and "circular" not in last:
        return module, ""
    return module, last


def test_every_module_imports_cold() -> None:
    modules = sorted(info.name for info in pkgutil.walk_packages(mark.__path__, "mark."))
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [(m, err) for m, err in pool.map(_cold_import, modules) if err]
    assert not failures, "\n".join(f"{m}: {err}" for m, err in failures)
