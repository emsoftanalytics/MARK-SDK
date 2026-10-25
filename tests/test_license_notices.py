# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Every source file names the copyright holder, and LICENSE/NOTICE agree."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDER = "Somtochukwu Owen Ikeanyi"
STALE = "MARK " + "Contributors"  # built so this file does not match itself


def test_license_and_notice_name_the_holder() -> None:
    assert f"Copyright 2025-2026 {HOLDER}" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = " ".join((ROOT / "NOTICE").read_text(encoding="utf-8").split())
    assert HOLDER in notice and "Apache License, Version 2.0" in notice


def test_no_source_file_names_an_anonymous_holder() -> None:
    stale = [str(path.relative_to(ROOT)) for folder in ("src", "tests", "examples", "benchmarks")
             for path in (ROOT / folder).rglob("*.py")
             if STALE in path.read_text(encoding="utf-8", errors="replace")]
    assert stale == [], f"use the project header (see CONTRIBUTING.md): {stale[:5]}"
