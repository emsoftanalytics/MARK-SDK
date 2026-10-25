# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""The ``mark.cli`` module docstring documents every command and option."""
from __future__ import annotations

import argparse

import mark.cli as cli


def _options(parser: argparse.ArgumentParser) -> set[str]:
    found = {opt for action in parser._actions for opt in action.option_strings if opt.startswith("--")}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                found.add(name)
                found |= _options(sub)
    return found - {"--help"}


def test_docstring_mentions_every_command_and_option() -> None:
    doc = cli.__doc__ or ""
    missing = sorted(name for name in _options(cli._parser()) if name not in doc)
    assert missing == []
    assert "fact retract" in doc and "--correction" in doc and "--extract-facts" in doc
    assert "memories and current facts" in doc
