# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared fakes for the test_compressor* modules."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# LLMContextualCompressor
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response
