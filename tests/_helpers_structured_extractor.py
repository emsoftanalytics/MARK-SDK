# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared LLM stub for the structured-extraction test modules."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# _parse_structured_response
# ---------------------------------------------------------------------------

_VALID_JSON = """{
  "entities": [
    {"type": "character", "label": "Elena",          "confidence": 0.95},
    {"type": "object",    "label": "Red Scarf",      "confidence": 0.90},
    {"type": "location",  "label": "North Warehouse","confidence": 0.90}
  ],
  "relations": [
    {"source": "Elena", "relation": "owns",       "target": "Red Scarf",      "confidence": 0.90},
    {"source": "Elena", "relation": "located_at", "target": "North Warehouse","confidence": 0.90}
  ],
  "tags": ["character:elena", "object:red-scarf", "location:north-warehouse"],
  "summary": "Elena enters the North Warehouse."
}"""


# ---------------------------------------------------------------------------
# LLMStructuredExtractor
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response
