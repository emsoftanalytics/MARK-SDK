# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared LLM stubs and canned extraction responses for the test_observe_* modules."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Minimal LLM stubs
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Returns a hardcoded extraction response for any prompt."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


# JSON format used by LLMStructuredExtractor in observe() integration tests
_ELENA_JSON_RESPONSE = """\
{
  "entities": [
    {"type": "character", "label": "Elena",          "confidence": 0.95},
    {"type": "object",    "label": "Red Scarf",      "confidence": 0.90},
    {"type": "location",  "label": "North Warehouse","confidence": 0.90}
  ],
  "relations": [
    {"source": "Elena",     "relation": "owns",       "target": "Red Scarf",      "confidence": 0.90},
    {"source": "Elena",     "relation": "located_at", "target": "North Warehouse","confidence": 0.90},
    {"source": "Red Scarf", "relation": "appears_in", "target": "North Warehouse","confidence": 0.85}
  ],
  "tags": ["character:elena", "object:red-scarf", "location:north-warehouse"],
  "summary": "Elena enters the North Warehouse wearing the red scarf."
}"""


_EMPTY_RESPONSE = """\
ENTITIES: NONE
RELATIONS: NONE
"""


_MALFORMED_RESPONSE = "I don't know what you mean. Here is some random text."
