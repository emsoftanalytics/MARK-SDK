# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# observe() result types and lightweight local entity extraction.
#
# Extraction is intentionally minimal — a best-effort parse of LLM output.
# It never raises on parse failure; on any error it returns empty lists so
# observe() always succeeds as a plain store_sync() call.
"""Observation auto-structuring results and observability events."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.types import MemoryEdge, MemoryNode
    from mark.types.llm import LLMProvider


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObserveResult:
    """Result of a mark.observe() call.

    fragment_id — the stored fragment ID
    content     — the original observed text
    session_id  — session this observation was tagged to (may be None)
    nodes       — nodes created or retrieved during this observation
    edges       — edges created during this observation
    auto_tags   — tags automatically generated from extracted entities
    inferred    — True if any extraction (deterministic or LLM) found entities
    """
    fragment_id: str
    content:     str
    session_id:  str | None
    nodes:       list["MemoryNode"] = field(default_factory=list)
    edges:       list["MemoryEdge"] = field(default_factory=list)
    auto_tags:   list[str]          = field(default_factory=list)
    inferred:    bool = False


@dataclass(frozen=True)
class ObserveEvent:
    """Event emitted through HOOK_OBSERVE_EVENT after every observe() call.

    The local SDK fires this with no registered handler (no-op).
    MARK Cloud registers a handler to collect events for the Memory Observatory.

    Schema is intentionally flat so it can be serialised and shipped cheaply.
    """
    agent_id:     str
    fragment_id:  str
    content:      str
    session_id:   str | None
    node_labels:  list[str]                    # entity labels extracted
    edge_count:   int                          # edges created
    inferred:     bool                         # True if any extraction found entities
    auto_tags:    list[str] = field(default_factory=list)  # tags auto-applied to fragment
    timestamp:    datetime  = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata:     dict[str, Any] = field(default_factory=dict)


# ── Extraction prompt and parser ──────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
Extract named entities and relationships from the text below.

Text: {text}

Output format — two sections, exactly as shown:

ENTITIES:
<type>: <name>
(valid types: character, object, location, scene, concept, event, agent)

RELATIONS:
<source name> -> <relation> -> <target name>
(valid relations: owns, appears_in, located_at, causes, part_of, related_to, supports, references, mentions)

Rules:
- Only extract what is explicitly stated in the text.
- Each entity on its own line; each relation on its own line.
- If no entities are found, write: ENTITIES: NONE
- If no relations are found, write: RELATIONS: NONE
- Do not add explanations or extra text.
"""

_TYPE_MAP: dict[str, str] = {
    "character": "character",
    "object":    "object",
    "location":  "location",
    "scene":     "scene",
    "concept":   "concept",
    "event":     "event",
    "agent":     "agent",
    "entity":    "entity",   # fallback alias
    "person":    "character",
    "place":     "location",
    "thing":     "object",
}

_RELATION_MAP: dict[str, str] = {
    "owns":         "owns",
    "appear_in":    "appears_in",
    "appears_in":   "appears_in",
    "located_at":   "located_at",
    "located at":   "located_at",
    "causes":       "causes",
    "part_of":      "part_of",
    "part of":      "part_of",
    "related_to":   "related_to",
    "related to":   "related_to",
    "supports":     "supports",
    "references":   "references",
    "mentions":     "mentions",
    "wears":        "owns",
    "carries":      "owns",
    "holds":        "owns",
    "enters":       "located_at",
    "is in":        "located_at",
    "at":           "located_at",
}


def _parse_extraction(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Parse LLM extraction output.

    Returns:
        entities  — list of (node_type_str, label)
        relations — list of (label_a, relation_str, label_b)
    Never raises; returns empty lists on any parse failure.
    """
    entities:  list[tuple[str, str]]        = []
    relations: list[tuple[str, str, str]]   = []

    try:
        in_entities = False
        in_relations = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("ENTITIES:"):
                in_entities  = True
                in_relations = False
                remainder = line[len("ENTITIES:"):].strip()
                if remainder and remainder.upper() != "NONE":
                    line = remainder
                else:
                    continue
            elif upper.startswith("RELATIONS:"):
                in_relations = True
                in_entities  = False
                remainder = line[len("RELATIONS:"):].strip()
                if remainder and remainder.upper() != "NONE":
                    line = remainder
                else:
                    continue

            if line.upper() == "NONE":
                continue

            if in_entities and ":" in line:
                parts = line.split(":", 1)
                type_raw = parts[0].strip().lower()
                label    = parts[1].strip()
                if label and type_raw in _TYPE_MAP:
                    entities.append((_TYPE_MAP[type_raw], label))

            elif in_relations and "->" in line:
                parts = [p.strip() for p in line.split("->")]
                if len(parts) == 3:
                    src, rel_raw, tgt = parts
                    rel = _RELATION_MAP.get(rel_raw.lower())
                    if src and tgt and rel:
                        relations.append((src, rel, tgt))

    except Exception:  # never propagate parse errors
        pass

    return entities, relations


def extract_entities(
    text: str,
    llm: "LLMProvider",
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Run LLM extraction. Returns (entities, relations) or ([], []) on failure."""
    try:
        prompt   = _EXTRACTION_PROMPT.format(text=text)
        response = llm.complete(prompt)
        return _parse_extraction(response)
    except Exception:
        return [], []
