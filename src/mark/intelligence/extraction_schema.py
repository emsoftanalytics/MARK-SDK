# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MARK-owned schema, prompt and alias tables for LLM-based graph extraction.

Separate from LLMContextualCompressor (post-retrieval compression).

Flow::

    observation text
      -> DeterministicExtractor  (always, free)
      -> LLMStructuredExtractor  (optional, developer LLM)
      -> ExtractionMerger        (dedup + combine)
      -> nodes / edges / tags    (stored in memory graph)
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

_VALID_ENTITY_TYPES = Literal[
    "character", "object", "location", "scene",
    "concept", "event", "agent", "entity",
]

_VALID_RELATIONS = Literal[
    "owns", "appears_in", "located_at", "related_to",
    "symbolizes", "causes", "part_of", "supports", "references", "mentions",
]


class ExtractedEntity(BaseModel):
    """One entity found in an observation."""
    model_config = ConfigDict(extra="ignore")

    type:       _VALID_ENTITY_TYPES
    label:      str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    """One directed relationship between two entities."""
    model_config = ConfigDict(extra="ignore")

    source:     str
    relation:   _VALID_RELATIONS
    target:     str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StructuredExtraction(BaseModel):
    """Full structured output from LLMStructuredExtractor.

    Validated against MARK's schema before merging with deterministic results.
    """
    model_config = ConfigDict(extra="ignore")

    entities:  list[ExtractedEntity]  = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    tags:      list[str]              = Field(default_factory=list)
    summary:   str | None             = None


# ── Extraction prompt ─────────────────────────────────────────────────────────
# Braces are doubled because the prompt is filled with str.format(text=...).

_STRUCTURED_PROMPT = """\
Extract MARK memory entities and relationships from the observation text below.
Return ONLY valid JSON — no explanation, no markdown code blocks.

Required JSON format:
{{
  "entities":  [{{"type": "character", "label": "Elena",          "confidence": 0.95}},
                {{"type": "object",    "label": "Red Scarf",      "confidence": 0.9}},
                {{"type": "location",  "label": "North Warehouse","confidence": 0.9}}],
  "relations": [{{"source": "Elena", "relation": "owns",       "target": "Red Scarf",      "confidence": 0.9}},
                {{"source": "Elena", "relation": "located_at", "target": "North Warehouse","confidence": 0.9}}],
  "tags":    ["character:elena", "object:red-scarf", "location:north-warehouse"],
  "summary": "One-line summary or null."
}}

Valid entity types : character, object, location, scene, concept, event, agent, entity
Valid relations    : owns, appears_in, located_at, related_to, symbolizes, causes,
                    part_of, supports, references, mentions

Rules:
- Only extract what is explicitly stated in the text.
- confidence: 0.0–1.0 (how certain you are)
- tags: "type:slug" format, lowercase, spaces as hyphens
- If nothing found, return: {{"entities": [], "relations": [], "tags": [], "summary": null}}

Observation: {text}"""


_ENTITY_TYPES: frozenset[str] = frozenset(get_args(_VALID_ENTITY_TYPES))
_RELATION_TYPES: frozenset[str] = frozenset(get_args(_VALID_RELATIONS))

_ENTITY_TYPE_ALIASES: dict[str, str] = {
    "person": "character", "people": "character", "human": "character",
    "character_name": "character", "actor": "character",
    "place": "location", "loc": "location", "venue": "location", "setting": "location",
    "thing": "object", "item": "object", "prop": "object", "artifact": "object", "tool": "object",
    "idea": "concept", "topic": "concept", "theme": "concept",
    "organization": "entity", "organisation": "entity", "org": "entity", "company": "entity",
    "named_entity": "entity",
    "episode": "scene", "shot": "scene",
    "action": "event", "incident": "event",
    "bot": "agent", "assistant": "agent",
}

_RELATION_ALIASES: dict[str, str] = {
    "has": "owns", "have": "owns", "possesses": "owns", "holds": "owns", "carries": "owns",
    "wears": "owns", "owned_by": "owns",
    "located_in": "located_at", "is_at": "located_at", "at": "located_at", "in": "located_at",
    "lives_in": "located_at", "enters": "located_at",
    "appears": "appears_in", "featured_in": "appears_in",
    "related": "related_to", "associated_with": "related_to", "linked_to": "related_to",
    "represents": "symbolizes", "symbolises": "symbolizes",
    "caused": "causes", "leads_to": "causes",
    "belongs_to": "part_of", "member_of": "part_of",
    "supported_by": "supports", "refers_to": "references", "cites": "references",
    "mention": "mentions", "mentioned": "mentions",
}
