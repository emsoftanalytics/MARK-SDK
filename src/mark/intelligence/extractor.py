# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Local deterministic entity and relation extractor.
#
# Extracts named entities and relationships from observation text using
# pattern matching — no LLM required. Designed for any long-running agent
# that tracks named concepts: characters/people, objects/tools, locations/places,
# session/scene context. Works for media production, game narratives, legal cases,
# research sessions, engineering incident timelines, and any project where
# continuity across many observations matters.
#
# When an LLMProvider is configured, its output is additive on top of this.
"""Deterministic and optional LLM-assisted entity/relation extraction."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from mark.types.llm import LLMProvider


# ── Utilities ─────────────────────────────────────────────────────────────────

def _slug(label: str) -> str:
    """'North Warehouse' → 'north-warehouse'"""
    return re.sub(r'[\s_]+', '-', label.strip().lower())


def _title(raw: str) -> str:
    """'red scarf' → 'Red Scarf'"""
    return ' '.join(w.capitalize() for w in raw.strip().split())


# ── Vocabulary ─────────────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "it", "he", "she", "they", "we", "you", "i", "his", "her", "their",
    "this", "that", "these", "those", "into", "from", "by", "not", "as",
    "then", "when", "where", "who", "which", "what", "how", "there",
    "now", "also", "just", "still", "already", "very", "while", "after",
    "before", "during", "over", "under", "above", "below", "between",
    "through", "about", "up", "down", "out", "off", "again", "once",
})

# Verbs that signal a LOCATION follows (entered X, arrived at X, headed to X)
_LOCATION_VERB_RE = re.compile(
    r'\b(?:'
    r'enter(?:s|ed|ing)?'
    r'|walk(?:ed|ing|s)?\s+(?:into|to|toward)'
    r'|arrive(?:s|d|ing)?\s+(?:at|in)'
    r'|go(?:es|ing)?\s+(?:to|into)'
    r'|went\s+(?:to|into)'
    r'|head(?:s|ed|ing)?\s+(?:to|toward|for)'
    r'|move(?:s|d|ing)?\s+(?:to|toward|into)'
    r'|step(?:s|ped|ping)?\s+(?:into|inside|in)'
    r'|rush(?:es|ed|ing)?\s+(?:to|into|toward)'
    r'|run(?:s|ning)?\s+(?:to|into|toward)'
    r'|travel(?:s|led|ing)?\s+(?:to|toward)'
    r'|return(?:s|ed|ing)?\s+to'
    r'|approach(?:es|ed|ing)?'
    r')\b',
    re.IGNORECASE,
)

# After a LOCATION verb: match optional article + CAPITALISED proper-noun sequence.
# Stops naturally when it hits a lowercase word — so "North Warehouse wearing" stops
# at "wearing" since it's lowercase and not part of the capitalised sequence.
_LOCATION_NP_RE = re.compile(
    r'^(?:the\s+|an?\s+)?([A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*)',
)

# Verbs that signal an OBJECT follows (ownership / possession / use)
_OWN_VERB_RE = re.compile(
    r'\b(?:'
    r'wear(?:s|ing)?|wore'
    r'|carries?|carrying|carried'
    r'|hold(?:s|ing)?|held'
    r'|grip(?:s|ped|ping)?'
    r'|clutch(?:es|ed|ing)?'
    r'|own(?:s|ed|ing)?'
    r'|use(?:s|d|ing)?'
    r'|wield(?:s|ed|ing)?'
    r'|pick(?:s|ed|ing)?\s+up'
    r'|grab(?:s|bed|bing)?'
    r'|take(?:s|n)?|took'
    r'|pull(?:s|ed|ing)?\s+out'
    r')\b',
    re.IGNORECASE,
)

# After an OWN verb: match optional article + lowercase noun phrase (1-3 words).
# Objects are typically lowercase common nouns ("the red scarf", "a data file").
_OBJECT_NP_RE = re.compile(
    r'^(?:the\s+|an?\s+)?([a-z][a-zA-Z-]+(?:\s+[a-z][a-zA-Z-]+){0,2})',
)

# Proper noun sequence: one or more capitalised words (mid-sentence use)
_PROPER_NOUN_RE = re.compile(r'(?<![.!?]\s)(?<!\A)\b([A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*)\b')

# Sentence-start proper noun: first word capitalised (may be a subject)
_SENTENCE_SUBJECT_RE = re.compile(
    r'(?:^|(?<=[.!?]\s))([A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*)'
    r'\s+(?:is|was|has|had|went|said|looks?|looked|walked|'
    r'enters?|entered|stands?|stood|wears?|wore|'
    r'carries?|carried|holds?|held|runs?|arrives?|approaches?|appears?|'
    r'turns?|sees?|leaves?|returns?|picks?|grabs?|takes?)',
)

# Tag type → prefix map
_TAG_PREFIXES: dict[str, str] = {
    "character": "character",
    "object":    "object",
    "location":  "location",
    "scene":     "scene",
    "episode":   "episode",
    "shot":      "shot",
    "session":   "session",
    "concept":   "concept",
    "entity":    "entity",
    "event":     "event",
    "agent":     "agent",
    "fact":      "fact",
}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Output of DeterministicExtractor.extract().

    entities  — list of (type_str, label) pairs.  type_str matches NodeType values.
    relations — list of (label_a, relation_str, label_b) triples.
    tags      — auto-generated fragment tags: "character:elena", "location:north-warehouse"
    """
    entities:  list[tuple[str, str]]       = field(default_factory=list)
    relations: list[tuple[str, str, str]]  = field(default_factory=list)
    tags:      list[str]                   = field(default_factory=list)


# ── Extractor ─────────────────────────────────────────────────────────────────

class DeterministicExtractor:
    """Pattern-based entity and relation extractor.

    Requires no LLM. Suitable for any long-running agent that tracks
    named concepts across many observations: people/characters, tools/objects,
    locations/places, session context. Media production is one example;
    others include research sessions, incident timelines, game narratives,
    legal case tracking, or any project where cross-session continuity matters.

    Extraction is best-effort. Precision > recall — it is better to extract
    fewer correct entities than many incorrect ones. An LLM (if configured)
    adds to this output, never replaces it.
    """

    def extract(
        self,
        text: str,
        *,
        session_id:  str | None = None,
        memory_type: str | None = None,
    ) -> ExtractionResult:
        """Extract entities, relations, and tags from an observation string.

        session_id   — if provided, creates a context entity from the last
                       path component (e.g. "project/phase-1/step-3" → "step-3").
        memory_type  — override the context entity type (default: "scene").
                       Use "episode", "shot", "session" for other hierarchies.
        """
        entities:  list[tuple[str, str]]       = []
        relations: list[tuple[str, str, str]]  = []
        seen:      set[str]                    = set()

        # 1. Context entity from session_id ────────────────────────────────────
        context_label: str | None = None
        if session_id:
            context_label = session_id.rstrip("/").split("/")[-1]
            ctx_type = memory_type if memory_type in _TAG_PREFIXES else "scene"
            entities.append((ctx_type, context_label))
            seen.add(context_label.lower())

        # 2. Location entities — capitalized proper-noun NP after location verbs ─
        locations: list[str] = []
        for m in _LOCATION_VERB_RE.finditer(text):
            after = text[m.end():].lstrip()
            np = _LOCATION_NP_RE.match(after)
            if np:
                label = np.group(1).strip()
                slug  = label.lower()
                if slug and slug not in _STOPWORDS and slug not in seen and len(label) > 2:
                    entities.append(("location", label))
                    locations.append(label)
                    seen.add(slug)

        # 3. Object entities — lowercase noun phrase after ownership / use verbs ─
        objects: list[str] = []
        for m in _OWN_VERB_RE.finditer(text):
            after = text[m.end():].lstrip()
            np = _OBJECT_NP_RE.match(after)
            if np:
                raw   = np.group(1).strip()
                label = _title(raw)
                slug  = label.lower()
                if slug and slug not in _STOPWORDS and slug not in seen and len(label) > 2:
                    entities.append(("object", label))
                    objects.append(label)
                    seen.add(slug)

        # 4. Named entity detection — two tiers ───────────────────────────────
        #
        # subjects:       sentence-start proper nouns followed by action verbs.
        #                 These are clearly active participants → source of relations.
        # mid_entities:   mid-sentence capitalised sequences not already classified.
        #                 Recorded as "entity" type but NOT used as relation sources
        #                 (they might be locations/objects already classified above).

        subjects: list[str] = []

        for m in _SENTENCE_SUBJECT_RE.finditer(text):
            label = m.group(1).strip()
            slug  = label.lower()
            if slug not in seen and slug not in _STOPWORDS:
                entities.append(("character", label))
                subjects.append(label)
                seen.add(slug)

        for m in _PROPER_NOUN_RE.finditer(text):
            label = m.group(1).strip()
            slug  = label.lower()
            if slug not in seen and slug not in _STOPWORDS and len(label) > 1:
                entities.append(("entity", label))
                # NOT added to subjects — mid-sentence proper nouns are not assumed
                # to be active participants that own objects or visit locations.
                seen.add(slug)

        # 5. Relation inference — subjects only as relation sources ───────────
        for subj in subjects:
            if context_label:
                relations.append((subj, "appears_in", context_label))
            for loc in locations:
                relations.append((subj, "located_at", loc))
            for obj in objects:
                relations.append((subj, "owns", obj))
        for obj in objects:
            if context_label:
                relations.append((obj, "appears_in", context_label))
        for loc in locations:
            if context_label:
                relations.append((loc, "appears_in", context_label))

        # 6. Tag generation ────────────────────────────────────────────────────
        tags: list[str] = []
        for ent_type, label in entities:
            prefix = _TAG_PREFIXES.get(ent_type, ent_type)
            tags.append(f"{prefix}:{_slug(label)}")

        return ExtractionResult(entities=entities, relations=relations, tags=tags)


# ── Structured extraction schema ──────────────────────────────────────────────
#
# MARK-owned schema for LLM-based graph extraction.
# Separate from LLMContextualCompressor (post-retrieval compression).
#
# Flow:
#   observation text
#     -> DeterministicExtractor  (always, free)
#     -> LLMStructuredExtractor  (optional, developer LLM)
#     -> ExtractionMerger        (dedup + combine)
#     -> nodes / edges / tags    (stored in memory graph)

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


# ── LLM Structured Extractor ──────────────────────────────────────────────────

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


def _parse_structured_response(raw: str) -> StructuredExtraction:
    """Parse an LLM response into StructuredExtraction.

    Tries JSON directly; strips markdown code fences if present.
    Returns an empty StructuredExtraction on any parse or validation failure —
    never raises, so observe() always succeeds.
    """
    text = raw.strip()

    # Strip optional markdown code fence (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
        return StructuredExtraction.model_validate(data)
    except Exception:
        return StructuredExtraction()


class LLMStructuredExtractor:
    """Optional LLM-based structured entity and relation extractor.

    Uses a developer-provided LLMProvider to extract entities, relationships,
    and tags from observation text. Returns a Pydantic-validated schema so
    output is always well-typed before reaching the memory graph.

    Works with any LLMProvider: Ollama, OpenAI-compatible endpoints, llama.cpp,
    or cloud-hosted models. Does NOT bundle or download any model.

    Use ExtractionMerger to combine this with DeterministicExtractor output.
    LangChain's with_structured_output() integration lives in mark-adapters.

    Example::

        mark = Mark.local(project_path=".", llm=my_ollama_llm)
        mark.observe(
            "Elena enters the North Warehouse wearing the red scarf.",
            session_id="season-01/ep-02/scene-04",
        )
    """

    def __init__(
        self,
        llm:            "LLMProvider",
        *,
        min_confidence: float = 0.5,
    ) -> None:
        self._llm            = llm
        self._min_confidence = min_confidence

    def extract_structured(self, text: str) -> StructuredExtraction:
        """Call the LLM and return a validated StructuredExtraction.

        Returns an empty StructuredExtraction on any failure so observe()
        always succeeds as a store_sync() call at minimum.
        """
        try:
            prompt   = _STRUCTURED_PROMPT.format(text=text)
            response = self._llm.complete(prompt)
            result   = _parse_structured_response(response)
            # Filter by confidence threshold
            entities = [e for e in result.entities if e.confidence >= self._min_confidence]
            relations = [r for r in result.relations if r.confidence >= self._min_confidence]
            allowed_tags = {f"{e.type}:{_slug(e.label)}" for e in entities}
            result = StructuredExtraction(
                entities  = entities,
                relations = relations,
                tags      = [tag for tag in result.tags if tag in allowed_tags],
                summary   = result.summary,
            )
            return result
        except Exception:
            return StructuredExtraction()


# ── Extraction Merger ─────────────────────────────────────────────────────────

class ExtractionMerger:
    """Merges DeterministicExtractor output with LLMStructuredExtractor output.

    Rules:
    - Deterministic entities are always included (they are precise and fast).
    - LLM entities are added only if their label is not already present
      (case-insensitive normalised comparison).
    - Relations from both sources are included; exact duplicates are dropped.
    - Tags from both sources are merged and deduplicated.
    """

    def merge(
        self,
        deterministic:  ExtractionResult,
        structured:     StructuredExtraction,
    ) -> ExtractionResult:
        """Return a merged ExtractionResult combining both extraction passes."""
        seen_labels: set[str] = {label.lower() for _, label in deterministic.entities}
        entities  = list(deterministic.entities)
        relations = list(deterministic.relations)
        tags      = list(deterministic.tags)
        tag_set   = set(tags)

        # Add LLM entities not already present
        for ent in structured.entities:
            slug = ent.label.lower()
            if slug not in seen_labels:
                entities.append((ent.type, ent.label))
                seen_labels.add(slug)
                # Generate tag for this entity if not already in set
                auto_tag = f"{ent.type}:{_slug(ent.label)}"
                if auto_tag not in tag_set:
                    tags.append(auto_tag)
                    tag_set.add(auto_tag)

        # Add LLM relations (deduplicated)
        relation_set = {(a, rel, b) for a, rel, b in relations}
        for rel in structured.relations:
            triple = (rel.source, rel.relation, rel.target)
            if triple not in relation_set:
                relations.append(triple)
                relation_set.add(triple)

        # Merge LLM-provided tags
        for tag in structured.tags:
            if tag not in tag_set:
                tags.append(tag)
                tag_set.add(tag)

        return ExtractionResult(entities=entities, relations=relations, tags=tags)
