# SPDX-License-Identifier: Apache-2.0
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
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from mark.types.llm import LLMProvider

logger = logging.getLogger(__name__)


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

# Idioms where an ownership/location verb does not introduce an object or place:
# "used to", "take care", "take place", "took part", "held on", "picks up speed"...
_IDIOM_RE = re.compile(
    r'\b(?:'
    r'(?:use[sd]?|using)\s+to'
    r'|(?:take[sn]?|taking|took)\s+(?:care|place|part|time|note|notice|advantage|over|off|turns?'
    r'|charge|hold|effect|action|steps?|place|shape|root|aim|sides|a\s+(?:look|break|seat|moment|while|chance|stand))'
    r'|(?:hold[s]?|holding|held)\s+(?:on|off|up|back|out|true)'
    r'|(?:own[s]?|owned|owning)\s+up'
    r'|(?:grab[s]?|grabbed|grabbing)\s+(?:attention|a\s+bite)'
    r'|(?:wear[s]?|wore|wearing)\s+(?:off|out|thin|down)'
    r'|(?:carr(?:y|ies|ied|ying))\s+(?:on|out)'
    r')\b',
    re.IGNORECASE,
)

# Sentence spans: text up to and including terminal punctuation (or end of text).
_SENTENCE_RE = re.compile(r'[^.!?\n]+(?:[.!?]+|$)', re.MULTILINE)

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

        # 2-4. Per-sentence extraction. Relations are only inferred inside a
        # sentence, from each object/location to the nearest subject, so
        # "Elena holds a lamp. Marco enters the Vault." never links Elena to
        # the Vault or Marco to the lamp.
        locations: list[str] = []
        objects:   list[str] = []
        subjects:  list[str] = []
        subject_links: list[tuple[str, str, str]] = []

        for sentence_match in _SENTENCE_RE.finditer(text):
            sentence = sentence_match.group(0)
            if not sentence.strip():
                continue
            lead = len(sentence) - len(sentence.lstrip())
            body = sentence.lstrip()

            # Subjects: sentence-start proper nouns followed by action verbs.
            sentence_subjects: list[tuple[int, str]] = []
            for m in _SENTENCE_SUBJECT_RE.finditer(body):
                label = m.group(1).strip()
                slug  = label.lower()
                if slug in _STOPWORDS:
                    continue
                sentence_subjects.append((lead + m.start(1), label))
                if slug not in seen:
                    entities.append(("character", label))
                    subjects.append(label)
                    seen.add(slug)

            def nearest_subject(pos: int) -> str | None:
                if not sentence_subjects:
                    return None
                before = [s for s in sentence_subjects if s[0] <= pos]
                pool = before or sentence_subjects
                return min(pool, key=lambda s: abs(pos - s[0]))[1]

            # Locations — capitalised proper-noun NP after location verbs.
            for m in _LOCATION_VERB_RE.finditer(sentence):
                after = sentence[m.end():].lstrip()
                np = _LOCATION_NP_RE.match(after)
                if not np:
                    continue
                label = np.group(1).strip()
                slug  = label.lower()
                if not slug or slug in _STOPWORDS or len(label) <= 2:
                    continue
                if slug not in seen:
                    entities.append(("location", label))
                    locations.append(label)
                    seen.add(slug)
                subj = nearest_subject(m.start())
                if subj is not None and subj.lower() != slug:
                    subject_links.append((subj, "located_at", label))

            # Objects — lowercase noun phrase after ownership / use verbs.
            for m in _OWN_VERB_RE.finditer(sentence):
                if _IDIOM_RE.match(sentence, m.start()):
                    continue
                after = sentence[m.end():].lstrip()
                np = _OBJECT_NP_RE.match(after)
                if not np:
                    continue
                raw   = np.group(1).strip()
                first = raw.split()[0].lower()
                if first in _STOPWORDS:
                    continue
                label = _title(raw)
                slug  = label.lower()
                if not slug or slug in _STOPWORDS or len(label) <= 2:
                    continue
                if slug not in seen:
                    entities.append(("object", label))
                    objects.append(label)
                    seen.add(slug)
                subj = nearest_subject(m.start())
                if subj is not None:
                    subject_links.append((subj, "owns", label))

        # Mid-sentence capitalised sequences not already classified are
        # recorded as "entity" but never used as relation sources.
        for m in _PROPER_NOUN_RE.finditer(text):
            label = m.group(1).strip()
            slug  = label.lower()
            if slug not in seen and slug not in _STOPWORDS and len(label) > 1:
                entities.append(("entity", label))
                seen.add(slug)

        # 5. Relation inference ────────────────────────────────────────────────
        for subj in subjects:
            if context_label:
                relations.append((subj, "appears_in", context_label))
        for triple in subject_links:
            if triple not in relations:
                relations.append(triple)
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

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)


def _iter_json_objects(text: str):
    """Yield every brace-balanced ``{...}`` candidate in *text*, in order."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:end + 1]
                    break


def _first_json_object(raw: str) -> dict[str, Any] | None:
    """Find the first JSON object in an LLM reply (fences, prose, trailing text)."""
    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(raw)]
    candidates.append(raw)
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            data = json.loads(chunk)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        for candidate in _iter_json_objects(chunk):
            try:
                data = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


def _coerce_confidence(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if number != number:  # NaN
        return 1.0
    return max(0.0, min(1.0, number))


def _norm_key(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def _coerce_entity(item: Any) -> ExtractedEntity | None:
    if not isinstance(item, dict):
        return None
    label = item.get("label") or item.get("name")
    if not isinstance(label, str) or not label.strip():
        return None
    raw_type = _norm_key(item.get("type") or "entity")
    etype = _ENTITY_TYPE_ALIASES.get(raw_type, raw_type)
    if etype not in _ENTITY_TYPES:
        return None
    return ExtractedEntity(type=etype, label=label.strip(), confidence=_coerce_confidence(item.get("confidence")))


def _coerce_relation(item: Any) -> ExtractedRelation | None:
    if not isinstance(item, dict):
        return None
    source = item.get("source")
    target = item.get("target")
    if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
        return None
    raw_rel = _norm_key(item.get("relation") or "related_to")
    relation = _RELATION_ALIASES.get(raw_rel, raw_rel)
    if relation not in _RELATION_TYPES:
        return None
    return ExtractedRelation(
        source=source.strip(),
        relation=relation,
        target=target.strip(),
        confidence=_coerce_confidence(item.get("confidence")),
    )


def _parse_structured_response(raw: str) -> StructuredExtraction:
    """Parse an LLM response into StructuredExtraction.

    Tolerates markdown code fences, prose before/after the JSON, unknown or
    differently-cased entity types (aliases are mapped, unknowns skipped per
    item), and ``null`` confidences (default 1.0). Invalid items are dropped
    individually instead of discarding the whole response. Failures are
    logged; this never raises, so observe() always succeeds.
    """
    if not isinstance(raw, str) or not raw.strip():
        logger.warning("structured extraction: empty LLM response")
        return StructuredExtraction()

    data = _first_json_object(raw)
    if data is None:
        logger.warning("structured extraction: no JSON object found in LLM response (%d chars)", len(raw))
        return StructuredExtraction()

    entities: list[ExtractedEntity] = []
    raw_entities = data.get("entities") or []
    if not isinstance(raw_entities, list):
        logger.warning("structured extraction: 'entities' is %s, expected list", type(raw_entities).__name__)
        raw_entities = []
    for item in raw_entities:
        entity = _coerce_entity(item)
        if entity is None:
            logger.debug("structured extraction: skipped invalid entity %r", item)
        else:
            entities.append(entity)

    relations: list[ExtractedRelation] = []
    raw_relations = data.get("relations") or []
    if not isinstance(raw_relations, list):
        logger.warning("structured extraction: 'relations' is %s, expected list", type(raw_relations).__name__)
        raw_relations = []
    for item in raw_relations:
        relation = _coerce_relation(item)
        if relation is None:
            logger.debug("structured extraction: skipped invalid relation %r", item)
        else:
            relations.append(relation)

    raw_tags = data.get("tags") or []
    tags = [t.strip() for t in raw_tags if isinstance(t, str) and t.strip()] if isinstance(raw_tags, list) else []
    summary = data.get("summary")
    skipped = (len(raw_entities) - len(entities)) + (len(raw_relations) - len(relations))
    if skipped:
        logger.info("structured extraction: skipped %d invalid item(s)", skipped)
    return StructuredExtraction(
        entities=entities,
        relations=relations,
        tags=tags,
        summary=summary if isinstance(summary, str) else None,
    )


class LLMStructuredExtractor:
    """Optional LLM-based structured entity and relation extractor.

    Uses a developer-provided LLMProvider to extract entities, relationships,
    and tags from observation text. Returns a Pydantic-validated schema so
    output is always well-typed before reaching the memory graph.

    Works with any LLMProvider: Ollama, OpenAI-compatible endpoints, llama.cpp,
    or remote models. Does NOT bundle or download any model.

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
        except Exception as exc:
            logger.warning(
                "LLMStructuredExtractor failed; using deterministic extraction only: %s: %s",
                type(exc).__name__, exc,
            )
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
