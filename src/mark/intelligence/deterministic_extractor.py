# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Local deterministic entity and relation extractor (no LLM required).

Extracts named entities and relationships from observation text using
pattern matching. Designed for any long-running agent that tracks named
concepts: characters/people, objects/tools, locations/places, session/scene
context. Works for media production, game narratives, legal cases, research
sessions, engineering incident timelines, and any project where continuity
across many observations matters.

When an LLMProvider is configured, its output is additive on top of this
(see ``llm_extractor``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mark.intelligence.extraction_patterns import (
    _IDIOM_RE,
    _LOCATION_NP_RE,
    _LOCATION_VERB_RE,
    _OBJECT_NP_RE,
    _OWN_VERB_RE,
    _PROPER_NOUN_RE,
    _SENTENCE_RE,
    _SENTENCE_SUBJECT_RE,
    _STOPWORDS,
    _TAG_PREFIXES,
    _slug,
    _title,
)


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
                # Prefer the closest subject *before* the verb ("Elena ... holds");
                # fall back to the closest one after it.
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
