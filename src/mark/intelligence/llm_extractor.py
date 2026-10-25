# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Optional LLM-assisted extraction and its merge with deterministic output.

The LLM pass is additive: deterministic entities always stay, LLM entities
are added only when new, and everything below the confidence threshold is
dropped before it can reach the memory graph.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mark.intelligence.deterministic_extractor import ExtractionResult
from mark.intelligence.extraction_parsing import _parse_structured_response
from mark.intelligence.extraction_patterns import _slug
from mark.intelligence.extraction_schema import _STRUCTURED_PROMPT, StructuredExtraction

if TYPE_CHECKING:
    from mark.types.llm import LLMProvider

# Same logger name as before the split, so log filtering (and caplog) keeps working.
logger = logging.getLogger("mark.intelligence.extractor")


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
