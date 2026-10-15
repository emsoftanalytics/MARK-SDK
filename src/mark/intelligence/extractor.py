# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Deterministic and optional LLM-assisted entity/relation extraction.

This module is the stable import surface for extraction. The implementation
is split by responsibility:

    extraction_patterns      vocabulary, regexes, slug/title helpers
    deterministic_extractor  ExtractionResult, DeterministicExtractor (no LLM)
    extraction_schema        Pydantic schema, LLM prompt, alias tables
    extraction_parsing       tolerant parsing of LLM replies
    llm_extractor            LLMStructuredExtractor, ExtractionMerger

Every name previously defined here, including private helpers used by tests,
is re-exported unchanged (same objects, same identities).
"""
from __future__ import annotations

import logging

from mark.intelligence.deterministic_extractor import DeterministicExtractor, ExtractionResult
from mark.intelligence.extraction_parsing import (
    _FENCE_RE,
    _coerce_confidence,
    _coerce_entity,
    _coerce_relation,
    _first_json_object,
    _iter_json_objects,
    _norm_key,
    _parse_structured_response,
)
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
from mark.intelligence.extraction_schema import (
    _ENTITY_TYPE_ALIASES,
    _ENTITY_TYPES,
    _RELATION_ALIASES,
    _RELATION_TYPES,
    _STRUCTURED_PROMPT,
    _VALID_ENTITY_TYPES,
    _VALID_RELATIONS,
    ExtractedEntity,
    ExtractedRelation,
    StructuredExtraction,
)
from mark.intelligence.llm_extractor import ExtractionMerger, LLMStructuredExtractor

logger = logging.getLogger(__name__)
