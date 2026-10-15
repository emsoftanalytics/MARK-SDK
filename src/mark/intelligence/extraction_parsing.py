# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Tolerant parsing of LLM extraction replies into StructuredExtraction.

LLMs wrap JSON in code fences, add prose around it, invent entity types and
emit ``null`` confidences. Parsing therefore finds the first JSON object
anywhere in the reply, maps known aliases onto MARK's vocabulary, and drops
invalid items one by one instead of rejecting the whole reply. It never
raises: a bad reply degrades to an empty extraction and a log line.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from mark.intelligence.extraction_schema import (
    _ENTITY_TYPE_ALIASES,
    _ENTITY_TYPES,
    _RELATION_ALIASES,
    _RELATION_TYPES,
    ExtractedEntity,
    ExtractedRelation,
    StructuredExtraction,
)

# Same logger name as before the split, so log filtering (and caplog) keeps working.
logger = logging.getLogger("mark.intelligence.extractor")

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)


def _iter_json_objects(text: str):
    """Yield every brace-balanced ``{...}`` candidate in *text*, in order.

    Braces inside JSON strings (including escaped quotes) are ignored, so a
    label such as ``"a {b}"`` does not end the object early.
    """
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
    """Clamp a confidence to [0, 1]; missing, boolean, non-numeric or NaN means 1.0."""
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
    """'Located In' / 'located-in' -> 'located_in', for alias lookup."""
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def _coerce_entity(item: Any) -> ExtractedEntity | None:
    """One raw entity dict -> ExtractedEntity, mapping type aliases; None when unusable."""
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
    """One raw relation dict -> ExtractedRelation, mapping relation aliases; None when unusable."""
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
