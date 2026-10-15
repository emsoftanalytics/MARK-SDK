"""Automatic fact extraction: turn stored memories into bi-temporal facts.

    from mark.intelligence.fact_extraction import PatternFactExtractor
    PatternFactExtractor().extract("[2026-03-30] CHANGE: atlas-core moved from port 9239 to port 4444.")
    # -> [FactCandidate(subject='atlas-core', attribute='port', value='4444', ...)]

Modules:
  - ``base``      FactCandidate, the FactExtractor protocol, candidate hygiene
  - ``rules``     RulePack and the shared regex building blocks
  - ``packs``     the built-in generic packs (pure data)
  - ``clauses``   clause splitting with subject carry-over
  - ``packs_people`` personal details, preferences, prices, measurements
  - ``vocab``     closed vocabularies (personal/dev attribute nouns, product kinds)
  - ``personal``  first-person rewriting ("I"/"my"/"me" -> the user subject)
  - ``negation``  negations as retractions, corrections
  - ``structured`` JSON / YAML / key: value / k=v records and numbered steps
  - ``labels``    line labels ("DECISION:", "TODO:", "HAL-1:") and note kinds
  - ``numbers``   number words and ordinals ("step four" -> "step 4")
  - ``sentences`` sentence splitting, stamps, labels and guards
  - ``normalise`` regex match -> validated candidate
  - ``patterns``  PatternFactExtractor (regex packs; fast, runs inline)
  - ``apply``     record candidates: assertions, corrections, retractions
  - ``llm``       LLMFactExtractor (OpenAI-compatible server; runs in background)
  - ``service``   FactExtractionService: inline/queued extraction into the store
  - ``worker``    the background thread that drains the durable job queue

Wiring: ``MarkRuntime.local(fact_extractor="patterns")`` or
``memory.store_sync(..., extract_facts=True)``.
"""
from mark.intelligence.fact_extraction.base import FactCandidate, FactExtractor
from mark.intelligence.fact_extraction.labels import note_kinds
from mark.intelligence.fact_extraction.llm import LLMFactExtractor
from mark.intelligence.fact_extraction.numbers import normalise_step_refs, parse_number
from mark.intelligence.fact_extraction.packs import BUILTIN_PACKS, RulePack
from mark.intelligence.fact_extraction.patterns import PatternFactExtractor
from mark.intelligence.fact_extraction.service import FactExtractionService, resolve_extractor

__all__ = [
    "BUILTIN_PACKS",
    "FactCandidate",
    "FactExtractionService",
    "FactExtractor",
    "LLMFactExtractor",
    "PatternFactExtractor",
    "RulePack",
    "normalise_step_refs",
    "note_kinds",
    "parse_number",
    "resolve_extractor",
]
