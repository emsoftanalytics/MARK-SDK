"""Automatic fact extraction: turn stored memories into bi-temporal facts.

    from mark.intelligence.fact_extraction import PatternFactExtractor
    PatternFactExtractor().extract("[2026-03-30] CHANGE: atlas-core moved from port 9239 to port 4444.")
    # -> [FactCandidate(subject='atlas-core', attribute='port', value='4444', ...)]

Modules:
  - ``base``      FactCandidate, the FactExtractor protocol, candidate hygiene
  - ``rules``     RulePack and the shared regex building blocks
  - ``packs``     the built-in generic packs (pure data)
  - ``clauses``   clause splitting with subject carry-over
  - ``patterns``  PatternFactExtractor (regex packs; fast, runs inline)
  - ``llm``       LLMFactExtractor (OpenAI-compatible server; runs in background)
  - ``service``   FactExtractionService: inline/queued extraction into the store
  - ``worker``    the background thread that drains the durable job queue

Wiring: ``MarkRuntime.local(fact_extractor="patterns")`` or
``memory.store_sync(..., extract_facts=True)``.
"""
from mark.intelligence.fact_extraction.base import FactCandidate, FactExtractor
from mark.intelligence.fact_extraction.llm import LLMFactExtractor
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
    "resolve_extractor",
]
