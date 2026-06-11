# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import pytest

from mark.intelligence.classifier import (
    QueryAnalysis,
    QueryClassifier,
    QueryComplexity,
    query_classifier,
)


# ---------------------------------------------------------------------------
# Imports / singleton
# ---------------------------------------------------------------------------

def test_query_classifier_importable() -> None:
    assert QueryClassifier is not None


def test_query_classifier_singleton_importable() -> None:
    assert query_classifier is not None
    assert isinstance(query_classifier, QueryClassifier)


# ---------------------------------------------------------------------------
# Simple queries → SIMPLE policy
# ---------------------------------------------------------------------------

def test_simple_wh_question_maps_to_simple() -> None:
    r = QueryClassifier().classify("What is Elena wearing?")
    assert r.policy == QueryComplexity.SIMPLE


def test_who_is_maps_to_simple() -> None:
    r = QueryClassifier().classify("Who is Elena?")
    assert r.policy == QueryComplexity.SIMPLE


def test_define_query_maps_to_simple() -> None:
    r = QueryClassifier().classify("Define the red scarf.")
    assert r.policy == QueryComplexity.SIMPLE


def test_list_query_maps_to_simple() -> None:
    r = QueryClassifier().classify("List all characters.")
    assert r.policy == QueryComplexity.SIMPLE


# ---------------------------------------------------------------------------
# Complex queries → DEEP policy
# ---------------------------------------------------------------------------

def test_multi_hop_query_maps_to_deep() -> None:
    r = QueryClassifier().classify(
        "Why did Elena choose the red scarf because of her mother, "
        "which then led to the confrontation at the warehouse?"
    )
    assert r.policy in (QueryComplexity.BALANCED, QueryComplexity.DEEP)


def test_evaluate_query_maps_to_deep_or_balanced() -> None:
    r = QueryClassifier().classify(
        "Evaluate and contrast Elena's emotional state across seasons 1 and 2."
    )
    assert r.policy in (QueryComplexity.BALANCED, QueryComplexity.DEEP)


def test_reasoning_query_has_high_complexity() -> None:
    # Long, multi-entity, reasoning query with contrast keyword
    r = QueryClassifier().classify(
        "Evaluate and contrast how Elena's relationship with the red scarf changed "
        "over time, because of her mother, which then led to her confrontation with Marcus "
        "and subsequently altered the trajectory of her story."
    )
    assert r.complexity > 0.3


# ---------------------------------------------------------------------------
# QueryAnalysis fields
# ---------------------------------------------------------------------------

def test_analysis_complexity_is_float_in_range() -> None:
    r = QueryClassifier().classify("What is the scarf color?")
    assert 0.0 <= r.complexity <= 1.0


def test_analysis_is_factual_true_for_wh_question() -> None:
    r = QueryClassifier().classify("What is Elena's name?")
    assert r.is_factual is True


def test_analysis_keywords_strips_stop_words() -> None:
    r = QueryClassifier().classify("What is the name of the character?")
    # "what", "is", "the", "of" are stop words
    assert "what" not in r.keywords
    assert "is"   not in r.keywords


def test_analysis_keywords_keeps_content_words() -> None:
    r = QueryClassifier().classify("Elena carries the red scarf artifact.")
    assert any("elena" in kw.lower() or "scarf" in kw.lower() or "artifact" in kw.lower()
               for kw in r.keywords)


def test_analysis_has_temporal_flag() -> None:
    r = QueryClassifier().classify("What happened last week in the warehouse?")
    assert r.has_temporal is True


def test_analysis_no_temporal_flag_for_atemporal_query() -> None:
    r = QueryClassifier().classify("What is Elena wearing?")
    assert r.has_temporal is False


def test_analysis_has_multi_entity_flag() -> None:
    r = QueryClassifier().classify("How does Elena compare to Marcus in the scene?")
    assert r.has_multi_entity is True


def test_analysis_has_negation_flag() -> None:
    r = QueryClassifier().classify("What did Elena never say to Marcus?")
    assert r.has_negation is True


def test_analysis_has_reasoning_flag() -> None:
    r = QueryClassifier().classify("Why did Elena leave the warehouse?")
    assert r.has_reasoning is True


def test_analysis_question_depth_simple() -> None:
    r = QueryClassifier().classify("What is a scarf?")
    assert r.question_depth == 1


def test_analysis_question_depth_deep_for_reasoning() -> None:
    r = QueryClassifier().classify(
        "Explain and analyze why Elena chose the red scarf given the backstory."
    )
    assert r.question_depth >= 2


# ---------------------------------------------------------------------------
# estimated_top_k alignment
# ---------------------------------------------------------------------------

def test_simple_policy_top_k_is_5() -> None:
    r = QueryClassifier().classify("What is Elena?")
    if r.policy == QueryComplexity.SIMPLE:
        assert r.estimated_top_k == 5


def test_deep_policy_top_k_is_15() -> None:
    r = QueryClassifier().classify(
        "Evaluate and contrast Elena's development across three seasons, "
        "considering how the scarf because of her mother led to the results."
    )
    if r.policy == QueryComplexity.DEEP:
        assert r.estimated_top_k == 15


# ---------------------------------------------------------------------------
# reasons and reasoning string
# ---------------------------------------------------------------------------

def test_reasoning_string_is_non_empty() -> None:
    r = QueryClassifier().classify("Why did Elena leave?")
    assert len(r.reasoning) > 0


def test_reasons_list_is_list() -> None:
    r = QueryClassifier().classify("What is Elena?")
    assert isinstance(r.reasons, list)


def test_reasons_include_simple_lookup_for_simple_query() -> None:
    r = QueryClassifier().classify("What is Elena?")
    assert "simple_lookup" in r.reasons


# ---------------------------------------------------------------------------
# analyze alias
# ---------------------------------------------------------------------------

def test_analyze_alias_equivalent_to_classify() -> None:
    c  = QueryClassifier()
    r1 = c.classify("What is Elena?")
    r2 = c.analyze("What is Elena?")
    assert r1.policy     == r2.policy
    assert r1.complexity == r2.complexity
