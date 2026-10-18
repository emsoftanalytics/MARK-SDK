# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Vocabulary and regular expressions for deterministic entity extraction.

The patterns favour precision over recall: locations are capitalised noun
phrases after movement verbs, objects are lowercase noun phrases after
possession/use verbs, subjects are sentence-initial proper nouns followed
by an action verb. Idioms ("used to", "take place") are excluded explicitly.
"""
from __future__ import annotations

import re

from mark.intelligence._lazy_re import lazy  # compiled on first use (cold start)


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
_LOCATION_VERB_RE = lazy(
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
_LOCATION_NP_RE = lazy(
    r'^(?:the\s+|an?\s+)?([A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*)',
)

# Verbs that signal an OBJECT follows (ownership / possession / use)
_OWN_VERB_RE = lazy(
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
_IDIOM_RE = lazy(
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
_SENTENCE_RE = lazy(r'[^.!?\n]+(?:[.!?]+|$)', re.MULTILINE)

# After an OWN verb: match optional article + lowercase noun phrase (1-3 words).
# Objects are typically lowercase common nouns ("the red scarf", "a data file").
_OBJECT_NP_RE = lazy(
    r'^(?:the\s+|an?\s+)?([a-z][a-zA-Z-]+(?:\s+[a-z][a-zA-Z-]+){0,2})',
)

# Proper noun sequence: one or more capitalised words (mid-sentence use)
_PROPER_NOUN_RE = lazy(r'(?<![.!?]\s)(?<!\A)\b([A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*)\b')

# Sentence-start proper noun: first word capitalised (may be a subject)
_SENTENCE_SUBJECT_RE = lazy(
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
