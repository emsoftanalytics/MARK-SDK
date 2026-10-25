"""Number words and ordinals: "four", "fourth", "4th", "twenty-one" -> 21.

Procedures are written both ways ("step four", "the fourth step", "4. Run
migrations"), so extraction keys list steps on the digit form (``step 4``)
and queries can be normalised the same way with :func:`normalise_step_refs`
before matching. English only, up to ninety-nine: enough for list steps and
small counts without ever guessing at larger compounds.
"""
from __future__ import annotations

import re
from mark.security.lazy_regex import lazy_compile

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
          " fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_ORDINAL_UNITS = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth"
                  " thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth").split()
_ORDINAL_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60,
                 "seventieth": 70, "eightieth": 80, "ninetieth": 90}

_WORDS: dict[str, int] = {w: i for i, w in enumerate(_UNITS)} | _TENS
_ORDINALS: dict[str, int] = {w: i for i, w in enumerate(_ORDINAL_UNITS)} | _ORDINAL_TENS
_DIGITS = lazy_compile(r"^(\d{1,4})(?:st|nd|rd|th)?$", re.IGNORECASE)

#: One number word or ordinal, including hyphenated compounds ("twenty-first").
NUMBER_WORD = (r"(?:(?:" + "|".join(_TENS) + r")(?:[-\s](?:" + "|".join(_UNITS[1:10] + _ORDINAL_UNITS[1:10]) + r"))?"
               r"|" + "|".join(sorted(set(_UNITS) | set(_ORDINAL_UNITS) | set(_ORDINAL_TENS), key=len, reverse=True))
               + r"|\d{1,4}(?:st|nd|rd|th)?)")

_STEP_WORD = lazy_compile(rf"\bstep\s+(?P<n>{NUMBER_WORD})\b", re.IGNORECASE)
_WORD_STEP = lazy_compile(rf"\b(?P<n>{NUMBER_WORD})\s+step\b", re.IGNORECASE)


def parse_number(word: str) -> int | None:
    """Integer value of a number word, ordinal or digit string; None if it is not one."""
    text = word.strip().lower()
    digits = _DIGITS.match(text)
    if digits:
        return int(digits.group(1))
    if text in _WORDS:
        return _WORDS[text]
    if text in _ORDINALS:
        return _ORDINALS[text]
    parts = re.split(r"[-\s]+", text)
    if len(parts) == 2 and parts[0] in _TENS:
        unit = _WORDS.get(parts[1], _ORDINALS.get(parts[1]))
        if unit is not None and 0 < unit < 10:
            return _TENS[parts[0]] + unit
    return None


def normalise_step_refs(text: str) -> str:
    """Rewrite "step four" and "fourth step" as "step 4" (other text unchanged)."""

    def _step(match: re.Match[str]) -> str:
        number = parse_number(match.group("n"))
        return match.group(0) if number is None else f"step {number}"

    return _WORD_STEP.sub(_step, _STEP_WORD.sub(_step, text))
