"""Dates written in a sentence that say *when* a stated value took (or takes) effect.

  - a leading "Starting / From / Effective / Beginning / As of <YYYY-MM-DD>,"
    dates the whole sentence and may lie in the future: the fact is then
    *scheduled* (``facts.scheduled()``) rather than current;
  - "in <Month> [<year>]" (or "since <Month>") in a past-tense clause dates
    the change to the first of that month: "I moved to Porto in September".
    Without a year it is the most recent such month not after the
    reference time (the memory's event time, else now), so "in November"
    said in October means last November.

Only the text itself is read; nothing here guesses a date that is not written.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

LEADING_DATE = re.compile(
    r"^(?:starting|from|effective|beginning|as\s+of|commencing)\s+(?:on\s+|from\s+)?(\d{4}-\d{2}-\d{2})\b\s*,?\s*",
    re.IGNORECASE)
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_MONTH_ALT = (r"jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?"
              r"|nov(?:ember)?|dec(?:ember)?")
_IN_MONTH = re.compile(rf"\b(?:in|since)\s+(?:early\s+|late\s+|mid-?)?(?P<month>{_MONTH_ALT})\b(?:\s+(?P<year>\d{{4}}))?",
                       re.IGNORECASE)
# A past-tense clause: "-ed" verbs and the common irregular ones. A present
# clause with a month ("the review is in May") is a plan, not a dated change.
_PAST = re.compile(r"\b(?:\w+ed|was|were|became|began|left|got|took|joined|went|came|made|bought|sold|quit)\b",
                   re.IGNORECASE)


def split_leading_date(sentence: str) -> tuple[str, datetime | None]:
    """"Starting 2027-01-01, Priya leads X" -> ("Priya leads X", 2027-01-01 UTC)."""
    match = LEADING_DATE.match(sentence)
    if match is None:
        return sentence, None
    try:
        when = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
    except ValueError:
        return sentence, None
    return sentence[match.end():], when


def month_date(clause: str, reference: datetime | None) -> datetime | None:
    """First day of the month a past-tense clause names ("moved to Porto in September"), or None."""
    if " in " not in clause and "since" not in clause.lower() and " In " not in clause:
        return None
    match = _IN_MONTH.search(clause)
    if match is None or not _PAST.search(clause[:match.start()]):
        return None
    month = next(i for i, name in enumerate(_MONTHS, 1) if name.startswith(match.group("month").lower()[:3]))
    now = reference or datetime.now(timezone.utc)
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if match.group("year"):
        return datetime(int(match.group("year")), month, 1, tzinfo=timezone.utc)
    year = now.year if (now.month, 1) >= (month, 1) else now.year - 1
    return datetime(year, month, 1, tzinfo=timezone.utc)
