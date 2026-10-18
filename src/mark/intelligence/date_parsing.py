"""Natural and relative date expressions, resolved to calendar ranges (stdlib only).

``find_dates(text, reference)`` returns every date expression in a query or a
memory as a :class:`DateSpan` (inclusive first and last day plus the matched
character span). Supported forms:

- ISO dates and months: ``2026-05-03``, ``2026-05``;
- written dates: ``3rd of May 2026``, ``3 May``, ``May 3``, ``May 3rd, 2026``;
- numeric dates: ``05/03/2026``, ``5.3.26``. **Locale assumption: day first**
  (05/03/2026 is 5 March), as in most of the world; a first number above 12
  is always a day and a second number above 12 makes the date month-first
  (US). Pass ``day_first=False`` for US-style text. ISO is never ambiguous
  and is preferred;
- months and years: ``May 2026``, ``in March``, ``in 2025``;
- quarters and halves: ``Q3``, ``Q3 2026``, ``third quarter``, ``H1 2026``;
- relative: ``today``, ``yesterday``, ``tomorrow``, ``3 days ago``,
  ``last/this/next week|month|year|quarter``, ``last/this/next Tuesday``,
  ``on Tuesday`` (the one in the reference week), ``past 7 days``.

Relative expressions resolve against ``reference`` (the query time, or a
memory's own event date). A year-less date takes the reference year.
"""
from __future__ import annotations

import calendar
import re

from mark.intelligence._lazy_re import LazyPattern, lazy  # compiled on first use (cold start)
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
_WEEKDAYS.update({name.lower(): i for i, name in enumerate(calendar.day_abbr)})
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

_MON = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d{2})"
_WD = r"(?P<wd>" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")"
_UNIT = r"(?P<unit>day|week|month|quarter|year)s?"

# Order matters: longer, more specific forms first; matched spans are claimed
# so "3rd of May 2026" is not also read as "May 2026".
_PATTERNS: list[tuple[str, LazyPattern]] = [
    ("iso", lazy(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")),
    ("numeric", lazy(r"(?<![\w./-])(?P<a>\d{1,2})[/.](?P<b>\d{1,2})[/.](?P<y>\d{4}|\d{2})(?![\w./-]|\.\d)")),
    ("day_month", lazy(rf"\b{_DAY}(?:\s+of)?\s+{_MON}(?:,?\s+{_YEAR})?\b", re.I)),
    ("month_day", lazy(rf"\b{_MON}\s+{_DAY}(?!\d|:)(?:,?\s+{_YEAR})?\b", re.I)),
    ("iso_month", lazy(r"\b(?P<y>\d{4})-(?P<m>\d{2})\b(?!-\d)")),
    ("quarter", lazy(r"\b(?:Q(?P<q>[1-4])|(?P<qw>first|second|third|fourth|1st|2nd|3rd|4th)[\s-]+quarter)"
                           r"(?:\s+(?:of\s+)?(?P<year>(?:19|20)\d{2}))?\b", re.I)),
    ("half", lazy(r"\bH(?P<h>[12])\s*(?P<year>(?:19|20)\d{2})?\b")),
    ("month_year", lazy(rf"\b{_MON}\s+{_YEAR}\b", re.I)),
    # A bare month name needs a preposition: "May I ...", "march on" are not dates.
    ("month", lazy(rf"\b(?:in|during|since|until|by|early|late|mid|of|last|next|this)[\s-]+{_MON}(?!\w)", re.I)),
    ("year", lazy(rf"\b(?:in|during|since|until|by)\s+{_YEAR}\b", re.I)),
    ("ago", lazy(rf"\b(?P<n>\d+|a|an|one|two|three)\s+{_UNIT}\s+ago\b", re.I)),
    ("past", lazy(rf"\b(?:past|last)\s+(?P<n>\d+)\s+{_UNIT}\b", re.I)),
    ("rel_unit", lazy(rf"\b(?P<rel>last|this|next|previous|coming)\s+{_UNIT}\b", re.I)),
    ("rel_wd", lazy(rf"\b(?P<rel>last|this|next|previous|coming)\s+{_WD}\b", re.I)),
    ("on_wd", lazy(rf"\b(?:on|since|by)\s+{_WD}\b", re.I)),
    ("word", lazy(r"\b(?P<w>today|yesterday|tomorrow|tonight)\b", re.I)),
]
_SMALL_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}

# Cheap pre-checks: a pattern only runs when the text has the words it needs
# (most memories hold no date at all, and this runs on every candidate).
_WORD_RE = lazy(r"[a-z0-9]+")
_NEED_WORDS = {
    "month": frozenset(_MONTHS),
    "weekday": frozenset(_WEEKDAYS),
    "unit": frozenset({"day", "days", "week", "weeks", "month", "months", "quarter", "quarters", "year", "years"}),
    "quarter": frozenset({"q1", "q2", "q3", "q4", "quarter"}),
    "half": frozenset({"h1", "h2"}),
    "word": frozenset({"today", "yesterday", "tomorrow", "tonight"}),
}
_NEEDS = {
    "iso": ("digit",), "numeric": ("digit",), "day_month": ("digit", "month"), "month_day": ("digit", "month"),
    "iso_month": ("digit",), "quarter": ("quarter",), "half": ("half",), "month_year": ("digit", "month"),
    "month": ("month",), "year": ("digit",), "ago": ("unit",), "past": ("digit", "unit"),
    "rel_unit": ("unit",), "rel_wd": ("weekday",), "on_wd": ("weekday",), "word": ("word",),
}


@dataclass(frozen=True)
class DateSpan:
    """A date expression: inclusive calendar range plus where it occurs in the text."""
    start: date
    end: date
    text: str
    pos: tuple[int, int]

    def overlaps(self, start: date, end: date) -> bool:
        """True when this span shares at least one day with [start, end]."""
        return self.start <= end and start <= self.end


def as_date(value: date | datetime | None) -> date:
    """A calendar date for *value* (today, UTC, when None)."""
    if value is None:
        return datetime.now(timezone.utc).date()
    return value.date() if isinstance(value, datetime) else value


def find_dates(text: str, reference: date | datetime | None = None, *, day_first: bool = True) -> list[DateSpan]:
    """Every date expression in *text*, resolved against *reference* (see module docstring)."""
    ref = as_date(reference)
    taken: list[tuple[int, int]] = []
    spans: list[DateSpan] = []
    words = set(_WORD_RE.findall(text.lower()))
    present = {need: not vocabulary.isdisjoint(words) for need, vocabulary in _NEED_WORDS.items()}
    present["digit"] = any(ch.isdigit() for ch in text)
    for kind, pattern in _PATTERNS:
        if not all(present[need] for need in _NEEDS[kind]):
            continue
        for match in pattern.finditer(text):
            s, e = match.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            rng = _resolve(kind, match, ref, day_first)
            if rng is None:
                continue
            taken.append((s, e))
            spans.append(DateSpan(rng[0], rng[1], match.group(0), (s, e)))
    spans.sort(key=lambda span: span.pos)
    return spans


def parse_when(value: str | date | datetime | None, reference: date | datetime | None = None,
               *, end: bool = False) -> datetime | None:
    """A ``since=`` / ``until=`` bound: datetime, date, ISO text or a natural expression.

    ``end=True`` gives the last instant of a natural range ("last week" ends on
    Sunday 23:59:59), so ``until="last week"`` includes the whole week.
    Raises ValueError for text that holds no date.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return _day_bound(value, end)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is not None and len(text) > 10:  # a real timestamp, not just a day
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    spans = find_dates(text, reference)
    if not spans:
        raise ValueError(f"no date found in {value!r}")
    return _day_bound(spans[0].end if end else spans[0].start, end)


def _day_bound(day: date, end: bool) -> datetime:
    clock = datetime.max.time() if end else datetime.min.time()
    return datetime.combine(day, clock, tzinfo=timezone.utc)


def _resolve(kind: str, m: re.Match[str], ref: date, day_first: bool) -> tuple[date, date] | None:
    """Calendar range for one match, or None when the numbers are not a valid date."""
    try:
        if kind == "iso":
            return _day(date(int(m["y"]), int(m["m"]), int(m["d"])))
        if kind == "numeric":
            a, b, y = int(m["a"]), int(m["b"]), int(m["y"])
            year = y + 2000 if y < 100 else y
            day_first_here = a > 12 or (day_first and b <= 12)
            day, month = (a, b) if day_first_here else (b, a)
            return _day(date(year, month, day))
        if kind in ("day_month", "month_day"):
            year = int(m["year"]) if m["year"] else ref.year
            return _day(date(year, _MONTHS[m["mon"].lower()], int(m["day"])))
        if kind == "iso_month":
            return _month(int(m["y"]), int(m["m"]))
        if kind == "month_year":
            return _month(int(m["year"]), _MONTHS[m["mon"].lower()])
        if kind == "month":
            return _month(ref.year, _MONTHS[m["mon"].lower()])
        if kind == "year":
            return date(int(m["year"]), 1, 1), date(int(m["year"]), 12, 31)
        if kind == "quarter":
            q = int(m["q"]) if m["q"] else _ORDINALS[m["qw"].lower()]
            return _months(int(m["year"]) if m["year"] else ref.year, 3 * q - 2, 3)
        if kind == "half":
            return _months(int(m["year"]) if m["year"] else ref.year, 6 * int(m["h"]) - 5, 6)
    except ValueError:
        return None
    return _resolve_relative(kind, m, ref)


def _resolve_relative(kind: str, m: re.Match[str], ref: date) -> tuple[date, date] | None:
    if kind == "word":
        offset = {"today": 0, "tonight": 0, "yesterday": -1, "tomorrow": 1}[m["w"].lower()]
        return _day(ref + timedelta(days=offset))
    if kind in ("ago", "past"):
        raw = m["n"].lower()
        n = int(raw) if raw.isdigit() else _SMALL_NUMBERS[raw]
        unit = m["unit"].lower()
        start = _shift(ref, unit, -n)
        # "3 days ago" is that day; "past 3 days" runs from then to the reference.
        return (start, ref) if kind == "past" else _unit_range(start, unit)
    if kind == "rel_unit":
        step = {"last": -1, "previous": -1, "this": 0, "next": 1, "coming": 1}[m["rel"].lower()]
        unit = m["unit"].lower()
        return _unit_range(_shift(ref, unit, step), unit)
    if kind in ("rel_wd", "on_wd"):
        target = _WEEKDAYS[m["wd"].lower()]
        rel = m["rel"].lower() if kind == "rel_wd" else "this"
        if rel in ("last", "previous"):  # the most recent such day strictly before the reference
            return _day(ref - timedelta(days=(ref.weekday() - target - 1) % 7 + 1))
        if rel in ("next", "coming"):   # the first such day strictly after the reference
            return _day(ref + timedelta(days=(target - ref.weekday() - 1) % 7 + 1))
        return _day(ref - timedelta(days=ref.weekday()) + timedelta(days=target))
    return None


def _day(day: date) -> tuple[date, date]:
    return day, day


def _month(year: int, month: int) -> tuple[date, date]:
    return _months(year, month, 1)


def _months(year: int, month: int, count: int) -> tuple[date, date]:
    """``count`` calendar months starting at year-month."""
    last_month = month + count - 1
    return date(year, month, 1), date(year, last_month, calendar.monthrange(year, last_month)[1])


def _shift(ref: date, unit: str, n: int) -> date:
    """*ref* moved by n units (months clamp to the month's last day)."""
    if unit == "day":
        return ref + timedelta(days=n)
    if unit == "week":
        return ref + timedelta(weeks=n)
    months = n * {"month": 1, "quarter": 3, "year": 12}[unit]
    index = ref.year * 12 + ref.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(ref.day, calendar.monthrange(year, month + 1)[1]))


def _unit_range(day: date, unit: str) -> tuple[date, date]:
    """The calendar day / ISO week (Mon-Sun) / month / quarter / year containing *day*."""
    if unit == "day":
        return _day(day)
    if unit == "week":
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=6)
    if unit == "month":
        return _month(day.year, day.month)
    if unit == "quarter":
        return _months(day.year, 3 * ((day.month - 1) // 3) + 1, 3)
    return date(day.year, 1, 1), date(day.year, 12, 31)
