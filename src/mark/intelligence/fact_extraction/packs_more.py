"""Built-in packs for habits, counts, tools and project settings.

Phrasing the 0.2.0a10 re-tests found unrecorded, each with a precision guard:

  - ``drinks``: "I drink oat milk lattes" -> ``user.drinks`` (multi-valued);
    adverbs of quantity/manner ("a lot", "socially") are never a drink;
  - ``count``: "atlas-core has 6 replicas" -> ``atlas-core.replicas = 6``; only a
    number (digits or a word) before a plural noun that ends the clause, and
    never a time span ("has 3 days left");
  - ``uses``: "ledger-api uses pnpm" -> ``package manager``; only when the value is
    a known product (see :func:`.vocab.value_type`), so "uses lots of memory" is dropped;
  - ``project``: "The test command is `make test`" -> ``project.test command``: a
    developer setting stated without a subject belongs to the current project
    (MARK stores are usually one per project);
  - ``status_of``: "Status of OPS-88: resolved" -> ``OPS-88.status``;
  - ``appointment``: "The dentist appointment was moved to 14 November at 10:00"
    -> ``dentist appointment.date``; passive verb and a day/date/time value only.
"""
from __future__ import annotations

from mark.intelligence.fact_extraction.numbers import NUMBER_WORD, parse_number
from mark.intelligence.fact_extraction.packs_people import DATE
from mark.intelligence.fact_extraction.personal import USER_TOKEN
from mark.intelligence.fact_extraction.rules import NOW, VALUE, RulePack

_NAME = r"(?-i:[A-Z][\w'’.-]*)(?:\s+(?-i:[A-Z][\w'’.-]*)){0,2}"
_PERSON = rf"(?P<subject>{USER_TOKEN}|{_NAME})"

DRINKS = RulePack(
    name="drinks",
    patterns=(rf"^{_PERSON}\s+(?:usually\s+|mostly\s+|only\s+|mainly\s+|also\s+)?(?:drinks|is\s+drinking)\s+"
              r"(?!(?:a\s+lot|lots|too|much|more|less|when|if|socially|heavily|occasionally|rarely|sometimes"
              r"|responsibly|regularly|it|that|this|every|daily|at|with|in|on|like)\b)"
              r"(?:a\s+|an\s+|some\s+)?(?P<value>[\w'’-]+(?:\s+[\w'’-]+){0,3}?)"
              r"(?=\s*$|\s+(?:every|each|daily|at|in|with|when|because|but|and|before|after|on|most|instead"
              r"|now|nowadays|these\s+days)\b)",),
    attribute="drinks",
    attribute_from_value=False,
    multi=True,
    confidence=0.7,
    trigger_words=('drinks', 'drinking'),
)

# Plural nouns that measure time are spans or deadlines, not counts of things.
_TIME_NOUNS = r"(?:years|months|weeks|days|hours|minutes|seconds|times|nights|mornings|evenings|left)"
COUNT = RulePack(
    name="count",
    patterns=(rf"^{{S}}\s+(?:now\s+|currently\s+|still\s+|also\s+)?(?:has|have|owns)\s+"
              r"(?:about\s+|around\s+|over\s+|exactly\s+|only\s+)?"
              rf"(?P<value>\d{{1,7}}|{NUMBER_WORD})\s+(?!\w+\s+{_TIME_NOUNS}\b|{_TIME_NOUNS}\b)"
              r"(?P<attribute_free>(?:[a-z][\w-]*\s+)?[a-z][a-z-]*s)\s*$",),
    attribute_from_value=False,
    value_map={word: str(parse_number(word)) for word in (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
        " seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety").split()},
    confidence=0.7,
    trigger=r"\b(?:has|have|owns)\s+(?:about\s+|around\s+|over\s+|exactly\s+|only\s+)?(?:\d|[a-z]+\s+[a-z])",
)

USES = RulePack(
    name="uses",
    patterns=(rf"^{{S}}\s+{NOW}(?:uses|is\s+using|is\s+built\s+with|is\s+written\s+in)\s+"
              r"(?P<value>[A-Za-z][\w.+#!-]*(?:\s+[A-Za-z0-9][\w.+#-]*){0,2}?)\s*$",),
    attribute_from_value=False,
    attribute_from_type=True,
    confidence=0.7,
    trigger_words=('uses', 'using', 'built', 'written'),
)

# "<word> command" is always a developer setting ("deploy command", "format command").
_PROJECT_SETTING = (r"(?:[a-z][\w-]*\s+){1,2}command|config\s+file|config\s+path|configuration\s+file"
                    r"|entry\s*point|package\s+manager|build\s+tool|test\s+runner|linter|formatter|type\s+checker"
                    r"|python\s+version|node\s+version|default\s+branch|docs\s+url|repo\s+url|homepage")
PROJECT = RulePack(
    name="project",
    patterns=(rf"^(?:the|our|this\s+project['’]s|the\s+project['’]s)\s+(?P<attribute>{_PROJECT_SETTING})\s*"
              r"(?::|=|is\s+now|is)\s*" + VALUE,),
    subject="project",
    attribute_from_value=False,
    confidence=0.7,
    trigger=r"(?i)^(?:the|our|this)\s.*?\b(?:command|config|configuration|entry|package|build|runner|linter"
            r"|formatter|checker|version|branch|url|homepage)\b",
)

STATUS_OF = RulePack(
    name="status_of",
    patterns=(r"^(?:the\s+)?(?P<attribute>status|state|priority|severity)\s+(?:of|for)\s+{S}\s*"
              r"(?::|=|->|is\s+now|is)\s*{V}",),
    value_case="lower",
    value_map={"in-progress": "in progress", "inprogress": "in progress", "wip": "in progress"},
    attribute_from_value=False,
    confidence=0.8,
    trigger_words=('status', 'state', 'priority', 'severity'),
)

# Events that get rescheduled. The value must be a day, date or time, and the
# verb passive ("was moved to"): "the meeting moved to room 4" is not a date.
_EVENT = (r"(?:[\w'’-]+\s+){0,2}?(?:appointment|meeting|call|interview|session|stand-?up|review|demo|dinner|lunch"
          r"|class|lesson|flight|booking|reservation|party|wedding|visit|check-?up|consultation|exam|launch"
          r"|kick-?off|retro|retrospective|sync|one-on-one|1:1|deadline|presentation|workshop)")
_DAY_NAME = (r"(?:(?:next|this|coming)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
             r"|tomorrow|today|tonight")
_CLOCK = r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{1,2}:\d{2}|noon|midday"
_WHEN = (rf"(?P<value>(?:{_DAY_NAME}|{DATE})(?:\s*(?:,|at)\s*(?:{_CLOCK}))?|(?:{_CLOCK})(?:\s+(?:on\s+)?"
         rf"(?:{_DAY_NAME}|{DATE}))?)")
APPOINTMENT = RulePack(
    name="appointment",
    # "My appointment" arrives as "<user>'s appointment" after first-person rewriting.
    patterns=(rf"^(?:the\s+|my\s+|our\s+|your\s+|{USER_TOKEN}['’]s\s+)?(?P<subject>{_EVENT})\s+(?:was|has\s+been|have\s+been|had\s+been"
              r"|got|is\s+now|is)\s+(?:moved|rescheduled|postponed|pushed(?:\s+back)?|brought\s+forward|changed"
              rf"|delayed|shifted|set)\s+(?:to|until|till|for)\s+{_WHEN}\s*$",),
    attribute="date",
    attribute_from_value=False,
    confidence=0.75,
    trigger_words=('moved', 'rescheduled', 'postponed', 'pushed', 'brought', 'changed', 'delayed', 'shifted', 'set'),
)

#: Tried after the people packs and before the generic ``key: value`` pack.
MORE_PACKS: tuple[RulePack, ...] = (STATUS_OF, PROJECT, DRINKS, COUNT, USES, APPOINTMENT)
