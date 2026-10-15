"""Rule packs: the data that drives :class:`PatternFactExtractor`.

A pack is a list of regular expressions plus a little mapping data. It
contains no code, so users can write their own for domain phrasing
("<S> was promoted to <V>" → title) and pass them alongside, or instead of,
the built-ins::

    RulePack("titles", patterns=(r"^{S}\\s+was\\s+promoted\\s+to\\s+{V}",), attribute="title")

Pattern conventions:
  - patterns are matched against one *clause* at a time (sentences split on
    ``. ! ? ;`` and ", "), case-insensitively, after a leading date stamp and
    an upper-case log label ("CHANGE:", "HR:") have been removed;
  - named groups: ``subject`` and ``value`` (required), ``attribute`` /
    ``attribute_free`` (optional), ``verb`` (looked up in
    ``verb_attributes``) and ``old`` (the "from ..." part, used only to infer
    the attribute);
  - placeholders ``{S}``, ``{V}`` and ``{A}`` expand to the shared subject,
    value and attribute-vocabulary building blocks below.

Built-in packs describe generic English phrasing of state changes. They are
deliberately not tuned to any one dataset.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Generic attribute nouns: recognised between a verb and "to", as the first
#: word of a value ("port 4444") or the last ("the Platform team").
ATTRIBUTE_WORDS: tuple[str, ...] = (
    "port", "region", "zone", "availability zone", "datacenter", "data center", "cluster", "host",
    "hostname", "datastore", "data store", "database", "db", "storage", "queue", "endpoint", "url",
    "domain", "address", "ip", "version", "release", "tier", "plan", "branch", "environment",
    "team", "department", "group", "squad", "division", "office", "city", "location", "country",
    "timezone", "role", "title", "manager", "owner", "maintainer", "assignee", "status", "state",
    "stage", "phase", "priority", "severity", "name", "email", "phone", "channel", "budget",
    "price", "quota", "schedule", "deadline", "language", "framework", "runtime", "provider",
    "vendor", "license", "currency",
)

#: Spelling variants folded onto one attribute key.
ATTRIBUTE_SYNONYMS: dict[str, str] = {
    "db": "database", "data store": "datastore", "data center": "datacenter",
    "hostname": "host", "dept": "department",
}

# Longest first so "availability zone" wins over "zone".
_ATTR_ALT = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted(ATTRIBUTE_WORDS, key=len, reverse=True))

# One identifier-like token: names, hyphenated ids, dotted hosts, emails.
_TOKEN = r"[^\W_][\w.@/+#&'-]*"
#: Subject: 1-4 tokens anchored at the clause start (patterns begin with ^).
SUBJECT = rf"(?P<subject>{_TOKEN}(?:\s+{_TOKEN}){{0,3}}?)(?:'s)?"
#: Value: the shortest run that ends the clause or meets a connective
#: ("... to port 4444 on Monday", "... to Lisbon because ...").
VALUE = (
    r"(?P<value>[^;()]+?)"
    r"(?=\s*(?:[;()]|,(?:\s|$)|\.(?:\s|$)|$)"
    r"|\s+(?:and|but|because|after|before|since|on|at|as\s+of|effective|following|due\s+to|until"
    r"|while|when|which|per|instead\s+of|from|for|with|via|so|then)\b)"
)
ATTRIBUTE = rf"(?P<attribute>{_ATTR_ALT})"
#: Optional auxiliaries before a past-tense verb ("was moved", "has been moved").
_AUX = r"(?:(?:was|were|has|have|had|got|has\s+been|have\s+been|had\s+been|is\s+now|are\s+now)\s+)?"
_DET = r"(?:its|their|his|her|the|our|my|a|an)"


@dataclass(frozen=True)
class RulePack:
    """One named family of extraction patterns (pure data)."""
    name: str
    patterns: tuple[str, ...]
    #: Fixed attribute for every match (wins over anything found in the text).
    attribute: str | None = None
    #: Fallback attribute keyed by the lower-cased ``verb`` group.
    verb_attributes: dict[str, str] = field(default_factory=dict)
    #: "keep" or "lower": case normalisation applied to values.
    value_case: str = "keep"
    #: Read the attribute from a noun at the start/end of the value ("port 4444",
    #: "the Data team"). Off where the verb fixes the attribute ("owned by Team X").
    attribute_from_value: bool = True
    #: Exact (lower-case) value rewrites applied after normalisation.
    value_map: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.8

    def compiled(self) -> tuple[re.Pattern[str], ...]:
        """Patterns with placeholders expanded, compiled case-insensitively."""
        return tuple(_compile(p) for p in self.patterns)


def _compile(pattern: str) -> re.Pattern[str]:
    expanded = pattern.replace("{S}", SUBJECT).replace("{V}", VALUE).replace("{A}", ATTRIBUTE)
    return re.compile(expanded, re.IGNORECASE)


_CHANGE_VERBS = (r"moved|changed|switched|migrated|transferred|relocated|reassigned|shifted|updated"
                 r"|upgraded|downgraded|renamed|bumped|rehomed|re-homed|repointed|re-pointed")

CHANGES = RulePack(
    name="changes",
    patterns=(
        # "<S> [was] <verb> [its <attr> | <attr>] [from <old>] to <V>"
        rf"^{{S}}\s+{_AUX}(?P<verb>{_CHANGE_VERBS})"
        rf"(?:\s+(?:{_DET}\s+)?{{A}}|\s+{_DET}\s+(?P<attribute_free>(?!(?:from|to|into|onto)\b)[a-z][\w-]*))?"
        r"(?:\s+from\s+(?P<old>.+?))?\s+(?:to|into|onto)\s+{V}",
    ),
    # No default for "moved": "the meeting moved to Thursday" is not a location.
    verb_attributes={"relocated": "location", "rehomed": "location",
                     "re-homed": "location", "reassigned": "assignee", "renamed": "name",
                     "upgraded": "version", "downgraded": "version", "bumped": "version"},
    confidence=0.8,
)

OWNERSHIP = RulePack(
    name="ownership",
    patterns=(
        r"^{S}\s+(?:is|are|was|were|has\s+been|have\s+been)\s+(?:now\s+)?(?P<verb>owned|maintained|managed)\s+by\s+{V}",
        r"^(?:the\s+)?ownership\s+of\s+{S}\s+" + _AUX
        + r"(?:transferred|moved|passed|handed\s+over|changed|went)\s+to\s+{V}",
    ),
    # "" is the fallback when a pattern has no verb group ("ownership of X passed to V").
    verb_attributes={"owned": "owner", "maintained": "maintainer", "managed": "manager", "": "owner"},
    attribute_from_value=False,
    confidence=0.85,
)

_STATUS_WORDS = r"status|state|stage|phase|priority|severity"
STATUS = RulePack(
    name="status",
    patterns=(
        rf"^{{S}}\s+(?P<attribute>{_STATUS_WORDS})\s+{_AUX}(?:changed|updated|moved|set|switched|went|flipped)"
        r"(?:\s+from\s+(?P<old>.+?))?\s+to\s+{V}",
        rf"^(?:the\s+)?(?P<attribute>{_STATUS_WORDS})\s+of\s+{{S}}\s+{_AUX}(?:changed|updated|moved|set|switched|went)"
        r"(?:\s+from\s+(?P<old>.+?))?\s+to\s+{V}",
        r"^{S}\s+(?:was|has\s+been|is\s+now)\s+(?:marked|set)\s+(?:as\s+)?{V}",
    ),
    verb_attributes={"": "status"},
    value_case="lower",
    value_map={"in-progress": "in progress", "inprogress": "in progress", "wip": "in progress"},
    confidence=0.85,
)

MEMBERSHIP = RulePack(
    name="membership",
    # Only the attribute noun in the value says what was joined ("the Data team").
    patterns=(r"^{S}\s+(?:has\s+)?(?:joined|was\s+added\s+to|was\s+moved\s+into)\s+(?:the\s+)?{V}",),
    confidence=0.75,
)

ASSIGNMENT = RulePack(
    name="assignment",
    patterns=(r"^{S}\s+" + _AUX + r"(?P<verb>reassigned|assigned|handed\s+over)\s+to\s+{V}",),
    verb_attributes={"assigned": "assignee", "reassigned": "assignee", "handed over": "assignee"},
    confidence=0.8,
)

KEY_VALUE = RulePack(
    name="key_value",
    patterns=(
        r"^{S}\s+{A}\s*(?::|=|->|is\s+now|is\s+set\s+to|is)\s*{V}",
        r"^(?P<subject>[^\W_][\w-]*)\.(?P<attribute_free>[a-z][\w-]*)\s*=\s*{V}",
    ),
    confidence=0.7,
)

#: The built-in packs, in the order they are tried.
BUILTIN_PACKS: tuple[RulePack, ...] = (STATUS, OWNERSHIP, ASSIGNMENT, MEMBERSHIP, CHANGES, KEY_VALUE)
PACKS_BY_NAME: dict[str, RulePack] = {pack.name: pack for pack in BUILTIN_PACKS}
