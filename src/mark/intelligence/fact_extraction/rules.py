"""Rule-pack building blocks: the :class:`RulePack` type and shared regex pieces.

Pattern conventions:
  - patterns are matched against one *clause* at a time (sentences split on
    ``. ! ? ;``, ", " and "and" before a verb), case-insensitively, after a
    leading date stamp and an upper-case log label ("CHANGE:", "HR:") have
    been removed;
  - named groups: ``subject`` and ``value`` (required), ``attribute`` /
    ``attribute_free`` (optional), ``verb`` (looked up in
    ``verb_attributes``) and ``old`` (the "from ..." part, used only to infer
    the attribute);
  - placeholders ``{S}``, ``{V}`` and ``{A}`` expand to the shared subject,
    value and attribute-vocabulary building blocks below.

The built-in packs themselves live in :mod:`.packs`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from mark.intelligence.fact_extraction.vocab import DEV_ATTRIBUTES

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
) + DEV_ATTRIBUTES

#: Spelling variants folded onto one attribute key.
ATTRIBUTE_SYNONYMS: dict[str, str] = {
    "db": "database", "data store": "datastore", "data center": "datacenter",
    "hostname": "host", "dept": "department",
    # Personal details written several ways.
    "phone number": "phone", "mobile": "phone", "mobile number": "phone", "email address": "email",
    "date of birth": "birthday", "birthdate": "birthday", "dob": "birthday", "home address": "address",
    "home town": "hometown", "time zone": "timezone", "mum": "mother", "mom": "mother", "dad": "father",
    "allergies": "allergy", "configuration file": "config file", "entrypoint": "entry point",
    # One key for where a config lives: agents write all of these for the same thing.
    "config path": "config file", "configuration path": "config file", "config file path": "config file",
    "config_path": "config file", "config_file": "config file",
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
    # A `code span` is one value, whatever it contains ("`ruff check .`", "`make test on ci`").
    r"(?P<value>`[^`\n]+`|[^;()]+?)"
    # A full stop ends a value unless it closes a title ("Dr. Silva").
    r"(?=\s*(?:[;()]|,(?:\s|$)|(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bProf)(?<!\bSt)\.(?:\s|$)|$)"
    r"|\s+(?:and|but|because|after|before|since|on|at|as\s+of|effective|following|due\s+to|until"
    r"|while|when|which|instead\s+of|from|for|with|via|so|then)\b"
    # "in" + a time ends a value ("relocated to Accra in June"); "in staging" does not.
    r"|\s+in\s+(?:early\s+|late\s+|mid-?)?(?:\d{4}|jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july"
    r"|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|spring|summer|autumn|fall|winter"
    r"|q[1-4])\b"
    # "per" ends a value ("... per the runbook") except after an amount, where
    # it is part of the unit: "$79 per month", "42 euros per square metre".
    r"|(?<![\d%])(?<!euros)(?<!euro)(?<!dollars)(?<!pounds)(?<!EUR)(?<!USD)(?<!GBP)\s+per\b)"
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
    #: Swap subject and value when only the value looks like an identifier
    #: ("Bo Kim is assigned to OPS-17" -> OPS-17.assignee = Bo Kim).
    swap_when_value_is_id: bool = False
    confidence: float = 0.8
    #: Optional cheap pre-checks (speed-ups only; they never change results):
    #: at least one of ``trigger_words`` must occur as a word in the clause, and
    #: ``trigger`` (a regex) must be found in it, for any pattern to match.
    trigger_words: tuple[str, ...] = ()
    trigger: str | None = None
    #: Fixed subject for patterns without a subject group ("The test command is X" -> project).
    subject: str | None = None
    #: Values are one of several current values (recorded with ``facts.add``).
    multi: bool = False
    #: Name the attribute after the kind of product in the value ("uses pnpm" -> package manager);
    #: the match is dropped when the value is not a known product.
    attribute_from_type: bool = False
    #: Only match when the sentence carries a date ("From 2027-02-01 the team moves to Lisbon"):
    #: present-tense moves without one are narration or plans.
    needs_date: bool = False
    #: The pattern states a change of value, so an undated value of the slot
    #: stamped later than the change is older than it (see ``FactCandidate.change``).
    change: bool = False

    def compiled(self) -> tuple[re.Pattern[str], ...]:
        """Patterns with placeholders expanded, compiled case-insensitively."""
        return tuple(_compile(p) for p in self.patterns)

    def compiled_trigger(self) -> re.Pattern[str] | None:
        """The trigger compiled case-insensitively (None: always try the patterns)."""
        return re.compile(self.trigger, re.IGNORECASE) if self.trigger else None


def _compile(pattern: str) -> re.Pattern[str]:
    expanded = pattern.replace("{S}", SUBJECT).replace("{V}", VALUE).replace("{A}", ATTRIBUTE)
    return re.compile(expanded, re.IGNORECASE)

# ── blocks for present-tense descriptions ("X runs on port 8080") ─────────────

#: Optional "now"/"currently" before a present-tense verb.
NOW = r"(?:(?:now|currently|still|also)\s+)?"
# Words that refer elsewhere or to nobody: never a party in "<party> owns X".
_NON_PARTY = r"(?!(?:it|he|she|they|we|i|you|this|that|these|those|there|everyone|everybody|someone"
_NON_PARTY += r"|somebody|nobody|no\s+one|anyone|anybody|who|which|what)\b)"
#: Value written *before* the verb ("Dana owns X"): 1-4 tokens, not a pronoun.
PARTY = rf"(?P<value>{_NON_PARTY}{_TOKEN}(?:\s+{_TOKEN}){{0,3}}?)"
#: Subject written *after* the verb, running to the clause end (or a trailing
#: time phrase): "Dana owns the atlas-core service since May".
OBJECT_SUBJECT = (rf"(?:the\s+)?(?P<subject>(?!to\b){_TOKEN}(?:\s+{_TOKEN}){{0,3}}?)"
                  r"(?=\s*$|\s+(?:now|today|since|as\s+of|effective|starting|from|on)\b)")
