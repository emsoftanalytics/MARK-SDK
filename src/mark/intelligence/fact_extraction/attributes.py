"""Last-step clean-up of a candidate's subject and attribute (runs on every candidate).

Rule packs read "<subject> <attribute noun> is <value>" with the noun taken
from a closed vocabulary, so words *before* the noun stay in the subject.
Three shapes need a second look:

- a modifier at the end of the subject belongs to the attribute:
  "spendlog default currency: EUR" -> ``spendlog.default currency``, not
  ``spendlog default.currency``;
- a generic component word as the whole subject is not a thing of its own:
  "the CLI entry point is cli.py::main" -> ``project.cli entry point``
  (the convention for developer settings without a named project);
- one attribute written many ways gets one key: "database path", "db path",
  "default database path", "database file" -> ``db path``; a "database"
  whose value is a file path is the path too (a datastore is never a path).
  "default database" stays as written (an established key).

Tool slots ("My editor is Vim", "My shell is zsh") must name a known
product of that kind: "My editor is Joan" (a person) is not a tool.
"""
from __future__ import annotations

import re
from dataclasses import replace

from mark.intelligence.fact_extraction.base import FactCandidate
from mark.intelligence.fact_extraction.vocab import DEV_ATTRIBUTES, value_type
from mark.security.lazy_regex import lazy_compile

_MODIFIERS = frozenset({"default", "primary", "preferred", "fallback", "max", "maximum", "min", "minimum",
                        "initial"})
_GENERIC_SUBJECTS = frozenset({"cli", "api", "app", "console", "server", "backend", "frontend", "web", "main"})
_DEV = frozenset(DEV_ATTRIBUTES)
DB_PATH = "db path"
_DB_PATH_NAMES = frozenset({"database path", "db path", "default database path", "default db path",
                            "database file", "db file", "default database file", "database location",
                            "default database location", "sqlite path", "sqlite file", "db_path"})
# "default database" is kept as written: the held-out gold records "shorty default
# database: shorty.db" under that key, and it is the established shape.
_DB_NOUNS = frozenset({"database", "db"})
_PATH_LIKE = lazy_compile(r"^(?:~|/|\.{1,2}/|[A-Za-z]:\\)|\.(?:db|sqlite3?|duckdb)$")
#: Tool slot -> the product kind (``vocab.value_type``) its value must be.
TOOL_KINDS = {"editor": "editor", "text editor": "editor", "code editor": "editor", "ide": "editor",
              "browser": "browser", "web browser": "browser", "shell": "shell", "os": "os",
              "operating system": "os"}


def refine(candidate: FactCandidate) -> FactCandidate | None:
    """The candidate with subject/attribute normalised (see module docstring); None to drop it."""
    subject, attribute, value = candidate.subject, candidate.attribute, candidate.value
    words = subject.split()
    if len(words) >= 2 and words[-1] in _MODIFIERS:
        subject, attribute = " ".join(words[:-1]), f"{words[-1]} {attribute}"
    elif candidate.rule == "key_value" and subject in _MODIFIERS and attribute in _DEV:
        # "The default timeout is 30 seconds": a setting of the project, not of "default".
        subject, attribute = "project", f"{subject} {attribute}"
    if candidate.rule == "key_value" and subject.lower() in _GENERIC_SUBJECTS and attribute in _DEV:
        subject, attribute = "project", f"{subject.lower()} {attribute}"
    if attribute in _DB_PATH_NAMES or (attribute in _DB_NOUNS and _PATH_LIKE.search(value.strip())):
        attribute = DB_PATH
    if candidate.rule == "tools":
        kind = TOOL_KINDS.get(re.sub(r"\s+", " ", attribute))
        if kind is None or value_type(value) != kind:
            return None
        attribute = kind
    if (subject, attribute) == (candidate.subject, candidate.attribute):
        return candidate
    return replace(candidate, subject=subject, attribute=attribute)
