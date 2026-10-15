"""First-person rewriting: "I"/"my"/"me" statements become statements about the user.

Rule packs are written for third-person clauses ("X lives in Y"). Rather
than duplicating every pack for "I live in Y", a sentence is rewritten
before matching:

    "I live in Lagos and work at Seefar" -> "MARKUSER lives in Lagos and work at Seefar"
    "My sister is Ines"                 -> "MARKUSER's sister is Ines"
    "Priya manages me"                  -> "Priya manages MARKUSER"

and :data:`USER_TOKEN` is replaced by the configured user subject (default
``"user"``) once candidates are built. Continuation clauses ("and work at
...") borrow the subject and are conjugated per clause.

Third-person pronouns are resolved only when trivially safe: exactly one
person has been named so far in the text. Otherwise the clause keeps its
pronoun and the subject hygiene in :mod:`.base` drops it.
"""
from __future__ import annotations

import re

from mark.intelligence.fact_extraction.vocab import RELATION_ALT

#: Placeholder subject for the speaker; one identifier-like token so every
#: pack's subject pattern accepts it, and unlikely to occur in real text.
USER_TOKEN = "MARKUSER"

# "I" only at a clause start (sentence start, after ", ; :" or "and/but/so"),
# so the roman numeral in "World War I" or "Phase I" is left alone.
_BOUNDARY = r"(?:^|(?<=[,;:])\s*|(?<=\s)(?:and|but|so|because|since|now|then)\s+|(?<=\())"
_I_FORMS = (
    (re.compile(rf"{_BOUNDARY}\bI\s*['’]m\b|{_BOUNDARY}\bI\s+am\b"), "is"),
    (re.compile(rf"{_BOUNDARY}\bI\s*['’]ve\b|{_BOUNDARY}\bI\s+have\b"), "has"),
    (re.compile(rf"{_BOUNDARY}\bI\s*['’]ll\b|{_BOUNDARY}\bI\s+will\b"), "will"),
    (re.compile(rf"{_BOUNDARY}\bI\s*['’]d\b"), "would"),
    (re.compile(rf"{_BOUNDARY}\bI\s+don['’]t\b|{_BOUNDARY}\bI\s+do\s+not\b"), "does not"),
)
_I_VERB = re.compile(rf"{_BOUNDARY}\bI\s+(?P<adverb>(?:(?:also|now|currently|still|really|actually|just|usually"
                     r"|mostly|no\s+longer|recently|finally|totally)\s+)*)(?P<verb>[a-z][a-z'-]*)\b")
_MY = re.compile(r"(?<![\w'])[Mm]y\s+(?=[a-z])")
_ME = re.compile(r"(?<![\w'])me\b(?!['’-])")
# Verbs that keep their form after "I": past tenses and modals.
_KEEP = frozenset("was were had did could would should might must can may will shall used".split())
_IRREGULAR = {"am": "is", "be": "is", "have": "has", "do": "does", "go": "goes"}

# "<user>'s sister Ines lives in Faro": a relation in apposition to a name.
_APPOSITION = re.compile(
    rf"^(?P<owner>{USER_TOKEN}|(?-i:[A-Z])[\w.-]*)['’]s\s+(?P<relation>{RELATION_ALT})\s+"
    r"(?P<name>(?-i:[A-Z])[\w'-]*(?:\s+(?-i:[A-Z])[\w'-]*){0,2})\s+(?P<rest>(?!(?:is|was|are)\b)(?-i:[a-z]).+)$")
_DIET_NEG = re.compile(
    r"^(?P<who>.+?)\s+(?:no\s+longer\s+eats?|does\s+not\s+eat|doesn['’]t\s+eat|(?:has\s+)?stopped\s+eating"
    r"|(?:has\s+)?quit\s+eating|(?:has\s+)?given\s+up|gave\s+up(?:\s+eating)?|never\s+eats?)\s+"
    r"(?P<food>meat|red\s+meat|fish|dairy|pork|beef|gluten|sugar|eggs|alcohol)(?:\s+any\s?more)?"
    # A trailing time phrase dates the change; it is not part of the diet.
    r"(?:\s+(?:last|this)\s+(?:year|month|week|spring|summer|autumn|fall|winter)|\s+recently|\s+in\s+\d{4}"
    r"|\s+since\s+\S+)?\s*$",
    re.IGNORECASE)
# Addresses keep their commas: a word joiner after the comma stops clause
# splitting and value truncation; :func:`restore` puts the space back.
_ADDRESS = re.compile(r"(\baddress\s*(?:is|:)\s+)([^.;!?]+)", re.IGNORECASE)
_JOINER = ",⁠"
_HE_SHE = re.compile(r"^(?:he|she)\b", re.IGNORECASE)
_HIS_HER = re.compile(r"^(?:his|her)\s+(?!other\b)", re.IGNORECASE)
_PERSON_NAME = re.compile(r"^(?:(?:Dr|Mr|Mrs|Ms|Prof)\.\s+)?[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){0,2}$")
_USER_WORDS = re.compile(r"^(?:the\s+)?user$", re.IGNORECASE)
# Base-form verbs that continue a first-person clause ("... and work at X"),
# matching the continuation verbs in :mod:`.clauses`.
_CONTINUATION_BASE = "work|live|own|lead|use|prefer|like|love|hate|enjoy|speak|drive"
_NOT_VERBS = frozenset("think guess believe suppose hope wish feel mean know want need wonder".split())


def conjugate(verb: str) -> str:
    """Third-person singular of a base-form verb ("live" -> "lives", "carry" -> "carries")."""
    low = verb.lower()
    if low in _IRREGULAR:
        return _IRREGULAR[low]
    if low in _KEEP or low.endswith("ed") or low.endswith("n't"):
        return verb
    if re.search(r"(?:s|sh|ch|x|z|o)$", low):
        return verb + "es"
    if re.search(r"[^aeiou]y$", low):
        return verb[:-1] + "ies"
    return verb + "s"


def rewrite_first_person(sentence: str) -> str:
    """Rewrite I/my/me as statements about :data:`USER_TOKEN` (see module docstring)."""
    if not re.search(r"\b(?:I|[Mm]y|me)\b|\bI['’]", sentence):
        return sentence
    sentence = _ADDRESS.sub(lambda m: m.group(1) + m.group(2).replace(", ", _JOINER), sentence)
    for pattern, verb in _I_FORMS:
        sentence = pattern.sub(lambda m, v=verb: _keep_boundary(m) + f"{USER_TOKEN} {v}", sentence)

    def _verb(match: re.Match[str]) -> str:
        verb = match.group("verb")
        if verb.lower() in _NOT_VERBS:
            return match.group(0)  # "I think ..." reports an opinion, not a fact
        return _keep_boundary(match) + f"{USER_TOKEN} {match.group('adverb')}{conjugate(verb)}"

    sentence = _I_VERB.sub(_verb, sentence)
    sentence = _MY.sub(f"{USER_TOKEN}'s ", sentence)
    return _ME.sub(USER_TOKEN, sentence)


def _keep_boundary(match: re.Match[str]) -> str:
    """The clause-boundary text a first-person pattern matched before "I"."""
    text = match.group(0)
    index = text.find("I")
    return text[:index] if index > 0 else ""


def conjugate_user_clause(clause: str) -> str:
    """A borrowed-subject clause "<user> work at X" gets its verb conjugated."""
    match = re.match(rf"^{USER_TOKEN}\s+({_CONTINUATION_BASE})\b", clause)
    if match is None:
        return clause
    return f"{USER_TOKEN} {conjugate(match.group(1))}{clause[match.end():]}"


def split_apposition(clause: str) -> tuple[tuple[str, str, str] | None, str]:
    """"<user>'s sister Ines lives in Faro" -> ((<user>, "sister", "Ines"), "Ines lives in Faro")."""
    match = _APPOSITION.match(clause)
    if match is None:
        return None, clause
    owner, relation, name = match.group("owner"), match.group("relation"), match.group("name")
    return (owner, relation, name), f"{name} {match.group('rest')}"


def rewrite_diet(clause: str) -> str:
    """"<who> no longer eats meat" -> "<who> diet is no meat" (a current diet, not a negation)."""
    match = _DIET_NEG.match(clause.rstrip(".!"))
    if match is None:
        return clause
    return f"{match.group('who')}'s diet is no {match.group('food').lower()}"


def resolve_pronoun(clause: str, antecedent: str | None) -> str:
    """Replace a leading he/she/his/her with the single person named so far (else unchanged)."""
    if not antecedent:
        return clause
    if _HE_SHE.match(clause):
        return _HE_SHE.sub(antecedent, clause, count=1)
    if _HIS_HER.match(clause):
        return _HIS_HER.sub(f"{antecedent}'s ", clause, count=1)
    return clause


def single_person(subjects: list[str]) -> str | None:
    """The antecedent for a pronoun: exactly one distinct person-like subject, else None."""
    people = {s for s in subjects if _PERSON_NAME.match(s) and s != USER_TOKEN}
    return next(iter(people)) if len(people) == 1 else None


def is_user(subject: str) -> bool:
    """True for the placeholder and for "user"/"the user" written in the text."""
    return subject == USER_TOKEN or bool(_USER_WORDS.match(subject))


def restore(value: str) -> str:
    """Undo the address comma protection in a final value."""
    return value.replace(_JOINER, ", ")
