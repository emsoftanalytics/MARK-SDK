"""Word-level relevance helpers for briefings: content words, floors, sequences, gaps.

Retrieval scores are relative (a lexical hash embedder gives 0.4-0.6 to
lines that share no word with the question), so a small model's briefing
needs its own cut: lines are kept for what they share with the question,
not for where they happen to land in a top-k (a10 small-model run: the
licence-plate question was briefed with the dog and the sister lines, and
1B models answered from them).

* :func:`content_words` — lower-case, lightly stemmed words without stop words.
* :func:`above_floor` — the request's records that clear the relevance floor.
* :func:`is_sequence` / :func:`sequence_order` — "Step 2", "days 4 to 6":
  multi-turn sequences, shown oldest first so step 1 precedes step 4.
* :func:`missing_relations` — "my brother", "my car" in a question that no
  memory mentions: reported as "Not in memory" instead of left to guesswork.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ROLE = re.compile(r"^\s*(user said|user|assistant replied|assistant said)\s*:\s*", re.IGNORECASE)
STOP = frozenset("""
a an and are as at be been being but by can could did do does doing done for from had has have
having he her hers him his how i i'm im if in into is it its it's me my mine myself no nor not now of
off on once only or our ours out over own same she should so some such than that the their them then
there these they this those through to too under until up very was we were what whats when where
which while who whom whose why will with would you your yours yourself s t d ll ve re m just also
about again all any both each few more most other here tell please one get got let lets
""".split())
#: Relations and possessions a personal question asks about ("my brother's name").
RELATIONS = frozenset("""
brother sister mother mom mum father dad parent parents son daughter child children kid kids wife
husband partner boyfriend girlfriend fiance fiancee grandmother grandma grandfather grandpa
grandparents aunt uncle cousin niece nephew boss manager colleague coworker neighbour neighbor
friend roommate flatmate doctor dentist therapist lawyer teacher dog cat pet pets car bike
bicycle motorbike house flat apartment home address phone laptop passport licence license
plate birthday anniversary email
""".split())
_SEQUENCE = re.compile(r"\b(step|steps|day|days|part|phase|stage|week|round)\s+\d+", re.IGNORECASE)
_MY_THING = re.compile(r"\bmy\s+((?:[a-z]+\s+){0,2}?[a-z]+?)(?:'s\b|\b)", re.IGNORECASE)
_INTRODUCING = re.compile(r"\bmy\s+[a-z]+(?:'s)?\s+(?:[A-Z][a-z]+\s+)?(is|are|was|were|has|had|will|'s)\b",
                          re.IGNORECASE)


@lru_cache(maxsize=16_384)   # a vocabulary is small; long memories repeat words
def stem(word: str) -> str:
    """A crude, dependency-free stem: enough to match "lives"/"live", "steps"/"step"."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return word.rstrip("e") if len(word) > 4 else word


#: Texts up to this length are memoised (the bound keeps the cache's memory
#: at ~512 x 32 KB worst case); longer ones are rare and stemmed each time.
_CACHE_MAX_CHARS = 32_768


def content_words(text: str) -> frozenset[str]:
    """Stemmed words of *text* without stop words or a "User said:" prefix.

    Memoised: one briefing asks for the same memory's words several times
    (floor, gaps, outdated lines), and stemming 16 KB memories each time cost
    seconds per briefing (a10 re-test issue 4).
    """
    text = text or ""
    return _content_words_cached(text) if len(text) <= _CACHE_MAX_CHARS else _content_words(text)


@lru_cache(maxsize=512)
def _content_words_cached(text: str) -> frozenset[str]:
    return _content_words(text)


def _content_words(text: str) -> frozenset[str]:
    text = _ROLE.sub("", text, count=1).lower().replace("'s ", " ")
    return frozenset(stem(w) for w in _WORD.findall(text) if w not in STOP and len(w) > 1)


def shares_words(text: str, words: frozenset[str]) -> bool:
    return bool(words and content_words(text) & words)


def above_floor(pairs: list[tuple[Any, float]], task: str, ratio: float) -> list[Any]:
    """Records of *pairs* (fragment, score; best first) worth briefing for *task*.

    A record that shares a content word with the question is kept down to
    half the best score; one that shares none must reach *ratio* of it. When
    the best record itself shares no word with a question that has content
    words, it is kept alone (a lexical embedder's best guess is often noise,
    a semantic one's is often right: "pets" -> "a cat named Pixel").
    ``ratio`` <= 0 disables the floor.
    """
    if ratio <= 0 or not pairs:
        return [f for f, _ in pairs]
    question = content_words(task)
    top = max(score for _, score in pairs) or 1e-9
    if question and not shares_words(getattr(pairs[0][0], "content", ""), question):
        return [pairs[0][0]]
    kept = []
    for fragment, score in pairs:
        shared = len(content_words(getattr(fragment, "content", "")) & question)
        if score >= ratio * top or (shared and score >= 0.5 * top):
            # Each shared question word is worth a quarter of the best score:
            # "my gym locker number" ranks the locker line above "Joao's number".
            kept.append((score / top + 0.25 * shared, fragment))
    kept.sort(key=lambda pair: -pair[0])
    return [fragment for _, fragment in kept]


def is_sequence(fragment: Any) -> bool:
    """True for one part of a multi-turn sequence ("Step 2: ...", "days 4 to 6 in Kyoto")."""
    return bool(_SEQUENCE.search(getattr(fragment, "content", "") or ""))


def sequence_order(fragments: list[Any], key: Any) -> list[Any]:
    """*fragments* with their sequence parts gathered, oldest first, where the best one was.

    Everything else keeps its place. *key* is the timeline sort key.
    """
    parts = [f for f in fragments if is_sequence(f)]
    if len(parts) < 2:
        return fragments
    ordered = sorted(parts, key=key)
    out: list[Any] = []
    for fragment in fragments:
        if fragment is parts[0]:
            out.extend(ordered)
        elif not is_sequence(fragment):
            out.append(fragment)
    return out


def missing_relations(task: str, texts: Iterable[str]) -> list[str]:
    """Relations/possessions the question asks about ("my brother") that no text mentions.

    Only questions count: "My brother Tom is visiting" introduces him, so a
    sentence that states something about "my X" never reports X as missing.
    """
    known = set()
    for text in texts:
        known |= content_words(text)
        known |= {w.lower() for w in _WORD.findall(text or "")}
    missing: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join((task or "").split())):
        if not sentence.endswith("?") or _INTRODUCING.search(sentence):
            continue
        for match in _MY_THING.finditer(sentence):
            for word in match.group(1).lower().split():
                if word in RELATIONS and word not in known and stem(word) not in known \
                        and f"the user's {word}" not in missing:
                    missing.append(f"the user's {word}")
    return missing


def inside_longer_name(text: str, label: str) -> bool:
    """True when *label* occurs in *text* only inside a longer capitalised name.

    "Joao" in "no longer at Sao Joao" is the hospital, not the friend (a10:
    the friend's anchor pulled the job line into every later briefing).
    """
    if not label or not label[:1].isupper():
        return False
    found = False
    pattern = r"(?<!\w)" + re.escape(label) + r"(?!\w)"
    for match in re.finditer(pattern, text or ""):
        found = True
        before = text[: match.start()].rstrip()
        after = text[match.end():].lstrip()
        prev_word = re.search(r"([^\W\d_][\w'-]*)$", before)
        next_word = re.match(r"([^\W\d_][\w-]*)", after)
        sentence_start = not before or before[-1] in ".!?:"
        joined_before = prev_word and prev_word.group(1)[:1].isupper() and not sentence_start
        joined_after = next_word and next_word.group(1)[:1].isupper() and not after.startswith(("'s", "s "))
        if not (joined_before or joined_after):
            return False
    return found


def introduces(text: str, label: str) -> bool:
    """Does the statement *text* introduce *label* ("My brother Tom is visiting",
    "Quokka-77 is the staging cluster")? Commands that merely name it
    ("restart checkout-api", "Tell me about Quokka-77") do not."""
    name = re.escape(label)
    return bool(re.search(r"(?<!\w)" + name + r"(?:'s)?\s+(is|are|was|were|has|have|had|will|lives|works|"
                          r"moved|joined|started)\b", text, re.IGNORECASE)
                or re.search(r"\bmy\s+\w+\s+" + name + r"(?!\w)", text, re.IGNORECASE))
