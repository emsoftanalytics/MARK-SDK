"""Provenance of conversation memories: who said it, and how far to trust it.

Agent loops store three kinds of conversation text, and they are not equally
reliable:

* **user** statements are the ground truth about the user ("I moved to
  Braga"); stored with ``source="user"``.
* **tool** results are evidence from the outside world; ``source="tool"``.
* **assistant** replies are the model's own words. They are often wrong
  (small models confabulate names, dates, even tool names) and repeat what
  memory already holds, so they are stored with ``source="assistant"``, the
  tag :data:`ASSISTANT_TAG`, low importance and low confidence.

Why it matters: in 0.2.0a9 assistant text was stored exactly like user text.
Its invented phrases became briefing anchors and its confident mistakes were
shown back to the model as if the user had said them (usage test M7). Every
briefing renderer therefore asks :func:`is_assistant` before using a record,
and pure questions (:func:`is_question`) are tagged so they never crowd out
statements.

The ``source`` values follow the cross-workstream contract: "user",
"assistant", "tool", "system", "document", "extracted".
"""
from __future__ import annotations

import re
from typing import Any

USER, ASSISTANT, TOOL = "user", "assistant", "tool"
#: Tag on every stored assistant reply (low trust, never a user fact).
ASSISTANT_TAG = "mark:assistant"
#: Tag on stored user messages that only ask something (no new information).
QUESTION_TAG = "mark:question"
#: Tag on stored user messages that only ask the assistant to do something
#: ("Explain what a goroutine is.", "Suggest a dinner dish."): like questions,
#: they state nothing about the user, so briefings never show them as notes.
REQUEST_TAG = "mark:request"
#: Tag on stored user statements: the "trusted lane" a briefing can retrieve
#: on its own when assistant replies and questions crowd the top results.
USER_TAG = "mark:user"
#: Tags on stored tool output; errors are low-trust evidence that must never
#: confirm a subject or anchor a briefing (integration test N3).
TOOL_TAG = "mark:tool"
ERROR_TAG = "mark:error"
#: Store options for a failed tool call (low importance and confidence).
TOOL_ERROR_STORE = {"importance": 0.2, "confidence": 0.3}
#: Tag set by the security workstream on instruction-like memories.
UNTRUSTED_TAG = "mark:untrusted"
#: Store options for assistant replies: low importance and confidence keep
#: them below user statements in ranking and mark them as unverified.
ASSISTANT_STORE = {"importance": 0.2, "confidence": 0.3}

# Sources older releases used for assistant output; still recognised on read.
_ASSISTANT_SOURCES = {ASSISTANT, "agent", "model", "ai"}
_ASSISTANT_PREFIXES = ("assistant replied:", "assistant said:", "agent reasoning:",
                       "agent completed task.")
_QUESTION_START = re.compile(
    r"^(what|whats|what's|who|whom|whose|when|where|which|why|how|do|does|did|is|are|am|was|"
    r"were|can|could|should|would|will|shall|may|have|has|had)\b", re.IGNORECASE)


#: Imperative openers of a request; "remember"/"note" are deliberately absent
#: ("Please remember I prefer mornings" states a preference worth keeping).
_REQUEST_START = re.compile(
    r"^(?:(?:please|just|now|ok|okay|so)[,\s]+)*(suggest|explain|give|tell|recommend|write|list|show|"
    r"describe|summari[sz]e|translate|find|calculate|compute|name|define|compare|draft|generate|"
    r"create|propose|outline|share|search|look up|check)\b", re.IGNORECASE)


def _get(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def is_assistant(fragment: Any) -> bool:
    """True when a stored memory is the model's own output (any release's format)."""
    tags = _get(fragment, "tags") or ()
    if ASSISTANT_TAG in tags:
        return True
    source = str(_get(fragment, "source") or "").lower()
    if source in _ASSISTANT_SOURCES:
        return True
    content = str(_get(fragment, "content") or "").lstrip().lower()
    return content.startswith(_ASSISTANT_PREFIXES)


def is_question_record(fragment: Any) -> bool:
    """True when a stored memory is a user message that only asks a question."""
    return QUESTION_TAG in (_get(fragment, "tags") or ())


def is_error_record(fragment: Any) -> bool:
    """True when a stored memory is a failed tool call (``mark:error``)."""
    return ERROR_TAG in (_get(fragment, "tags") or ())


def tool_options(*, error: bool = False, tags: list[str] | None = None) -> dict[str, Any]:
    """``store_sync`` keyword arguments for a tool result (low trust when it is an error)."""
    extra = [TOOL_TAG, *([ERROR_TAG] if error else [])]
    return {"source": TOOL, "tags": [*(tags or ["conversation"]), *extra],
            **(TOOL_ERROR_STORE if error else {})}


def is_request_record(fragment: Any) -> bool:
    """True when a stored memory is a user request to the assistant (no information)."""
    return REQUEST_TAG in (_get(fragment, "tags") or ())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join((text or "").split())) if s.strip()]


#: Imperatives that are often conventions worth keeping ("Use 4 spaces in Python."):
#: they mark a request only next to a question ("How many sprockets? Use the tool.").
_WEAK_REQUEST = re.compile(r"^(?:(?:please|just|ok|okay)[,\s]+)*(use|run|call|try|ask|go)\b", re.IGNORECASE)


def is_request(text: str) -> bool:
    """True when *text* only asks for something: every sentence is a question or an
    imperative request ("Explain what a goroutine is in one sentence.").

    At least one sentence must be imperative (pure questions are :func:`is_question`).
    """
    sentences = _sentences(text)
    asks = any(_is_question_sentence(s) for s in sentences)

    def imperative(sentence: str) -> bool:
        return bool(_REQUEST_START.match(sentence) or (asks and _WEAK_REQUEST.match(sentence)))

    if not sentences or not any(imperative(s) for s in sentences):
        return False
    return all(imperative(s) or _is_question_sentence(s) for s in sentences)


def _is_question_sentence(sentence: str) -> bool:
    return sentence.endswith("?") or bool(_QUESTION_START.match(sentence) and not sentence.endswith((".", "!")))


def is_question(text: str) -> bool:
    """True when *text* only asks something: every sentence is a question.

    "What's my dog's name?" is a question; "My dog is Biscuit. When is his
    vet visit?" is not (it states something), so it is stored as a statement.
    """
    sentences = _sentences(text)
    return bool(sentences) and all(_is_question_sentence(s) for s in sentences)


def user_options(text: str, tags: list[str] | None = None) -> dict[str, Any]:
    """``store_sync`` keyword arguments for one user message.

    Questions get ``mark:question``, requests ``mark:request`` (both low
    importance, never briefing notes); statements get ``mark:user``.
    """
    tag = QUESTION_TAG if is_question(text) else REQUEST_TAG if is_request(text) else USER_TAG
    return {"source": USER, "tags": [*(tags or ["conversation"]), tag],
            **({"importance": 0.3} if tag != USER_TAG else {})}


def assistant_options(tags: list[str] | None = None) -> dict[str, Any]:
    """``store_sync`` keyword arguments for one assistant reply (low trust)."""
    return {"source": ASSISTANT, "tags": [*(tags or ["conversation"]), ASSISTANT_TAG],
            **ASSISTANT_STORE}


def observe_options(role: str, text: str = "") -> dict[str, Any]:
    """``observe()`` keyword arguments for a message of *role* (observe has no confidence)."""
    if role == "assistant":
        return {"source": ASSISTANT, "tags": ["conversation", ASSISTANT_TAG],
                "importance": ASSISTANT_STORE["importance"], "metadata": {"trust": "low"}}
    if role == "tool":
        return {"source": TOOL, "tags": ["conversation"]}
    if role == "user":
        return user_options(text)
    return {}
