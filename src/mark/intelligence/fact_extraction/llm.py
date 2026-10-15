"""LLMFactExtractor: fact extraction with any OpenAI-compatible chat server.

Works with local servers that speak ``POST {base_url}/chat/completions``
(Ollama, llama.cpp, vLLM, LM Studio, ...). It uses only the standard library
(urllib) and is never used unless configured explicitly.

Because a model call takes seconds, the extractor declares
``background = True``: writes only enqueue a durable job and the runtime's
worker thread calls :meth:`extract` later, so ``store_sync`` stays fast.

Replies are parsed tolerantly (code fences, prose around the JSON, a bare
list, aliases such as ``entity``/``property``) and every candidate passes the
same hygiene checks as pattern candidates. Transport errors *raise*, so the
job queue records them and retries; an unparseable reply yields no facts.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from mark.intelligence.extraction_parsing import _coerce_confidence, _first_json_object
from mark.intelligence.fact_extraction.base import FactCandidate, clean_text, is_redacted, valid_candidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract facts that a text states as true. A fact is the CURRENT value of an attribute of a "
    "subject after the event described, e.g. 'atlas-core moved from port 9239 to port 4444' -> "
    '{"subject": "atlas-core", "attribute": "port", "value": "4444"}. '
    "For statements in the first person (I, me, my) use the subject \"user\". "
    "Ignore questions, negations, plans, hypotheticals and old values. Use short lower-case attribute "
    "names (port, owner, team, status, region, ...). If the text gives the date the change took effect, "
    'add "valid_from" as YYYY-MM-DD. Reply with JSON only: {"facts": [...]} (an empty list if none).'
)

_FIRST_PERSON = frozenset({"i", "me", "my", "myself", "user", "the user", "speaker", "the speaker"})

# Field aliases models commonly use instead of the requested names.
_ALIASES = {
    "subject": ("subject", "entity", "name"),
    "attribute": ("attribute", "property", "predicate", "key", "field", "relation"),
    "value": ("value", "new_value", "current_value", "object_value"),
}


class LLMFactExtractor:
    """Ask a local OpenAI-compatible model for facts (slow; runs in background)."""

    name = "llm"
    background = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        min_confidence: float = 0.5,
        user_subject: str = "user",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.min_confidence = min_confidence
        #: Subject for first-person facts; the model is asked to answer "user".
        self.user_subject = user_subject

    def extract(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Call the model and parse its reply; raises on transport errors."""
        if not (text or "").strip():
            return []
        reply = self._chat(text, occurred_at)
        return parse_fact_reply(reply, occurred_at=occurred_at, min_confidence=self.min_confidence,
                                user_subject=self.user_subject)

    def _chat(self, text: str, occurred_at: datetime | None) -> str:
        """One chat completion; returns the assistant message content."""
        user = text if occurred_at is None else f"(written on {occurred_at.date().isoformat()})\n{text}"
        body = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user}],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers,
                                         method="POST")
        # Local servers only: never route through an environment proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            return str(payload["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected chat completion payload: {str(payload)[:200]}") from exc


def parse_fact_reply(
    reply: str, *, occurred_at: datetime | None = None, min_confidence: float = 0.0, user_subject: str = "user",
) -> list[FactCandidate]:
    """Model reply -> candidates. Never raises; invalid items are dropped one by one.

    First-person subjects a model may still write ("I", "me", "the user")
    become *user_subject*; other pronouns are dropped by the hygiene checks.
    """
    items = _fact_items(reply)
    found: list[FactCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fields = {key: _pick(item, names) for key, names in _ALIASES.items()}
        if is_redacted(fields["subject"]) or is_redacted(fields["value"]):
            continue
        confidence = _coerce_confidence(item.get("confidence"))
        if confidence < min_confidence:
            continue
        subject = clean_text(fields["subject"])
        if subject.lower() in _FIRST_PERSON:
            subject = user_subject
        candidate = FactCandidate(
            subject=subject,
            attribute=clean_text(fields["attribute"]).lower(),
            value=clean_text(fields["value"]),
            valid_from=_date(item.get("valid_from")) or occurred_at,
            confidence=confidence,
            rule="llm",
        )
        # The user subject may be a word hygiene rejects elsewhere ("me"): check the rest.
        check = replace(candidate, subject="user") if subject == user_subject else candidate
        if valid_candidate(check):
            found.append(candidate)
    return found


def _fact_items(reply: str) -> list[Any]:
    """The list of fact objects in a reply: {"facts": [...]}, a bare list, or one object."""
    text = (reply or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        data = _first_json_object(text)
    if not isinstance(data, dict):
        if text:
            logger.warning("LLM fact reply had no JSON object: %.120s", text)
        return []
    facts = data.get("facts", data.get("items"))
    if isinstance(facts, list):
        return facts
    return [data] if any(name in data for name in _ALIASES["subject"]) else []


def _pick(item: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = item.get(name)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return ""


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
