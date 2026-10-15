"""MARK as the context window: the model only sees the last two messages plus MARK.

Early facts fall out of the window after a few turns; MARK stored them and
puts the relevant ones back (within a fixed token budget) when they matter.

Run with a local model (Ollama: `ollama pull llama3.2:3b`):
  MARK_LLM_URL=http://localhost:11434/v1 MARK_LLM_MODEL=llama3.2:3b python examples/09_context_replacement.py
Without MARK_LLM_URL no model is called: it prints what the model would be sent.
"""
import os
from tempfile import TemporaryDirectory

from mark import Mark, MarkChat

TURNS = [
    "Hi! I'm Ana. I keep bees on my roof in Lisbon.",
    "My only hive is called Colmeia Azul.",
    "Which flowers do bees like in spring? One sentence please.",
    "And in summer? One sentence.",
    "Quick question: what is my hive called, and in which city do I live?",
]

url = os.environ.get("MARK_LLM_URL")
with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    chat = MarkChat(mark.runtime.memory("chat"), base_url=url or "http://localhost:11434/v1",
                    model=os.environ.get("MARK_LLM_MODEL", "llama3.2:3b"),
                    system="You are a friendly assistant. Answer briefly.",
                    keep_last=2)   # preset="small" (default): a 200-token briefing
    with chat:
        for text in TURNS:
            print(f"USER: {text}")
            if url:
                print(f"MODEL: {chat.send(text)}\n")
            else:
                chat.prepare(text)
                chat.commit(text, None)   # no model configured: only the user turn is stored

        # What the model saw on the last turn: no "Colmeia Azul" message in the
        # window, but the MARK briefing (at most 200 tokens) carries it.
        window = [m["content"] for m in chat.last_prompt[1:]]
        print("Window sent to the model:", window)
        print(chat.last_prompt[0]["content"])
        print("Briefing usage:", chat.last_briefing.usage.used_tokens, "of", chat.budget.max_tokens, "tokens")
        if not url:
            print("(set MARK_LLM_URL to get model replies)")

    assert not any("Colmeia Azul" in text for text in window)
    assert "Colmeia Azul" in chat.last_prompt[0]["content"]
    assert chat.last_briefing.usage.used_tokens <= 200
