"""Let MARK prepare context before the agent asks for it.

Run:  python examples/05_context_watch.py
With a local model:  MARK_LLM_URL=http://localhost:11434/v1 python examples/05_context_watch.py
"""
from tempfile import TemporaryDirectory

from _llm import ask
from mark import ContextWatch, Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    ops = mark.runtime.memory("ops")
    ops.remember("[2026-03-12] TICKET HAL-4433: checkout returns 502 errors; affects checkout-api.")
    ops.facts.set("checkout-api", "port", "8080", valid_from="2026-03-01")
    ops.facts.set("checkout-api", "port", "9090", valid_from="2026-03-20")

    with ContextWatch(ops) as watch:
        # 1. The user mentions a ticket; MARK looks up the ticket and what it links to.
        user = "Customers see 502s. Please look into HAL-4433."
        watch.observe(user, role="user")
        briefing = watch.briefing(wait=1.0).text
        print(briefing, "\n")

        # 2. The briefing goes into the model's system prompt; no search tool needed.
        reply = ask(user, system=f"You are an on-call engineer. Answer in two sentences.\n{briefing}")
        print(reply if reply is not None else "(set MARK_LLM_URL to get a model reply)", "\n")

        # 3. If the agent states an outdated value, the next briefing corrects it.
        watch.observe("I'll restart checkout-api on port 8080.", role="assistant")
        print(watch.briefing().corrections)

    assert "HAL-4433" in briefing and "9090" in briefing
