"""Push MARK context into any chat loop (plain OpenAI-style message lists).

Run:  python examples/07_briefing_any_loop.py
With a local model:  MARK_LLM_URL=http://localhost:11434/v1 python examples/07_briefing_any_loop.py
"""
from tempfile import TemporaryDirectory

from _llm import ask
from mark import ContextWatch, Mark
from mark.context.watch import inject

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    ops = mark.runtime.memory("ops")
    ops.remember("[2026-03-12] TICKET HAL-4433: checkout returns 502 errors; affects checkout-api.")
    ops.facts.set("checkout-api", "port", "8080", valid_from="2026-03-01")
    ops.facts.set("checkout-api", "port", "9090", valid_from="2026-03-20")

    messages = [{"role": "system", "content": "You are an on-call engineer. Answer in two sentences."},
                {"role": "user", "content": "Customers see 502s. Please look into HAL-4433."}]
    # Small local models: a tighter briefing that still flags unknowns and corrections.
    with ContextWatch.for_small_models(ops) as watch:
        watch.observe_messages(messages)            # feeds only messages it has not seen
        prompt = inject(messages, watch.briefing(wait=1.0))  # new list; `messages` is untouched
        print(prompt[0]["content"], "\n")

        reply = ask(prompt[-1]["content"], system=prompt[0]["content"])
        print(reply if reply is not None else "(set MARK_LLM_URL to get a model reply)", "\n")
        if reply is not None:
            messages.append({"role": "assistant", "content": reply})
            watch.observe_messages(messages)        # a stale port in the reply gets corrected next turn
            print(watch.briefing().corrections)

    assert "HAL-4433" in prompt[0]["content"] and "9090" in prompt[0]["content"]
