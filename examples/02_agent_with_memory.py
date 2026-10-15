"""Ask an LLM the same question without and with MARK memory.

Run:  python examples/02_agent_with_memory.py
With a local model:  MARK_LLM_URL=http://localhost:11434/v1 python examples/02_agent_with_memory.py
"""
from tempfile import TemporaryDirectory

from _llm import ask
from mark import Mark

QUESTION = "Add a health-check endpoint to our API. Which framework and route should I use? Answer in two sentences."

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    mark.memory.block("project").write("The API is built with FastAPI; health checks go at /api/v1/health.")

    # Without MARK: the model only sees the question.
    plain_prompt = QUESTION

    # With MARK: put the relevant memories in front of the question.
    memory = mark.memory.retrieve(QUESTION).as_text()
    mark_prompt = f"{memory}\n\n{QUESTION}"

    for label, prompt in [("Without MARK", plain_prompt), ("With MARK", mark_prompt)]:
        answer = ask(prompt)
        print(f"--- {label} ---")
        print(answer if answer is not None else f"(set MARK_LLM_URL to get an answer)\nPrompt:\n{prompt}")
        print()

    assert "FastAPI" in mark_prompt and "/api/v1/health" in mark_prompt
