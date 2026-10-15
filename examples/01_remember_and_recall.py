"""Remember a fact, then recall it with a question.

Run:  python examples/01_remember_and_recall.py
"""
from tempfile import TemporaryDirectory

from mark import Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    # Write memories into a named block.
    notes = mark.memory.block("project")
    notes.write("The API is built with FastAPI.")
    notes.write("Health checks live at /health.")

    # Ask in your own words; MARK finds the relevant memories.
    context = mark.memory.retrieve("Which web framework does the API use?")
    print(context.as_text())

    assert "FastAPI" in context.as_text()
