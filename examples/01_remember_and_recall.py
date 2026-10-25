"""Remember a few notes, then recall them with a question.

Run:  python examples/01_remember_and_recall.py
"""
from tempfile import TemporaryDirectory

from mark import Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    memory = mark.memory

    # remember() stores one note and returns its id.
    memory.remember("The API is built with FastAPI.")
    memory.remember("Health checks live at /health.")

    # Ask in your own words; recall() finds the relevant notes.
    result = memory.recall("Which framework is the API built with?", k=2)
    print(result.as_context())     # ready to paste into a prompt

    assert "FastAPI" in result.top().content
