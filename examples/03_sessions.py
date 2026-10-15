"""Keep memories per session, then recall across a whole season.

Run:  python examples/03_sessions.py
"""
from tempfile import TemporaryDirectory

from mark import Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    story = mark.runtime.memory("video-agent")

    # Each scene is a session; observe() records what happened.
    story.observe("Elena wears a red scarf in the warehouse.", session_id="season-01/scene-04")
    story.observe("Marco loses his left glove on the bridge.", session_id="season-01/scene-07")
    story.observe("Elena dyes her hair blue.", session_id="season-02/scene-01")

    # Recall only from season 1.
    result = story.retrieve_sync("What does Elena wear?", session_prefix="season-01/")
    print(result.as_context())

    assert "red scarf" in result.as_context()
    assert "blue" not in result.as_context()  # season 2 is out of scope
