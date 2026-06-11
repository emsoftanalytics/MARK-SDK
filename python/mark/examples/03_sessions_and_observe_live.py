"""Live example: observe session facts and retrieve continuity context.

Run:
    python examples/03_sessions_and_observe_live.py
"""
from __future__ import annotations

from tempfile import TemporaryDirectory

from mark import Mark


def main() -> None:
    with TemporaryDirectory(prefix="mark-example-session-") as tmp:
        with Mark.local(project_path=tmp) as mark:
            observed = mark.observe(
                "Elena enters the North Warehouse wearing the red scarf.",
                agent_id="video-agent",
                session_id="season-01/episode-01/scene-04",
                memory_type="scene",
                importance=0.8,
            )

            memory = mark.runtime.memory("video-agent")
            result = memory.retrieve_sync(
                "What should stay consistent for Elena?",
                session_prefix="season-01/",
            )
            text = result.as_context()

            assert observed.fragment_id
            assert "Elena" in text
            assert "red scarf" in text
            assert "season-01/episode-01/scene-04" in memory.list_sessions()

            print("Sessions and observe live example passed")
            print(text)


if __name__ == "__main__":
    main()
