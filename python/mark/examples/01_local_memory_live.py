"""Live example: store and retrieve local project memory.

Run:
    python examples/01_local_memory_live.py
"""
from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from mark import Mark


def main() -> None:
    with TemporaryDirectory(prefix="mark-example-local-") as tmp:
        project = Path(tmp)
        with Mark.local(project_path=project) as mark:
            mark.memory.block("project").write(
                "The API framework is FastAPI and health checks live at /health.",
                importance=0.9,
                confidence=0.95,
                source="example",
            )

            context = mark.memory.retrieve("Which framework handles the API?")
            text = context.as_text()

            assert "FastAPI" in text
            assert (project / ".mark" / "memory.db").exists()

            print("Local memory live example passed")
            print(text)


if __name__ == "__main__":
    main()
