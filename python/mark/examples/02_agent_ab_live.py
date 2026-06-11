"""Live example: compare a plain callable with a MARK-wrapped callable.

Run:
    python examples/02_agent_ab_live.py
"""
from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

from mark import Mark


def coding_agent(prompt: str) -> str:
    if "MARK context:" in prompt and "FastAPI" in prompt:
        return "Use FastAPI dependency injection and add the route to /health."
    return "I need the project framework before changing the endpoint."


async def main() -> None:
    with TemporaryDirectory(prefix="mark-example-agent-") as tmp:
        with Mark.local(project_path=tmp) as mark:
            mark.memory.block("project").write(
                "This backend uses FastAPI dependency injection.",
                importance=0.9,
            )

            baseline = coding_agent("Add a health check endpoint.")
            wrapped = mark.wrap_agent(coding_agent, blocks=["project"])
            result = await wrapped.run("Add a health check endpoint.")

            assert "need the project framework" in baseline
            assert "FastAPI" in result.output
            assert result.context.memories

            print("Agent A/B live example passed")
            print(f"Without MARK: {baseline}")
            print(f"With MARK:    {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
