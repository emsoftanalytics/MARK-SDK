import pytest

from mark import Mark


@pytest.mark.asyncio
async def test_agent_wrapper_injects_memory(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("project").write("The project framework is FastAPI.", importance=0.9)

    async def llm(prompt: str) -> str:
        assert "FastAPI" in prompt
        return "The project uses FastAPI."

    agent = mark.wrap_agent(llm, blocks=["project"])
    result = await agent.run("What framework do we use?")

    assert "FastAPI" in result.output
    assert result.context.memories
    assert result.events[0]["type"] == "policy.decision"
