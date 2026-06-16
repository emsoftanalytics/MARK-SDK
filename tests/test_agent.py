import pytest

from mark import AgentRun, BaseMiddleware, Mark, MiddlewareContext


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


@pytest.mark.asyncio
async def test_agent_runs_registered_skills(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    async def llm(prompt: str) -> str:
        assert "<skill_result>" in prompt
        assert "hello from skill" in prompt
        return "skill context received"

    agent = mark.wrap_agent(llm, skills=["echo"])
    result = await agent.run(
        "Use the skill result.",
        skill_inputs={"echo": {"message": "hello from skill"}},
    )

    assert result.output == "skill context received"
    assert result.skill_results["echo"].success is True
    assert any(event["type"] == "skill.run" for event in result.events)


@pytest.mark.asyncio
async def test_agent_middleware_wraps_run(tmp_path):
    class RecorderMiddleware(BaseMiddleware):
        name = "agent-recorder"

        def __init__(self) -> None:
            self.events: list[tuple[str, str]] = []

        def before(self, context: MiddlewareContext) -> None:
            self.events.append(("before", context.operation))
            request = context.payload["request"]
            assert isinstance(request, AgentRun)
            request.metadata["seen_by_middleware"] = True

        def after(self, context: MiddlewareContext) -> None:
            self.events.append(("after", context.operation))

    recorder = RecorderMiddleware()
    mark = Mark.local(project_path=tmp_path)

    async def llm(prompt: str) -> str:
        return "middleware ok"

    agent = mark.wrap_agent(llm, middleware=[recorder])
    result = await agent.run("Check middleware.")

    assert result.output == "middleware ok"
    assert result.metadata["seen_by_middleware"] is True
    assert recorder.events == [("before", "agent.run"), ("after", "agent.run")]


@pytest.mark.asyncio
async def test_agent_can_store_output_when_requested(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    async def llm(prompt: str) -> str:
        return "Remember this answer."

    agent = mark.wrap_agent(llm, agent_id="writer")
    result = await agent.run("Store this.", store_output=True)

    assert result.stored_record_ids
    recalled = mark.memory.retrieve("Remember this answer.", blocks=["observe"])
    assert recalled.memories


@pytest.mark.asyncio
async def test_agent_reuses_creative_continuity_memory_without_media_dependencies(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("visual-canon").write(
        "Elena wears a red scarf, carries a brass key, and stands under cyan warehouse lights.",
        importance=0.95,
    )
    mark.memory.block("visual-canon").write(
        "The scene style is photorealistic cinematic noir with rain on concrete.",
        importance=0.9,
    )

    async def image_prompt_agent(prompt: str) -> str:
        assert "red scarf" in prompt
        assert "brass key" in prompt
        assert "cyan warehouse lights" in prompt
        assert "photorealistic cinematic noir" in prompt
        return (
            "Photorealistic cinematic noir still: Elena keeps the red scarf and "
            "brass key visible under cyan warehouse lights."
        )

    agent = mark.wrap_agent(image_prompt_agent, blocks=["visual-canon"], agent_id="continuity-agent")
    result = await agent.run(
        "Create the next shot prompt for Elena entering the warehouse.",
        store_output=True,
        metadata={"example": "offline-continuity"},
    )

    assert "red scarf" in result.output
    assert "MARK context:" in result.rendered_prompt
    assert "brass key" in result.rendered_prompt
    assert result.stored_record_ids
    assert any(event["type"] == "memory.retrieve" for event in result.events)
