from mark import Mark


def test_skill_registry_loads_echo_skill(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    result = mark.skills.get("echo").run_sync({"hello": "world"}, {"source": "test"})

    assert result.metadata["input"] == {"hello": "world"}
    assert result.success is True


def test_mark_memory_skill_retrieves_and_writes_memory(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    skill = mark.skills.get("mark_memory")

    write = skill.run_sync(
        {
            "action": "write",
            "block": "project",
            "content": "The coding template uses FastAPI and pytest.",
            "importance": 0.9,
        },
        {},
    )
    retrieve = skill.run_sync({"action": "retrieve", "query": "coding template framework"}, {})

    assert write.metadata["record"]["content"] == "The coding template uses FastAPI and pytest."
    assert "FastAPI" in retrieve.content
