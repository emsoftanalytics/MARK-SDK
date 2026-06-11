from mark import Mark


def test_block_write_and_list_records(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    block = mark.memory.block("project", kind="project_context")
    record = block.write("MARK starts as a local Python SDK.", importance=0.9, confidence=0.8)

    assert record in block.records()
    assert mark.memory.block("project").id == block.id


def test_retrieve_returns_ranked_context(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    project = mark.memory.block("project")
    project.write("The backend framework is FastAPI.", importance=0.9, confidence=0.9)
    project.write("The preferred color is green.", importance=0.2, confidence=0.2)

    bundle = mark.memory.retrieve("What backend framework do we use?", blocks=["project"], top_k=2)

    assert bundle.memories[0].content == "The backend framework is FastAPI."
    assert "FastAPI" in bundle.as_text()


def test_retrieve_can_scope_blocks(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("project").write("The backend framework is FastAPI.", importance=0.9)
    mark.memory.block("personal").write("The favorite framework is Django.", importance=0.9)

    bundle = mark.memory.retrieve("framework", blocks=["personal"], top_k=5)

    assert len(bundle.memories) == 1
    assert "Django" in bundle.memories[0].content
