from mark import Mark


def test_context_empty_when_no_records(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    bundle = mark.memory.retrieve("anything")

    assert bundle.memories == []
    assert "no relevant local memory" in bundle.as_text()


def test_context_respects_character_budget(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    block = mark.memory.block("project")
    block.write("alpha " * 100, importance=0.9)
    block.write("beta " * 100, importance=0.9)

    bundle = mark.memory.retrieve("alpha beta", max_chars=120)

    assert len(bundle.as_text()) <= 120
