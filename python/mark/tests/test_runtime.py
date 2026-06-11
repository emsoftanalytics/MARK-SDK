from mark import Mark


def test_runtime_creates_local_store(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    assert mark.config.store_path == tmp_path / ".mark"
    assert (tmp_path / ".mark" / "memory.db").exists()
    assert "default" in mark.policies.names()
    assert "echo" in mark.skills.names()
    mark.shutdown()
