from mark import Mark
from mark.policies import AgentState


def test_policy_registry_loads_builtins(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    assert mark.policies.names() == ["coding-agent", "default", "research"]


def test_unknown_policy_raises_clear_error(tmp_path):
    mark = Mark.local(project_path=tmp_path)

    try:
        mark.policies.get("missing")
    except KeyError as exc:
        assert "Available policies" in str(exc)
    else:
        raise AssertionError("Expected missing policy to raise KeyError")


def test_coding_policy_requests_more_context(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    policy = mark.policies.get("coding-agent")

    decision = policy.choose(AgentState(prompt="fix tests", block_labels=["project"], skill_names=[]))

    assert decision.retrieve_top_k == 8
