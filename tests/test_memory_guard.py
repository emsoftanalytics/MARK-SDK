from mark import GovernanceMiddleware, Mark
from mark.types import MemoryState


def test_store_sync_quarantines_failure_artifact(tmp_path):
    mark = Mark.local(project_path=tmp_path, middleware=[GovernanceMiddleware()])
    memory = mark.runtime.memory("agent")

    fragment_id = memory.store_sync(
        "Scene 7 image artifact failed: HTTP 402 PAYMENT_REQUIRED",
        importance=0.9,
        source="generator",
    )

    fragment = mark.runtime.store.get(fragment_id)
    assert fragment is not None
    assert fragment.state == MemoryState.QUARANTINED
    assert fragment.importance == 0.0
    assert fragment.metadata["mark_governance"]["passed"] is False
    result = memory.retrieve_sync("Scene 7 image artifact")
    assert fragment_id not in {item.id for item in result.fragments}


def test_bare_store_does_not_quarantine_without_governance_middleware(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    memory = mark.runtime.memory("agent")

    fragment_id = memory.store_sync(
        "Scene 7 image artifact failed: HTTP 402 PAYMENT_REQUIRED",
        importance=0.9,
        source="generator",
    )

    fragment = mark.runtime.store.get(fragment_id)
    assert fragment is not None
    assert fragment.state == MemoryState.UNVERIFIED
    assert "mark_governance" not in fragment.metadata


def test_quarantine_sync_stores_non_retrievable_diagnostic(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    memory = mark.runtime.memory("agent")

    fragment_id = memory.quarantine_sync(
        "Review scene 7 failed or returned invalid score.",
        reason="vision review failed",
        source="continuity-reviewer",
    )

    assert memory.quarantined(limit=10)[0].id == fragment_id
    result = memory.retrieve_sync("scene 7 failed")
    assert fragment_id not in {item.id for item in result.fragments}
