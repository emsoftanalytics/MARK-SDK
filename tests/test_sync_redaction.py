from mark import Mark
from mark.middlewares.sync import redact_records_for_sync


def test_sync_redacts_secrets(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    record = mark.memory.block("project").write("api_key=abc123 should not sync raw")

    payload = redact_records_for_sync([record])

    assert "abc123" not in payload[0]["content"]
    assert "[REDACTED]" in payload[0]["content"]


def test_sync_redacts_record_metadata(tmp_path):
    mark = Mark.local(project_path=tmp_path)
    record = mark.memory.block("project").write(
        "public fact",
        metadata={"nested": {"token": "token=metadata-secret-value"}},
    )

    payload = redact_records_for_sync([record])

    assert "metadata-secret-value" not in str(payload)
