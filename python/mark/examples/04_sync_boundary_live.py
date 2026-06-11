"""Live example: prepare a redacted sync envelope without cloud transport.

Run:
    python examples/04_sync_boundary_live.py
"""
from __future__ import annotations

from tempfile import TemporaryDirectory

from mark import CloudSync, Mark, SyncOptions


def main() -> None:
    with TemporaryDirectory(prefix="mark-example-sync-") as tmp:
        with Mark.local(project_path=tmp) as mark:
            mark.memory.block("project").write(
                "Deploy uses staging. API key: sk-exampleSecret123456",
                importance=0.8,
                metadata={"token_note": "token=example-token-value"},
                source="https://docs.example.test/setup?token=example-token-value",
            )
            mark.runtime.tracer().emit(
                "example.sync",
                note="password=example-password-value",
            )

            delta = CloudSync().prepare_delta(
                mark.runtime,
                options=SyncOptions(include_blocks=["project"], include_events=True),
            )
            payload = str(delta.fragments) + str(delta.events)

            assert delta.fragments
            assert delta.stats.synced >= 1
            assert "sk-exampleSecret123456" not in payload
            assert "example-token-value" not in payload
            assert "example-password-value" not in payload

            print("Sync boundary live example passed")
            print(f"Prepared {delta.stats.synced} redacted fragment(s); no cloud client used.")


if __name__ == "__main__":
    main()
