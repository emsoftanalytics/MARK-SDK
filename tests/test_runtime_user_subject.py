"""MarkRuntime.local applies the user subject stored with the store (scripts, LangChain backend)."""
from __future__ import annotations

from mark.memory.runtime import MarkRuntime
from mark.runtime_config import apply_user_subject


def test_direct_runtime_users_get_the_stored_user_subject(tmp_path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    apply_user_subject(runtime, "Ana")
    runtime.shutdown()
    seen: list[str] = []
    original = MarkRuntime.configure_user_subject
    MarkRuntime.configure_user_subject = lambda self, name: seen.append(name) or original(self, name)
    try:
        MarkRuntime.local(store_path=tmp_path / "m.db").shutdown()
    finally:
        MarkRuntime.configure_user_subject = original
    assert seen == ["Ana"]
