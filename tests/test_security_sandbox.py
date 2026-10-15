# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Security hardening: DockerSandbox command hardening, timeouts, missing
binary handling, workspace allowlist, and env validation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from mark.middlewares.sandbox import DockerSandbox


# ── 7. sandbox ────────────────────────────────────────────────────────────────

class _Runner:
    def __init__(self, *, run_result=None, raise_on_run: BaseException | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self._run_result = run_result
        self._raise = raise_on_run

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[1] == "run":
            if self._raise is not None:
                raise self._raise
            return self._run_result or subprocess.CompletedProcess(command, 0, "out\n", "")
        return subprocess.CompletedProcess(command, 0, "ok", "")


def test_sandbox_command_is_hardened_and_code_goes_via_stdin() -> None:
    runner = _Runner()
    result = DockerSandbox(runner=runner).execute("print('hi')", env={"MODE": "test"})
    assert result.ok and result.stdout == "out\n"
    command, kwargs = next(c for c in runner.calls if c[0][1] == "run")
    assert kwargs["input"] == "print('hi')"
    for flag in ("--rm", "-i", "--init", "--read-only", "--pids-limit", "--user", "--network", "--cap-drop"):
        assert flag in command
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--pids-limit") + 1] == "64"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--name") + 1].startswith("mark-sbx-")
    assert command[-2:] == ["python", "-"]
    assert not any(part.endswith("code.py") or ".py:" in part for part in command)
    assert "MODE=test" in command


def test_sandbox_timeout_kills_named_container_and_decodes_bytes() -> None:
    runner = _Runner(raise_on_run=subprocess.TimeoutExpired(["docker"], 1, output=b"partial", stderr=b"slow"))
    result = DockerSandbox(runner=runner).execute("while True: pass", timeout=1)
    assert result.timed_out and result.stdout == "partial" and result.stderr == "slow"
    run_cmd = next(c[0] for c in runner.calls if c[0][1] == "run")
    kill_cmd = next(c[0] for c in runner.calls if c[0][1] == "kill")
    assert kill_cmd[2] == run_cmd[run_cmd.index("--name") + 1]


def test_sandbox_missing_docker_binary_is_blocked() -> None:
    def runner(command, **kwargs):
        raise FileNotFoundError(2, "No such file", "docker")

    result = DockerSandbox(runner=runner).execute("print(1)")
    assert result.sandbox_mode == "blocked/docker-unavailable"
    assert result.exit_code == -2


def test_sandbox_workspace_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "root"
    inside = allowed / "ws"
    inside.mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    runner = _Runner()
    assert DockerSandbox(runner=runner).execute("1", workspace=inside).sandbox_mode == "blocked/invalid-arguments"

    box = DockerSandbox(runner=runner, workspace_root=allowed)
    assert box.execute("1", workspace=outside).sandbox_mode == "blocked/invalid-arguments"
    assert box.execute("1", workspace="/").sandbox_mode == "blocked/invalid-arguments"
    assert box.execute("1", workspace=str(inside) + ":/etc").sandbox_mode == "blocked/invalid-arguments"
    assert box.execute("1", workspace=allowed / ".." / "elsewhere").sandbox_mode == "blocked/invalid-arguments"
    ok = box.execute("1", workspace=inside)
    assert ok.ok
    command = runner.calls[-1][0]
    assert f"{inside.resolve()}:{DockerSandbox.CONTAINER_WORKSPACE}:rw" in command

    with pytest.raises(ValueError):
        DockerSandbox(workspace_root="/")


@pytest.mark.parametrize("env", [{"BAD KEY": "x"}, {"1ABC": "x"}, {"A=B": "x"}, {"OK": "line\nbreak"}])
def test_sandbox_rejects_invalid_env(env: dict[str, str]) -> None:
    runner = _Runner()
    result = DockerSandbox(runner=runner).execute("1", env=env)
    assert result.sandbox_mode == "blocked/invalid-arguments"
    assert not any(c[0][1] == "run" for c in runner.calls)
