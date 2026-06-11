"""Docker-backed execution sandbox; never executes code on the host."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxHealth:
    """Report on Docker daemon and sandbox image availability."""
    healthy: bool
    docker_available: bool
    image_available: bool
    image: str
    message: str = ""


@dataclass(frozen=True)
class SandboxResult:
    """Captured stdout/stderr/exit-code of one sandboxed execution."""
    stdout: str
    stderr: str
    exit_code: int
    sandbox_mode: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Return True when execution succeeded."""
        return self.exit_code == 0 and not self.timed_out


class DockerSandbox:
    """
    Docker-backed execution sandbox.

    This never falls back to host subprocess execution. If Docker or the image
    is unavailable, execution returns a blocked result.
    """

    DEFAULT_IMAGE = "mark-sdk-sandbox:0.1"
    CONTAINER_CODE_PATH = "/sandbox/code.py"
    CONTAINER_WORKSPACE = "/sandbox/workspace"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        memory: str = "256m",
        cpus: str = "0.5",
        network: str = "none",
        default_timeout: int = 30,
        docker_context: str | None = None,
        runner=subprocess.run,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.default_timeout = default_timeout
        self.docker_context = docker_context or os.environ.get("MARK_DOCKER_CONTEXT")
        self._runner = runner

    def health(self, *, pull: bool = False, timeout: int = 20) -> SandboxHealth:
        """Probe availability and return a health report."""
        docker = self._run(self._docker_command("info", "--format", "{{.ServerVersion}}"), timeout=timeout)
        if docker.returncode != 0:
            return SandboxHealth(
                healthy=False,
                docker_available=False,
                image_available=False,
                image=self.image,
                message=(docker.stderr or docker.stdout or "Docker daemon is not reachable").strip(),
            )
        image_available = self._image_exists(timeout=timeout)
        if not image_available and pull:
            pulled = self._run(self._docker_command("pull", self.image), timeout=max(timeout, 120))
            image_available = pulled.returncode == 0
            if not image_available:
                return SandboxHealth(
                    healthy=False,
                    docker_available=True,
                    image_available=False,
                    image=self.image,
                    message=(pulled.stderr or pulled.stdout or f"Could not pull {self.image}").strip(),
                )
        return SandboxHealth(
            healthy=image_available,
            docker_available=True,
            image_available=image_available,
            image=self.image,
            message="healthy" if image_available else f"Image {self.image!r} is not available locally",
        )

    def ensure_healthy(self, *, pull: bool = True) -> SandboxHealth:
        """Probe health, pulling the image when needed."""
        return self.health(pull=pull)

    def execute(
        self,
        code: str,
        *,
        timeout: int | None = None,
        workspace: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run code in the sandbox and return the captured result."""
        health = self.health(pull=False)
        if not health.docker_available:
            return SandboxResult("", health.message, -2, "blocked/docker-unavailable")
        if not health.image_available:
            return SandboxResult("", health.message, -2, f"blocked/missing-image:{self.image}")

        timeout = timeout or self.default_timeout
        host_workspace = self._resolve_workspace(workspace) if workspace is not None else None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as file:
            file.write(code)
            host_code = Path(file.name)
        try:
            command = [
                *self._docker_command("run"),
                "--rm",
                "--network",
                self.network,
                "--memory",
                self.memory,
                "--cpus",
                self.cpus,
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{host_code}:{self.CONTAINER_CODE_PATH}:ro",
            ]
            if host_workspace is not None:
                command.extend(["-v", f"{host_workspace}:{self.CONTAINER_WORKSPACE}:rw"])
            for key, value in (env or {}).items():
                command.extend(["-e", f"{key}={value}"])
            command.extend([self.image, "python", self.CONTAINER_CODE_PATH])
            try:
                result = self._run(command, timeout=timeout)
                return SandboxResult(result.stdout, result.stderr, result.returncode, f"docker/{self.image}")
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    exc.stdout or "",
                    exc.stderr or "Timeout",
                    -1,
                    f"docker/{self.image}",
                    timed_out=True,
                )
        finally:
            try:
                host_code.unlink()
            except FileNotFoundError:
                pass

    def _image_exists(self, *, timeout: int = 20) -> bool:
        return self._run(self._docker_command("image", "inspect", self.image), timeout=timeout).returncode == 0

    def _docker_command(self, *args: str) -> list[str]:
        if self.docker_context:
            return ["docker", "--context", self.docker_context, *args]
        return ["docker", *args]

    def _run(self, command: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        return self._runner(command, capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def _resolve_workspace(workspace: str | Path) -> Path:
        path = Path(workspace).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"workspace must be an existing directory: {workspace}")
        return path
