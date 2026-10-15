"""Docker-backed execution sandbox; never executes code on the host."""
from __future__ import annotations

import os
import re
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
# Characters that would alter `docker run -v src:dst:mode` parsing or mount options.
_FORBIDDEN_PATH_CHARS = (":", ",", "\n", "\r", "\x00")


def _decode(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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

    Code is streamed to ``python -`` over stdin, so no host file has to be
    readable by the unprivileged container user. Containers run as
    ``65534:65534`` (nobody) with no network, a read-only root filesystem,
    dropped capabilities, a PID limit, and an init process. Each run gets a
    unique ``--name`` so a timed-out container can be killed explicitly.

    Host workspaces are only mounted when they resolve inside one of the
    explicit ``workspace_root`` directories.
    """

    DEFAULT_IMAGE = "mark-sdk-sandbox:0.1"
    CONTAINER_WORKSPACE = "/sandbox/workspace"
    CONTAINER_USER = "65534:65534"
    PIDS_LIMIT = 64

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        memory: str = "256m",
        cpus: str = "0.5",
        network: str = "none",
        default_timeout: int = 30,
        docker_context: str | None = None,
        workspace_root: str | Path | Iterable[str | Path] | None = None,
        runner=subprocess.run,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.default_timeout = default_timeout
        self.docker_context = docker_context or os.environ.get("MARK_DOCKER_CONTEXT")
        self.workspace_roots = self._normalise_roots(workspace_root)
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
                message=(_decode(docker.stderr) or _decode(docker.stdout) or "Docker daemon is not reachable").strip(),
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
                    message=(
                        _decode(pulled.stderr) or _decode(pulled.stdout) or f"Could not pull {self.image}"
                    ).strip(),
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

    def build_command(
        self,
        *,
        name: str,
        workspace: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Return the ``docker run`` argv for one execution (code arrives on stdin)."""
        command = [
            *self._docker_command("run"),
            "--rm",
            "-i",
            "--name",
            name,
            "--init",
            "--network",
            self.network,
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.PIDS_LIMIT),
            "--user",
            self.CONTAINER_USER,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--env",
            "HOME=/tmp",
        ]
        if workspace is not None:
            command.extend(
                ["-v", f"{workspace}:{self.CONTAINER_WORKSPACE}:rw", "--workdir", self.CONTAINER_WORKSPACE]
            )
        for key, value in (env or {}).items():
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.image, "python", "-"])
        return command

    def execute(
        self,
        code: str,
        *,
        timeout: int | None = None,
        workspace: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run code in the sandbox and return the captured result."""
        try:
            host_workspace = self._resolve_workspace(workspace) if workspace is not None else None
            clean_env = self._validate_env(env)
        except ValueError as exc:
            return SandboxResult("", str(exc), -2, "blocked/invalid-arguments")

        health = self.health(pull=False)
        if not health.docker_available:
            return SandboxResult("", health.message, -2, "blocked/docker-unavailable")
        if not health.image_available:
            return SandboxResult("", health.message, -2, f"blocked/missing-image:{self.image}")

        timeout = timeout or self.default_timeout
        name = f"mark-sbx-{uuid.uuid4().hex}"
        command = self.build_command(name=name, workspace=host_workspace, env=clean_env)
        try:
            result = self._run(command, timeout=timeout, input=code)
        except subprocess.TimeoutExpired as exc:
            self._kill(name)
            return SandboxResult(
                _decode(exc.stdout),
                _decode(exc.stderr) or "Timeout",
                -1,
                f"docker/{self.image}",
                timed_out=True,
            )
        if result.args is None:  # docker binary vanished between health and run
            return SandboxResult("", _decode(result.stderr), -2, "blocked/docker-unavailable")
        return SandboxResult(
            _decode(result.stdout), _decode(result.stderr), result.returncode, f"docker/{self.image}"
        )

    def _kill(self, name: str) -> None:
        try:
            self._run(self._docker_command("kill", name), timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _image_exists(self, *, timeout: int = 20) -> bool:
        return self._run(self._docker_command("image", "inspect", self.image), timeout=timeout).returncode == 0

    def _docker_command(self, *args: str) -> list[str]:
        if self.docker_context:
            return ["docker", "--context", self.docker_context, *args]
        return ["docker", *args]

    def _run(
        self,
        command: list[str],
        *,
        timeout: int,
        input: str | None = None,
    ) -> subprocess.CompletedProcess:
        kwargs: dict[str, object] = {"capture_output": True, "text": True, "timeout": timeout}
        if input is not None:
            kwargs["input"] = input
        try:
            return self._runner(command, **kwargs)
        except OSError as exc:
            # Missing/unusable docker binary: report as unavailable, never raise.
            return subprocess.CompletedProcess(None, 127, "", f"docker binary unavailable: {exc}")  # type: ignore[arg-type]

    @staticmethod
    def _normalise_roots(roots: str | Path | Iterable[str | Path] | None) -> tuple[Path, ...]:
        if roots is None:
            return ()
        if isinstance(roots, (str, Path)):
            roots = [roots]
        resolved: list[Path] = []
        for root in roots:
            path = Path(root).resolve()
            if path == Path(path.anchor):
                raise ValueError("workspace_root must not be the filesystem root")
            if any(ch in str(path) for ch in _FORBIDDEN_PATH_CHARS):
                raise ValueError(f"workspace_root contains a forbidden character: {path}")
            resolved.append(path)
        return tuple(resolved)

    @staticmethod
    def _validate_env(env: dict[str, str] | None) -> dict[str, str]:
        clean: dict[str, str] = {}
        for key, value in (env or {}).items():
            if not isinstance(key, str) or not _ENV_KEY_RE.match(key):
                raise ValueError(f"invalid environment variable name: {key!r}")
            text = str(value)
            if "\x00" in text or "\n" in text or "\r" in text:
                raise ValueError(f"invalid environment variable value for {key!r}")
            clean[key] = text
        return clean

    def _resolve_workspace(self, workspace: str | Path) -> Path:
        raw = str(workspace)
        if any(ch in raw for ch in _FORBIDDEN_PATH_CHARS):
            raise ValueError(f"workspace path contains a forbidden character: {workspace!r}")
        if not self.workspace_roots:
            raise ValueError("workspace mounts require an explicit workspace_root allowlist")
        path = Path(workspace).resolve()
        if path == Path(path.anchor):
            raise ValueError("workspace must not be the filesystem root")
        if any(ch in str(path) for ch in _FORBIDDEN_PATH_CHARS):
            raise ValueError(f"workspace path contains a forbidden character: {path}")
        if not any(path == root or path.is_relative_to(root) for root in self.workspace_roots):
            raise ValueError(f"workspace {path} is outside the allowed workspace_root")
        if not path.exists() or not path.is_dir():
            raise ValueError(f"workspace must be an existing directory: {workspace}")
        return path
