"""Environment backends: where the agent's shell commands run.

An environment only needs `execute`. Lifecycle (start/stop) belongs to whoever
creates the environment, so the agent loop never manages containers itself.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExecResult:
    output: str  # stdout and stderr, merged
    returncode: int
    timed_out: bool = False


class Environment(Protocol):
    async def execute(self, command: str, timeout: float | None = None) -> ExecResult: ...


@dataclass
class DockerEnvironment:
    """A fresh, throwaway Docker container per episode.

    Every command runs in a new `bash -c` inside the same container: the
    filesystem persists across commands, shell state (cwd, variables) does not.
    The container is removed on `stop()`, so each run starts from the image.
    """

    image: str = "ubuntu:24.04"
    cwd: str = "/"
    network: str | None = None  # e.g. "none" to cut the container off the network
    env: dict[str, str] = field(default_factory=dict)
    run_args: list[str] = field(default_factory=list)  # extra `docker run` flags, e.g. ["--cpus", "2"]
    container: str | None = None
    image_id: str | None = None  # resolved image digest, recorded for reproducibility

    async def start(self) -> None:
        name = f"mta-{uuid.uuid4().hex[:12]}"
        args = ["run", "-d", "--rm", "--name", name, "-w", self.cwd]
        if self.network:
            args += ["--network", self.network]
        for key, value in self.env.items():
            args += ["-e", f"{key}={value}"]
        args += [*self.run_args, "--entrypoint", "sleep", self.image, "infinity"]
        await _docker(*args)
        self.container = name
        self.image_id = (await _docker("inspect", "--format", "{{.Image}}", name)).strip()

    async def execute(self, command: str, timeout: float | None = None) -> ExecResult:
        assert self.container, "call start() first"
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-w", self.cwd, self.container, "bash", "-c", command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            # Kills the `docker exec` client; the process inside may linger until stop().
            proc.kill()
            await proc.wait()
            return ExecResult(output="", returncode=-1, timed_out=True)
        return ExecResult(output=out.decode(errors="replace"), returncode=proc.returncode)

    async def stop(self) -> None:
        if self.container:
            await _docker("rm", "-f", self.container)
            self.container = None

    async def __aenter__(self) -> DockerEnvironment:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()


async def _docker(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {err.decode().strip()}")
    return out.decode()
