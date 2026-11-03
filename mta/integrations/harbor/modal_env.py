"""Harbor's Modal environment, made tolerant of non-UTF-8 command output.

Harbor 0.25.0 reads sandbox stdout/stderr as strict UTF-8 text, so a command
that prints binary bytes (e.g. `cat` on a git object) raises UnicodeDecodeError
and ends the episode as an infrastructure error. This subclass reads raw bytes
and decodes with errors="replace" instead; everything else is unchanged.

    harbor run ... -e mta.integrations.harbor.modal_env:ModalEnvironment

This is a compatibility shim: it overrides a private Harbor method and is
pinned to harbor==0.25.0. Drop it once Harbor decodes output leniently.
"""

from __future__ import annotations

from harbor.environments.base import ExecResult
from harbor.environments.modal import ModalEnvironment as _ModalEnvironment
from harbor.environments.modal import _cached_env_secret


class ModalEnvironment(_ModalEnvironment):
    async def _sdk_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash",
        login: bool = False,
    ) -> ExecResult:
        # Same as the parent, except text=False plus a lenient decode.
        env = self._merge_env(env)
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        try:
            process = await self._sandbox.exec.aio(
                shell,
                "-lc" if login else "-c",
                command,
                workdir=cwd,
                secrets=[_cached_env_secret(env)] if env else [],
                timeout=timeout_sec,
                text=False,
            )
            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            return_code = await process.wait.aio()
        except Exception as exc:
            await self._raise_if_unexpected_exit_137(cause=exc)
            raise
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=return_code,
        )

