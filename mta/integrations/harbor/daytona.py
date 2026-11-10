"""Harbor's Daytona environment with safer cleanup and resource floors, plus a sandbox cleanup tool.

Harbor's built-in Daytona backend (`-e daytona`) already runs both single-container
and docker-compose tasks. This subclass changes three things:

- **Auto-stop.** Harbor's sandboxes never auto-stop, so a `harbor run` that is
  killed leaves them running (and billing) indefinitely. Here a sandbox stops,
  and because Harbor makes it ephemeral is then deleted, after
  `auto_stop_interval_mins` (default 60) without API activity. Harbor polls every
  running command, so only an idle sandbox stops; keep the value above the
  longest single model call.
- **Leftover sandboxes.** Harbor only deletes the sandbox it holds a handle to.
  A sandbox whose build failed, or one created by an attempt that Harbor then
  retried, stays behind. On stop, every sandbox labelled with this trial's
  session id is deleted.
- **Resource floors.** Tasks that set no `cpus`/`memory_mb`/`storage_mb` get
  Daytona's smallest sandbox (1 vCPU, 1 GiB, 3 GiB, as hard limits).
  `min_cpus`, `min_memory_mb` and `min_storage_mb` raise those values to a floor
  without lowering what a task asks for (unlike Harbor's `--override-*`).
  For docker-compose tasks the floors size the Docker-in-Docker sandbox; the
  `main` container keeps the limits its task declares.

    harbor run ... --env-file .env \\
        -e mta.integrations.harbor.daytona:DaytonaEnvironment \\
        [--ek auto_stop_interval_mins=30] [--ek min_memory_mb=4096] [--ek min_storage_mb=10240]

`DAYTONA_API_KEY` must be set (e.g. in the file passed to `--env-file`).

To list Harbor's sandboxes that are still alive, and delete them:

    python -m mta.integrations.harbor.daytona [--delete]
"""

from __future__ import annotations

import argparse
import asyncio

from harbor.environments.daytona import DaytonaClientManager
from harbor.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment

DEFAULT_AUTO_STOP_MINS = 60
LEFTOVER_LOOKUPS = 4  # when the start failed: look 4 times, 3 s apart
LEFTOVER_LOOKUP_DELAY_SEC = 3


def _floor(value, minimum):
    if minimum is None:
        return value
    return minimum if value is None else max(value, minimum)


class DaytonaEnvironment(_DaytonaEnvironment):
    def __init__(
        self,
        *args,
        auto_stop_interval_mins: int = DEFAULT_AUTO_STOP_MINS,
        min_cpus: float | None = None,
        min_memory_mb: int | None = None,
        min_storage_mb: int | None = None,
        **kwargs,
    ):
        self._min_cpus, self._min_memory_mb, self._min_storage_mb = min_cpus, min_memory_mb, min_storage_mb
        super().__init__(*args, auto_stop_interval_mins=auto_stop_interval_mins, **kwargs)

    @property
    def _effective_cpus(self):
        return _floor(super()._effective_cpus, self._min_cpus)

    @property
    def _effective_memory_mb(self):
        return _floor(super()._effective_memory_mb, self._min_memory_mb)

    @property
    def _effective_storage_mb(self):
        return _floor(super()._effective_storage_mb, self._min_storage_mb)

    async def stop(self, delete: bool):
        handled = self._sandbox.id if self._sandbox else None  # the sandbox Harbor deletes itself
        try:
            await super().stop(delete)
        finally:
            if delete:
                await self._delete_session_sandboxes(skip=handled)

    async def _delete_session_sandboxes(self, skip: str | None) -> None:
        """Delete sandboxes of this trial that Harbor lost track of (failed builds, retried creates).

        When Harbor never got a handle (the start failed), the sandbox may take a few
        seconds to show up in listings, so look a few times before giving up.
        """
        from daytona import ListSandboxesQuery, SandboxState

        gone = {SandboxState.DESTROYING, SandboxState.DESTROYED}
        query = ListSandboxesQuery(labels={"harbor.session_id": self.session_id})
        try:
            client = await (await DaytonaClientManager.get_instance()).get_client()
            for attempt in range(1 if skip else LEFTOVER_LOOKUPS):
                if attempt:
                    await asyncio.sleep(LEFTOVER_LOOKUP_DELAY_SEC)
                leftovers = [s async for s in client.list(query) if s.id != skip and s.state not in gone]
                for sandbox in leftovers:
                    self.logger.warning("Deleting leftover Daytona sandbox %s (%s)", sandbox.id, sandbox.state)
                    try:
                        await sandbox.delete()
                    except Exception as e:
                        self.logger.warning("Could not delete leftover Daytona sandbox %s: %s", sandbox.id, e)
                if leftovers:
                    return
        except Exception as e:  # cleanup must never fail the trial
            self.logger.warning("Could not check for leftover Daytona sandboxes: %s", e)


async def _harbor_sandboxes(delete: bool) -> None:
    from daytona import AsyncDaytona, ListSandboxesQuery

    async with AsyncDaytona() as daytona:
        found = [s async for s in daytona.list(ListSandboxesQuery(labels={"harbor.managed": "true"}))]
        for s in found:
            name = s.labels.get("harbor.session_id", "")
            print(f"{s.id}  {s.state}  cpu={s.cpu} mem={s.memory}G disk={s.disk}G  auto_stop={s.auto_stop_interval}m  {name}")
            if delete:
                await s.delete(wait=True)
        verb = "deleted" if delete else "found"
        print(f"{len(found)} Harbor sandbox(es) {verb}")


def main() -> None:
    p = argparse.ArgumentParser(description="List (and optionally delete) Harbor's Daytona sandboxes.")
    p.add_argument("--delete", action="store_true", help="delete every sandbox listed")
    args = p.parse_args()

    from mta.config import load_dotenv

    load_dotenv()  # DAYTONA_API_KEY from ./.env if not already set
    asyncio.run(_harbor_sandboxes(args.delete))


if __name__ == "__main__":
    main()
