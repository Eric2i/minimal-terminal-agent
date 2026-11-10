"""Harbor's Daytona environment with a safe auto-stop default, and a sandbox cleanup tool.

Harbor's built-in Daytona backend (`-e daytona`) already runs both single-container
and docker-compose tasks. Its sandboxes are deleted when a trial ends, but by
default they never auto-stop, so a `harbor run` that is killed leaves them running
(and billing) indefinitely. This subclass changes one default: a sandbox stops,
and because Harbor makes it ephemeral is then deleted, after
`auto_stop_interval_mins` (default 60) without any API activity. Harbor polls
every running command, so only an idle sandbox stops; keep the value above the
longest single model call.

    harbor run ... --env-file .env \\
        -e mta.integrations.harbor.daytona:DaytonaEnvironment \\
        [--ek auto_stop_interval_mins=30]

`DAYTONA_API_KEY` must be set (e.g. in the file passed to `--env-file`).

To list Harbor's sandboxes that are still alive, and delete them:

    python -m mta.integrations.harbor.daytona [--delete]
"""

from __future__ import annotations

import argparse
import asyncio

from harbor.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment

DEFAULT_AUTO_STOP_MINS = 60


class DaytonaEnvironment(_DaytonaEnvironment):
    def __init__(self, *args, auto_stop_interval_mins: int = DEFAULT_AUTO_STOP_MINS, **kwargs):
        super().__init__(*args, auto_stop_interval_mins=auto_stop_interval_mins, **kwargs)


async def _harbor_sandboxes(delete: bool) -> None:
    from daytona import AsyncDaytona, ListSandboxesQuery

    async with AsyncDaytona() as daytona:
        found = [s async for s in daytona.list(ListSandboxesQuery(labels={"harbor.managed": "true"}))]
        for s in found:
            name = s.labels.get("harbor.session_id", "")
            print(f"{s.id}  {s.state}  cpu={s.cpu} mem={s.memory}G disk={s.disk}G  auto_stop={s.auto_stop_interval}m  {name}")
            if delete:
                await s.delete()
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
