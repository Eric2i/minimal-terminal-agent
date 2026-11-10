import asyncio

import pytest

pytest.importorskip("daytona")

from harbor.environments.daytona import DaytonaEnvironment as Base  # noqa: E402

from mta.integrations.harbor import daytona as mod  # noqa: E402
from mta.integrations.harbor.daytona import DEFAULT_AUTO_STOP_MINS, DaytonaEnvironment  # noqa: E402


def test_defaults_to_auto_stop(monkeypatch):
    """Harbor's sandboxes never auto-stop by default; ours do, unless overridden."""
    seen = []
    monkeypatch.setattr(Base, "__init__", lambda self, *a, **kw: seen.append(kw["auto_stop_interval_mins"]))
    DaytonaEnvironment()
    DaytonaEnvironment(auto_stop_interval_mins=15)
    assert seen == [DEFAULT_AUTO_STOP_MINS, 15]


def test_resource_floors_raise_but_never_lower(monkeypatch):
    for name, value in [("_effective_cpus", None), ("_effective_memory_mb", 8192), ("_effective_storage_mb", 2048)]:
        monkeypatch.setattr(Base, name, property(lambda self, v=value: v))
    env = object.__new__(DaytonaEnvironment)
    env._min_cpus, env._min_memory_mb, env._min_storage_mb = 2, 4096, None
    assert env._effective_cpus == 2  # unset -> floor
    assert env._effective_memory_mb == 8192  # task asks for more than the floor
    assert env._effective_storage_mb == 2048  # no floor


class _Sandbox:
    def __init__(self, id, state="build_failed"):
        from daytona import SandboxState

        self.id, self.state, self.deleted = id, SandboxState(state), False

    async def delete(self):
        if self.state.value == "destroying":
            raise RuntimeError("Sandbox state change in progress")
        self.deleted = True


def _stop_with(monkeypatch, sandboxes, handle, base_stop):
    queries = []

    class Client:
        async def list(self, query):
            queries.append(query.labels)
            for s in sandboxes:
                yield s

    class Manager:
        async def get_client(self):
            return Client()

    async def get_instance():
        return Manager()

    async def no_sleep(_):
        pass

    monkeypatch.setattr(mod.DaytonaClientManager, "get_instance", staticmethod(get_instance))
    monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(Base, "stop", base_stop)
    env = object.__new__(DaytonaEnvironment)
    env.session_id = "task__abc__env"
    env.logger = __import__("logging").getLogger("test")
    env._sandbox = handle
    return env, queries


def test_stop_deletes_sandboxes_harbor_lost(monkeypatch):
    """A failed build leaves Harbor without a handle; the sandbox is found by its session label."""
    leftover = _Sandbox("a")

    async def base_stop(self, delete):
        raise RuntimeError("Sandbox not found")

    env, queries = _stop_with(monkeypatch, [leftover], None, base_stop)
    with pytest.raises(RuntimeError):
        asyncio.run(env.stop(delete=True))
    assert queries[0] == {"harbor.session_id": "task__abc__env"}
    assert leftover.deleted


def test_stop_skips_the_sandbox_harbor_deletes(monkeypatch):
    """Harbor's own sandbox is still listed while it is being deleted; only the orphan goes."""
    own, orphan = _Sandbox("own", "started"), _Sandbox("orphan", "started")
    busy = _Sandbox("busy", "destroying")

    async def base_stop(self, delete):
        pass

    env, queries = _stop_with(monkeypatch, [own, busy, orphan], own, base_stop)
    asyncio.run(env.stop(delete=True))
    assert not own.deleted and not busy.deleted and orphan.deleted
    assert len(queries) == 1
