"""DockerEnvironment against a real container. Skipped when Docker is unavailable."""

import asyncio
import shutil
import subprocess

import pytest

from mta import DockerEnvironment

docker_ok = shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not docker_ok, reason="docker not available")


def test_files_persist_shell_state_does_not():
    async def scenario():
        async with DockerEnvironment(image="ubuntu:24.04", cwd="/tmp") as env:
            name = env.container
            r1 = await env.execute("echo hi > a.txt && cd / && export X=1 && echo $X")
            r2 = await env.execute('pwd; cat a.txt; echo "X=$X"; ls /nope')
            r3 = await env.execute("sleep 5", timeout=1)
        return name, r1, r2, r3

    name, r1, r2, r3 = asyncio.run(scenario())
    assert (r1.output, r1.returncode) == ("1\n", 0)
    assert r2.output.startswith("/tmp\nhi\nX=\n") and "No such file" in r2.output  # stderr merged
    assert r2.returncode != 0
    assert r3.timed_out
    ps = subprocess.run(["docker", "ps", "-a", "-q", "-f", f"name={name}"], capture_output=True, text=True)
    assert ps.stdout.strip() == ""  # container removed
