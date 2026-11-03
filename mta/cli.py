"""Run one task in a fresh Docker container:

    mta "Create hello.txt containing 'hi'" --image ubuntu:24.04
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from mta import __version__
from mta.agent import Agent, AgentConfig
from mta.config import load_dotenv, model_config_from_env
from mta.environment import DockerEnvironment
from mta.model import OpenAIModel
from mta.tools import default_tools
from mta.trajectory import Trajectory, TrajectoryRecorder


async def run(args: argparse.Namespace) -> Trajectory:
    model_config = model_config_from_env()
    env = DockerEnvironment(image=args.image, cwd=args.cwd, network=args.network)
    async with env:
        recorder = TrajectoryRecorder(args.output or Path("runs") / f"{env.container}.json")
        agent = Agent(
            OpenAIModel(model_config),
            env,
            default_tools(command_timeout=args.command_timeout),
            AgentConfig(step_limit=args.step_limit),
            recorder,
        )
        info = {
            "mta_version": __version__,
            "model": model_config.public(),
            "environment": {"type": "docker", "image": args.image, "image_id": env.image_id, "cwd": args.cwd, "network": args.network},
        }
        traj = await agent.run(args.task, info=info)
    print(f"exit_status={traj.exit_status} steps={len(traj.steps)} trajectory={recorder.path}")
    return traj


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal Terminal Agent")
    p.add_argument("task", help="task instruction")
    p.add_argument("--image", default="ubuntu:24.04")
    p.add_argument("--cwd", default="/root")
    p.add_argument("--network", default=None, help='docker network, e.g. "none"')
    p.add_argument("--step-limit", type=int, default=50)
    p.add_argument("--command-timeout", type=float, default=180)
    p.add_argument("--env-file", default=".env")
    p.add_argument("-o", "--output", type=Path, help="trajectory JSON path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(args.env_file)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
