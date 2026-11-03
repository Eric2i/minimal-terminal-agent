"""Fetch Terminal-Bench 1.0 tasks and convert them to Harbor's task format.

Terminal-Bench 1.0 is the `terminal-bench-core==0.1.1` dataset, pinned below to
the commit its registry entry points at. Conversion uses Harbor's own
`harbor task migrate`; the only thing we add is turning the legacy
`solution.yaml` (a list of keystrokes, used by the oracle agent) into
`solution.sh`, which the migrator requires.

    uv run --extra harbor python examples/terminal_bench/prepare_tb1.py          # smoke subset
    uv run --extra harbor python examples/terminal_bench/prepare_tb1.py --all    # all 80 tasks
    uv run --extra harbor python examples/terminal_bench/prepare_tb1.py --tasks fix-git

Note that some tasks at this commit no longer work anywhere: their setup or
tests depend on packages, mirrors or web pages that have since changed. Run the
oracle agent (`-a oracle`) first to see which tasks are solvable in your setup.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

TB_REPO = "https://github.com/laude-institute/terminal-bench.git"
TB_COMMIT = "91e10457b5410f16c44364da1a34cb6de8c488a5"  # terminal-bench-core 0.1.1

# Small images with arm64 builds, no GPU, one task per category. (fix-git was
# dropped: its setup clones a GitHub repo that no longer exists.)
SMOKE_TASKS = [
    "hello-world",  # file-operations: sanity check of the whole pipeline
    "fix-permissions",  # system-administration: inspect and chmod
    "sqlite-db-truncate",  # debugging: recover rows from a corrupted SQLite file
    "heterogeneous-dates",  # file-operations / data: parse files, compute an answer
    "openssl-selfsigned-cert",  # security: drive a CLI tool and write a script
]


def fetch(src: Path) -> None:
    if (src / ".git").exists():
        return
    src.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=src, check=True)  # noqa: E731
    git("init", "-q")
    git("fetch", "-q", "--depth", "1", TB_REPO, TB_COMMIT)
    git("checkout", "-q", "FETCH_HEAD")


def core_task_dirs(src: Path) -> list[str]:
    """The 80 task ids of terminal-bench-core 0.1.1. Variants such as
    `crack-7z-hash.easy` are their own directories; `.base` means the plain one."""
    ids = yaml.safe_load((src / "datasets" / "terminal-bench-core-v0.yaml").read_text())["task_ids"]
    return [i.removesuffix(".base") for i in ids]


def solution_yaml_to_sh(task: Path) -> None:
    steps = yaml.safe_load((task / "solution.yaml").read_text())
    lines = ["#!/bin/bash", "# Converted from solution.yaml by prepare_tb1.py", *(s["command"] for s in steps)]
    (task / "solution.sh").write_text("\n".join(lines) + "\n")
    (task / "solution.yaml").unlink()


def main() -> None:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tasks", nargs="+", default=SMOKE_TASKS)
    group.add_argument("--all", action="store_true", help="all 80 terminal-bench-core 0.1.1 tasks")
    p.add_argument("--out", type=Path, default=Path("data/tb1"))
    args = p.parse_args()

    src, staging, tasks_dir = args.out / "src", args.out / "staging", args.out / "tasks"
    fetch(src)
    if args.all:
        args.tasks = core_task_dirs(src)
    shutil.rmtree(staging, ignore_errors=True)
    for name in args.tasks:
        shutil.copytree(src / "tasks" / name, staging / name)
        if (staging / name / "solution.yaml").exists():
            solution_yaml_to_sh(staging / name)
    for name in args.tasks:
        shutil.rmtree(tasks_dir / name, ignore_errors=True)
    subprocess.run(["harbor", "task", "migrate", "-i", str(staging), "-o", str(tasks_dir)], check=True)
    shutil.rmtree(staging)
    print(f"Tasks ready in {tasks_dir}")


if __name__ == "__main__":
    main()
