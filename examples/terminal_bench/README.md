# Terminal-Bench via Harbor

[Harbor](https://github.com/laude-institute/harbor) owns the task containers,
time limits and grading; `mta.integrations.harbor.MTAAgent` exposes each task
container to MTA through the plain `Environment` interface.

`prepare_tb1.py` fetches Terminal-Bench 1.0 (`terminal-bench-core==0.1.1`,
80 tasks) at the commit its registry entry points to, and converts it to
Harbor's task format with Harbor's own `harbor task migrate`.

```bash
uv sync --extra harbor
uv run python examples/terminal_bench/prepare_tb1.py          # 5-task smoke subset → data/tb1/tasks
uv run python examples/terminal_bench/prepare_tb1.py --all    # all 80 tasks

# Check which tasks are solvable in your setup with the reference solutions:
uv run harbor run -p data/tb1/tasks -a oracle -o jobs/oracle

# Run MTA (local Docker):
uv run harbor run -p data/tb1/tasks -a mta.integrations.harbor:MTAAgent -n 4 -o jobs/mta \
    --ak step_limit=50 --ak command_timeout=180
```

Each trial writes the MTA trajectory to `jobs/.../<trial>/agent/mta_trajectory.json`
and Harbor's verdict to `.../verifier/reward.txt`. `-m <model>` overrides
`MTA_MODEL`; `--agent-timeout-multiplier` scales the per-task time limits.

## Cloud sandboxes

**Modal.** With `uv sync --extra modal` and Modal credentials, add
`-e mta.integrations.harbor.modal_env:ModalEnvironment` to run each task in a
[Modal](https://modal.com) sandbox. This shim fixes a Harbor 0.25.0 issue where
command output that is not valid UTF-8 crashed the episode.

**Daytona.** With `uv sync --extra daytona` and `DAYTONA_API_KEY` in `.env`:

```bash
uv run harbor run -p data/tb1/tasks --env-file .env \
    -e mta.integrations.harbor.daytona:DaytonaEnvironment \
    -a mta.integrations.harbor:MTAAgent -n 4 -o jobs/mta-daytona
uv run python -m mta.integrations.harbor.daytona            # list leftover sandboxes
uv run python -m mta.integrations.harbor.daytona --delete   # ... and delete them
```

Harbor's own `-e daytona` runs single-container tasks directly and
docker-compose tasks with Docker-in-Docker. Its sandboxes never auto-stop, so
a killed run leaves them billing; MTA's subclass stops (and deletes) a sandbox
after 60 idle minutes (`--ek auto_stop_interval_mins=N` to change it).

- **Resources.** Tasks that set no `cpus`/`memory_mb`/`storage_mb` get
  Daytona's default sandbox: 1 vCPU, 1 GiB RAM, 3 GiB disk, as hard limits.
  `--override-cpus`, `--override-memory-mb` and `--override-storage-mb`
  raise them, but apply to every task. Your account tier caps the size of one
  sandbox and the total across concurrent sandboxes, which bounds `-n`.
- **Network.** Lower Daytona tiers restrict sandbox internet access to an
  allowlist of common services. Package registries (PyPI, npm, apt, Go, crates,
  Docker Hub, ghcr.io), GitHub, Hugging Face and `astral.sh` (used by the
  verifiers to install `uv`) are reachable; arbitrary websites are not. Tasks
  that fetch from elsewhere fail for that reason alone, so run the oracle on
  Daytona before comparing scores with other backends.

## Before reporting scores

- **Run the oracle first.** Some Terminal-Bench 1.0 tasks no longer work
  anywhere: their setup or tests depend on packages, mirrors or web pages that
  have changed since the dataset was frozen. Report scores on the tasks the
  oracle passes.
- **Check for leaked answers.** A few tasks expose their solution or grading
  data to the agent, for example by copying `solution.sh` or `tests/` into the
  image, or by hiding answers behind permissions that do not stop an agent
  running as root. Inspect each task's Dockerfile, and search the trajectories
  for reads of `/tests`, `/solution` or similar paths.
- **State the time budget and the inference setup.** With a slow model server,
  many failures are timeouts rather than wrong answers.
