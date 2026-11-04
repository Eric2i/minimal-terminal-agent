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

With `uv sync --extra modal` and Modal credentials, add
`-e mta.integrations.harbor.modal_env:ModalEnvironment` to run each task in a
[Modal](https://modal.com) sandbox. This shim fixes a Harbor 0.25.0 issue where
command output that is not valid UTF-8 crashed the episode.

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
