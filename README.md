# MTA — Minimal Terminal Agent

A small, readable terminal-use agent meant as a base for agent research and
study. Inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent):
the agent's only real tool is a shell, and the whole core is about 650 lines.

- Talks to any **OpenAI-compatible** chat-completions endpoint (vLLM, SGLang,
  Ollama, hosted APIs).
- Runs commands in a **fresh Docker container** per episode by default.
- Records every episode as one **serializable trajectory**: SFT-ready
  messages, plus per-step slots for token IDs, logprobs and loss masks.
- Runs **Terminal-Bench** tasks through [Harbor](https://github.com/laude-institute/harbor)
  via a thin, optional integration; the core never imports Harbor.

## Layout

```
mta/                      core: depends only on `openai`
  agent.py                Agent loop             Agent, AgentConfig, SYSTEM_PROMPT
  model.py                Model backend          Model (Protocol), OpenAIModel, ModelOutput
  environment.py          Environment backend    Environment (Protocol), DockerEnvironment, ExecResult
  tools.py                Tool interface         Tool (Protocol), BashTool, SubmitTool
  trajectory.py           Trajectory recorder    Trajectory, Step, ToolCallRecord, TrajectoryRecorder
  config.py               .env / env-var loading
  cli.py                  `mta "task" --image ...`
  integrations/harbor/    Harbor agent + environment shims (the only code that imports harbor)
examples/terminal_bench/  fetch Terminal-Bench 1.0 tasks in Harbor format
tests/
```

Dependencies point one way: `mta.integrations → mta`. Each boundary is one small Protocol:

```python
class Model(Protocol):
    async def generate(self, messages: list[dict], tools: list[dict]) -> ModelOutput: ...

class Environment(Protocol):
    async def execute(self, command: str, timeout: float | None = None) -> ExecResult: ...

class Tool(Protocol):
    name: str
    schema: dict                      # OpenAI tool spec
    async def run(self, env: Environment, args: dict) -> ToolResult: ...
```

The loop (`Agent.step`) is: call the model with the conversation and tool
schemas → append the assistant message → run each tool call against the
environment → append the observations → save the trajectory. It ends when the
model calls `submit`, at `step_limit`, or on an error.

## Setup

```bash
git clone https://github.com/Eric2i/minimal-terminal-agent && cd minimal-terminal-agent
uv sync                    # core only; add --extra harbor (or --extra modal) for benchmarks
cp .env.example .env       # then set MTA_BASE_URL and MTA_MODEL (never commit .env)
```

Configuration is environment variables only (a `.env` file is read if present;
real environment variables win). Nothing about the endpoint is hard-coded:

| Variable | Example | |
|---|---|---|
| `MTA_BASE_URL` | `http://localhost:8000/v1` | required |
| `MTA_MODEL` | `Qwen3.5-9B` | required |
| `MTA_API_KEY` | | optional for local servers |
| `MTA_TEMPERATURE`, `MTA_MAX_TOKENS` | `0.6`, `32768` | optional |
| `MTA_LOGPROBS` | `true` | optional, request logprobs |
| `MTA_EXTRA_BODY` | `{"top_k": 20}` | optional, extra request JSON (e.g. vLLM sampling params) |

## Run a single task

```bash
uv run mta "Install cowsay and make it say hi into /root/hi.txt" --image ubuntu:24.04
# → runs/mta-<id>.json
```

Options: `--image`, `--cwd`, `--network none` (no network), `--step-limit`,
`--command-timeout`, `-o path.json`. The container is removed afterwards.

From Python:

```python
import asyncio
from mta import Agent, DockerEnvironment, OpenAIModel, TrajectoryRecorder, default_tools
from mta.config import load_dotenv, model_config_from_env

async def main():
    load_dotenv()
    async with DockerEnvironment(image="python:3.12-slim", cwd="/root") as env:
        agent = Agent(OpenAIModel(model_config_from_env()), env, default_tools(),
                      recorder=TrajectoryRecorder("runs/demo.json"))
        return await agent.run("Write fib.py printing the first 10 Fibonacci numbers and run it")

traj = asyncio.run(main())
```

## Terminal-Bench via Harbor

Harbor owns the task containers; `mta.integrations.harbor.MTAAgent` exposes
them to MTA through the plain `Environment` protocol. Terminal-Bench 1.0
(`terminal-bench-core==0.1.1`, 80 tasks) can be fetched and converted with:

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
and Harbor's verdict to `.../verifier/reward.txt`. `-m <model>` overrides `MTA_MODEL`.

Cloud sandboxes: with `--extra modal` and Modal credentials, pass
`-e mta.integrations.harbor.modal_env:ModalEnvironment`. This shim fixes a
Harbor 0.25.0 issue where command output that is not valid UTF-8 crashed the
episode.

Some Terminal-Bench 1.0 tasks no longer work anywhere, because their setup or
tests depend on packages, mirrors or web pages that have changed since the
dataset was frozen. Run the oracle first and report scores on the tasks it
passes.

## Trajectory format

One JSON file per episode (`schema_version: 1`):

```jsonc
{
  "id": "…", "task": "…", "exit_status": "submitted",  // | step_limit | error | cancelled
  "error": null, "started_at": "…", "finished_at": "…",
  "info": {"mta_version": "0.1.0", "model": {…no api key…}, "environment": {…, "image_id": "sha256:…"}},
  "tools": [ /* OpenAI tool schemas offered to the model */ ],
  "messages": [ /* exact OpenAI-format conversation: system, user, assistant(+tool_calls,
                  reasoning_content), tool, … */ ],
  "steps": [{
    "index": 0, "message_index": 2,          // where this step's assistant message sits in `messages`
    "finish_reason": "tool_calls", "usage": {…}, "latency_s": 1.9,
    "tool_calls": [{"id": "…", "name": "bash", "arguments": "{\"command\": \"ls\"}",
                    "output": "exit code: 0\n…", "metadata": {"returncode": 0, "timed_out": false},
                    "duration_s": 0.2}],
    // online-RL slots, filled only when the server returns token-level data:
    "prompt_token_ids": null, "completion_token_ids": null, "logprobs": null, "loss_mask": null
  }]
}
```

- **Debugging**: `steps[].tool_calls` has every command, its raw output, exit code and timing.
- **SFT**: `{"messages", "tools"}` is a standard chat-with-tools example; train on assistant turns.
- **RL (experimental)**: when the server returns token-level data (for vLLM,
  `MTA_LOGPROBS=true` and `MTA_EXTRA_BODY={"return_token_ids": true}`), each
  step stores prompt/completion token IDs, completion logprobs and a loss mask,
  and the Harbor integration forwards them as Harbor `rollout_details`. This
  path is not yet covered by tests.

Reasoning returned by the server (`reasoning_content` or `reasoning`) is stored
on the assistant message as `reasoning_content`. Load a trajectory with
`Trajectory.load(path)`. The file is rewritten after every step, so a crashed
or timed-out run still has everything up to its last step.

## Design choices

- **Native tool calling** (`bash`, `submit`) instead of parsing code blocks out
  of text: arguments arrive as JSON, and the messages are already in the
  format chat templates use for tool-use SFT.
- **Stateless shell**, as in mini-swe-agent: every `bash` call is a new
  `bash -c`, so the files persist but `cd` and env vars do not. This matches
  Harbor's `exec` and avoids terminal-state bugs (hung prompts, pagers).
- **Async** agent loop: Harbor is async, and concurrent rollouts for RL need it.
- **Errors**: model mistakes (unknown tool, bad JSON, missing args, no tool call)
  become observations so the model can recover. Infrastructure errors end the
  episode with `exit_status: "error"` and the traceback in `error`.
- **Kept out of the core**: planning, reflection, memory, context compaction
  and training. Build them on top by implementing a `Model`, `Environment` or
  `Tool`, or by subclassing `Agent`.

## Tests

```bash
uv run pytest     # agent loop with a scripted model; DockerEnvironment against real Docker
```

## License

[Apache-2.0](LICENSE)
