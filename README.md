# MTA — Minimal Terminal Agent

A small, readable terminal-use agent meant as a base for agent research and
study. The core is about 490 lines (excluding blank lines, comments and
docstrings).

- **Inference Engine Agnostic**: works with any OpenAI-compatible
  chat-completions endpoint (vLLM, SGLang, Ollama, hosted APIs).
- **Native Sandbox Environment**: runs commands in a fresh Docker container
  per episode by default for reproducibility.
- **Token-level Replay**: records every episode as one serializable
  trajectory: SFT-ready messages, plus per-step slots for token IDs, logprobs
  and loss masks.

## Setup

```bash
git clone https://github.com/Eric2i/minimal-terminal-agent && cd minimal-terminal-agent
uv sync                    # core only; add --extra harbor (or --extra modal / daytona) for benchmarks
cp .env.example .env       # then set MTA_BASE_URL and MTA_MODEL (never commit .env)
```

Configuration is environment variables only (a `.env` file in the current
directory is read if present; real environment variables win). Nothing about
the endpoint is hard-coded:

| Variable | Example | |
|---|---|---|
| `MTA_BASE_URL` | `http://localhost:8000/v1` | required |
| `MTA_MODEL` | `Qwen3.5-9B` | required |
| `MTA_API_KEY` | | optional for local servers |
| `MTA_TEMPERATURE`, `MTA_MAX_TOKENS` | `0.6`, `32768` | optional |
| `MTA_LOGPROBS` | `true` | optional, request logprobs |
| `MTA_EXTRA_BODY` | `{"top_k": 20}` | optional, extra request JSON (e.g. vLLM sampling params) |

## Give it a Task

### CLI

```bash
uv run mta "Install cowsay and make it say hi into /root/hi.txt" --image ubuntu:24.04
```
Other options:
`--image`, `--cwd`, `--network none` (no network), `--step-limit`,
`--command-timeout`, `--env-file`, `-o path.json`.

The agent works in a fresh
container until it calls `submit` or reaches the step limit; the container is then removed, and the full trajectory is written to `runs/`. 

### Python Interface

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

## Acknowledgements

MTA's design is inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
which showed that a shell-only agent in a few hundred lines can go a long way.

## License

[Apache-2.0](LICENSE)
