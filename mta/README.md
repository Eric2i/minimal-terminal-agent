# How MTA works

The core is one agent loop and three small interfaces. It depends only on
`openai`; everything else is optional.

## The loop

`Agent.step()` does one model call and the tool calls it asks for:

1. Call the model with the conversation so far and the tool schemas.
2. Append the assistant message to the conversation.
3. Run each tool call against the environment, and append each output as a
   `tool` message.
4. Save the trajectory.

`Agent.run()` repeats `step()` until the model calls `submit`, `step_limit`
model calls have been made, or an infrastructure error occurs.

The model decides when it is done. If a reply contains no tool call, the loop
adds a reminder to call `bash` or `submit` and continues. Model mistakes
(unknown tool, invalid JSON, missing argument) are returned to the model as
observations so it can recover; only infrastructure errors end the episode.

## Components

| File | What it does | Main names |
|---|---|---|
| `agent.py` | The loop above, and the system prompt | `Agent`, `AgentConfig`, `SYSTEM_PROMPT` |
| `model.py` | Turns a conversation into the next assistant message | `Model`, `OpenAIModel`, `ModelConfig`, `ModelOutput` |
| `environment.py` | Runs a shell command somewhere and returns its output | `Environment`, `DockerEnvironment`, `ExecResult` |
| `tools.py` | Turns a tool call's JSON arguments into an observation | `Tool`, `BashTool`, `SubmitTool`, `default_tools` |
| `trajectory.py` | The record of an episode, written to disk after every step | `Trajectory`, `Step`, `ToolCallRecord`, `TrajectoryRecorder` |
| `config.py` | Reads `MTA_*` settings from the environment or a `.env` file | `model_config_from_env`, `load_dotenv` |
| `cli.py` | The `mta` command: one task in a fresh Docker container | `main` |
| `integrations/harbor/` | Runs MTA as a [Harbor](https://github.com/laude-institute/harbor) agent; the only code that imports Harbor | `MTAAgent`, `HarborEnvironment` |

Each boundary is one `Protocol`, so any part can be replaced without touching
the loop:

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

**Model.** `OpenAIModel` works with any OpenAI-compatible chat-completions
server (vLLM, SGLang, Ollama, hosted APIs). Actions come from native tool
calls, never from parsing the reply text, so arguments arrive as JSON and the
conversation is already in the format chat templates use for tool-use training.
Reasoning returned by the server (`reasoning_content` or `reasoning`) is kept on
the assistant message as `reasoning_content`.

**Environment.** `DockerEnvironment` starts a fresh container per episode and
removes it afterwards. Environments only execute commands; whoever creates one
also starts and stops it, so the loop never manages containers. Under Harbor,
`HarborEnvironment` adapts Harbor's task container to the same interface.

**Tools.** The default set is two tools. `bash` runs one command string with
`bash -c`, so anything installed in the container is available, including
pipes and heredocs; output is capped at 10,000 characters (head and tail
kept). Each call starts a new shell: files persist between calls, but `cd`
and exported variables do not. This avoids terminal-state problems such as
hung prompts and pagers. `submit` ends the episode.

**Async.** The loop is async because Harbor is, and because collecting many
rollouts at once needs concurrency.

Planning, reflection, memory, context compaction and training are kept out of
the core. Build them on top by implementing a `Model`, `Environment` or
`Tool`, or by subclassing `Agent`.

## Trajectory format

Each episode is one JSON file (`schema_version: 1`):

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

- **Debugging**: `steps[].tool_calls` has every command, its raw output, exit
  code and timing.
- **SFT**: `{"messages", "tools"}` is a standard chat-with-tools example; train
  on the assistant turns.
- **RL (experimental)**: when the server returns token-level data (for vLLM,
  `MTA_LOGPROBS=true` and `MTA_EXTRA_BODY={"return_token_ids": true}`), each
  step stores prompt and completion token IDs, completion logprobs and a loss
  mask, and the Harbor integration forwards them as Harbor `rollout_details`.
  This path is not yet covered by tests.

Load a trajectory with `Trajectory.load(path)`. The file is rewritten after
every step, so a crashed or timed-out run keeps everything up to its last step.
