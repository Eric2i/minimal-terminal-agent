"""Agent loop tests with a scripted model and an in-memory environment (no network, no Docker)."""

import asyncio
import json

from mta import Agent, AgentConfig, ExecResult, ModelOutput, Trajectory, TrajectoryRecorder, default_tools


class ScriptedModel:
    def __init__(self, messages):
        self.messages = list(messages)
        self.seen = []

    async def generate(self, messages, tools):
        self.seen.append(list(messages))
        return ModelOutput(message=self.messages.pop(0), finish_reason="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 2})


class EchoEnv:
    def __init__(self):
        self.commands = []

    async def execute(self, command, timeout=None):
        self.commands.append(command)
        return ExecResult(output=f"ran {command}\n", returncode=0)


def call(name, args, id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def assistant(*calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def run(agent, task="do it"):
    return asyncio.run(agent.run(task))


def test_bash_then_submit(tmp_path):
    model = ScriptedModel([assistant(call("bash", {"command": "ls"})), assistant(call("submit", {}))])
    env = EchoEnv()
    recorder = TrajectoryRecorder(tmp_path / "t.json")
    traj = run(Agent(model, env, default_tools(), recorder=recorder))

    assert traj.exit_status == "submitted"
    assert env.commands == ["ls"]
    assert [m["role"] for m in traj.messages] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert traj.messages[3]["content"] == "exit code: 0\nran ls\n"
    assert traj.steps[0].tool_calls[0].metadata == {"returncode": 0, "timed_out": False}
    assert traj.steps[1].message_index == 4
    # The model saw the tool result before its second call.
    assert model.seen[1][-1]["role"] == "tool"
    # The recorded file round-trips.
    assert Trajectory.load(recorder.path) == traj


def test_no_tool_call_gets_nudge_and_step_limit():
    model = ScriptedModel([assistant(content="thinking..."), assistant(content="still thinking")])
    traj = run(Agent(model, EchoEnv(), default_tools(), AgentConfig(step_limit=2)))
    assert traj.exit_status == "step_limit"
    assert traj.messages[3]["role"] == "user" and "No tool call" in traj.messages[3]["content"]


def test_model_mistakes_become_observations():
    bad_json = {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{oops"}}
    model = ScriptedModel([
        assistant(call("python", {"code": "1"}), bad_json, call("bash", {}, id="c3")),
        assistant(call("submit", {})),
    ])
    traj = run(Agent(model, EchoEnv(), default_tools()))
    outputs = [m["content"] for m in traj.messages if m["role"] == "tool"]
    assert outputs[0].startswith("Unknown tool 'python'")
    assert outputs[1].startswith("Invalid arguments for bash")
    assert outputs[2].startswith("Missing argument 'command'")
    assert traj.exit_status == "submitted"


def test_model_exception_is_recorded():
    class Broken:
        async def generate(self, messages, tools):
            raise ConnectionError("down")

    traj = run(Agent(Broken(), EchoEnv(), default_tools()))
    assert traj.exit_status == "error" and "ConnectionError: down" in traj.error


def test_timeout_and_truncation():
    class SlowEnv:
        async def execute(self, command, timeout=None):
            if command == "sleep":
                return ExecResult("", -1, timed_out=True)
            return ExecResult("x" * 50_000, 0)

    model = ScriptedModel([assistant(call("bash", {"command": "sleep"}), call("bash", {"command": "big"}, id="c2")), assistant(call("submit", {}))])
    traj = run(Agent(model, SlowEnv(), default_tools(command_timeout=5)))
    timeout_obs, big_obs = [m["content"] for m in traj.messages if m["role"] == "tool"][:2]
    assert "timed out after 5s" in timeout_obs
    assert "characters omitted" in big_obs and len(big_obs) < 11_000


def test_reasoning_field_is_recorded():
    from openai.types.chat import ChatCompletionMessage

    from mta.model import _assistant_message

    for field in ("reasoning_content", "reasoning"):  # DeepSeek; recent vLLM and Ollama
        m = ChatCompletionMessage.model_validate({"role": "assistant", "content": "hi", field: "thought"})
        assert _assistant_message(m)["reasoning_content"] == "thought"
