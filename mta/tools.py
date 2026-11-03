"""Tools: what the model can call. A tool turns JSON arguments into a text observation.

The default toolset is deliberately tiny: a shell, and a way to say "done".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from mta.environment import Environment


@dataclass
class ToolResult:
    output: str  # observation shown to the model
    done: bool = False  # True ends the episode
    metadata: dict[str, Any] = field(default_factory=dict)  # recorded, not shown to the model


class Tool(Protocol):
    name: str
    schema: dict[str, Any]  # OpenAI tool spec: {"type": "function", "function": {...}}

    async def run(self, env: Environment, args: dict[str, Any]) -> ToolResult: ...


class BashTool:
    name = "bash"

    def __init__(self, timeout: float = 180, max_output_chars: int = 10_000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Run a command with `bash -c` in the task's Linux environment and return its "
                    f"exit code and combined stdout/stderr. Times out after {timeout:g}s. "
                    "Each call starts a new shell: files persist, but cd and environment "
                    "variables do not."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "The command to run."}},
                    "required": ["command"],
                },
            },
        }

    async def run(self, env: Environment, args: dict[str, Any]) -> ToolResult:
        r = await env.execute(args["command"], timeout=self.timeout)
        meta = {"returncode": r.returncode, "timed_out": r.timed_out}
        if r.timed_out:
            return ToolResult(
                f"Command timed out after {self.timeout:g}s. Run long jobs in the background "
                "(`nohup cmd > log 2>&1 &`) or break them into smaller steps.",
                metadata=meta,
            )
        output = truncate(r.output, self.max_output_chars)
        return ToolResult(f"exit code: {r.returncode}\n{output}", metadata=meta)


class SubmitTool:
    name = "submit"
    schema = {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Call this once the task is complete and verified. It ends the episode.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    async def run(self, env: Environment, args: dict[str, Any]) -> ToolResult:
        return ToolResult("Submitted.", done=True)


def default_tools(command_timeout: float = 180) -> list[Tool]:
    return [BashTool(timeout=command_timeout), SubmitTool()]


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of long outputs; the middle is usually least useful."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters omitted] ...\n{text[-half:]}"
