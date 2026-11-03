"""The agent loop: ask the model, run the tools it calls, repeat until it submits."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any

from mta.environment import Environment
from mta.model import Model
from mta.tools import Tool, ToolResult
from mta.trajectory import Step, ToolCallRecord, Trajectory, TrajectoryRecorder, now

logger = logging.getLogger("mta")

SYSTEM_PROMPT = """\
You are an autonomous agent that completes tasks by operating a Linux terminal.

Act only through tool calls:
- `bash` runs one command and returns its exit code and output. Each call starts
  a fresh shell in the default working directory: files persist between calls,
  but `cd` and exported variables do not, so chain commands (`cd /app && make`).
- Commands are non-interactive. Do not open editors, pagers or REPLs; write files
  with heredocs or scripts, and pass flags like `-y` to installers.
- Start servers or other long jobs in the background with output redirected,
  e.g. `nohup python server.py > server.log 2>&1 &`.
- `submit` ends the episode. Call it only after you have checked that the task is done.

Work step by step: explore the environment, make changes, verify the result.
Nobody will answer questions; make reasonable assumptions and keep going."""

NO_TOOL_CALL = (
    "No tool call found in your last message. Continue the task by calling `bash`, "
    "or call `submit` if the task is complete."
)


@dataclass
class AgentConfig:
    system_prompt: str = SYSTEM_PROMPT
    step_limit: int = 50  # max model calls per episode


class Agent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        tools: list[Tool],
        config: AgentConfig | None = None,
        recorder: TrajectoryRecorder | None = None,
    ):
        self.model = model
        self.env = env
        self.tools = {tool.name: tool for tool in tools}
        self.config = config or AgentConfig()
        self.recorder = recorder

    async def run(self, task: str, info: dict[str, Any] | None = None) -> Trajectory:
        traj = Trajectory(
            task=task,
            tools=[tool.schema for tool in self.tools.values()],
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": task},
            ],
            info=info or {},
        )
        traj.exit_status = "step_limit"
        try:
            for _ in range(self.config.step_limit):
                if await self.step(traj):
                    traj.exit_status = "submitted"
                    break
        except asyncio.CancelledError:  # e.g. the benchmark harness hit its time limit
            traj.exit_status = "cancelled"
            raise
        except Exception:
            traj.exit_status = "error"
            traj.error = traceback.format_exc()
            logger.error("episode failed:\n%s", traj.error)
        finally:
            traj.finished_at = now()
            self._record(traj)
        return traj

    async def step(self, traj: Trajectory) -> bool:
        """One model call plus its tool calls. Returns True when the episode is done."""
        start = time.monotonic()
        out = await self.model.generate(traj.messages, traj.tools)
        traj.messages.append(out.message)
        step = Step(
            index=len(traj.steps),
            message_index=len(traj.messages) - 1,
            finish_reason=out.finish_reason,
            usage=out.usage,
            latency_s=time.monotonic() - start,
            prompt_token_ids=out.prompt_token_ids,
            completion_token_ids=out.completion_token_ids,
            logprobs=out.logprobs,
            loss_mask=[1] * len(out.completion_token_ids) if out.completion_token_ids else None,
        )
        traj.steps.append(step)
        logger.info("step %d: %s", step.index, out.message.get("content", "")[:200])

        calls = out.message.get("tool_calls") or []
        if not calls:
            traj.messages.append({"role": "user", "content": NO_TOOL_CALL})
        done = False
        for call in calls:
            start = time.monotonic()
            result = await self.call_tool(call["function"]["name"], call["function"]["arguments"])
            traj.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.output})
            step.tool_calls.append(
                ToolCallRecord(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=call["function"]["arguments"],
                    output=result.output,
                    metadata=result.metadata,
                    duration_s=time.monotonic() - start,
                )
            )
            logger.info("  %s %s -> %s", call["function"]["name"], call["function"]["arguments"][:200], result.metadata)
            done = done or result.done
        self._record(traj)
        return done

    async def call_tool(self, name: str, arguments: str) -> ToolResult:
        """Model mistakes (unknown tool, bad JSON) become observations, not crashes."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool {name!r}. Available: {', '.join(self.tools)}.", metadata={"error": "unknown_tool"})
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            return ToolResult(f"Invalid arguments for {name}: {e}", metadata={"error": "bad_arguments"})
        try:
            return await tool.run(self.env, args)
        except KeyError as e:
            return ToolResult(f"Missing argument {e} for {name}.", metadata={"error": "bad_arguments"})

    def _record(self, traj: Trajectory) -> None:
        if self.recorder:
            self.recorder.record(traj)
