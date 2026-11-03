"""Trajectory: the single serializable record of an episode, and a recorder that persists it.

`messages` is the exact OpenAI-format conversation the model saw, so
{"messages", "tools"} is directly an SFT example. `steps` adds one entry per
model call with timing, usage, tool executions and, when the server provides
them, token IDs / logprobs / loss mask for online RL.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCallRecord:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model produced it
    output: str  # observation returned to the model
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. returncode, timed_out
    duration_s: float = 0.0


@dataclass
class Step:
    """One model call and the tool calls it triggered."""

    index: int
    message_index: int  # position of this step's assistant message in Trajectory.messages
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    # Online-RL fields, aligned with completion_token_ids. None when the server
    # does not return token-level data (e.g. hosted APIs).
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    loss_mask: list[int] | None = None  # 1 = train on this token


@dataclass
class Trajectory:
    task: str
    tools: list[dict[str, Any]]  # tool schemas offered to the model
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    exit_status: str | None = None  # submitted | step_limit | error | cancelled
    error: str | None = None
    info: dict[str, Any] = field(default_factory=dict)  # model/env config, versions, task ids, ...
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = SCHEMA_VERSION
    started_at: str = field(default_factory=now)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        data = dict(data)
        data["steps"] = [
            Step(**{**s, "tool_calls": [ToolCallRecord(**t) for t in s["tool_calls"]]})
            for s in data["steps"]
        ]
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        return cls.from_dict(json.loads(Path(path).read_text()))


class TrajectoryRecorder:
    """Writes the trajectory to a JSON file after every step, so a crash or
    timeout still leaves everything up to the last step on disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, trajectory: Trajectory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(trajectory.to_dict(), indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)
