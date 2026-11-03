"""Harbor adapter for MTA. This is the only place that imports Harbor.

    harbor run -p <tasks> -a mta.integrations.harbor:MTAAgent

Harbor owns the task container; HarborEnvironment exposes it to MTA through
the plain `Environment` protocol, so the agent loop never sees Harbor.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

import mta
from mta.agent import Agent, AgentConfig
from mta.config import load_dotenv, model_config_from_env
from mta.environment import ExecResult
from mta.model import OpenAIModel
from mta.tools import default_tools
from mta.trajectory import Trajectory, TrajectoryRecorder


class HarborEnvironment:
    """MTA Environment backed by a Harbor environment (commands run in the task's container)."""

    def __init__(self, env: BaseEnvironment):
        self.env = env

    async def execute(self, command: str, timeout: float | None = None) -> ExecResult:
        try:
            r = await self.env.exec(command, timeout_sec=int(timeout) if timeout else None)
        except RuntimeError as e:
            if "timed out" in str(e):
                return ExecResult(output="", returncode=-1, timed_out=True)
            raise
        return ExecResult(output=(r.stdout or "") + (r.stderr or ""), returncode=r.return_code)


class MTAAgent(BaseAgent):
    """Options via `--ak key=value`: step_limit, command_timeout.
    The model endpoint comes from MTA_* environment variables (or .env);
    `-m <model>` overrides MTA_MODEL."""

    def __init__(self, *args: Any, step_limit: int = 50, command_timeout: float = 180, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.step_limit = int(step_limit)
        self.command_timeout = float(command_timeout)

    @staticmethod
    def name() -> str:
        return "mta"

    def version(self) -> str:
        return mta.__version__

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        load_dotenv()
        model_config = model_config_from_env()
        if self.model_name:
            model_config.model = self.model_name
        agent = Agent(
            OpenAIModel(model_config),
            HarborEnvironment(environment),
            default_tools(command_timeout=self.command_timeout),
            AgentConfig(step_limit=self.step_limit),
            TrajectoryRecorder(self.logs_dir / "mta_trajectory.json"),
        )
        info = {
            "mta_version": mta.__version__,
            "model": model_config.public(),
            "environment": {"type": "harbor", "harbor_environment": environment.type()},
            "harbor_session_id": self.session_id,
        }
        # If Harbor cancels us at its agent timeout, the recorder has already
        # written every finished step (exit_status "cancelled").
        traj = await agent.run(instruction, info=info)
        _fill_context(context, traj)


def _cached_tokens(usage: dict) -> int:
    """Prompt tokens served from cache: OpenAI-style `prompt_tokens_details.cached_tokens`
    (vLLM, Ollama, OpenAI) or DeepSeek's `prompt_cache_hit_tokens`."""
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0


def _fill_context(context: AgentContext, traj: Trajectory) -> None:
    usage = [s.usage for s in traj.steps]
    context.n_input_tokens = sum(u.get("prompt_tokens") or 0 for u in usage)
    context.n_output_tokens = sum(u.get("completion_tokens") or 0 for u in usage)
    context.n_cache_tokens = sum(_cached_tokens(u) for u in usage)
    context.metadata = {"exit_status": traj.exit_status, "n_steps": len(traj.steps), "error": traj.error}
    if traj.steps and all(s.completion_token_ids is not None for s in traj.steps):
        context.rollout_details = [
            {
                "prompt_token_ids": [s.prompt_token_ids or [] for s in traj.steps],
                "completion_token_ids": [s.completion_token_ids for s in traj.steps],
                "logprobs": [s.logprobs or [] for s in traj.steps],
            }
        ]
