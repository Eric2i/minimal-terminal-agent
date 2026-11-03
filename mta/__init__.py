"""MTA: a minimal terminal agent."""

__version__ = "0.1.0"

from mta.agent import Agent, AgentConfig
from mta.environment import DockerEnvironment, Environment, ExecResult
from mta.model import Model, ModelConfig, ModelOutput, OpenAIModel
from mta.tools import BashTool, SubmitTool, Tool, ToolResult, default_tools
from mta.trajectory import Step, ToolCallRecord, Trajectory, TrajectoryRecorder

__all__ = [
    "Agent", "AgentConfig",
    "Environment", "ExecResult", "DockerEnvironment",
    "Model", "ModelConfig", "ModelOutput", "OpenAIModel",
    "Tool", "ToolResult", "BashTool", "SubmitTool", "default_tools",
    "Trajectory", "Step", "ToolCallRecord", "TrajectoryRecorder",
]
