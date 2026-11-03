"""Model backends: turn a conversation into the next assistant message.

Messages and tools use the OpenAI chat-completions format throughout, so the
same trajectory can be replayed against any OpenAI-compatible server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ModelOutput:
    message: dict[str, Any]  # assistant message: role, content, tool_calls, reasoning_content
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # Token-level data for online RL. Only filled when the server returns it
    # (e.g. vLLM with `logprobs` and `return_token_ids`); None otherwise.
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None


class Model(Protocol):
    async def generate(self, messages: list[dict], tools: list[dict]) -> ModelOutput: ...


@dataclass
class ModelConfig:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float | None = None
    max_tokens: int | None = None
    logprobs: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)  # server-specific request fields
    timeout: float = 600.0
    max_retries: int = 3

    def public(self) -> dict[str, Any]:
        """Config without the API key, safe to write into trajectories."""
        return {k: v for k, v in vars(self).items() if k != "api_key"}


class OpenAIModel:
    """Any server that speaks the OpenAI chat-completions API (DeepSeek, vLLM, Ollama, ...)."""

    def __init__(self, config: ModelConfig):
        from openai import AsyncOpenAI

        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def generate(self, messages: list[dict], tools: list[dict]) -> ModelOutput:
        c = self.config
        kwargs: dict[str, Any] = {"model": c.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        if c.max_tokens is not None:
            kwargs["max_tokens"] = c.max_tokens
        if c.logprobs:
            kwargs["logprobs"] = True
        if c.extra_body:
            kwargs["extra_body"] = c.extra_body

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return ModelOutput(
            message=_assistant_message(choice.message),
            finish_reason=choice.finish_reason,
            usage=response.usage.model_dump() if response.usage else {},
            prompt_token_ids=getattr(response, "prompt_token_ids", None),
            completion_token_ids=getattr(choice, "token_ids", None),
            logprobs=[t.logprob for t in choice.logprobs.content]
            if choice.logprobs and choice.logprobs.content
            else None,
        )


def _assistant_message(m: Any) -> dict[str, Any]:
    """Keep only the fields that belong in the conversation history."""
    message: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
    # DeepSeek uses `reasoning_content`; recent vLLM and Ollama use `reasoning`
    reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
    if reasoning:
        message["reasoning_content"] = reasoning
    if m.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in m.tool_calls
        ]
    return message
