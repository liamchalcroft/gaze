"""OpenAI chat/vision adapter for the VLM harness."""

from __future__ import annotations

from typing import Any

from loguru import logger
from openai import APITimeoutError
from openai import AsyncOpenAI
from openai import OpenAIError

from radiant_harness.exceptions import APIError
from radiant_harness.exceptions import ModelError
from radiant_harness.models._types import GenerationLog
from radiant_harness.models.adapter_protocol import AdapterProtocol


class OpenAIAdapter(AdapterProtocol):
    """Adapter around OpenAI's Chat Completions API (text + vision)."""

    def __init__(
        self,
        model_name: str,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
    ) -> None:
        self.model_name = model_name
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        """Call OpenAI chat completions with optional tool calling."""
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
                reasoning=(
                    {"effort": self.reasoning_effort} if self.reasoning_enabled else None
                ),
                cache_control={"type": "ephemeral"} if self.enable_caching else None,
            )
        except APITimeoutError as e:
            raise APIError(f"OpenAI request timed out: {e}", model_name=self.model_name) from e
        except OpenAIError as e:  # pragma: no cover - dependency error surface
            raise APIError(f"OpenAI request failed: {e}", model_name=self.model_name) from e

        if not completion.choices:
            raise ModelError("OpenAI returned no choices", model_name=self.model_name)

        choice = completion.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        usage = completion.usage
        gen_log = GenerationLog(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )

        logger.debug(
            f"OpenAI completion finished with reason={choice.finish_reason}, tokens={gen_log.tokens}"
        )

        return content, tool_calls, gen_log
