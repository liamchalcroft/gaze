# pyright: basic
"""OpenAI chat/vision adapter for the VLM harness."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from beartype import beartype
from loguru import logger
from openai import APITimeoutError
from openai import AsyncOpenAI
from openai import OpenAIError
from openai import RateLimitError
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from radiant_harness.exceptions import APIError
from radiant_harness.exceptions import ModelError
from radiant_harness.models.adapter_protocol import AdapterProtocol
from radiant_harness.models.adapter_protocol import GenerationLog

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAIAdapter(AdapterProtocol):
    """Adapter around OpenAI's Chat Completions API (text + vision)."""

    @beartype
    def __init__(
        self,
        model_name: str,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching
        self._base_url = base_url
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client.

        The client is lazily initialized to avoid unnecessary API key
        validation at module import time. When OPENROUTER_API_KEY is used
        (without OPENAI_API_KEY), base_url is automatically set to the
        OpenRouter endpoint.

        Returns:
            Configured AsyncOpenAI client instance

        Raises:
            ModelError: If no API key is configured
        """
        if self._client is None:
            import os

            openai_key = os.getenv("OPENAI_API_KEY")
            openrouter_key = os.getenv("OPENROUTER_API_KEY")

            api_key = openai_key or openrouter_key
            if not api_key:
                raise ModelError(
                    "No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable",
                    model_name=self.model_name,
                )

            # Resolve base_url: explicit > auto-detect OpenRouter > default (OpenAI)
            base_url = self._base_url
            if base_url is None and not openai_key and openrouter_key:
                base_url = OPENROUTER_BASE_URL
                logger.info("Using OpenRouter base URL (OPENROUTER_API_KEY detected)")

            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": 60.0,
                "max_retries": 3,
            }
            if base_url is not None:
                kwargs["base_url"] = base_url

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @beartype
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/5 for OpenAI API: {retry_state.outcome.exception()}"
        ),
    )
    async def _create_completion_with_retry(self, **kwargs):
        """Create a completion with retry logic."""
        return await self.client.chat.completions.create(**kwargs)

    @beartype
    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]:
        """Call OpenAI chat completions with optional tool calling."""
        # Build request kwargs - only include optional params if they have values
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools is not None:
            kwargs["tools"] = tools

        if response_format is not None:
            kwargs["response_format"] = response_format

        # Provider-specific extensions (OpenRouter, etc.)
        # These are passed via extra_body to avoid breaking standard OpenAI API
        extra_body: dict[str, Any] = {}
        if self.reasoning_enabled:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if self.enable_caching:
            extra_body["cache_control"] = {"type": "ephemeral"}

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Handle streaming
        if stream:
            kwargs["stream"] = True
            return self._stream_completion(**kwargs)

        try:
            completion = await self._create_completion_with_retry(**kwargs)
        except OpenAIError as e:  # pragma: no cover - dependency error surface
            # If retries failed, raise APIError
            if isinstance(e, APITimeoutError | RateLimitError):
                raise APIError(
                    f"OpenAI API error after retries: {e}", model_name=self.model_name
                ) from e
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

    @beartype
    async def _stream_completion(self, **kwargs) -> AsyncIterator[str]:
        """Stream completion with retry logic."""
        try:
            stream = await self._create_completion_with_retry(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIError as e:
            if isinstance(e, APITimeoutError | RateLimitError):
                raise APIError(
                    f"OpenAI API streaming error after retries: {e}", model_name=self.model_name
                ) from e
            raise APIError(f"OpenAI streaming failed: {e}", model_name=self.model_name) from e
