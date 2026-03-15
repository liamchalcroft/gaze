# pyright: basic
"""OpenAI chat/vision adapter for the VLM harness."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import httpx
from beartype import beartype
from loguru import logger
from openai import APIStatusError
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


def _safe_error_summary(e: Exception) -> str:
    """Build a safe, bounded error summary that never leaks credentials.

    OpenAI SDK exceptions can contain request headers (including
    ``Authorization: Bearer sk-...``) in their ``__str__`` output.  This
    helper extracts only the class name, status code, and a truncated
    ``message`` attribute (if present) to avoid leaking secrets into logs
    or re-raised exception messages.
    """
    cls_name = type(e).__name__
    status = getattr(e, "status_code", None)
    body = getattr(e, "message", "")
    if isinstance(body, str) and len(body) > 200:
        body = body[:200] + "..."
    parts = [cls_name]
    if status is not None:
        parts.append(f"status={status}")
    if body:
        parts.append(body)
    return ": ".join(parts)


class OpenAIAdapter(AdapterProtocol):
    """Adapter around OpenAI's Chat Completions API (text + vision)."""

    supports_multipart_tool_content: bool = True

    _ALLOWED_BASE_URLS: frozenset[str] = frozenset(
        {
            OPENROUTER_BASE_URL,
            "https://api.openai.com/v1",
        }
    )

    @staticmethod
    def _validate_base_url(url: str) -> None:
        """Validate *url* is HTTPS and reject unlisted hosts unless opted-in.

        Custom base URLs route the API key to a third-party host.  To
        prevent accidental credential leakage, non-allowlisted URLs are
        only permitted when ``RADIANT_ALLOW_CUSTOM_BASE_URL=1`` is set in
        the environment.
        """
        import os
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ModelError(
                f"base_url must use HTTPS scheme, got {parsed.scheme!r}",
                model_name=None,
            )
        if url.rstrip("/") not in {u.rstrip("/") for u in OpenAIAdapter._ALLOWED_BASE_URLS}:
            if os.environ.get("RADIANT_ALLOW_CUSTOM_BASE_URL") == "1":
                logger.warning(
                    f"Custom base_url {parsed.netloc!r} is not on the built-in allowlist. "
                    "API key will be sent to this host (allowed via RADIANT_ALLOW_CUSTOM_BASE_URL)."
                )
            else:
                raise ModelError(
                    f"Custom base_url {parsed.netloc!r} is not on the built-in allowlist. "
                    "Set RADIANT_ALLOW_CUSTOM_BASE_URL=1 to allow sending your API key "
                    "to third-party hosts.",
                    model_name=None,
                )

    @beartype
    def __init__(
        self,
        model_name: str,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
        base_url: str | None = None,
    ) -> None:
        if base_url is not None:
            self._validate_base_url(base_url)
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
                    "No API key found. Set OPENAI_API_KEY or "
                    "OPENROUTER_API_KEY environment variable",
                    model_name=self.model_name,
                )

            # Resolve base_url: explicit > auto-detect OpenRouter > default (OpenAI)
            base_url = self._base_url
            if base_url is None and not openai_key and openrouter_key:
                base_url = OPENROUTER_BASE_URL
                logger.info("Using OpenRouter base URL (OPENROUTER_API_KEY detected)")

            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": httpx.Timeout(
                    connect=10.0,
                    read=90.0,
                    write=10.0,
                    pool=30.0,
                ),
                "max_retries": 0,
            }
            if base_url is not None:
                kwargs["base_url"] = base_url

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @beartype
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/5 for OpenAI API: "
            f"{type(retry_state.outcome.exception()).__name__}"
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
        seed: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]:
        """Call OpenAI chat completions with optional tool calling."""
        # Build request kwargs - only include optional params if they have values
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if seed is not None:
            kwargs["seed"] = seed

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
            kwargs["stream_options"] = {"include_usage": True}
            return self._stream_completion(**kwargs)

        try:
            completion = await self._create_completion_with_retry(**kwargs)
        except OpenAIError as e:  # pragma: no cover - dependency error surface
            status_code = e.status_code if isinstance(e, APIStatusError) else None
            safe_msg = _safe_error_summary(e)
            if isinstance(e, APITimeoutError | RateLimitError):
                raise APIError(
                    f"OpenAI API error after retries: {safe_msg}",
                    model_name=self.model_name,
                    status_code=status_code,
                ) from e
            raise APIError(
                f"OpenAI request failed: {safe_msg}",
                model_name=self.model_name,
                status_code=status_code,
            ) from e

        if not completion.choices:
            raise ModelError("OpenAI returned no choices", model_name=self.model_name)

        choice = completion.choices[0]
        message = choice.message

        content = message.content or ""
        # Thinking models (Qwen3.5, etc.) put chain-of-thought in reasoning_content.
        # If content is empty, fall back to reasoning_content for the actual answer.
        reasoning = getattr(message, "reasoning_content", None) or None
        if not content and reasoning:
            logger.info("Content empty, falling back to reasoning_content from thinking model")
            content = reasoning

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
            reasoning_content=reasoning,
        )

        logger.debug(
            f"OpenAI completion finished with "
            f"reason={choice.finish_reason}, tokens={gen_log.tokens}"
        )

        return content, tool_calls, gen_log

    @beartype
    async def _stream_completion(self, **kwargs) -> AsyncIterator[str]:
        """Stream completion with retry logic.

        When stream_options={"include_usage": True} is set (the default for
        this adapter), the final chunk carries a ``usage`` field with token
        counts.  We log this as a DEBUG message so callers who need telemetry
        can observe it.  The protocol ``AsyncIterator[str]`` return type does
        not allow returning a ``GenerationLog`` directly.
        """
        try:
            stream = await self._create_completion_with_retry(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                # The final chunk (with include_usage) has usage but empty choices
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage = chunk.usage
                    logger.debug(
                        f"OpenAI stream usage: prompt_tokens={usage.prompt_tokens}, "
                        f"completion_tokens={usage.completion_tokens}"
                    )
        except OpenAIError as e:
            status_code = e.status_code if isinstance(e, APIStatusError) else None
            safe_msg = _safe_error_summary(e)
            if isinstance(e, APITimeoutError | RateLimitError):
                raise APIError(
                    f"OpenAI API streaming error after retries: {safe_msg}",
                    model_name=self.model_name,
                    status_code=status_code,
                ) from e
            raise APIError(
                f"OpenAI streaming failed: {safe_msg}",
                model_name=self.model_name,
                status_code=status_code,
            ) from e

    async def aclose(self) -> None:
        """Close the underlying async client when a caller owns the adapter."""
        if self._client is None:
            return

        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            self._client = None
            return

        aclose = getattr(self._client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await result
        self._client = None
