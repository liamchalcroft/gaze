from __future__ import annotations

import asyncio
import base64
import copy
import os
import time
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import openai  # For exception classes: RateLimitError, APIError, APIConnectionError
from beartype import beartype
from loguru import logger
from openai import OpenAI  # Import the client class
from PIL import Image as PILImage

from nova_retrieval_vlm.types import APIError

from .base import BaseAdapter
from .base import GenerationLog


class OpenAIAdapter(BaseAdapter):
    """Adapter for calling OpenRouter-compatible models via the OpenAI SDK."""

    @beartype
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
    ) -> None:
        # Prefer OPENAI_API_KEY over OPENROUTER_API_KEY unless an explicit api_key is provided
        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not set")
        super().__init__(key)
        # Configure the OpenAI client to point at OpenRouter or other base_url
        default_headers = {
            "HTTP-Referer": os.getenv("APP_URL", "https://nova-retrieval-vlm.app"),
            "X-Title": os.getenv("APP_NAME", "NOVA Retrieval VLM"),
        }

        # Note: Reasoning is handled per OpenRouter's API format in the request body
        # Headers are kept for compatibility but main reasoning control is in kwargs

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching
        logger.info(
            f"Initialized OpenAIAdapter with model: {model_name}, reasoning: {reasoning_enabled}, effort: {reasoning_effort}, caching: {enable_caching}"
        )

    async def _call_with_retry(
        self, call_fn: Any, method_name: str = "API call"
    ) -> Any:
        """Execute API call with exponential backoff retry logic.

        Args:
            call_fn: Callable that executes the API call
            method_name: Name for logging purposes

        Returns:
            API response object

        Raises:
            RuntimeError: After all retry attempts are exhausted
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await asyncio.to_thread(call_fn)
            except openai.RateLimitError as e:
                last_error = e
                logger.warning(
                    f"Rate limit on {method_name} attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt + 1)
            except (TimeoutError, openai.APIError, openai.APIConnectionError) as e:
                last_error = e
                logger.warning(
                    f"API error on {method_name} attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt + 1)

        raise RuntimeError(
            f"{method_name} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _ensure_strict_json_schema(self, response_format: dict[str, Any]) -> dict[str, Any]:
        """Ensure strict mode is enabled for JSON schema structured outputs."""
        response_format_copy = copy.deepcopy(response_format)
        if (
            response_format_copy.get("type") == "json_schema"
            and "json_schema" in response_format_copy
        ):
            response_format_copy["json_schema"]["strict"] = True
        return response_format_copy

    @beartype
    def _estimate_tokens(self, response: Any) -> int:
        """Extract token count from API response, or return 0 if unavailable."""
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.total_tokens
        logger.debug("API response missing usage data; token count unavailable")
        return 0

    @beartype
    def _create_generation_log(self, tokens: int) -> GenerationLog:
        """Create a generation log with token count (no cost estimation)."""
        return GenerationLog(
            model_name=self.model_name,
            tokens=tokens,
            cost=0.0,  # Cost estimation removed - use OpenRouter dashboard instead
            timestamp=time.time(),
        )

    @beartype
    def _add_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add cache control to messages for OpenRouter prompt caching.

        Adds cache_control breakpoints to system messages and consistent content
        to enable prompt caching and reduce costs for repeated prompts.
        """
        if not self.enable_caching:
            return messages

        cached_messages = []
        for message in messages:
            cached_message = message.copy()

            # Add cache control to system messages (consistent across requests)
            if message.get("role") == "system":
                cached_message["cache_control"] = {"type": "cache_breakpoint"}

            # For multi-content messages, add cache control to text parts
            elif isinstance(message.get("content"), list):
                cached_content = []
                for content_part in message["content"]:
                    cached_part = content_part.copy()
                    # Add cache control to consistent text content
                    if content_part.get(
                        "type"
                    ) == "text" and "Analyze the provided image" in content_part.get("text", ""):
                        cached_part["cache_control"] = {"type": "cache_breakpoint"}
                    cached_content.append(cached_part)
                cached_message["content"] = cached_content

            cached_messages.append(cached_message)

        return cached_messages

    @beartype
    async def generate(
        self,
        image_path: Path,
        passages: Sequence[str],  # noqa: ARG002 - Reserved for future RAG integration
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, object] | None = None,
    ) -> tuple[str, GenerationLog]:
        """Generate response for an image with optional retrieval passages.

        Args:
            image_path: Path to the image file
            passages: Retrieved passages for RAG (reserved for future use)
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Optional JSON schema for structured output

        Returns:
            Tuple of (response_text, generation_log)

        Raises:
            APIError: If image loading or API call fails
            ValueError: If system_prompt is empty
        """
        # Validate required prompt
        if system_prompt is None:
            raise ValueError("system_prompt is required for generate()")

        # Load and optimize image with proper resource management
        try:
            with PILImage.open(image_path) as pil_img:
                # Only convert if necessary to avoid extra memory allocation
                rgb_img = pil_img if pil_img.mode == "RGB" else pil_img.convert("RGB")
                with BytesIO() as buf:
                    rgb_img.save(buf, format="JPEG", quality=85, optimize=True)
                    img_bytes = buf.getvalue()
                    img_b64 = base64.b64encode(img_bytes).decode()
                    compressed_size = len(img_bytes)
            logger.debug(f"Image size after compression: {compressed_size} bytes")
        except OSError as e:
            raise APIError(f"Failed to read image at {image_path}: {e}") from e

        # User message content with image
        user_content: list[dict[str, object]] = [
            {"type": "text", "text": "Analyze the provided image based on the instructions."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]

        messages = self._add_cache_control([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "timeout": self.timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format is not None:
                kwargs["response_format"] = self._ensure_strict_json_schema(response_format)
            if self.reasoning_enabled:
                kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
            return self.client.chat.completions.create(**kwargs)

        resp = await self._call_with_retry(_call, "generate")
        text = resp.choices[0].message.content
        if text is None:
            raise APIError(f"Model {self.model_name} returned no content")
        return text, self._create_generation_log(self._estimate_tokens(resp))

    @beartype
    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        """Generate a response from a multi-turn conversation.

        Args:
            messages: List of conversation messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional list of tool definitions for function calling
            response_format: Optional structured output format

        Returns:
            Tuple of (response_text, tool_calls, generation_log)
        """
        cached_messages = self._add_cache_control(messages)

        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": cached_messages,
                "timeout": self.timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if response_format is not None:
                kwargs["response_format"] = self._ensure_strict_json_schema(response_format)
            return self.client.chat.completions.create(**kwargs)

        resp = await self._call_with_retry(_call, "generate_chat")
        message = resp.choices[0].message
        text = message.content or ""

        # Extract tool calls if present
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

        return text, tool_calls, self._create_generation_log(self._estimate_tokens(resp))

    @beartype
    async def generate_text(
        self,
        prompt_text: str,
        system_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, GenerationLog]:
        """Generate text response without image input.

        Args:
            prompt_text: User prompt text
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Optional JSON schema for structured output

        Returns:
            Tuple of (response_text, generation_log)
        """
        messages = self._add_cache_control([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ])

        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "timeout": self.timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format is not None:
                kwargs["response_format"] = self._ensure_strict_json_schema(response_format)
            if self.reasoning_enabled:
                kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
            return self.client.chat.completions.create(**kwargs)

        resp = await self._call_with_retry(_call, "generate_text")
        text = resp.choices[0].message.content
        if text is None:
            raise APIError(f"Model {self.model_name} returned no content")
        return text, self._create_generation_log(self._estimate_tokens(resp))
