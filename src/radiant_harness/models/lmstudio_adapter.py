# pyright: basic
"""LM Studio adapter for local inference via OpenAI-compatible API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from beartype import beartype
from loguru import logger
from openai import AsyncOpenAI

from radiant_harness.models.adapter_protocol import GenerationLog
from radiant_harness.models.openai_adapter import OpenAIAdapter

_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_DEFAULT_API_KEY = "lm-studio"
_DEFAULT_TIMEOUT = 300.0


class LMStudioAdapter(OpenAIAdapter):
    """Adapter for LM Studio's OpenAI-compatible local inference server.

    Subclasses :class:`OpenAIAdapter` with these differences:

    1. HTTP base URLs are allowed (no HTTPS requirement).
    2. No real API key is required (LM Studio doesn't authenticate by default).
    3. Longer default timeout (300s) for local inference on consumer hardware.
    4. Tool messages use text-only content (no multipart image payloads).
    5. ``response_format`` is stripped — many local models (especially those
       with built-in thinking/reasoning) mishandle the ``json_schema``
       response format, putting output into ``reasoning_content`` instead
       of ``content``.  The prompts already instruct JSON output.

    All generation logic (tool calling, vision, streaming, retries) is
    inherited unchanged from :class:`OpenAIAdapter`.
    """

    supports_multipart_tool_content: bool = False

    @staticmethod
    def _validate_base_url(url: str) -> None:
        """Allow HTTP and HTTPS — LM Studio uses HTTP by default.

        Rejects other schemes (file://, ftp://, etc.) to prevent SSRF or
        unintended local file access through the OpenAI SDK.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            from radiant_harness.exceptions import ModelError

            raise ModelError(
                f"LM Studio base_url must use http:// or https://, got {parsed.scheme!r}",
                model_name=None,
            )

    @beartype
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        # Resolve base_url: explicit arg > env var > default
        resolved_url = base_url or os.getenv("LMSTUDIO_BASE_URL", _DEFAULT_BASE_URL)
        resolved_key = api_key or os.getenv("LMSTUDIO_API_KEY", _DEFAULT_API_KEY)

        # Delegate to parent — our _validate_base_url override allows HTTP.
        super().__init__(
            model_name=model_name,
            reasoning_enabled=False,
            reasoning_effort="high",
            enable_caching=False,
            base_url=resolved_url,
        )
        self._api_key = resolved_key
        self._timeout = timeout

    @property
    def client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client configured for LM Studio.

        No HTTPS validation, no cloud API key requirement.
        """
        if self._client is None:
            logger.info(f"Connecting to LM Studio at {self._base_url}")
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=0,
            )
        return self._client

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
        """Generate chat completion, stripping ``response_format``.

        Local models with built-in thinking (Qwen3.5, etc.) misroute
        structured output into ``reasoning_content`` when
        ``response_format`` is set, leaving ``content`` empty.  Dropping
        the parameter lets the prompt handle JSON formatting instead.
        """
        if response_format is not None:
            logger.debug(
                "LMStudioAdapter: stripping response_format (local model compatibility)"
            )
        return await super().generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=None,
            stream=stream,
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """List models currently loaded in LM Studio.

        Convenience method for verifying the connection and seeing
        which models are available before starting inference.

        Returns:
            List of model info dicts with at least an ``id`` key.
        """
        response = await self.client.models.list()
        return [{"id": m.id, "object": m.object} for m in response.data]
