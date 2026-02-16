# pyright: basic
"""LM Studio adapter for local inference via OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any

from beartype import beartype
from loguru import logger
from openai import AsyncOpenAI

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

    All generation logic (tool calling, vision, streaming, retries) is
    inherited unchanged from :class:`OpenAIAdapter`.
    """

    supports_multipart_tool_content: bool = False

    @staticmethod
    def _validate_base_url(_url: str) -> None:
        """Allow any URL scheme — LM Studio uses HTTP by default."""

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

    async def list_models(self) -> list[dict[str, Any]]:
        """List models currently loaded in LM Studio.

        Convenience method for verifying the connection and seeing
        which models are available before starting inference.

        Returns:
            List of model info dicts with at least an ``id`` key.
        """
        response = await self.client.models.list()
        return [{"id": m.id, "object": m.object} for m in response.data]
