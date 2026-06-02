# pyright: basic
"""LM Studio adapter for local inference via OpenAI-compatible API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from beartype import beartype
from loguru import logger
from openai import AsyncOpenAI

from gaze.exceptions import ModelError
from gaze.models.adapter_protocol import GenerationLog
from gaze.models.openai_adapter import OpenAIAdapter

_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_DEFAULT_API_KEY = "lm-studio"
_DEFAULT_TIMEOUT = 300.0
_CONNECT_TIMEOUT = 10.0
_WRITE_TIMEOUT = 10.0
_POOL_TIMEOUT = 30.0


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

    6. No retries on completion — local timeouts usually indicate OOM or
       model overload, not transient network issues.
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
            from gaze.exceptions import ModelError

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
                timeout=httpx.Timeout(
                    connect=_CONNECT_TIMEOUT,
                    read=self._timeout,
                    write=_WRITE_TIMEOUT,
                    pool=_POOL_TIMEOUT,
                ),
                max_retries=0,
            )
        return self._client

    @beartype
    async def _create_completion_with_retry(self, **kwargs):  # type: ignore[override]
        """Local models rarely recover from transient errors — no retries.

        Overrides the parent's 5-retry strategy.  For LM Studio, timeouts
        typically mean the model is too large or OOM — retrying just wastes
        minutes.

        Detects context window overflow (common with local models that have
        small context windows) and raises a clear error.
        """
        from openai import BadRequestError

        try:
            return await self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            msg = str(exc).lower()
            if (
                "context size" in msg
                or "context length" in msg
                or "maximum context" in msg
                or "n_ctx" in msg
                or "n_keep" in msg
            ):
                raise ModelError(
                    f"Context window exceeded for model {self.model_name!r}. "
                    f"Reduce input length or use a model with a larger context window.",
                    model_name=self.model_name,
                ) from exc
            raise

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
        """Generate chat completion, stripping ``response_format``.

        Local models with built-in thinking (Qwen3.5, etc.) misroute
        structured output into ``reasoning_content`` when
        ``response_format`` is set, leaving ``content`` empty.  Dropping
        the parameter lets the prompt handle JSON formatting instead.
        """
        if response_format is not None:
            logger.debug("LMStudioAdapter: stripping response_format (local model compatibility)")
        return await super().generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=None,
            stream=stream,
            seed=seed,
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


@beartype
async def list_lmstudio_model_ids(
    base_url: str | None = None,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Return model IDs from an OpenAI-compatible LM Studio endpoint."""
    resolved_url = base_url or os.getenv("LMSTUDIO_BASE_URL", _DEFAULT_BASE_URL)
    LMStudioAdapter._validate_base_url(resolved_url)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            f"{resolved_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key or _DEFAULT_API_KEY}"},
        )
        response.raise_for_status()

    payload = response.json()
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise ModelError(
            "LM Studio /models response did not contain a 'data' list",
            model_name=None,
        )

    model_ids: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return model_ids


@beartype
async def require_lmstudio_model(
    model_name: str,
    base_url: str | None = None,
    *,
    timeout: float = 10.0,
    health_check: bool = True,
) -> list[str]:
    """Fail fast when the requested model is not available in LM Studio.

    When *health_check* is True (default), a 1-token completion is attempted
    after verifying the model ID is listed.  LM Studio lists all available
    models but only loads them on demand — this catches OOM failures that
    ``/v1/models`` alone cannot detect.
    """
    model_ids = await list_lmstudio_model_ids(base_url=base_url, timeout=timeout)
    if model_name not in model_ids:
        available = ", ".join(model_ids) if model_ids else "<none>"
        raise ModelError(
            f"Model {model_name!r} is not loaded in LM Studio. Available models: {available}",
            model_name=model_name,
        )

    if health_check:
        resolved_url = base_url or os.getenv("LMSTUDIO_BASE_URL", _DEFAULT_BASE_URL)
        try:
            async with httpx.AsyncClient(timeout=max(timeout, 60.0)) as client:
                resp = await client.post(
                    f"{resolved_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {_DEFAULT_API_KEY}"},
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )
                if resp.status_code == 400:
                    body = resp.json()
                    err_msg = body.get("error", {}).get("message", resp.text)
                    if "insufficient" in err_msg.lower() or "failed to load" in err_msg.lower():
                        raise ModelError(
                            f"Model {model_name!r} is listed but cannot be loaded "
                            f"(likely insufficient memory): {err_msg}",
                            model_name=model_name,
                        )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelError(
                f"Health check failed for {model_name!r}: HTTP {exc.response.status_code}",
                model_name=model_name,
            ) from exc
        except httpx.TimeoutException:
            logger.warning(
                f"Health check timed out for {model_name!r} (may be loading for the first time)"
            )

    return model_ids
