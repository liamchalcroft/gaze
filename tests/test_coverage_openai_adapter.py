"""Coverage tests for OpenAIAdapter uncovered paths.

Targets: _stream_completion (296-314), aclose (322-338).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from openai import APITimeoutError
from openai import OpenAIError
from openai import RateLimitError

from radiant_harness.exceptions import APIError
from radiant_harness.models.openai_adapter import OpenAIAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    if content is not None:
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice], usage=usage)
    return SimpleNamespace(choices=[], usage=usage)


class _AsyncChunkIterator:
    """Async iterator that yields pre-built chunks."""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


# ---------------------------------------------------------------------------
# _stream_completion (lines 296-314)
# ---------------------------------------------------------------------------


class TestStreamCompletion:
    @pytest.mark.asyncio
    async def test_yields_content_from_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        chunks = [
            _make_chunk("Hello"),
            _make_chunk(" world"),
            _make_chunk(None, usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)),
        ]
        adapter._create_completion_with_retry = AsyncMock(return_value=_AsyncChunkIterator(chunks))

        result = await adapter.generate_chat(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.0,
            stream=True,
        )

        collected = [token async for token in result]
        assert collected == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_timeout_raises_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        adapter._create_completion_with_retry = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )

        stream = await adapter.generate_chat(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.0,
            stream=True,
        )

        with pytest.raises(APIError, match="streaming error after retries"):
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_stream_rate_limit_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        adapter._create_completion_with_retry = AsyncMock(
            side_effect=RateLimitError(
                message="rate limited",
                response=mock_resp,
                body=None,
            )
        )

        stream = await adapter.generate_chat(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.0,
            stream=True,
        )

        with pytest.raises(APIError, match="streaming error after retries"):
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_stream_generic_openai_error_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        adapter._create_completion_with_retry = AsyncMock(
            side_effect=OpenAIError("connection failed")
        )

        stream = await adapter.generate_chat(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.0,
            stream=True,
        )

        with pytest.raises(APIError, match="streaming failed"):
            async for _ in stream:
                pass


# ---------------------------------------------------------------------------
# aclose (lines 322-338)
# ---------------------------------------------------------------------------


class TestAclose:
    @pytest.mark.asyncio
    async def test_no_client_returns_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        assert adapter._client is None
        await adapter.aclose()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_client_with_sync_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        mock_client = MagicMock()
        mock_client.close = MagicMock(return_value=None)  # sync close
        adapter._client = mock_client

        await adapter.aclose()
        mock_client.close.assert_called_once()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_client_with_async_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        mock_client = MagicMock()
        mock_client.close = AsyncMock(return_value=None)
        adapter._client = mock_client

        await adapter.aclose()
        mock_client.close.assert_awaited_once()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_client_with_aclose_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        mock_client = MagicMock(spec=[])  # no close attribute
        mock_client.aclose = AsyncMock(return_value=None)
        adapter._client = mock_client

        await adapter.aclose()
        mock_client.aclose.assert_awaited_once()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_client_with_no_close_methods(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")
        mock_client = MagicMock(spec=[])  # no close or aclose
        adapter._client = mock_client

        await adapter.aclose()
        assert adapter._client is None
