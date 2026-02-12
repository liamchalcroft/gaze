"""Tests for OpenAI adapter: client initialization, base_url routing, and response parsing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from radiant_harness.exceptions import ModelError
from radiant_harness.models.openai_adapter import OpenAIAdapter

# ---------------------------------------------------------------------------
# Client initialization and base_url routing
# ---------------------------------------------------------------------------


class TestClientBaseUrl:
    """Verify that the correct base_url is used depending on env vars."""

    def test_openai_key_uses_default_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        adapter = OpenAIAdapter(model_name="gpt-4o")
        client = adapter.client
        # Default OpenAI base URL
        assert "api.openai.com" in str(client.base_url)

    def test_openrouter_key_auto_sets_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-456")

        adapter = OpenAIAdapter(model_name="openai/gpt-4o")
        client = adapter.client
        assert "openrouter.ai" in str(client.base_url)

    def test_both_keys_prefers_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-456")

        adapter = OpenAIAdapter(model_name="gpt-4o")
        client = adapter.client
        # When both keys are present, OPENAI_API_KEY wins → default base_url
        assert "api.openai.com" in str(client.base_url)

    def test_explicit_base_url_overrides_auto_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-456")

        custom_url = "https://my-proxy.example.com/v1"
        adapter = OpenAIAdapter(model_name="gpt-4o", base_url=custom_url)
        client = adapter.client
        assert "my-proxy.example.com" in str(client.base_url)

    def test_no_api_key_raises_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        adapter = OpenAIAdapter(model_name="gpt-4o")
        with pytest.raises(ModelError, match="No API key found"):
            _ = adapter.client

    def test_client_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        adapter = OpenAIAdapter(model_name="gpt-4o")
        client1 = adapter.client
        client2 = adapter.client
        assert client1 is client2


# ---------------------------------------------------------------------------
# generate_chat response parsing
# ---------------------------------------------------------------------------


def _make_completion(
    content: str = "hello",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    """Build a mock ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tool_call(tc_id: str, name: str, arguments: str) -> SimpleNamespace:
    """Build a mock tool call object."""
    return SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=arguments))


class TestGenerateChat:
    """Test response parsing from generate_chat."""

    @pytest.mark.asyncio
    async def test_basic_text_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        mock_completion = _make_completion(content="The scan shows no abnormalities.")
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        content, tool_calls, gen_log = await adapter.generate_chat(
            messages=[{"role": "user", "content": "Analyze this scan."}],
            max_tokens=100,
            temperature=0.0,
        )

        assert content == "The scan shows no abnormalities."
        assert tool_calls is None
        assert gen_log.prompt_tokens == 10
        assert gen_log.completion_tokens == 5
        assert gen_log.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_calls_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        tc = _make_tool_call("call_1", "zoom", '{"x": 100, "y": 200, "level": 2}')
        mock_completion = _make_completion(content="", tool_calls=[tc], finish_reason="tool_calls")
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        content, tool_calls, gen_log = await adapter.generate_chat(
            messages=[{"role": "user", "content": "Zoom in."}],
            max_tokens=100,
            temperature=0.0,
            tools=[{"type": "function", "function": {"name": "zoom"}}],
        )

        assert content == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[0]["name"] == "zoom"
        assert tool_calls[0]["arguments"] == '{"x": 100, "y": 200, "level": 2}'
        assert gen_log.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_no_choices_raises_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        empty_completion = SimpleNamespace(choices=[], usage=None)
        adapter._create_completion_with_retry = AsyncMock(return_value=empty_completion)

        with pytest.raises(ModelError, match="no choices"):
            await adapter.generate_chat(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0.0,
            )

    @pytest.mark.asyncio
    async def test_none_content_becomes_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        mock_completion = _make_completion(content=None)
        # Override content to None (SimpleNamespace allows it)
        mock_completion.choices[0].message.content = None
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        content, _, _ = await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.0,
        )
        assert content == ""

    @pytest.mark.asyncio
    async def test_none_usage_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        mock_completion = _make_completion()
        mock_completion.usage = None
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        _, _, gen_log = await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.0,
        )
        assert gen_log.prompt_tokens == 0
        assert gen_log.completion_tokens == 0


# ---------------------------------------------------------------------------
# generate_chat kwarg construction
# ---------------------------------------------------------------------------


class TestGenerateChatKwargs:
    """Verify that optional kwargs are only passed when set."""

    @pytest.mark.asyncio
    async def test_tools_not_passed_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        mock_completion = _make_completion()
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.0,
            tools=None,
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_response_format_passed_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o")

        mock_completion = _make_completion()
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        fmt = {"type": "json_object"}
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.0,
            response_format=fmt,
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert call_kwargs["response_format"] == fmt

    @pytest.mark.asyncio
    async def test_reasoning_extra_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(
            model_name="gpt-4o", reasoning_enabled=True, reasoning_effort="medium"
        )

        mock_completion = _make_completion()
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.0,
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert call_kwargs["extra_body"]["reasoning"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_no_extra_body_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(model_name="gpt-4o", reasoning_enabled=False, enable_caching=False)

        mock_completion = _make_completion()
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.0,
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert "extra_body" not in call_kwargs
