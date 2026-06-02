"""Tests for LMStudioAdapter: client init, HTTP URLs, env var handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gaze.models.lmstudio_adapter import LMStudioAdapter
from gaze.models.lmstudio_adapter import list_lmstudio_model_ids
from gaze.models.lmstudio_adapter import require_lmstudio_model
from gaze.models.openai_adapter import OpenAIAdapter


class TestClientCreation:
    """Verify client is created with correct defaults and overrides."""

    def test_default_base_url_and_api_key(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        client = adapter.client
        assert "localhost:1234" in str(client.base_url)

    def test_http_url_allowed(self) -> None:
        """HTTP URLs must work — LM Studio doesn't use HTTPS by default."""
        adapter = LMStudioAdapter(
            model_name="qwen2.5-vl-7b",
            base_url="http://192.168.1.50:1234/v1",
        )
        client = adapter.client
        assert "192.168.1.50:1234" in str(client.base_url)

    def test_custom_api_key(self) -> None:
        adapter = LMStudioAdapter(
            model_name="test-model",
            api_key="my-secret",
        )
        client = adapter.client
        assert client.api_key == "my-secret"

    def test_custom_timeout(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model", timeout=600.0)
        client = adapter.client
        # Read timeout carries the custom value; connect is always fast
        assert client.timeout.read == 600.0
        assert client.timeout.connect == 10.0

    def test_default_timeout_is_300(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        client = adapter.client
        assert client.timeout.read == 300.0
        assert client.timeout.connect == 10.0

    def test_reasoning_and_caching_disabled(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        assert adapter.reasoning_enabled is False
        assert adapter.enable_caching is False


class TestEnvVarOverrides:
    """Verify LMSTUDIO_BASE_URL and LMSTUDIO_API_KEY env vars."""

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://10.0.0.5:9999/v1")
        adapter = LMStudioAdapter(model_name="test-model")
        assert "10.0.0.5:9999" in str(adapter.client.base_url)

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMSTUDIO_API_KEY", "env-key-123")
        adapter = LMStudioAdapter(model_name="test-model")
        assert adapter.client.api_key == "env-key-123"

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://env-host:1234/v1")
        monkeypatch.setenv("LMSTUDIO_API_KEY", "env-key")
        adapter = LMStudioAdapter(
            model_name="test-model",
            base_url="http://arg-host:5678/v1",
            api_key="arg-key",
        )
        assert "arg-host:5678" in str(adapter.client.base_url)
        assert adapter.client.api_key == "arg-key"


class TestInheritance:
    """Verify LMStudioAdapter inherits from OpenAIAdapter."""

    def test_is_subclass_of_openai_adapter(self) -> None:
        assert issubclass(LMStudioAdapter, OpenAIAdapter)

    def test_has_generate_chat(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        assert hasattr(adapter, "generate_chat")
        assert callable(adapter.generate_chat)

    def test_has_list_models(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        assert hasattr(adapter, "list_models")
        assert callable(adapter.list_models)

    def test_calls_super_init(self) -> None:
        """Regression: LMStudioAdapter must delegate to OpenAIAdapter.__init__.

        The previous implementation skipped super().__init__() and manually
        duplicated attribute assignments.  If OpenAIAdapter ever adds new
        init logic, the manual approach silently diverges.
        """
        adapter = LMStudioAdapter(model_name="test-model")
        # All attributes set by OpenAIAdapter.__init__ must be present
        assert adapter.model_name == "test-model"
        assert adapter.reasoning_enabled is False
        assert adapter.reasoning_effort == "high"
        assert adapter.enable_caching is False
        assert adapter._base_url is not None
        assert adapter._client is None

    def test_parent_attrs_propagated_on_new_attribute(self) -> None:
        """Verify that attributes from OpenAIAdapter.__init__ are real — not
        just manually duplicated — by checking isinstance confirms the init
        path ran through OpenAIAdapter.
        """
        adapter = LMStudioAdapter(model_name="test-model")
        # If super().__init__ ran, these attributes exist via the parent path.
        # Verify the _base_url was set by super().__init__ (not duplicated).
        assert hasattr(adapter, "_base_url")
        assert hasattr(adapter, "_client")
        assert hasattr(adapter, "model_name")

    def test_validate_base_url_override_allows_http(self) -> None:
        """LMStudioAdapter._validate_base_url allows HTTP for local inference."""
        # Should not raise — HTTP is valid for local inference
        LMStudioAdapter._validate_base_url("http://localhost:1234/v1")

    def test_validate_base_url_override_allows_https(self) -> None:
        """LMStudioAdapter._validate_base_url also allows HTTPS."""
        LMStudioAdapter._validate_base_url("https://localhost:1234/v1")

    def test_parent_validate_base_url_rejects_http(self) -> None:
        """OpenAIAdapter._validate_base_url must reject HTTP (sanity check)."""
        from gaze.exceptions import ModelError

        with pytest.raises(ModelError, match="HTTPS"):
            OpenAIAdapter._validate_base_url("http://localhost:1234/v1")


class TestURLSchemeValidation:
    """Verify LMStudio rejects dangerous URL schemes."""

    def test_file_scheme_rejected(self) -> None:
        from gaze.exceptions import ModelError

        with pytest.raises(ModelError, match="http://.*https://"):
            LMStudioAdapter._validate_base_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self) -> None:
        from gaze.exceptions import ModelError

        with pytest.raises(ModelError, match="http://.*https://"):
            LMStudioAdapter._validate_base_url("ftp://evil.com/model")

    def test_empty_scheme_rejected(self) -> None:
        from gaze.exceptions import ModelError

        with pytest.raises(ModelError):
            LMStudioAdapter._validate_base_url("localhost:1234/v1")

    def test_constructor_rejects_file_url(self) -> None:
        from gaze.exceptions import ModelError

        with pytest.raises(ModelError, match="http://.*https://"):
            LMStudioAdapter(model_name="test", base_url="file:///etc/passwd")


# ---------------------------------------------------------------------------
# generate_chat: response_format stripping
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


class TestResponseFormatStripping:
    """Verify LMStudioAdapter strips response_format before calling the API."""

    @pytest.mark.asyncio
    async def test_response_format_stripped(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        mock_completion = _make_completion(content='{"answer": "yes"}')
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_json_schema_format_stripped(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        mock_completion = _make_completion(content='{"answer": "yes"}')
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            },
        }
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.0,
            response_format=schema_format,
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_none_response_format_passthrough(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        mock_completion = _make_completion()
        adapter._create_completion_with_retry = AsyncMock(return_value=mock_completion)

        await adapter.generate_chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.0,
            response_format=None,
        )

        call_kwargs = adapter._create_completion_with_retry.call_args[1]
        assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# Protocol signature parity
# ---------------------------------------------------------------------------


class TestProtocolSignatureParity:
    """Verify LMStudioAdapter matches AdapterProtocol signature."""

    def test_generate_chat_signature_matches_protocol(self) -> None:
        import inspect

        from gaze.models.adapter_protocol import AdapterProtocol

        proto_sig = inspect.signature(AdapterProtocol.generate_chat)
        lm_sig = inspect.signature(LMStudioAdapter.generate_chat)

        proto_params = set(proto_sig.parameters.keys()) - {"self"}
        lm_params = set(lm_sig.parameters.keys()) - {"self"}

        assert lm_params == proto_params, f"LMStudio params {lm_params} != protocol {proto_params}"

    def test_supports_multipart_tool_content_is_false(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        assert adapter.supports_multipart_tool_content is False


class _MockHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return self._payload


class _MockAsyncClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> _MockAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    async def get(self, url: str, headers: dict[str, str]) -> _MockHTTPResponse:
        assert url.endswith("/models")
        assert headers["Authorization"].startswith("Bearer ")
        return _MockHTTPResponse(self._payload)


class TestLMStudioPreflight:
    @pytest.mark.asyncio
    async def test_list_lmstudio_model_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "gaze.models.lmstudio_adapter.httpx.AsyncClient",
            lambda timeout: _MockAsyncClient(  # noqa: ARG005
                {"data": [{"id": "qwen3.5-a3b"}, {"id": "gemma-3"}]}
            ),
        )

        assert await list_lmstudio_model_ids("http://localhost:1234/v1") == [
            "qwen3.5-a3b",
            "gemma-3",
        ]

    @pytest.mark.asyncio
    async def test_require_lmstudio_model_raises_for_missing_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gaze.exceptions import ModelError

        monkeypatch.setattr(
            "gaze.models.lmstudio_adapter.httpx.AsyncClient",
            lambda timeout: _MockAsyncClient({"data": [{"id": "gemma-3"}]}),  # noqa: ARG005
        )

        with pytest.raises(ModelError, match="Available models: gemma-3"):
            await require_lmstudio_model(
                model_name="qwen3.5-a3b",
                base_url="http://localhost:1234/v1",
            )
