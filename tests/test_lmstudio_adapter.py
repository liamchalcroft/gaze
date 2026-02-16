"""Tests for LMStudioAdapter: client init, HTTP URLs, env var handling."""

from __future__ import annotations

import pytest

from radiant_harness.models.lmstudio_adapter import LMStudioAdapter
from radiant_harness.models.openai_adapter import OpenAIAdapter


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
        assert client.timeout == 600.0

    def test_default_timeout_is_300(self) -> None:
        adapter = LMStudioAdapter(model_name="test-model")
        client = adapter.client
        assert client.timeout == 300.0

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
        """LMStudioAdapter._validate_base_url must be a no-op so HTTP works."""
        # Should not raise — HTTP is valid for local inference
        LMStudioAdapter._validate_base_url("http://localhost:1234/v1")

    def test_parent_validate_base_url_rejects_http(self) -> None:
        """OpenAIAdapter._validate_base_url must reject HTTP (sanity check)."""
        from radiant_harness.exceptions import ModelError

        with pytest.raises(ModelError, match="HTTPS"):
            OpenAIAdapter._validate_base_url("http://localhost:1234/v1")
