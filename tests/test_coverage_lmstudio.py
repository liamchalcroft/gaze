"""Coverage tests for LMStudioAdapter uncovered paths.

Targets: _create_completion_with_retry context overflow (112-130),
list_models (171-172), list_lmstudio_model_ids edge cases (196, 204),
require_lmstudio_model health check (234-267).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx
import pytest

from radiant_harness.exceptions import ModelError
from radiant_harness.models.lmstudio_adapter import LMStudioAdapter
from radiant_harness.models.lmstudio_adapter import list_lmstudio_model_ids
from radiant_harness.models.lmstudio_adapter import require_lmstudio_model

# ---------------------------------------------------------------------------
# _create_completion_with_retry — context overflow detection (lines 112-130)
# ---------------------------------------------------------------------------


class TestContextOverflowDetection:
    @staticmethod
    def _make_bad_request(message: str):
        from openai import BadRequestError

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": {"message": message}}
        mock_resp.headers = {}
        return BadRequestError(message=message, response=mock_resp, body=None)

    @pytest.mark.asyncio
    async def test_context_size_triggers_model_error(self) -> None:
        adapter = LMStudioAdapter(model_name="qwen-7b")
        exc = self._make_bad_request("This model's maximum context size is 4096 tokens")
        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(ModelError, match="Context window exceeded.*'qwen-7b'"):
            await adapter._create_completion_with_retry(model="qwen-7b", messages=[])

    @pytest.mark.asyncio
    async def test_context_length_triggers_model_error(self) -> None:
        adapter = LMStudioAdapter(model_name="test")
        exc = self._make_bad_request("maximum context length exceeded")
        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(ModelError, match="Context window exceeded"):
            await adapter._create_completion_with_retry(model="test", messages=[])

    @pytest.mark.asyncio
    async def test_maximum_context_triggers_model_error(self) -> None:
        adapter = LMStudioAdapter(model_name="test")
        exc = self._make_bad_request("maximum context window is too small")
        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(ModelError, match="Context window exceeded"):
            await adapter._create_completion_with_retry(model="test", messages=[])

    @pytest.mark.asyncio
    async def test_n_ctx_triggers_model_error(self) -> None:
        adapter = LMStudioAdapter(model_name="test")
        exc = self._make_bad_request("n_ctx must be at least 512")
        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(ModelError, match="Context window exceeded"):
            await adapter._create_completion_with_retry(model="test", messages=[])

    @pytest.mark.asyncio
    async def test_n_keep_triggers_model_error(self) -> None:
        adapter = LMStudioAdapter(model_name="test")
        exc = self._make_bad_request("n_keep exceeds n_ctx")
        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(ModelError, match="Context window exceeded"):
            await adapter._create_completion_with_retry(model="test", messages=[])

    @pytest.mark.asyncio
    async def test_unrelated_bad_request_propagates(self) -> None:
        from openai import BadRequestError

        adapter = LMStudioAdapter(model_name="test")
        exc = self._make_bad_request("invalid request body format")
        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(BadRequestError, match="invalid request body"):
            await adapter._create_completion_with_retry(model="test", messages=[])


# ---------------------------------------------------------------------------
# list_models (lines 171-172)
# ---------------------------------------------------------------------------


class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_id_and_object_fields(self) -> None:
        adapter = LMStudioAdapter(model_name="test")
        mock_data = [
            SimpleNamespace(id="qwen3.5-a3b", object="model"),
            SimpleNamespace(id="glm-4.6v", object="model"),
        ]
        adapter._client = MagicMock()
        adapter._client.models.list = AsyncMock(return_value=SimpleNamespace(data=mock_data))

        result = await adapter.list_models()
        assert len(result) == 2
        assert result[0] == {"id": "qwen3.5-a3b", "object": "model"}
        assert result[1] == {"id": "glm-4.6v", "object": "model"}

    @pytest.mark.asyncio
    async def test_empty_models(self) -> None:
        adapter = LMStudioAdapter(model_name="test")
        adapter._client = MagicMock()
        adapter._client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        result = await adapter.list_models()
        assert result == []


# ---------------------------------------------------------------------------
# list_lmstudio_model_ids edge cases (lines 196, 204)
# ---------------------------------------------------------------------------


class _MockHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _MockAsyncClient:
    """Mock httpx.AsyncClient supporting GET (models) and POST (health check)."""

    def __init__(self, get_payload: dict, post_response=None, **_kw) -> None:
        self._get_payload = get_payload
        self._post_response = post_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        pass

    async def get(self, url: str, headers: dict) -> _MockHTTPResponse:
        return _MockHTTPResponse(self._get_payload)

    async def post(self, url: str, headers: dict, json: dict):
        if self._post_response is None:
            return _MockHTTPResponse({"choices": []})
        return self._post_response


class TestListModelIdsEdgeCases:
    @pytest.mark.asyncio
    async def test_non_list_data_raises_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _MockAsyncClient({"data": "not-a-list"}),
        )
        with pytest.raises(ModelError, match="did not contain a 'data' list"):
            await list_lmstudio_model_ids("http://localhost:1234/v1")

    @pytest.mark.asyncio
    async def test_non_dict_items_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _MockAsyncClient(
                {"data": [42, "string", {"id": "real"}, {"id": ""}, {"no_id": True}]}
            ),
        )
        result = await list_lmstudio_model_ids("http://localhost:1234/v1")
        assert result == ["real"]

    @pytest.mark.asyncio
    async def test_missing_data_key_raises_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _MockAsyncClient({"models": []}),
        )
        with pytest.raises(ModelError, match="did not contain a 'data' list"):
            await list_lmstudio_model_ids("http://localhost:1234/v1")


# ---------------------------------------------------------------------------
# require_lmstudio_model health check (lines 234-267)
# ---------------------------------------------------------------------------


class _HealthCheckResponse:
    """Mock POST response for health check scenarios."""

    def __init__(self, status_code: int, body: dict, raise_on_status: bool = False):
        self.status_code = status_code
        self._body = body
        self._raise = raise_on_status
        self.text = str(body)

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self._raise:
            resp = httpx.Response(
                self.status_code,
                request=httpx.Request("POST", "http://localhost"),
            )
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=resp.request, response=resp
            )


class _HealthCheckClient(_MockAsyncClient):
    """Extends _MockAsyncClient with configurable POST behavior."""

    def __init__(self, models: list[str], health_resp=None, post_raises=None, **kw):
        super().__init__(
            get_payload={"data": [{"id": m} for m in models]},
            post_response=health_resp,
        )
        self._post_raises = post_raises

    async def post(self, url: str, headers: dict, json: dict):
        if self._post_raises is not None:
            raise self._post_raises
        return await super().post(url, headers, json)


class TestRequireHealthCheck:
    @pytest.mark.asyncio
    async def test_400_insufficient_memory_raises_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _HealthCheckResponse(400, {"error": {"message": "Insufficient VRAM to load model"}})
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _HealthCheckClient(["big-model"], health_resp=resp),
        )
        with pytest.raises(ModelError, match="cannot be loaded.*insufficient memory"):
            await require_lmstudio_model("big-model", "http://localhost:1234/v1")

    @pytest.mark.asyncio
    async def test_400_failed_to_load_raises_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _HealthCheckResponse(400, {"error": {"message": "Failed to load model weights"}})
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _HealthCheckClient(["bad-model"], health_resp=resp),
        )
        with pytest.raises(ModelError, match="cannot be loaded"):
            await require_lmstudio_model("bad-model", "http://localhost:1234/v1")

    @pytest.mark.asyncio
    async def test_400_unrelated_falls_through_to_raise_for_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _HealthCheckResponse(
            400,
            {"error": {"message": "Invalid model format"}},
            raise_on_status=True,
        )
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _HealthCheckClient(["my-model"], health_resp=resp),
        )
        with pytest.raises(ModelError, match="Health check failed.*HTTP 400"):
            await require_lmstudio_model("my-model", "http://localhost:1234/v1")

    @pytest.mark.asyncio
    async def test_500_http_error_raises_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = _HealthCheckResponse(500, {}, raise_on_status=True)
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _HealthCheckClient(["my-model"], health_resp=resp),
        )
        with pytest.raises(ModelError, match="Health check failed.*HTTP 500"):
            await require_lmstudio_model("my-model", "http://localhost:1234/v1")

    @pytest.mark.asyncio
    async def test_timeout_logs_warning_returns_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _HealthCheckClient(
                ["slow-model"], post_raises=httpx.ReadTimeout("timed out")
            ),
        )
        result = await require_lmstudio_model("slow-model", "http://localhost:1234/v1")
        assert result == ["slow-model"]

    @pytest.mark.asyncio
    async def test_successful_health_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = _HealthCheckResponse(200, {"choices": [{"message": {"content": "h"}}]})
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _HealthCheckClient(["good-model"], health_resp=resp),
        )
        result = await require_lmstudio_model("good-model", "http://localhost:1234/v1")
        assert result == ["good-model"]

    @pytest.mark.asyncio
    async def test_skip_health_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "radiant_harness.models.lmstudio_adapter.httpx.AsyncClient",
            lambda **kw: _MockAsyncClient({"data": [{"id": "fast-model"}]}),
        )
        result = await require_lmstudio_model(
            "fast-model", "http://localhost:1234/v1", health_check=False
        )
        assert result == ["fast-model"]
