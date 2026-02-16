"""Tests for performance fixes (patch set 1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from radiant_harness.tools.image_manager import ImageManager
from radiant_harness.tools.registry import encode_image


class TestOpenAIAdapterTimeout:
    def test_client_uses_structured_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from radiant_harness.models.openai_adapter import OpenAIAdapter

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        adapter = OpenAIAdapter(model_name="gpt-4o")
        timeout = adapter.client.timeout

        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 10.0
        assert timeout.read == 90.0
        assert timeout.write == 10.0
        assert timeout.pool == 30.0


class TestPubMedConsolidatedSleep:
    @pytest.mark.asyncio
    async def test_single_sleep_before_gather(self) -> None:
        from radiant_harness.retrieval.web_search import PubMedSearchEngine

        engine = PubMedSearchEngine()
        sleep_calls: list[float] = []

        async def _mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        async def _fake_summary(_pmids: list[str]) -> dict:
            return {"result": {}}

        async def _fake_abstracts(_pmids: list[str]) -> dict[str, str]:
            return {}

        engine._fetch_summary = _fake_summary  # type: ignore[assignment]
        engine._fetch_abstracts = _fake_abstracts  # type: ignore[assignment]

        with patch("radiant_harness.retrieval.web_search.asyncio.sleep", _mock_sleep):
            await engine._fetch_article_details(["12345"])

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == engine._rate_limit_delay


class TestSharedDownloadSession:
    @pytest.mark.asyncio
    async def test_download_session_reused(self, tmp_path: Path) -> None:
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        session1 = await mgr._get_download_session()
        session2 = await mgr._get_download_session()
        assert session1 is session2
        await mgr.close()

    @pytest.mark.asyncio
    async def test_download_session_closed_on_cleanup(self, tmp_path: Path) -> None:
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        session = await mgr._get_download_session()
        assert not session.closed
        await mgr.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_close_without_session_does_not_raise(self, tmp_path: Path) -> None:
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        await mgr.close()


def _create_test_image(
    tmp_path: Path, name: str = "test.png", size: tuple[int, int] = (50, 50)
) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


class TestOriginalEncodingCache:
    def test_encoding_can_be_set_and_read(self, tmp_path: Path) -> None:
        path = _create_test_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not None

        encoded = encode_image(mgr.current_image)
        mgr.original_encoding = encoded
        assert mgr.original_encoding is encoded

    def test_close_clears_encoding(self, tmp_path: Path) -> None:
        path = _create_test_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not None
        mgr.original_encoding = encode_image(mgr.current_image)

        mgr.close()
        assert mgr.original_encoding is None

    def test_set_image_clears_encoding(self, tmp_path: Path) -> None:
        img1 = _create_test_image(tmp_path, "a.png", (30, 30))
        img2 = _create_test_image(tmp_path, "b.png", (60, 60))
        mgr = ImageManager()
        mgr.set_image(img1)
        assert mgr.current_image is not None
        mgr.original_encoding = encode_image(mgr.current_image)

        mgr.set_image(img2)
        assert mgr.original_encoding is None

    def test_reset_preserves_encoding(self, tmp_path: Path) -> None:
        path = _create_test_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not None
        cached = encode_image(mgr.current_image)
        mgr.original_encoding = cached

        mgr.transform_image(lambda img: img.resize((10, 10)))
        mgr.reset_to_original()
        assert mgr.original_encoding is cached

    def test_set_preloaded_image_clears_encoding(self, tmp_path: Path) -> None:
        path1 = _create_test_image(tmp_path, "a.png")
        mgr = ImageManager()
        mgr.set_image(path1)
        assert mgr.current_image is not None
        mgr.original_encoding = encode_image(mgr.current_image)

        new_img = Image.new("RGB", (80, 80), color=(0, 0, 0))
        path2 = _create_test_image(tmp_path, "b.png", (80, 80))
        mgr.set_preloaded_image(new_img, path2)
        assert mgr.original_encoding is None


class TestResetUsesCache:
    @pytest.mark.asyncio
    async def test_reset_skips_encode_when_cached(self, tmp_path: Path) -> None:
        from radiant_harness.tools.registry import ToolRegistry
        from radiant_harness.tools.visual import _execute_reset

        path = _create_test_image(tmp_path, size=(100, 100))
        registry = ToolRegistry(image_path=path, tools=[])

        mgr = registry.get_image_manager()
        await mgr.ensure_loaded()
        assert mgr.current_image is not None

        cached = encode_image(mgr.current_image)
        mgr.original_encoding = cached
        mgr.transform_image(lambda img: img.resize((10, 10)))

        encode_call_count = 0
        _real_encode = encode_image

        def _counting_encode(*args, **kwargs):
            nonlocal encode_call_count
            encode_call_count += 1
            return _real_encode(*args, **kwargs)

        with patch("radiant_harness.tools.visual.encode_image", _counting_encode):
            result = await _execute_reset(registry)

        assert encode_call_count == 0
        assert result.image_base64 == cached.data
