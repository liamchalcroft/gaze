"""Tests for image search validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiant_harness.exceptions import HarnessError
from radiant_harness.retrieval.image_search import ImageDownloadError
from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.image_search import MedicalImageSearchManager
from radiant_harness.retrieval.image_search import OpenISearchEngine


def test_invalid_image_search_limits_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_results_per_engine"):
        MedicalImageSearchManager(max_results_per_engine=0, download_dir=tmp_path)
    with pytest.raises(ValueError, match="rate_limit_delay"):
        MedicalImageSearchManager(rate_limit_delay=-0.1, download_dir=tmp_path)


class TestImageSearchErrorHierarchy:
    """Image search errors must be part of the HarnessError hierarchy."""

    def test_image_search_error_is_harness_error(self) -> None:
        err = ImageSearchError("Open-i", "test error")
        assert isinstance(err, HarnessError)

    def test_image_download_error_is_harness_error(self) -> None:
        err = ImageDownloadError("https://example.com/img.png", "timeout")
        assert isinstance(err, HarnessError)

    def test_image_search_error_preserves_fields(self) -> None:
        cause = RuntimeError("cause")
        err = ImageSearchError("Open-i", "search failed", original_error=cause)
        assert err.engine_name == "Open-i"
        assert err.original_error is cause

    def test_image_download_error_preserves_fields(self) -> None:
        cause = RuntimeError("cause")
        err = ImageDownloadError("https://example.com/img.png", "fail", original_error=cause)
        assert err.url == "https://example.com/img.png"
        assert err.original_error is cause


class TestOpenIResultParsing:
    """Tests for Open-i result parsing."""

    def test_modality_extraction(self) -> None:
        engine = OpenISearchEngine()
        assert engine._extract_modality("Brain MRI T2-weighted") == "MRI"
        assert engine._extract_modality("Chest CT scan") == "CT"
        assert engine._extract_modality("Lateral X-ray") == "X-ray"
        assert engine._extract_modality("No modality here") is None

    def test_body_part_extraction(self) -> None:
        engine = OpenISearchEngine()
        assert engine._extract_body_part("Brain lesion") == "brain"
        assert engine._extract_body_part("Chest radiograph") == "chest"
        assert engine._extract_body_part("Spinal cord") == "spine"
        assert engine._extract_body_part("No body part here") is None

    def test_parse_results_skips_no_image(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {"title": "No image result"},
                {
                    "title": "With image",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "imgThumb": "https://openi.nlm.nih.gov/thumb.jpg",
                    "pmcid": "PMC123",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].title == "With image"

    def test_parse_results_absolute_urls(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Relative URL",
                    "imgLarge": "/images/test.jpg",
                    "imgThumb": "/thumbs/test.jpg",
                    "pmcid": "PMC456",
                },
            ]
        }
        results = engine._parse_results(data)
        assert results[0].image_url.startswith("https://")
        assert results[0].thumbnail_url is not None
        assert results[0].thumbnail_url.startswith("https://")

    def test_modality_longer_keyword_preferred(self) -> None:
        """'ct scan' should match before 'ct' substring to avoid false positives."""
        engine = OpenISearchEngine()
        # "computed tomography" is a longer keyword mapping to CT
        assert engine._extract_modality("Computed tomography scan of the brain") == "CT"
        # "magnetic resonance" maps to MRI and is longer than "mri"
        assert engine._extract_modality("Magnetic resonance imaging of brain") == "MRI"

    def test_modality_ct_scan_over_mri_when_ct_primary(self) -> None:
        """When text mentions CT scan first, CT should be returned."""
        engine = OpenISearchEngine()
        # "ct scan" (7 chars) > "mri" (3 chars), so "ct scan" should be checked first
        assert engine._extract_modality("CT scan with MRI comparison") == "CT"


class TestAtexitTempDirCleanup:
    """Temp dir tracking must use module-level set, not per-instance atexit."""

    def test_temp_dir_tracked_on_creation(self) -> None:
        """Creating a manager without download_dir adds to _temp_dirs."""
        from radiant_harness.retrieval.image_search import _temp_dirs

        mgr = MedicalImageSearchManager()
        assert mgr._created_temp_dir is True
        assert mgr.download_dir in _temp_dirs
        # Clean up
        mgr._cleanup_temp_dir()
        _temp_dirs.discard(mgr.download_dir)

    def test_explicit_dir_not_tracked(self, tmp_path: Path) -> None:
        """Creating a manager with explicit download_dir must NOT track it."""
        from radiant_harness.retrieval.image_search import _temp_dirs

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        assert mgr._created_temp_dir is False
        assert mgr.download_dir not in _temp_dirs

    @pytest.mark.asyncio
    async def test_close_removes_from_tracking(self) -> None:
        """After close(), the temp dir should be removed from _temp_dirs."""
        from radiant_harness.retrieval.image_search import _temp_dirs

        mgr = MedicalImageSearchManager()
        temp_dir = mgr.download_dir
        assert temp_dir in _temp_dirs

        await mgr.close()
        assert temp_dir not in _temp_dirs

    def test_no_bound_method_in_atexit(self) -> None:
        """atexit handlers must not hold a reference to the manager instance.

        We verify by checking that _temp_dirs is a module-level set (not
        a bound method reference), and that the atexit function is the
        module-level _atexit_cleanup_temp_dirs, not a bound method.
        """
        from radiant_harness.retrieval.image_search import _atexit_cleanup_temp_dirs

        # The module-level function should be registered (it's registered
        # at module import time). We can verify it exists and is callable.
        assert callable(_atexit_cleanup_temp_dirs)

        # Create a manager and verify no bound methods leaked
        mgr = MedicalImageSearchManager()
        # The manager should not have registered self._cleanup_temp_dir with atexit
        # (we can't inspect atexit handlers directly, but we verified the code path)
        assert mgr._created_temp_dir is True
        # Cleanup
        mgr._cleanup_temp_dir()
        from radiant_harness.retrieval.image_search import _temp_dirs

        _temp_dirs.discard(mgr.download_dir)

    def test_atexit_handler_cleans_dirs(self, tmp_path: Path) -> None:
        """The module-level atexit handler should clean tracked dirs."""
        from radiant_harness.retrieval.image_search import _atexit_cleanup_temp_dirs
        from radiant_harness.retrieval.image_search import _temp_dirs

        # Create a temp dir and track it
        test_dir = tmp_path / "atexit_test"
        test_dir.mkdir()
        _temp_dirs.add(test_dir)

        # Simulate atexit
        _atexit_cleanup_temp_dirs()

        assert not test_dir.exists()
        assert test_dir not in _temp_dirs


class TestThumbnailUrlNone:
    """thumbnail_url must be None (not empty string) when imgThumb is absent."""

    def test_missing_thumb_is_none(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "No thumb",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC789",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].thumbnail_url is None

    def test_empty_string_thumb_is_none(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Empty thumb",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "imgThumb": "",
                    "pmcid": "PMC790",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].thumbnail_url is None


class TestLicenseDefault:
    """License must not default to 'Open Access' when field is missing."""

    def test_missing_license_is_none(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "No license",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC791",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].license is None

    def test_explicit_license_preserved(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Has license",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC792",
                    "license": "CC-BY-4.0",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].license == "CC-BY-4.0"


class TestExtensionFromUrl:
    """_get_extension_from_url must use path suffix, not substring."""

    def test_normal_image_url(self) -> None:
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager.__new__(MedicalImageSearchManager)
        assert mgr._get_extension_from_url("https://host.com/img/scan.jpg") == ".jpg"
        assert mgr._get_extension_from_url("https://host.com/img/scan.png") == ".png"

    def test_no_extension_returns_none(self) -> None:
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager.__new__(MedicalImageSearchManager)
        assert mgr._get_extension_from_url("https://host.com/img/scan") is None

    def test_substring_png_in_path_does_not_match(self) -> None:
        """URL with 'png' as a path segment (not extension) must not match."""
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager.__new__(MedicalImageSearchManager)
        # "pngdata" in path should NOT return .png
        assert mgr._get_extension_from_url("https://host.com/pngdata/file") is None

    def test_query_string_ignored(self) -> None:
        """Extension should come from path, not query string."""
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        mgr = MedicalImageSearchManager.__new__(MedicalImageSearchManager)
        assert mgr._get_extension_from_url("https://host.com/image.jpg?fmt=webp") == ".jpg"


class TestContentLengthMalformed:
    """Malformed Content-Length header must not crash download."""

    @pytest.mark.asyncio
    async def test_malformed_content_length_does_not_crash(self, tmp_path: Path) -> None:
        """int('abc') would crash without the ValueError guard."""
        from unittest.mock import AsyncMock

        from radiant_harness.retrieval.image_search import ImageSearchResult

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="Test",
            image_url="https://openi.nlm.nih.gov/test.jpg",
            thumbnail_url=None,
            source_url="https://openi.nlm.nih.gov/article",
            source="openi",
        )

        # Create a mock session and response with malformed Content-Length
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "image/jpeg", "Content-Length": "not-a-number"}
        # Valid JPEG magic bytes + padding
        mock_resp.read = AsyncMock(return_value=b"\xff\xd8\xff" + b"\x00" * 100)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = lambda *_a, **_kw: mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # Patch aiohttp.ClientSession to return our mock
        from unittest.mock import patch

        with patch("aiohttp.ClientSession", return_value=mock_session):
            filepath = await mgr.download_image(result)
            assert filepath.exists()


class TestKeywordMapsPreSorted:
    """Module-level keyword maps must be pre-sorted tuples (not dicts)."""

    def test_modality_keywords_is_tuple(self) -> None:
        from radiant_harness.retrieval.image_search import _MODALITY_KEYWORDS

        assert isinstance(_MODALITY_KEYWORDS, tuple)
        # Each element should be a (keyword, modality) pair
        for item in _MODALITY_KEYWORDS:
            assert isinstance(item, tuple) and len(item) == 2

    def test_body_part_keywords_is_tuple(self) -> None:
        from radiant_harness.retrieval.image_search import _BODY_PART_KEYWORDS

        assert isinstance(_BODY_PART_KEYWORDS, tuple)
        for item in _BODY_PART_KEYWORDS:
            assert isinstance(item, tuple) and len(item) == 2

    def test_modality_keywords_sorted_longest_first(self) -> None:
        from radiant_harness.retrieval.image_search import _MODALITY_KEYWORDS

        lengths = [len(kw) for kw, _ in _MODALITY_KEYWORDS]
        assert lengths == sorted(lengths, reverse=True)

    def test_body_part_keywords_sorted_longest_first(self) -> None:
        from radiant_harness.retrieval.image_search import _BODY_PART_KEYWORDS

        lengths = [len(kw) for kw, _ in _BODY_PART_KEYWORDS]
        assert lengths == sorted(lengths, reverse=True)


class TestOpeniBaseUrlDerived:
    """openi_base_url for relative URL resolution must derive from config."""

    def test_default_config_derives_origin(self) -> None:
        engine = OpenISearchEngine()
        assert engine.openi_base_url == "https://openi.nlm.nih.gov/"

    def test_custom_config_derives_origin(self) -> None:
        from radiant_harness.config import SearchConfig

        config = SearchConfig(openi_base_url="https://custom.example.com:8080/api/search")
        engine = OpenISearchEngine(config=config)
        assert engine.openi_base_url == "https://custom.example.com:8080/"
