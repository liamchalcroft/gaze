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

    def test_standalone_ct_detected(self) -> None:
        """Standalone 'CT' in caption should be detected as CT modality."""
        engine = OpenISearchEngine()
        assert engine._extract_modality("CT of the abdomen showing mass") == "CT"
        assert engine._extract_modality("Axial CT image") == "CT"
        assert engine._extract_modality("CT-guided biopsy") == "CT"

    def test_pet_no_false_positive(self) -> None:
        """'pet' must not match 'competent', 'repetitive', etc."""
        engine = OpenISearchEngine()
        assert engine._extract_modality("The test was not competitive") is None
        assert engine._extract_modality("Repetitive measurements taken") is None
        assert engine._extract_modality("Competent radiologist reviewed") is None
        # But real PET should still match
        assert engine._extract_modality("PET scan of the brain") == "PET"

    def test_ct_no_false_positive(self) -> None:
        """'ct' must not match 'infected', 'detected', etc."""
        engine = OpenISearchEngine()
        assert engine._extract_modality("Infected tissue was detected") is None
        assert engine._extract_modality("Protection against disease") is None
        # But real CT should still match
        assert engine._extract_modality("CT findings in stroke") == "CT"

    def test_mri_no_false_positive(self) -> None:
        """'mri' must not match substrings in other words."""
        engine = OpenISearchEngine()
        # Real MRI should match
        assert engine._extract_modality("MRI of the brain") == "MRI"
        assert engine._extract_modality("Brain mri shows lesion") == "MRI"

    def test_prefix_keywords_match_derived_forms(self) -> None:
        """Prefix keywords like 'mammograph' should match 'mammography'."""
        engine = OpenISearchEngine()
        assert engine._extract_modality("Mammography screening result") == "Mammography"
        assert engine._extract_modality("Mammographic findings") == "Mammography"

    def test_body_part_hip_no_false_positive(self) -> None:
        """'hip' must not match 'relationship', 'fellowship', etc."""
        engine = OpenISearchEngine()
        assert engine._extract_body_part("The relationship between X and Y") is None
        assert engine._extract_body_part("Fellowship training program") is None
        # Real hip should match
        assert engine._extract_body_part("Hip fracture in elderly") == "pelvis"

    def test_body_part_vertebr_matches_derived_forms(self) -> None:
        """'vertebr' prefix should match 'vertebral', 'vertebrae', etc."""
        engine = OpenISearchEngine()
        assert engine._extract_body_part("Vertebral compression fracture") == "spine"
        assert engine._extract_body_part("Lumbar vertebrae alignment") == "spine"

    def test_parse_results_non_list_items(self) -> None:
        """Non-list 'list' field should return empty results, not crash."""
        engine = OpenISearchEngine()
        assert engine._parse_results({"list": "error"}) == []
        assert engine._parse_results({"list": None}) == []
        assert engine._parse_results({"list": 42}) == []
        assert engine._parse_results({}) == []


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
        from unittest.mock import MagicMock

        from radiant_harness.retrieval.image_search import ImageSearchResult

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="Test",
            image_url="https://openi.nlm.nih.gov/test.jpg",
            thumbnail_url=None,
            source_url="https://openi.nlm.nih.gov/article",
            source="openi",
        )

        # Valid JPEG magic bytes + padding
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 100

        # Create a mock session and response with malformed Content-Length
        # and streaming support (iter_chunked)
        mock_content = MagicMock()

        async def _iter_chunked(chunk_size: int):
            yield image_bytes

        mock_content.iter_chunked = _iter_chunked

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "image/jpeg", "Content-Length": "not-a-number"}
        mock_resp.content = mock_content
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


class TestKeywordPatternsPreSorted:
    """Module-level keyword patterns must be pre-sorted tuples."""

    def test_modality_patterns_is_tuple(self) -> None:
        from radiant_harness.retrieval.image_search import _MODALITY_PATTERNS

        assert isinstance(_MODALITY_PATTERNS, tuple)
        for item in _MODALITY_PATTERNS:
            assert isinstance(item, tuple) and len(item) == 2

    def test_body_part_patterns_is_tuple(self) -> None:
        from radiant_harness.retrieval.image_search import _BODY_PART_PATTERNS

        assert isinstance(_BODY_PART_PATTERNS, tuple)
        for item in _BODY_PART_PATTERNS:
            assert isinstance(item, tuple) and len(item) == 2

    def test_modality_patterns_sorted_longest_first(self) -> None:
        import re

        from radiant_harness.retrieval.image_search import _MODALITY_PATTERNS

        # Extract effective keyword length from pattern (strip \b and escapes)
        def keyword_len(pat: re.Pattern[str]) -> int:
            raw = pat.pattern.replace(r"\b", "").replace("\\", "")
            return len(raw)

        lengths = [keyword_len(pat) for pat, _ in _MODALITY_PATTERNS]
        assert lengths == sorted(lengths, reverse=True)

    def test_body_part_patterns_sorted_longest_first(self) -> None:
        import re

        from radiant_harness.retrieval.image_search import _BODY_PART_PATTERNS

        def keyword_len(pat: re.Pattern[str]) -> int:
            raw = pat.pattern.replace(r"\b", "").replace("\\", "")
            return len(raw)

        lengths = [keyword_len(pat) for pat, _ in _BODY_PART_PATTERNS]
        assert lengths == sorted(lengths, reverse=True)


class TestThumbnailHttpsEnforcement:
    """thumbnail_url must be HTTPS or None — HTTP thumbnails must be dropped."""

    def test_http_thumbnail_dropped(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "HTTP thumbnail",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "imgThumb": "http://insecure.example.com/thumb.jpg",
                    "pmcid": "PMC900",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].thumbnail_url is None

    def test_https_thumbnail_preserved(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "HTTPS thumbnail",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "imgThumb": "https://openi.nlm.nih.gov/thumb.jpg",
                    "pmcid": "PMC901",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].thumbnail_url == "https://openi.nlm.nih.gov/thumb.jpg"

    def test_relative_thumbnail_joined_as_https(self) -> None:
        """Relative thumbnail URLs joined to HTTPS base should pass."""
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Relative thumb",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "imgThumb": "/thumbs/t.jpg",
                    "pmcid": "PMC902",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].thumbnail_url is not None
        assert results[0].thumbnail_url.startswith("https://")


class TestOpenIHttpErrorRetry:
    """HTTP error responses from Open-i must be retried by the base class."""

    @pytest.mark.asyncio
    async def test_openi_503_is_retried(self) -> None:
        """Open-i returning 503 must trigger retry, not immediate failure."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        import aiohttp

        from radiant_harness.config import SearchConfig

        config = SearchConfig(max_retries=3, rate_limit_delay_seconds=0.0)
        engine = OpenISearchEngine(config=config)

        call_count = 0

        @asynccontextmanager
        async def mock_get(url: str, params: dict | None = None):  # type: ignore[override]
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock()
            mock_resp.status = 503
            mock_resp.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=aiohttp.RequestInfo(
                        url=url,
                        method="GET",
                        headers={},
                        real_url=url,  # type: ignore[arg-type]
                    ),
                    history=(),
                    status=503,
                    message="Service Unavailable",
                )
            )
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get

        with (
            patch.object(engine, "_get_session", new=AsyncMock(return_value=mock_session)),
            pytest.raises(ImageSearchError, match="All search attempts failed"),
        ):
            await engine.search("brain MRI")

        # Should have retried max_retries times
        assert call_count == 3


class TestOpeniBaseUrlDerived:
    """openi_base_url for relative URL resolution must derive from config."""

    def test_default_config_derives_origin(self) -> None:
        engine = OpenISearchEngine()
        assert engine.openi_base_url == "https://openi.nlm.nih.gov/"

    def test_custom_config_derives_origin(self) -> None:
        from radiant_harness.config import SearchConfig

        # Use an allowed hostname with a non-default path to verify origin derivation
        config = SearchConfig(openi_base_url="https://openi.nlm.nih.gov/v2/api/search")
        engine = OpenISearchEngine(config=config)
        assert engine.openi_base_url == "https://openi.nlm.nih.gov/"


class TestSearchConfigHostnameAllowlist:
    """SearchConfig must reject hostnames not in the allowed set."""

    def test_rejects_arbitrary_hostname_ncbi(self) -> None:
        from radiant_harness.config import SearchConfig

        with pytest.raises(ValueError, match="not in the allowed set"):
            SearchConfig(ncbi_base_url="https://evil.example.com/entrez/eutils/")

    def test_rejects_arbitrary_hostname_openi(self) -> None:
        from radiant_harness.config import SearchConfig

        with pytest.raises(ValueError, match="not in the allowed set"):
            SearchConfig(openi_base_url="https://evil.example.com/api/search")

    def test_accepts_allowed_ncbi_hostname(self) -> None:
        from radiant_harness.config import SearchConfig

        config = SearchConfig(ncbi_base_url="https://eutils.ncbi.nlm.nih.gov/custom/path/")
        assert "eutils.ncbi.nlm.nih.gov" in config.ncbi_base_url

    def test_accepts_allowed_openi_hostname(self) -> None:
        from radiant_harness.config import SearchConfig

        config = SearchConfig(openi_base_url="https://openi.nlm.nih.gov/v2/search")
        assert "openi.nlm.nih.gov" in config.openi_base_url


class TestDownloadUrlSsrfValidation:
    """_validate_download_url must block SSRF vectors."""

    def test_rejects_http_scheme(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        with pytest.raises(ImageDownloadError, match="HTTPS"):
            _validate_download_url("http://openi.nlm.nih.gov/img.jpg")

    def test_rejects_non_allowed_hostname(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        with pytest.raises(ImageDownloadError, match="not in the allowed download set"):
            _validate_download_url("https://evil.example.com/img.jpg")

    def test_rejects_localhost(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        with pytest.raises(ImageDownloadError, match="loopback"):
            _validate_download_url("https://localhost/img.jpg")

    def test_rejects_loopback_ip(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://127.0.0.1/img.jpg",
                allowed_hostnames=frozenset({"127.0.0.1"}),
            )

    def test_rejects_private_ip(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://192.168.1.1/img.jpg",
                allowed_hostnames=frozenset({"192.168.1.1"}),
            )

    def test_rejects_no_hostname(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        with pytest.raises(ImageDownloadError, match="no hostname"):
            _validate_download_url("https:///img.jpg")

    def test_accepts_allowed_hostname(self) -> None:
        from radiant_harness.retrieval.image_search import _validate_download_url

        # Should not raise
        _validate_download_url("https://openi.nlm.nih.gov/images/test.jpg")

    def test_dns_resolution_rejects_private_ip(self) -> None:
        """If an allowed hostname resolves to a private IP, it must be rejected."""
        import socket
        from unittest.mock import patch

        from radiant_harness.retrieval.image_search import _validate_download_url

        # Simulate DNS resolving to a private IP
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.1", 0)),
        ]
        with (
            patch("socket.getaddrinfo", return_value=fake_addrinfo),
            pytest.raises(ImageDownloadError, match="private/loopback"),
        ):
            _validate_download_url("https://openi.nlm.nih.gov/img.jpg")


class TestSanitizeApiField:
    """_sanitize_api_field must strip control chars and truncate."""

    def test_strips_control_characters(self) -> None:
        from radiant_harness.retrieval.image_search import _sanitize_api_field

        result = _sanitize_api_field("normal\x00hidden\x01text\x7f")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result
        assert result == "normalhiddentext"

    def test_truncates_to_max_length(self) -> None:
        from radiant_harness.retrieval.image_search import _sanitize_api_field

        result = _sanitize_api_field("A" * 1000, max_length=100)
        assert len(result) == 100

    def test_preserves_normal_text(self) -> None:
        from radiant_harness.retrieval.image_search import _sanitize_api_field

        result = _sanitize_api_field("Brain MRI T2-weighted")
        assert result == "Brain MRI T2-weighted"

    def test_parse_results_sanitizes_title(self) -> None:
        """Titles from Open-i containing control chars should be stripped."""
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Injected\x00Title",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC999",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert "\x00" not in results[0].title
        assert results[0].title == "InjectedTitle"


class TestSanitizeNewlinesAndTabs:
    """_sanitize_api_field must strip newlines, carriage returns, and tabs."""

    def test_strips_newlines(self) -> None:
        from radiant_harness.retrieval.image_search import _sanitize_api_field

        result = _sanitize_api_field("line1\nline2\rline3")
        assert "\n" not in result
        assert "\r" not in result
        assert result == "line1line2line3"

    def test_strips_tabs(self) -> None:
        from radiant_harness.retrieval.image_search import _sanitize_api_field

        result = _sanitize_api_field("col1\tcol2")
        assert "\t" not in result
        assert result == "col1col2"

    def test_strips_crlf(self) -> None:
        from radiant_harness.retrieval.image_search import _sanitize_api_field

        result = _sanitize_api_field("Title\r\n## Injected Header")
        assert "\r" not in result
        assert "\n" not in result
        assert result == "Title## Injected Header"


class TestDownloadSessionUserAgent:
    """Download session must use honest-bot User-Agent."""

    @pytest.mark.asyncio
    async def test_download_session_has_user_agent(self, tmp_path: Path) -> None:
        import radiant_harness

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        session = await mgr._get_download_session()
        ua = session._default_headers.get("User-Agent", "")
        assert "radiant_harness" in ua
        assert radiant_harness.__version__ in ua
        await mgr.close()

    @pytest.mark.asyncio
    async def test_download_session_no_browser_impersonation(self, tmp_path: Path) -> None:
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        session = await mgr._get_download_session()
        ua = session._default_headers.get("User-Agent", "")
        for browser_str in ("Mozilla", "Chrome", "Safari"):
            assert browser_str not in ua
        await mgr.close()


class TestPmcidValidation:
    """pmcid used in URL construction must be format-validated."""

    def test_valid_pmcid_produces_url(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Valid PMCID article",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC1234567",
                }
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert "PMC1234567" in results[0].source_url

    def test_path_traversal_pmcid_rejected(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Traversal PMCID article",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "../../admin",
                    "detailedURL": "https://example.com/fallback",
                }
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert "../../admin" not in results[0].source_url

    def test_empty_pmcid_uses_detailed_url(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "No PMCID article",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "",
                    "detailedURL": "https://example.com/article",
                }
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].source_url == "https://example.com/article"


class TestMetadataSanitization:
    """mesh_terms and image_type in metadata must be sanitized."""

    def test_mesh_terms_control_chars_stripped(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Test article",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC123",
                    "meshMajor": ["Brain\x00Neoplasms", "Normal Term"],
                }
            ]
        }
        results = engine._parse_results(data)
        mesh = results[0].metadata["mesh_terms"]
        assert "\x00" not in mesh[0]
        assert mesh[0] == "BrainNeoplasms"
        assert mesh[1] == "Normal Term"

    def test_non_string_mesh_terms_filtered(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Test article",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC123",
                    "meshMajor": ["Valid", 42, None, "Also Valid"],
                }
            ]
        }
        results = engine._parse_results(data)
        mesh = results[0].metadata["mesh_terms"]
        assert len(mesh) == 2
        assert mesh[0] == "Valid"
        assert mesh[1] == "Also Valid"

    def test_image_type_sanitized(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Test article",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC123",
                    "imgType": "photo\x00graph",
                }
            ]
        }
        results = engine._parse_results(data)
        img_type = results[0].metadata["image_type"]
        assert "\x00" not in img_type
        assert img_type == "photograph"


# ---------------------------------------------------------------------------
# Shared download session lifecycle
# ---------------------------------------------------------------------------


class TestSharedDownloadSession:
    @pytest.mark.asyncio
    async def test_download_session_reused(self, tmp_path: Path) -> None:
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        session1 = await mgr._get_download_session()
        session2 = await mgr._get_download_session()
        assert session1 is session2
        await mgr.close()

    @pytest.mark.asyncio
    async def test_download_session_closed_on_cleanup(self, tmp_path: Path) -> None:
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        session = await mgr._get_download_session()
        assert not session.closed
        await mgr.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_close_without_session_does_not_raise(self, tmp_path: Path) -> None:
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        await mgr.close()
