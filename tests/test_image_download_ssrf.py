"""Tests for SSRF protection on image download URLs (PS-2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gaze.retrieval.image_search import ImageDownloadError
from gaze.retrieval.image_search import ImageSearchResult
from gaze.retrieval.image_search import MedicalImageSearchManager
from gaze.retrieval.image_search import OpenISearchEngine
from gaze.retrieval.image_search import _validate_download_url


# ---------------------------------------------------------------------------
# _validate_download_url unit tests
# ---------------------------------------------------------------------------
class TestValidateDownloadUrl:
    """_validate_download_url must enforce HTTPS and reject private IPs."""

    def test_https_url_passes(self) -> None:
        _validate_download_url("https://openi.nlm.nih.gov/image.jpg")

    def test_http_url_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="Only HTTPS"):
            _validate_download_url("http://openi.nlm.nih.gov/image.jpg")

    def test_ftp_url_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="Only HTTPS"):
            _validate_download_url("ftp://openi.nlm.nih.gov/image.jpg")

    def test_no_scheme_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="Only HTTPS"):
            _validate_download_url("openi.nlm.nih.gov/image.jpg")

    def test_localhost_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="loopback"):
            _validate_download_url("https://localhost/image.jpg")

    def test_zero_addr_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="loopback"):
            _validate_download_url("https://0.0.0.0/image.jpg")

    def test_loopback_ip_rejected(self) -> None:
        # Pass bare IP in allowlist so we test the IP-address check, not the hostname allowlist
        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://127.0.0.1/image.jpg",
                allowed_hostnames=frozenset({"127.0.0.1"}),
            )

    def test_private_ip_10_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://10.0.0.1/image.jpg",
                allowed_hostnames=frozenset({"10.0.0.1"}),
            )

    def test_private_ip_172_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://172.16.0.1/image.jpg",
                allowed_hostnames=frozenset({"172.16.0.1"}),
            )

    def test_private_ip_192_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://192.168.1.1/image.jpg",
                allowed_hostnames=frozenset({"192.168.1.1"}),
            )

    def test_link_local_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://169.254.169.254/latest/meta-data/",
                allowed_hostnames=frozenset({"169.254.169.254"}),
            )

    def test_ipv6_loopback_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="private/loopback"):
            _validate_download_url(
                "https://[::1]/image.jpg",
                allowed_hostnames=frozenset({"::1"}),
            )

    def test_public_dns_hostname_passes(self) -> None:
        """A public DNS hostname (in allowlist) with a public resolved IP should pass."""
        import socket
        from unittest.mock import patch

        # Mock DNS to return a public IP so we don't depend on real resolution
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            _validate_download_url(
                "https://cdn.example.com/images/scan.jpg",
                allowed_hostnames=frozenset({"cdn.example.com"}),
            )

    def test_public_ip_passes(self) -> None:
        _validate_download_url(
            "https://130.14.29.110/image.jpg",
            allowed_hostnames=frozenset({"130.14.29.110"}),
        )

    def test_no_hostname_rejected(self) -> None:
        with pytest.raises(ImageDownloadError, match="no hostname"):
            _validate_download_url("https:///image.jpg")


# ---------------------------------------------------------------------------
# Parse-time HTTPS enforcement
# ---------------------------------------------------------------------------
class TestParseResultsHttpsEnforcement:
    """_parse_results must reject non-HTTPS image URLs."""

    def test_https_urls_accepted(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Valid HTTPS image",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "pmcid": "PMC100",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1

    def test_http_urls_rejected(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "HTTP image should be skipped",
                    "imgLarge": "http://insecure.example.com/img.jpg",
                    "pmcid": "PMC200",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 0

    def test_relative_urls_joined_as_https(self) -> None:
        """Relative URLs are joined to the HTTPS base URL and should pass."""
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Relative URL image",
                    "imgLarge": "/imgs/scan.jpg",
                    "pmcid": "PMC300",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].image_url.startswith("https://")

    def test_mixed_urls_filters_correctly(self) -> None:
        """Only HTTPS results should survive."""
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Good HTTPS",
                    "imgLarge": "https://openi.nlm.nih.gov/good.jpg",
                    "pmcid": "PMC1",
                },
                {
                    "title": "Bad HTTP",
                    "imgLarge": "http://evil.example.com/bad.jpg",
                    "pmcid": "PMC2",
                },
                {
                    "title": "Another good HTTPS",
                    "imgLarge": "https://openi.nlm.nih.gov/good2.jpg",
                    "pmcid": "PMC3",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 2
        assert all(r.image_url.startswith("https://") for r in results)


# ---------------------------------------------------------------------------
# Download-time SSRF validation
# ---------------------------------------------------------------------------
class TestDownloadSsrfValidation:
    """download_image must reject SSRF-prone URLs at download time."""

    @pytest.mark.asyncio
    async def test_private_ip_blocked_at_download(self, tmp_path: Path) -> None:
        """Private IPs are blocked by the hostname allowlist."""
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="SSRF test",
            image_url="https://192.168.1.1/internal-scan.jpg",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
        )
        with pytest.raises(ImageDownloadError, match="not in the allowed download set"):
            await mgr.download_image(result)

    @pytest.mark.asyncio
    async def test_localhost_blocked_at_download(self, tmp_path: Path) -> None:
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="SSRF test",
            image_url="https://localhost/scan.jpg",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
        )
        with pytest.raises(ImageDownloadError, match="loopback"):
            await mgr.download_image(result)

    @pytest.mark.asyncio
    async def test_cloud_metadata_blocked(self, tmp_path: Path) -> None:
        """AWS metadata endpoint (169.254.169.254) must be blocked by allowlist."""
        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="Cloud metadata SSRF",
            image_url="https://169.254.169.254/latest/meta-data/",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
        )
        with pytest.raises(ImageDownloadError, match="not in the allowed download set"):
            await mgr.download_image(result)

    @pytest.mark.asyncio
    async def test_valid_https_url_proceeds_to_download(self, tmp_path: Path) -> None:
        """A valid HTTPS URL should pass validation and attempt download."""
        from unittest.mock import MagicMock
        from unittest.mock import patch

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="Valid image",
            image_url="https://openi.nlm.nih.gov/test.jpg",
            thumbnail_url=None,
            source_url="https://openi.nlm.nih.gov/article",
            source="openi",
        )

        # Valid JPEG magic bytes + padding
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 100

        # Create mock with streaming support (iter_chunked)
        mock_content = MagicMock()

        async def _iter_chunked(_chunk_size: int):
            yield image_bytes

        mock_content.iter_chunked = _iter_chunked

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.content = mock_content
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = lambda *_a, **_kw: mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            filepath = await mgr.download_image(result)
            assert filepath.exists()

    @pytest.mark.asyncio
    async def test_download_validation_runs_via_to_thread(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """URL validation should be offloaded so DNS resolution can't block the loop."""
        from unittest.mock import MagicMock

        import gaze.retrieval.image_search as image_search_module

        mgr = MedicalImageSearchManager(download_dir=tmp_path)
        result = ImageSearchResult(
            title="Valid image",
            image_url="https://openi.nlm.nih.gov/test.jpg",
            thumbnail_url=None,
            source_url="https://openi.nlm.nih.gov/article",
            source="openi",
        )

        offloaded_funcs: list[object] = []

        def _fake_validate(url: str) -> None:
            assert url == result.image_url

        async def _tracking_to_thread(func, *args, **kwargs):
            offloaded_funcs.append(func)
            return func(*args, **kwargs)

        monkeypatch.setattr(image_search_module, "_validate_download_url", _fake_validate)
        monkeypatch.setattr(image_search_module.asyncio, "to_thread", _tracking_to_thread)

        image_bytes = b"\xff\xd8\xff" + b"\x00" * 100
        mock_content = MagicMock()

        async def _iter_chunked(_chunk_size: int):
            yield image_bytes

        mock_content.iter_chunked = _iter_chunked

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.content = mock_content
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = lambda *_a, **_kw: mock_resp

        async def _fake_get_download_session():
            return mock_session

        monkeypatch.setattr(mgr, "_get_download_session", _fake_get_download_session)

        filepath = await mgr.download_image(result)

        assert filepath.exists()
        assert offloaded_funcs == [_fake_validate]
