"""Tests targeting uncovered lines in retrieval/web_search.py and retrieval/image_search.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest

from gaze.retrieval.image_search import ImageDownloadError
from gaze.retrieval.image_search import ImageSearchError
from gaze.retrieval.image_search import ImageSearchResult
from gaze.retrieval.image_search import MedicalImageSearchManager
from gaze.retrieval.image_search import OpenISearchEngine
from gaze.retrieval.image_search import _validate_download_url
from gaze.retrieval.web_search import PubMedSearchEngine

# ---------------------------------------------------------------------------
# PubMed: _fetch_article_details  (lines 275-336)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
class TestFetchArticleDetails:
    async def test_basic_article_parsing(self) -> None:
        """Covers lines 275-336: full article parsing path."""
        engine = PubMedSearchEngine()
        engine._rate_limit_delay = 0  # skip delay for tests

        summary_data = {
            "result": {
                "12345": {
                    "title": "Brain MRI findings in glioblastoma",
                    "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                    "fulljournalname": "Journal of Neuroradiology",
                    "pubdate": "2024 Jan",
                    "doi": "10.1234/test",
                    "articleids": [
                        {"idtype": "pubmed", "value": "12345"},
                        {"idtype": "pmc", "value": "PMC999"},
                    ],
                    "pubtype": ["Journal Article"],
                },
            }
        }
        abstracts = {
            "12345": "This study examined MRI findings in patients with glioblastoma multiforme."
        }

        with (
            patch.object(
                engine, "_fetch_summary", new_callable=AsyncMock, return_value=summary_data
            ),
            patch.object(
                engine, "_fetch_abstracts", new_callable=AsyncMock, return_value=abstracts
            ),
        ):
            results = await engine._fetch_article_details(["12345"])

        assert len(results) == 1
        r = results[0]
        assert r.title == "Brain MRI findings in glioblastoma"
        assert r.source == "pubmed"
        assert "pubmed.ncbi.nlm.nih.gov/12345" in r.url
        assert r.open_access is True  # PMC ID present
        assert r.publication_date == "2024 Jan"
        assert r.doi == "10.1234/test"
        assert r.journal == "Journal of Neuroradiology"
        assert "Smith J" in r.author
        assert 0 < r.reliability_score <= 1.0
        assert r.medical_relevance >= 0.7  # base for PubMed
        # Has abstract → content is abstract, not title
        assert "glioblastoma" in r.content.lower()

    async def test_missing_pmid_skipped(self) -> None:
        """PMIDs not in summary result dict are skipped."""
        engine = PubMedSearchEngine()
        engine._rate_limit_delay = 0

        summary_data = {"result": {"99999": {"title": "Other"}}}
        with (
            patch.object(
                engine, "_fetch_summary", new_callable=AsyncMock, return_value=summary_data
            ),
            patch.object(engine, "_fetch_abstracts", new_callable=AsyncMock, return_value={}),
        ):
            results = await engine._fetch_article_details(["12345"])

        assert len(results) == 0

    async def test_no_result_key_returns_empty(self) -> None:
        """Missing 'result' key in summary → empty list."""
        engine = PubMedSearchEngine()
        engine._rate_limit_delay = 0

        with (
            patch.object(
                engine, "_fetch_summary", new_callable=AsyncMock, return_value={"error": "bad"}
            ),
            patch.object(engine, "_fetch_abstracts", new_callable=AsyncMock, return_value={}),
        ):
            results = await engine._fetch_article_details(["12345"])

        assert results == []

    async def test_no_abstract_lower_relevance(self) -> None:
        """Article without abstract → content is title, lower medical_relevance."""
        engine = PubMedSearchEngine()
        engine._rate_limit_delay = 0

        summary_data = {
            "result": {
                "111": {
                    "title": "A simple note",
                    "authors": [],
                    "fulljournalname": "J Test",
                    "pubdate": "2024",
                    "doi": "",
                    "articleids": [],
                    "pubtype": ["Journal Article"],
                },
            }
        }

        with (
            patch.object(
                engine, "_fetch_summary", new_callable=AsyncMock, return_value=summary_data
            ),
            patch.object(engine, "_fetch_abstracts", new_callable=AsyncMock, return_value={}),
        ):
            results = await engine._fetch_article_details(["111"])

        assert len(results) == 1
        r = results[0]
        assert r.content == r.title  # No abstract → content equals title
        assert r.open_access is False  # No PMC ID
        # Without abstract, no abstract_bonus → relevance is lower
        assert r.medical_relevance < 1.0


# ---------------------------------------------------------------------------
# PubMed: search edge cases  (lines 225, 228, 240, 246)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
class TestPubMedSearchEdgeCases:
    async def test_empty_idlist_returns_empty(self, json_route, make_mock_http_client) -> None:
        """Empty PMID list → returns [] without calling _fetch_article_details."""

        engine = PubMedSearchEngine()
        client = make_mock_http_client(json_route({"esearch": {"esearchresult": {"idlist": []}}}))

        with patch.object(engine, "_get_session", new_callable=AsyncMock, return_value=client):
            assert await engine._search_impl("test query", max_results=5) == []
        await client.aclose()

    async def test_missing_esearchresult_returns_empty(
        self, json_route, make_mock_http_client
    ) -> None:
        """Missing esearchresult key → returns []."""
        engine = PubMedSearchEngine()

        client = make_mock_http_client(json_route({"esearch": {"error": "server error"}}))

        with patch.object(engine, "_get_session", new_callable=AsyncMock, return_value=client):
            assert await engine._search_impl("test query", max_results=5) == []
        await client.aclose()


# ---------------------------------------------------------------------------
# ImageSearch: MedicalImageSearchManager  (lines 525, 573-574, 689)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMedicalImageSearchManagerEdgeCases:
    def test_unknown_engine_raises(self) -> None:
        """Unknown engine name → ValueError."""
        with pytest.raises(ValueError, match="Unknown image search engine"):
            MedicalImageSearchManager(engines=["nonexistent_engine"])

    def test_cleanup_temp_dir_oserror_handled(self, tmp_path: Path) -> None:
        """OSError during cleanup is caught and logged, not raised."""
        mgr = MedicalImageSearchManager(download_dir=tmp_path / "dl")
        mgr._created_temp_dir = True
        mgr.download_dir.mkdir(parents=True, exist_ok=True)

        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            # Should not raise
            mgr._cleanup_temp_dir()

    @pytest.mark.asyncio
    async def test_download_client_error_wrapped(self, make_mock_http_client) -> None:
        """A transport error during download is wrapped in ImageDownloadError."""
        mgr = MedicalImageSearchManager()
        result = ImageSearchResult(
            title="test",
            image_url="https://openi.nlm.nih.gov/imgs/test.jpg",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed", request=request)

        client = make_mock_http_client(handler)

        with (
            patch.object(mgr, "_get_download_session", new_callable=AsyncMock, return_value=client),
            pytest.raises(ImageDownloadError, match="connection failed"),
        ):
            await mgr.download_image(result)
        await client.aclose()

        await mgr.close()


# ---------------------------------------------------------------------------
# _do_download error paths  (lines 740, 747, 759, 772, 789-790, 807-808)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
class TestDoDownload:
    @staticmethod
    def _install_download_client(
        make_mock_http_client: object, mgr: MedicalImageSearchManager, handler: object
    ) -> object:
        """Point the manager's download client at a MockTransport handler."""

        client = make_mock_http_client(handler)
        patch.object(
            mgr, "_get_download_session", new_callable=AsyncMock, return_value=client
        ).__enter__()
        return client

    @staticmethod
    def _result() -> ImageSearchResult:
        return ImageSearchResult(
            title="test",
            image_url="https://openi.nlm.nih.gov/imgs/test.jpg",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
        )

    async def test_non_200_raises(
        self, tmp_path: Path, download_route, make_mock_http_client
    ) -> None:

        mgr = MedicalImageSearchManager(download_dir=tmp_path / "dl")
        client = self._install_download_client(
            make_mock_http_client, mgr, download_route(status=404, content_type="text/html")
        )

        with (
            patch("gaze.retrieval.image_search._validate_download_url"),
            pytest.raises(ImageDownloadError, match="HTTP 404"),
        ):
            await mgr._do_download(client, self._result(), "abc123", ".jpg")
        await mgr.close()

    async def test_non_image_content_type_raises(
        self, tmp_path: Path, download_route, make_mock_http_client
    ) -> None:

        mgr = MedicalImageSearchManager(download_dir=tmp_path / "dl")
        client = self._install_download_client(
            make_mock_http_client, mgr, download_route(content_type="text/html")
        )

        with (
            patch("gaze.retrieval.image_search._validate_download_url"),
            pytest.raises(ImageDownloadError, match="not an image"),
        ):
            await mgr._do_download(client, self._result(), "abc", ".jpg")
        await mgr.close()

    async def test_content_length_too_large_raises(
        self, tmp_path: Path, download_route, make_mock_http_client
    ) -> None:
        """A declared size over the cap is rejected before the body is read."""

        mgr = MedicalImageSearchManager(download_dir=tmp_path / "dl")
        client = self._install_download_client(
            make_mock_http_client,
            mgr,
            download_route(content_length=str(mgr._MAX_DOWNLOAD_BYTES + 1)),
        )

        with (
            patch("gaze.retrieval.image_search._validate_download_url"),
            pytest.raises(ImageDownloadError, match="too large"),
        ):
            await mgr._do_download(client, self._result(), "abc", ".jpg")
        await mgr.close()

    async def test_streamed_body_over_the_cap_is_rejected(
        self, tmp_path: Path, download_route, make_mock_http_client
    ) -> None:
        """The guard must hold even when the server understates Content-Length."""

        mgr = MedicalImageSearchManager(download_dir=tmp_path / "dl")
        oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (mgr._MAX_DOWNLOAD_BYTES + 1024)
        client = self._install_download_client(
            make_mock_http_client, mgr, download_route(content=oversized, content_length="10")
        )

        with (
            patch("gaze.retrieval.image_search._validate_download_url"),
            pytest.raises(ImageDownloadError, match="too large"),
        ):
            await mgr._do_download(client, self._result(), "abc", ".jpg")
        await mgr.close()


def test_get_extension_from_content_type() -> None:
    """_get_extension_from_content_type returns correct extension."""
    mgr = MedicalImageSearchManager()
    assert mgr._get_extension_from_content_type("image/jpeg") == ".jpg"
    assert mgr._get_extension_from_content_type("image/png") == ".png"
    assert mgr._get_extension_from_content_type("image/png; charset=utf-8") == ".png"
    assert mgr._get_extension_from_content_type("image/unknown") == ".jpg"  # default


# ---------------------------------------------------------------------------
# OpenISearchEngine: JSON parse error  (lines 316-325)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_openi_invalid_json_raises(json_route, make_mock_http_client) -> None:
    """Open-i returning invalid JSON raises ImageSearchError."""
    engine = OpenISearchEngine()

    client = make_mock_http_client(json_route({"": "<html>not json</html>"}))

    with (
        patch.object(engine, "_get_session", new_callable=AsyncMock, return_value=client),
        pytest.raises(ImageSearchError, match="invalid JSON"),
    ):
        await engine._search_impl("brain MRI", max_results=5)

    await client.aclose()
    await engine.close()


# ---------------------------------------------------------------------------
# _validate_download_url: DNS resolution error  (lines 259-260)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_download_url_dns_failure() -> None:
    """DNS resolution failure raises ImageDownloadError."""
    import socket

    with (
        patch("socket.getaddrinfo", side_effect=socket.gaierror("DNS failed")),
        pytest.raises(ImageDownloadError, match="DNS resolution failed"),
    ):
        _validate_download_url(
            "https://openi.nlm.nih.gov/imgs/test.jpg",
            allowed_hostnames=frozenset({"openi.nlm.nih.gov"}),
        )


# ---------------------------------------------------------------------------
# _validate_image_magic  (line 714)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_image_magic_rejects_non_image() -> None:
    """Non-image content fails magic byte validation."""
    with pytest.raises(ImageDownloadError, match="does not match"):
        MedicalImageSearchManager._validate_image_magic(
            b"this is not an image at all", "https://example.com/fake.jpg"
        )


@pytest.mark.unit
def test_validate_image_magic_accepts_jpeg() -> None:
    """JPEG magic bytes pass validation."""
    # Should not raise
    MedicalImageSearchManager._validate_image_magic(
        b"\xff\xd8\xff\xe0" + b"\x00" * 100, "https://example.com/real.jpg"
    )
