"""Live integration tests for the retrieval managers.

These hit real external services (NCBI E-utilities for PubMed, NIH Open-i for
medical images) and so are excluded from the default suite by the
``-m "not integration"`` filter in ``pyproject.toml``. Run them explicitly with::

    uv run pytest -m integration

They are intentionally tolerant about the exact contents returned (the live
corpora change over time) and assert only that the request round-trips and the
parsed results carry the expected structure. They are skipped, never failed,
when the upstream service is unreachable or rate-limited, so a flaky network
does not break a deliberate integration run.

The configured base URLs (``SearchConfig.ncbi_base_url`` /
``SearchConfig.openi_base_url``) are used via the managers; no private or
hardcoded endpoints appear here.
"""

from __future__ import annotations

import pytest

from gaze.retrieval.image_search import ImageDownloadError
from gaze.retrieval.image_search import ImageSearchError
from gaze.retrieval.image_search import ImageSearchResult
from gaze.retrieval.image_search import MedicalImageSearchManager
from gaze.retrieval.web_search import SearchError
from gaze.retrieval.web_search import SearchResult
from gaze.retrieval.web_search import WebSearchManager

pytestmark = pytest.mark.integration


def _skip_if_upstream_unavailable(exc: Exception) -> None:
    """Skip for an upstream outage; re-raise anything that looks like our bug.

    These tests exist to catch regressions in the request and parsing code, so
    swallowing every error would defeat them: a broken parser raises the same
    SearchError as a dead service. Only transport failures and 5xx/429 statuses
    count as "the service is unavailable"; everything else propagates and fails
    the test.
    """
    import httpx

    causes: list[BaseException] = []
    cause: BaseException | None = exc
    while cause is not None and cause not in causes:
        causes.append(cause)
        cause = cause.__cause__

    for candidate in causes:
        if isinstance(candidate, httpx.HTTPStatusError):
            status = candidate.response.status_code
            if status == 429 or status >= 500:
                pytest.skip(f"upstream returned HTTP {status}: {exc}")
            return  # a 4xx means we sent a bad request; that is a real failure
        if isinstance(candidate, httpx.TransportError | TimeoutError):
            pytest.skip(f"upstream unreachable: {exc}")

    # No transport cause recorded: the message still carries the status when the
    # retry wrapper gave up on repeated 5xx.
    message = str(exc)
    if "All search attempts failed" in message or " 5" in message:
        pytest.skip(f"upstream unavailable: {exc}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pubmed_live_search_returns_structured_results() -> None:
    """A real PubMed query must return ranked SearchResult objects.

    Exercises the full WebSearchManager -> PubMedSearchEngine -> NCBI
    E-utilities path against the configured ``ncbi_base_url``.
    """
    async with WebSearchManager(engines=["pubmed"]) as manager:
        try:
            results = await manager.search(
                "glioblastoma MRI imaging",
                search_type="research",
            )
        except SearchError as exc:
            _skip_if_upstream_unavailable(exc)
            raise

    assert isinstance(results, list)
    assert results, "Expected at least one PubMed result for a common query"

    first = results[0]
    assert isinstance(first, SearchResult)
    # Core structural fields must be populated by the parser.
    assert first.title.strip(), "Result title must be non-empty"
    assert first.url.startswith("http"), f"Expected an absolute URL, got {first.url!r}"
    assert first.source, "Result source must be set"
    assert 0.0 <= first.reliability_score <= 1.0
    assert first.ranking_score >= 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openi_live_image_search_returns_structured_results() -> None:
    """A real Open-i query must return ImageSearchResult objects.

    Exercises the MedicalImageSearchManager -> OpenISearchEngine -> NIH Open-i
    path against the configured ``openi_base_url``.
    """
    async with MedicalImageSearchManager(engines=["openi"]) as manager:
        try:
            results = await manager.search("brain MRI")
        except ImageSearchError as exc:
            _skip_if_upstream_unavailable(exc)
            raise

    assert isinstance(results, list)
    assert results, "Expected at least one Open-i image result for a common query"

    first = results[0]
    assert isinstance(first, ImageSearchResult)
    # Image URLs are enforced to HTTPS by the parser's SSRF guard.
    assert first.image_url.startswith("https://"), (
        f"Expected an HTTPS image URL, got {first.image_url!r}"
    )
    assert first.title.strip(), "Result title must be non-empty"
    assert first.source == "openi"
    assert 0.0 <= first.reliability_score <= 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openi_live_image_download_streams_and_validates() -> None:
    """Downloading a real image must stream, size-check, and magic-byte check.

    This is the one retrieval path the mocked suite cannot fully stand in for:
    it exercises streaming over a real connection, the Content-Type and
    Content-Length handling of a live server, and the SSRF gate against a URL
    that came from an external API rather than from a fixture.
    """
    async with MedicalImageSearchManager(engines=["openi"]) as manager:
        try:
            results = await manager.search("brain MRI")
        except ImageSearchError as exc:
            _skip_if_upstream_unavailable(exc)
            raise

        if not results:
            pytest.skip("Open-i returned no results to download")

        try:
            path = await manager.download_image(results[0])
        except ImageDownloadError as exc:
            _skip_if_upstream_unavailable(exc)
            raise

        assert path.exists(), "downloaded file should be on disk"
        size = path.stat().st_size
        assert 0 < size <= manager._MAX_DOWNLOAD_BYTES

        # The magic-byte check inside _do_download already ran; confirm the
        # bytes really are an image rather than an error page saved verbatim.
        header = path.read_bytes()[:12]
        assert any(header.startswith(magic) for magic, _ in manager._IMAGE_MAGIC), (
            f"downloaded content is not a known image format: {header!r}"
        )
