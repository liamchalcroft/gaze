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

from gaze.retrieval.image_search import ImageSearchError
from gaze.retrieval.image_search import ImageSearchResult
from gaze.retrieval.image_search import MedicalImageSearchManager
from gaze.retrieval.web_search import SearchError
from gaze.retrieval.web_search import SearchResult
from gaze.retrieval.web_search import WebSearchManager

pytestmark = pytest.mark.integration


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
            pytest.skip(f"PubMed live service unavailable: {exc}")

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
            pytest.skip(f"Open-i live service unavailable: {exc}")

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
