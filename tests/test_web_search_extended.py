"""Extended tests for gaze.retrieval.web_search — covering uncovered lines.

Targets: reliability scoring for .edu/.gov/publisher domains (lines 115, 123),
_fetch_article_details (lines 265-322), WebSearchManager.search query variant
fallback (lines 660-743), _filter_results dedup/quality/medical (lines 765-793),
and context manager / close (lines 606-625).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from gaze.retrieval.web_search import PubMedSearchEngine
from gaze.retrieval.web_search import SearchError
from gaze.retrieval.web_search import SearchResult
from gaze.retrieval.web_search import WebSearchManager

# ---------------------------------------------------------------------------
# Reliability scoring — .edu, .gov, publisher domains (lines 112-123)
# ---------------------------------------------------------------------------


class TestReliabilityScoring:
    def setup_method(self) -> None:
        self.engine = PubMedSearchEngine()

    def test_edu_domain(self) -> None:
        assert self.engine._calculate_reliability("https://stanford.edu/paper") == 0.80

    def test_ac_uk_domain(self) -> None:
        assert self.engine._calculate_reliability("https://ox.ac.uk/study") == 0.80

    def test_gov_domain(self) -> None:
        assert self.engine._calculate_reliability("https://data.gov/dataset") == 0.80

    def test_nih_gov_domain(self) -> None:
        assert self.engine._calculate_reliability("https://nci.nih.gov/trial") == 0.80

    def test_elsevier_publisher(self) -> None:
        assert self.engine._calculate_reliability("https://www.elsevier.com/article") == 0.85

    def test_springer_publisher(self) -> None:
        assert self.engine._calculate_reliability("https://link.springer.com/paper") == 0.85

    def test_wiley_publisher(self) -> None:
        assert self.engine._calculate_reliability("https://onlinelibrary.wiley.com/doi") == 0.85

    def test_thelancet_publisher(self) -> None:
        assert self.engine._calculate_reliability("https://www.thelancet.com/article") == 0.85

    def test_bmj_publisher(self) -> None:
        assert self.engine._calculate_reliability("https://www.bmj.com/content") == 0.85

    def test_unknown_domain_baseline(self) -> None:
        assert self.engine._calculate_reliability("https://example.com/page") == 0.60

    def test_known_high_reliability_domain(self) -> None:
        assert self.engine._calculate_reliability("https://pubmed.ncbi.nlm.nih.gov/123/") == 0.95

    def test_radiopaedia(self) -> None:
        assert self.engine._calculate_reliability("https://radiopaedia.org/cases/123") == 0.85


# ---------------------------------------------------------------------------
# _filter_results — dedup, quality, medical focus (lines 765-793)
# ---------------------------------------------------------------------------


def _make_result(
    title: str = "Valid Medical Title Here",
    url: str = "https://pubmed.ncbi.nlm.nih.gov/1/",
    medical_relevance: float = 0.9,
    **kwargs: Any,
) -> SearchResult:
    defaults = {
        "content": "content",
        "snippet": "snippet",
        "source": "pubmed",
        "reliability_score": 0.9,
        "content_type": "article",
        "extracted_entities": (),
        **kwargs,
    }
    return SearchResult(title=title, url=url, medical_relevance=medical_relevance, **defaults)


class TestFilterResults:
    def setup_method(self) -> None:
        self.manager = WebSearchManager()

    def test_dedup_by_url(self) -> None:
        r1 = _make_result(
            title="Brain MRI Analysis Study A", url="https://pubmed.ncbi.nlm.nih.gov/1/"
        )
        r2 = _make_result(
            title="Brain MRI Analysis Study B", url="https://pubmed.ncbi.nlm.nih.gov/1/"
        )
        filtered = self.manager._filter_results([r1, r2], medical_focus=False)
        assert len(filtered) == 1
        assert filtered[0].title == "Brain MRI Analysis Study A"

    def test_dedup_by_title(self) -> None:
        r1 = _make_result(title="Same Brain MRI Title Here", url="https://example.com/1")
        r2 = _make_result(title="Same Brain MRI Title Here", url="https://example.com/2")
        filtered = self.manager._filter_results([r1, r2], medical_focus=False)
        assert len(filtered) == 1

    def test_short_title_filtered(self) -> None:
        r = _make_result(title="Short", url="https://example.com/1")
        filtered = self.manager._filter_results([r], medical_focus=False)
        assert len(filtered) == 0

    def test_empty_title_filtered(self) -> None:
        r = _make_result(title="", url="https://example.com/1")
        filtered = self.manager._filter_results([r], medical_focus=False)
        assert len(filtered) == 0

    def test_bad_url_filtered(self) -> None:
        r = _make_result(url="ftp://bad-protocol.com/file")
        filtered = self.manager._filter_results([r], medical_focus=False)
        assert len(filtered) == 0

    def test_empty_url_filtered(self) -> None:
        r = _make_result(url="")
        filtered = self.manager._filter_results([r], medical_focus=False)
        assert len(filtered) == 0

    def test_low_medical_relevance_filtered_when_focused(self) -> None:
        r = _make_result(medical_relevance=0.1)
        filtered = self.manager._filter_results([r], medical_focus=True)
        assert len(filtered) == 0

    def test_low_medical_relevance_kept_when_not_focused(self) -> None:
        r = _make_result(medical_relevance=0.1)
        filtered = self.manager._filter_results([r], medical_focus=False)
        assert len(filtered) == 1

    def test_url_dedup_ignores_trailing_slash(self) -> None:
        r1 = _make_result(title="Brain MRI Title One Here", url="https://example.com/a/")
        r2 = _make_result(title="Brain MRI Title Two Here", url="https://example.com/a")
        filtered = self.manager._filter_results([r1, r2], medical_focus=False)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# WebSearchManager.search — query variants, caching (lines 660-743)
# ---------------------------------------------------------------------------


class TestWebSearchManagerSearch:
    @pytest.mark.asyncio
    async def test_search_validates_empty_query(self) -> None:
        async with WebSearchManager() as manager:
            with pytest.raises(ValueError, match="non-empty"):
                await manager.search("")

    @pytest.mark.asyncio
    async def test_search_validates_search_type(self) -> None:
        async with WebSearchManager() as manager:
            with pytest.raises(ValueError, match="search_type"):
                await manager.search("brain tumor", search_type="invalid_type")

    @pytest.mark.asyncio
    async def test_search_returns_cached_results(self) -> None:
        manager = WebSearchManager()
        try:
            result = _make_result(title="Cached Result Title")
            # First call: engine returns results
            fake_engine = AsyncMock()
            fake_engine.name = "PubMed"
            fake_engine.search = AsyncMock(return_value=[result])
            manager.engines = [fake_engine]

            first_results = await manager.search("brain tumor", enhance_query=False)
            assert len(first_results) == 1
            assert first_results[0].title == "Cached Result Title"

            # Second call: should hit cache (engine not called again)
            fake_engine.search.reset_mock()
            second_results = await manager.search("brain tumor", enhance_query=False)
            assert len(second_results) == 1
            fake_engine.search.assert_not_called()
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_search_tries_shorter_variants_on_empty(self) -> None:
        """Long queries try progressively shorter variants when first returns empty."""
        manager = WebSearchManager()
        try:
            call_count = 0
            queries_seen: list[str] = []

            async def fake_search(query: str, max_results: int) -> list[SearchResult]:
                nonlocal call_count
                queries_seen.append(query)
                call_count += 1
                # Only return results on the 3rd+ attempt (shorter query)
                if call_count >= 3:
                    return [_make_result(title="Found on Short Query")]
                return []

            fake_engine = AsyncMock()
            fake_engine.name = "PubMed"
            fake_engine.search = fake_search
            manager.engines = [fake_engine]

            long_query = "glioblastoma multiforme MRI features T1 T2 contrast enhancement pattern"
            results = await manager.search(long_query, enhance_query=False)
            assert len(results) == 1
            assert results[0].title == "Found on Short Query"
            # Should have tried at least 2 variants before finding results
            assert len(queries_seen) >= 2
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_search_all_engines_fail_raises(self) -> None:
        manager = WebSearchManager()
        try:
            fake_engine = AsyncMock()
            fake_engine.name = "PubMed"
            fake_engine.search = AsyncMock(side_effect=SearchError("PubMed", "timeout"))
            manager.engines = [fake_engine]

            with pytest.raises(SearchError, match="All search engines failed"):
                await manager.search("brain tumor")
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_search_all_variants_empty_returns_empty(self) -> None:
        manager = WebSearchManager()
        try:
            fake_engine = AsyncMock()
            fake_engine.name = "PubMed"
            fake_engine.search = AsyncMock(return_value=[])
            manager.engines = [fake_engine]

            results = await manager.search("xyznonexistent query here")
            assert results == []
        finally:
            await manager.close()


# ---------------------------------------------------------------------------
# WebSearchManager context manager / close (lines 606-625)
# ---------------------------------------------------------------------------


class TestWebSearchManagerLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with WebSearchManager() as manager:
            assert len(manager.engines) > 0
        # After exit, engines should be closed (no assertion needed, just no crash)

    @pytest.mark.asyncio
    async def test_close_clears_cache(self) -> None:
        manager = WebSearchManager()
        manager._cache.set("test_key", [_make_result()])
        assert manager._cache.get("test_key") is not None
        await manager.close()
        assert manager._cache.get("test_key") is None

    def test_invalid_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown search engine"):
            WebSearchManager(engines=["google"])

    def test_max_results_per_engine_validation(self) -> None:
        with pytest.raises(ValueError, match="max_results_per_engine"):
            WebSearchManager(max_results_per_engine=0)

    def test_max_total_results_validation(self) -> None:
        with pytest.raises(ValueError, match="max_total_results"):
            WebSearchManager(max_total_results=0)


# ---------------------------------------------------------------------------
# _enhance_query (line 745-750)
# ---------------------------------------------------------------------------


class TestEnhanceQuery:
    def setup_method(self) -> None:
        self.manager = WebSearchManager()

    def test_diagnosis_enhancement(self) -> None:
        result = self.manager._enhance_query("brain tumor", "diagnosis")
        assert result == "brain tumor diagnosis imaging"

    def test_guidelines_enhancement(self) -> None:
        result = self.manager._enhance_query("stroke", "guidelines")
        assert result == "stroke clinical guidelines"

    def test_unknown_type_uses_default(self) -> None:
        result = self.manager._enhance_query("tumor", "general")
        assert result == "tumor medical imaging"
