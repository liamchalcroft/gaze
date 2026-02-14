"""Tests for web search module."""

from __future__ import annotations

import pytest

from radiant_harness.config import SearchConfig
from radiant_harness.exceptions import HarnessError
from radiant_harness.retrieval.web_search import PubMedSearchEngine
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.retrieval.web_search import SearchResult
from radiant_harness.retrieval.web_search import WebSearchManager


class TestSearchConfigValidation:
    def test_invalid_values_raise(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            SearchConfig(timeout_seconds=0)
        with pytest.raises(ValueError, match="max_retries"):
            SearchConfig(max_retries=0)
        with pytest.raises(ValueError, match="rate_limit_delay_seconds"):
            SearchConfig(rate_limit_delay_seconds=-0.1)
        with pytest.raises(ValueError, match="max_results_per_engine"):
            SearchConfig(max_results_per_engine=0)
        with pytest.raises(ValueError, match="max_total_results"):
            SearchConfig(max_total_results=0)


class TestPubMedSearchEngine:
    """Tests for PubMedSearchEngine."""

    def test_rate_limit_uses_config(self) -> None:
        """Verify PubMed engine uses configured rate limit delay."""
        config = SearchConfig(rate_limit_delay_seconds=2.5)
        engine = PubMedSearchEngine(config=config)
        assert engine._rate_limit_delay == 2.5

    def test_rate_limit_default_config(self) -> None:
        """Verify PubMed engine uses default config rate limit."""
        engine = PubMedSearchEngine()
        assert engine._rate_limit_delay == engine.config.rate_limit_delay_seconds

    def test_custom_timeout(self) -> None:
        """Verify custom timeout is respected."""
        config = SearchConfig(timeout_seconds=60)
        engine = PubMedSearchEngine(config=config)
        assert engine.timeout.total == 60

    def test_no_bearer_header(self) -> None:
        """API key must NOT be sent as a Bearer header (NCBI uses api_key param)."""
        engine = PubMedSearchEngine()
        assert "Authorization" not in engine.headers

    def test_classify_content_type_practice_guideline(self) -> None:
        """PubMed returns 'Practice Guideline', not 'Guideline'."""
        engine = PubMedSearchEngine()
        assert engine._classify_content_type(["Practice Guideline"]) == "guidelines"
        assert engine._classify_content_type(["Guideline"]) == "guidelines"

    def test_classify_content_type_systematic_review(self) -> None:
        """PubMed returns 'Systematic Review' as well as 'Review'."""
        engine = PubMedSearchEngine()
        assert engine._classify_content_type(["Systematic Review"]) == "review"
        assert engine._classify_content_type(["Review"]) == "review"

    def test_classify_content_type_case_reports(self) -> None:
        engine = PubMedSearchEngine()
        assert engine._classify_content_type(["Case Reports"]) == "case_report"

    def test_classify_content_type_defaults_to_article(self) -> None:
        engine = PubMedSearchEngine()
        assert engine._classify_content_type(["Journal Article"]) == "article"
        assert engine._classify_content_type([]) == "article"

    def test_classify_content_type_priority(self) -> None:
        """Guidelines take priority over review over case_report."""
        engine = PubMedSearchEngine()
        assert engine._classify_content_type(["Review", "Practice Guideline"]) == "guidelines"
        assert engine._classify_content_type(["Case Reports", "Review"]) == "review"

    def test_parse_abstracts_xml(self) -> None:
        """Verify XML abstract parsing extracts PMID → abstract mappings."""
        engine = PubMedSearchEngine()
        xml = """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID Version="1">12345</PMID>
              <Article>
                <Abstract>
                  <AbstractText Label="BACKGROUND">Background text.</AbstractText>
                  <AbstractText Label="METHODS">Methods text.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
          <PubmedArticle>
            <MedlineCitation>
              <PMID Version="1">67890</PMID>
              <Article>
                <ArticleTitle>No abstract here</ArticleTitle>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        result = engine._parse_abstracts_xml(xml)
        assert "12345" in result
        assert "Background text." in result["12345"]
        assert "Methods text." in result["12345"]
        # Article without abstract should not appear
        assert "67890" not in result

    def test_parse_abstracts_xml_strips_inner_tags(self) -> None:
        """AbstractText may contain inline tags like <b> or <i>."""
        engine = PubMedSearchEngine()
        xml = """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>11111</PMID>
              <Article>
                <Abstract>
                  <AbstractText>Text with <b>bold</b> and <i>italic</i> tags.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        result = engine._parse_abstracts_xml(xml)
        assert result["11111"] == "Text with bold and italic tags."


class TestWebSearchManager:
    """Tests for WebSearchManager."""

    def test_invalid_engine_raises(self) -> None:
        """Test that invalid engine name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown search engine"):
            WebSearchManager(engines=["invalid_engine"])

    def test_empty_engines_raises(self) -> None:
        """Test that empty engines list still provides default."""
        manager = WebSearchManager()
        assert len(manager.engines) == 1
        assert manager.engines[0].name == "PubMed"

    def test_invalid_limits_raise(self) -> None:
        with pytest.raises(ValueError, match="max_results_per_engine"):
            WebSearchManager(max_results_per_engine=0)
        with pytest.raises(ValueError, match="max_total_results"):
            WebSearchManager(max_total_results=0)

    @pytest.mark.asyncio
    async def test_invalid_search_type_raises(self) -> None:
        """Test that invalid search type raises ValueError."""
        manager = WebSearchManager()

        with pytest.raises(ValueError, match="search_type must be one of"):
            await manager.search("test query", search_type="invalid_type")

    @pytest.mark.asyncio
    async def test_empty_query_raises(self) -> None:
        """Test that empty query raises ValueError."""
        manager = WebSearchManager()

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            await manager.search("")

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            await manager.search("   ")


class TestSearchErrorHierarchy:
    """SearchError must be part of the HarnessError hierarchy."""

    def test_search_error_is_harness_error(self) -> None:
        err = SearchError("PubMed", "test error")
        assert isinstance(err, HarnessError)

    def test_search_error_preserves_fields(self) -> None:
        cause = RuntimeError("cause")
        err = SearchError("PubMed", "search failed", original_error=cause)
        assert err.engine_name == "PubMed"
        assert err.original_error is cause
        assert "PubMed" in str(err)


class TestRankingNormalization:
    """Ranking scores must preserve differentiation, not clamp to 1.0."""

    def test_scores_normalized_not_clamped(self) -> None:
        """Two results with different features must have different scores."""
        manager = WebSearchManager()

        # Result with high relevance signals
        high = SearchResult(
            title="Brain MRI glioblastoma diagnosis imaging",
            url="https://pubmed.ncbi.nlm.nih.gov/111/",
            content="This study examines brain MRI glioblastoma features.",
            snippet="Summary",
            source="pubmed",
            reliability_score=0.95,
            publication_date="2025",
            content_type="review",
            medical_relevance=0.9,
            extracted_entities=["glioblastoma", "mri"],
            open_access=True,
        )
        # Result with lower relevance signals
        low = SearchResult(
            title="General pathology overview",
            url="https://pubmed.ncbi.nlm.nih.gov/222/",
            content="A broad overview.",
            snippet="Overview",
            source="pubmed",
            reliability_score=0.60,
            publication_date="2005",
            content_type="article",
            medical_relevance=0.5,
            extracted_entities=[],
            open_access=False,
        )

        ranked = manager._rank_results(
            [low, high], query="brain MRI glioblastoma", search_type="diagnosis"
        )

        # High-signal result must be ranked first
        assert ranked[0].url.endswith("/111/")
        assert ranked[1].url.endswith("/222/")

        # Scores must differ (clamping would make both 1.0)
        assert ranked[0].ranking_score > ranked[1].ranking_score

        # Top score must be exactly 1.0 (normalized)
        assert ranked[0].ranking_score == 1.0
        # Lower score must be strictly less
        assert ranked[1].ranking_score < 1.0

    def test_single_result_gets_score_1(self) -> None:
        """A single result should normalize to 1.0."""
        manager = WebSearchManager()
        result = SearchResult(
            title="Single result",
            url="https://pubmed.ncbi.nlm.nih.gov/333/",
            content="Content",
            snippet="Snippet",
            source="pubmed",
            reliability_score=0.7,
        )
        ranked = manager._rank_results([result], query="test", search_type="general")
        assert ranked[0].ranking_score == 1.0


class TestNCBIParamsOnAllRequests:
    """esummary and efetch must include tool and email params (NCBI requirement)."""

    @pytest.mark.asyncio
    async def test_esummary_efetch_include_tool_param(self) -> None:
        """Verify _fetch_article_details builds params with 'tool' key."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import patch

        engine = PubMedSearchEngine()
        engine.email = "test@example.com"

        captured_params: list[dict[str, str]] = []

        @asynccontextmanager
        async def mock_get(url: str, params: dict[str, str] | None = None):  # type: ignore[override]
            captured_params.append(dict(params) if params else {})
            mock_resp = AsyncMock()
            mock_resp.status = 200
            if "esummary" in url:
                mock_resp.json = AsyncMock(return_value={"result": {}})
            else:
                mock_resp.text = AsyncMock(return_value="<PubmedArticleSet></PubmedArticleSet>")
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get

        with patch.object(engine, "_get_session", new=AsyncMock(return_value=mock_session)):
            await engine._fetch_article_details(["12345"])

        # Should have 2 calls: esummary, efetch
        assert len(captured_params) >= 2
        for params in captured_params:
            assert "tool" in params, f"Missing 'tool' param in {params}"
            assert params["tool"] == "radiant_harness"
            assert "email" in params, f"Missing 'email' param in {params}"
            assert params["email"] == "test@example.com"


class TestConcurrentSummaryAndAbstracts:
    """esummary and efetch must run concurrently, not sequentially."""

    @pytest.mark.asyncio
    async def test_fetch_summary_and_abstracts_overlap(self) -> None:
        """Verify _fetch_summary and _fetch_abstracts are awaited concurrently.

        We mock both methods to record start/end times via asyncio.sleep.
        If they run concurrently, total time ≈ max(delay, delay) not sum.
        """
        import asyncio
        import time
        from unittest.mock import AsyncMock, patch

        engine = PubMedSearchEngine(config=SearchConfig(rate_limit_delay_seconds=0.0))

        call_log: list[tuple[str, float]] = []

        async def fake_fetch_summary(pmid_list: list[str]) -> dict:
            call_log.append(("summary_start", time.monotonic()))
            await asyncio.sleep(0.15)
            call_log.append(("summary_end", time.monotonic()))
            return {"result": {}}

        async def fake_fetch_abstracts(pmid_list: list[str]) -> dict[str, str]:
            call_log.append(("abstracts_start", time.monotonic()))
            await asyncio.sleep(0.15)
            call_log.append(("abstracts_end", time.monotonic()))
            return {}

        with (
            patch.object(engine, "_fetch_summary", side_effect=fake_fetch_summary),
            patch.object(engine, "_fetch_abstracts", side_effect=fake_fetch_abstracts),
        ):
            start = time.monotonic()
            await engine._fetch_article_details(["12345"])
            elapsed = time.monotonic() - start

        # Both should have started
        start_events = [e for e, _ in call_log if e.endswith("_start")]
        assert len(start_events) == 2

        # Concurrent: total ~0.15s. Sequential would be ~0.30s.
        assert elapsed < 0.25, (
            f"_fetch_article_details took {elapsed:.2f}s — "
            "esummary and efetch appear to be running sequentially"
        )


class TestXMLParsing:
    """Tests for the ET-based XML abstract parser."""

    def test_parse_malformed_xml_returns_empty(self) -> None:
        """Malformed XML should not crash, just return empty dict."""
        engine = PubMedSearchEngine()
        result = engine._parse_abstracts_xml("<broken><xml")
        assert result == {}

    def test_parse_empty_xml_returns_empty(self) -> None:
        engine = PubMedSearchEngine()
        result = engine._parse_abstracts_xml("<PubmedArticleSet></PubmedArticleSet>")
        assert result == {}

    def test_parse_xml_with_cdata_like_content(self) -> None:
        """Articles with special characters in abstracts should parse."""
        engine = PubMedSearchEngine()
        xml = """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>99999</PMID>
              <Article>
                <Abstract>
                  <AbstractText>Signal &gt; noise ratio was &lt; 0.5.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        result = engine._parse_abstracts_xml(xml)
        assert "99999" in result
        assert "Signal > noise ratio was < 0.5." in result["99999"]


class TestWordBoundaryRanking:
    """Query term matching must use word boundaries, not substring."""

    def test_ct_does_not_match_infected(self) -> None:
        """Short term 'ct' must not match 'infected' or 'detection'."""
        manager = WebSearchManager()

        result_with_ct = SearchResult(
            title="CT scan findings in stroke",
            url="https://pubmed.ncbi.nlm.nih.gov/444/",
            content="CT imaging reveals hypodensity.",
            snippet="CT",
            source="pubmed",
            reliability_score=0.95,
        )
        result_without_ct = SearchResult(
            title="Detection of infected tissue",
            url="https://pubmed.ncbi.nlm.nih.gov/555/",
            content="Infected areas were detected via microscopy.",
            snippet="Detection",
            source="pubmed",
            reliability_score=0.95,
        )

        ranked = manager._rank_results(
            [result_without_ct, result_with_ct],
            query="ct",
            search_type="general",
        )

        # The result with actual "CT" should rank higher
        assert ranked[0].url.endswith("/444/")
        assert ranked[0].ranking_score > ranked[1].ranking_score
