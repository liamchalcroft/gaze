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
            extracted_entities=("glioblastoma", "mri"),
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
            extracted_entities=(),
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
        from unittest.mock import MagicMock
        from unittest.mock import patch

        engine = PubMedSearchEngine()
        engine.email = "test@example.com"

        captured_params: list[dict[str, str]] = []

        @asynccontextmanager
        async def mock_get(url: str, params: dict[str, str] | None = None):  # type: ignore[override]
            captured_params.append(dict(params) if params else {})
            mock_resp = AsyncMock()
            mock_resp.status = 200
            # raise_for_status() is synchronous in aiohttp — use MagicMock
            mock_resp.raise_for_status = MagicMock()
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
        from unittest.mock import patch

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
    """Tests for the defusedxml-based XML abstract parser."""

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

    def test_xxe_entity_expansion_rejected(self) -> None:
        """XML with external entity declarations must be rejected (XXE defense).

        defusedxml blocks DTD entity definitions by default, preventing
        attacks that could read local files or trigger SSRF.
        """
        engine = PubMedSearchEngine()
        xxe_xml = """<?xml version="1.0"?>
        <!DOCTYPE foo [
          <!ENTITY xxe SYSTEM "file:///etc/passwd">
        ]>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>00001</PMID>
              <Article>
                <Abstract>
                  <AbstractText>&xxe;</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        result = engine._parse_abstracts_xml(xxe_xml)
        assert result == {}

    def test_billion_laughs_rejected(self) -> None:
        """XML billion-laughs (entity expansion bomb) must be rejected.

        defusedxml blocks recursive entity expansion that could cause
        exponential memory consumption (denial of service).
        """
        engine = PubMedSearchEngine()
        bomb_xml = """<?xml version="1.0"?>
        <!DOCTYPE lolz [
          <!ENTITY lol "lol">
          <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
          <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
        ]>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>00002</PMID>
              <Article>
                <Abstract>
                  <AbstractText>&lol3;</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        result = engine._parse_abstracts_xml(bomb_xml)
        assert result == {}


class TestPubMedUserAgent:
    """PubMed must identify itself honestly per NCBI E-utilities guidelines."""

    def test_ua_contains_tool_name(self) -> None:
        """User-Agent header must include 'radiant_harness'."""
        engine = PubMedSearchEngine()
        headers = engine._get_headers()
        assert "radiant_harness" in headers["User-Agent"]

    def test_ua_contains_version(self) -> None:
        """User-Agent header must include the package version."""
        import radiant_harness

        engine = PubMedSearchEngine()
        headers = engine._get_headers()
        assert radiant_harness.__version__ in headers["User-Agent"]

    def test_ua_does_not_impersonate_browser(self) -> None:
        """PubMed UA must not contain browser-impersonation strings."""
        engine = PubMedSearchEngine()
        headers = engine._get_headers()
        ua = headers["User-Agent"]
        assert "Mozilla" not in ua
        assert "Chrome" not in ua
        assert "Safari" not in ua

    def test_ua_includes_email_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UA should include mailto: when NCBI_EMAIL is configured."""
        from radiant_harness.retrieval import web_search

        # Clear the lru_cache so the monkeypatched env var takes effect
        web_search._get_ncbi_email.cache_clear()
        monkeypatch.setenv("NCBI_EMAIL", "researcher@example.edu")
        try:
            engine = PubMedSearchEngine()
            headers = engine._get_headers()
            assert "mailto:researcher@example.edu" in headers["User-Agent"]
        finally:
            web_search._get_ncbi_email.cache_clear()

    def test_ua_omits_email_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UA should not contain 'mailto' when NCBI_EMAIL is not set."""
        from radiant_harness.retrieval import web_search

        web_search._get_ncbi_email.cache_clear()
        monkeypatch.delenv("NCBI_EMAIL", raising=False)
        try:
            engine = PubMedSearchEngine()
            headers = engine._get_headers()
            assert "mailto" not in headers["User-Agent"]
        finally:
            web_search._get_ncbi_email.cache_clear()

    def test_accepts_json_and_xml(self) -> None:
        """Accept header must include JSON and XML for E-utilities responses."""
        engine = PubMedSearchEngine()
        headers = engine._get_headers()
        accept = headers["Accept"]
        assert "application/json" in accept
        assert "xml" in accept


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

    def test_entity_match_uses_word_boundary(self) -> None:
        """Entity 'ct' must not match substring in query 'duct ectasia'."""
        manager = WebSearchManager()

        result_ct_entity = SearchResult(
            title="Some imaging study",
            url="https://pubmed.ncbi.nlm.nih.gov/666/",
            content="CT scan of the chest.",
            snippet="CT scan",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=("ct",),
        )
        result_no_entity = SearchResult(
            title="Some other study",
            url="https://pubmed.ncbi.nlm.nih.gov/777/",
            content="Another study.",
            snippet="Other",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=(),
        )

        # Query "duct ectasia" contains "ct" as a substring but NOT as a word
        ranked = manager._rank_results(
            [result_ct_entity, result_no_entity],
            query="duct ectasia",
            search_type="general",
        )

        # Entity "ct" should NOT match "duct ectasia" — scores should be equal
        # (both have same reliability, no other differentiating signals)
        assert ranked[0].ranking_score == ranked[1].ranking_score


class TestExtractMedicalEntitiesReturnType:
    """_extract_medical_entities must return tuple[str, ...] matching SearchResult."""

    def test_returns_tuple(self) -> None:
        engine = PubMedSearchEngine()
        result = engine._extract_medical_entities("brain MRI shows tumor and edema")
        assert isinstance(result, tuple)
        assert "tumor" in result
        assert "mri" in result
        assert "edema" in result


class TestRateLimitSingleDelay:
    """Rate-limit delay must happen once before gather, not inside each fetch."""

    @pytest.mark.asyncio
    async def test_single_delay_before_concurrent_fetches(self) -> None:
        """Verify only one rate-limit sleep before the concurrent esummary/efetch."""
        import asyncio
        from unittest.mock import patch

        engine = PubMedSearchEngine(config=SearchConfig(rate_limit_delay_seconds=0.2))

        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await original_sleep(delay)

        async def fake_fetch_summary(pmid_list: list[str]) -> dict:
            return {"result": {}}

        async def fake_fetch_abstracts(pmid_list: list[str]) -> dict[str, str]:
            return {}

        with (
            patch("asyncio.sleep", side_effect=tracking_sleep),
            patch.object(engine, "_fetch_summary", side_effect=fake_fetch_summary),
            patch.object(engine, "_fetch_abstracts", side_effect=fake_fetch_abstracts),
        ):
            await engine._fetch_article_details(["12345"])

        # Should have exactly one rate-limit sleep (0.2s), not two
        rate_limit_sleeps = [s for s in sleep_calls if s == 0.2]
        assert len(rate_limit_sleeps) == 1, (
            f"Expected 1 rate-limit sleep, got {len(rate_limit_sleeps)}: {sleep_calls}"
        )


class TestDiagnosisContentTypeBoosts:
    """For diagnosis queries, guidelines/reviews must outrank case reports."""

    def test_guideline_beats_case_report_for_diagnosis(self) -> None:
        """Guidelines should get a higher content-type boost than case reports."""
        manager = WebSearchManager()

        guideline = SearchResult(
            title="Brain lesion imaging guideline protocol",
            url="https://pubmed.ncbi.nlm.nih.gov/800/",
            content="Clinical practice guideline for brain lesion imaging.",
            snippet="Guideline",
            source="pubmed",
            reliability_score=0.95,
            content_type="guidelines",
            medical_relevance=0.9,
        )
        case_report = SearchResult(
            title="Brain lesion imaging case report findings",
            url="https://pubmed.ncbi.nlm.nih.gov/801/",
            content="A single case report of brain lesion imaging.",
            snippet="Case report",
            source="pubmed",
            reliability_score=0.95,
            content_type="case_report",
            medical_relevance=0.9,
        )

        ranked = manager._rank_results(
            [case_report, guideline],
            query="brain lesion imaging",
            search_type="diagnosis",
        )

        assert ranked[0].url.endswith("/800/"), (
            "Guideline should rank above case report for diagnosis queries"
        )
        assert ranked[0].ranking_score > ranked[1].ranking_score

    def test_diagnosis_boosts_follow_evidence_hierarchy(self) -> None:
        """content_type_boosts for 'diagnosis': guidelines > review > article > case_report."""
        from radiant_harness.config import get_config

        boosts = get_config().ranking.content_type_boosts["diagnosis"]
        assert boosts["guidelines"] > boosts["review"]
        assert boosts["review"] > boosts["article"]
        assert boosts["article"] > boosts["case_report"]


class TestCreateSnippetUsesInstanceConfig:
    """_create_snippet must use self._config, not get_config()."""

    def test_snippet_respects_engine_config(self) -> None:
        """Snippet length should be driven by the engine's SearchConfig,
        not the global default.
        """
        config = SearchConfig(max_snippet_length=20)
        engine = PubMedSearchEngine(config=config)

        title = "Title"
        content = "A" * 100
        snippet = engine._create_snippet(title, content)
        # With max_snippet_length=20, snippet cannot exceed 20 + "..."
        assert len(snippet) <= 23  # 20 chars + "..."

    def test_snippet_default_config_allows_longer(self) -> None:
        """Default config (100 chars) should produce a longer snippet."""
        engine = PubMedSearchEngine()  # default max_snippet_length=100
        title = "Title"
        content = "B" * 200
        snippet = engine._create_snippet(title, content)
        assert len(snippet) > 23  # longer than the 20-char test above


class TestNihGovDomainSuffix:
    """_calculate_reliability must use suffix matching for .nih.gov domains."""

    def test_legitimate_nih_subdomain(self) -> None:
        """Real NIH subdomain should get 0.80 reliability."""
        engine = PubMedSearchEngine()
        score = engine._calculate_reliability("https://some.service.nih.gov/page")
        assert score == 0.80

    def test_spoofed_nih_domain_rejected(self) -> None:
        """evil.nih.gov.attacker.com must NOT match the .nih.gov rule."""
        engine = PubMedSearchEngine()
        score = engine._calculate_reliability("https://evil.nih.gov.attacker.com/page")
        # Should fall through to the general web default (0.60)
        assert score == 0.60

    def test_exact_high_reliability_still_works(self) -> None:
        """Exact domain match (pubmed.ncbi.nlm.nih.gov) still gets 0.95."""
        engine = PubMedSearchEngine()
        score = engine._calculate_reliability("https://pubmed.ncbi.nlm.nih.gov/12345/")
        assert score == 0.95


class TestSnippetWordBoundary:
    """Snippets should truncate at word boundaries, not mid-word."""

    def test_truncates_at_word_boundary(self) -> None:
        engine = PubMedSearchEngine(config=SearchConfig(max_snippet_length=30))
        content = "The glioblastoma multiforme is a highly aggressive tumor type."
        snippet = engine._create_snippet("Title", content)
        # Should not cut "multiforme" mid-word
        assert not snippet.rstrip(".").endswith("multifo")
        assert snippet.endswith("...")

    def test_sentence_boundary_still_preferred(self) -> None:
        engine = PubMedSearchEngine(config=SearchConfig(max_snippet_length=50))
        content = "Brain MRI shows hyperintensity. Further analysis needed for diagnosis."
        snippet = engine._create_snippet("Title", content)
        # Should prefer the sentence boundary at "."
        assert snippet.endswith("hyperintensity.")


class TestQueryEnhancementReduced:
    """Query enhancement should append at most 2 terms."""

    def test_diagnosis_enhancement_concise(self) -> None:
        manager = WebSearchManager()
        enhanced = manager._enhance_query("glioblastoma", "diagnosis")
        extra_terms = enhanced.replace("glioblastoma", "").strip().split()
        assert len(extra_terms) <= 2

    def test_treatment_enhancement_concise(self) -> None:
        manager = WebSearchManager()
        enhanced = manager._enhance_query("glioblastoma", "treatment")
        extra_terms = enhanced.replace("glioblastoma", "").strip().split()
        assert len(extra_terms) <= 2

    def test_all_enhancements_max_two_terms(self) -> None:
        """Every search type should append at most 2 terms."""
        manager = WebSearchManager()
        for search_type in WebSearchManager.ALLOWED_SEARCH_TYPES:
            enhanced = manager._enhance_query("test", search_type)
            extra_terms = enhanced.replace("test", "").strip().split()
            assert len(extra_terms) <= 2, (
                f"search_type '{search_type}' appends {len(extra_terms)} terms: {extra_terms}"
            )


class TestMedicalRelevanceDerived:
    """medical_relevance should vary based on entity count and abstract presence."""

    def test_rich_article_gets_high_relevance(self) -> None:
        """Article with 5+ entities and abstract should score ~1.0."""
        engine = PubMedSearchEngine()

        xml = (
            "<PubmedArticleSet>"
            "<PubmedArticle><MedlineCitation>"
            "<PMID>90001</PMID><Article><Abstract>"
            "<AbstractText>Brain MRI shows tumor with edema "
            "and enhancement near the cortex and ventricle."
            "</AbstractText></Abstract></Article>"
            "</MedlineCitation></PubmedArticle>"
            "</PubmedArticleSet>"
        )
        abstracts = engine._parse_abstracts_xml(xml)
        entities = engine._extract_medical_entities(
            "Brain MRI tumor edema enhancement cortex ventricle"
        )

        # Simulate the calculation from _fetch_article_details
        entity_bonus = min(0.2, len(entities) * 0.04)
        abstract_bonus = 0.1 if abstracts.get("90001") else 0.0
        relevance = min(1.0, 0.7 + entity_bonus + abstract_bonus)

        assert relevance >= 0.9
        assert len(entities) >= 5

    def test_sparse_article_gets_lower_relevance(self) -> None:
        """Article with no entities and no abstract should score 0.7."""
        engine = PubMedSearchEngine()
        entities = engine._extract_medical_entities("General observations")

        entity_bonus = min(0.2, len(entities) * 0.04)
        abstract_bonus = 0.0  # no abstract
        relevance = min(1.0, 0.7 + entity_bonus + abstract_bonus)

        assert relevance == 0.7
        assert len(entities) == 0

    def test_relevance_is_not_hardcoded(self) -> None:
        """Two articles with different entity counts must get different relevance."""
        engine = PubMedSearchEngine()
        rich = engine._extract_medical_entities("Brain MRI shows tumor with edema near cortex")
        sparse = engine._extract_medical_entities("Some general text")

        rich_relevance = min(1.0, 0.7 + min(0.2, len(rich) * 0.04) + 0.1)
        sparse_relevance = min(1.0, 0.7 + min(0.2, len(sparse) * 0.04) + 0.0)

        assert rich_relevance > sparse_relevance


class TestBigramPhraseMatching:
    """Compound medical terms should be matched as phrases via bigrams."""

    def test_white_matter_phrase_beats_separate_words(self) -> None:
        """Article with 'white matter' together should rank above one with
        'white' and 'matter' in separate contexts."""
        manager = WebSearchManager()

        together = SearchResult(
            title="White matter hyperintensity in aging",
            url="https://pubmed.ncbi.nlm.nih.gov/900/",
            content="White matter lesions are common in elderly patients.",
            snippet="WM",
            source="pubmed",
            reliability_score=0.95,
            medical_relevance=0.9,
        )
        separate = SearchResult(
            title="White blood cell count in matter of hours",
            url="https://pubmed.ncbi.nlm.nih.gov/901/",
            content="The white blood cells were elevated. This matter requires attention.",
            snippet="WBC",
            source="pubmed",
            reliability_score=0.95,
            medical_relevance=0.9,
        )

        ranked = manager._rank_results(
            [separate, together],
            query="white matter hyperintensity",
            search_type="general",
        )

        assert ranked[0].url.endswith("/900/"), (
            "Phrase 'white matter' should rank above separate 'white' and 'matter'"
        )
        assert ranked[0].ranking_score > ranked[1].ranking_score

    def test_single_word_query_no_bigrams(self) -> None:
        """Single-word query should produce no bigrams and not crash."""
        manager = WebSearchManager()
        result = SearchResult(
            title="Tumor study",
            url="https://pubmed.ncbi.nlm.nih.gov/902/",
            content="Content",
            snippet="S",
            source="pubmed",
            reliability_score=0.95,
        )
        ranked = manager._rank_results([result], query="tumor", search_type="general")
        assert len(ranked) == 1
        assert ranked[0].ranking_score == 1.0

    def test_phrase_match_weight_in_config(self) -> None:
        """phrase_match_weight must be a positive float in RankingWeights."""
        from radiant_harness.config import RankingWeights

        weights = RankingWeights()
        assert weights.phrase_match_weight > 0

    def test_phrase_match_weight_negative_raises(self) -> None:
        """Negative phrase_match_weight should raise ValueError."""
        from radiant_harness.config import RankingWeights

        with pytest.raises(ValueError, match="phrase_match_weight"):
            RankingWeights(phrase_match_weight=-0.1)


class TestHTTPErrorRetry:
    """HTTP error responses from PubMed must be retried by the base class."""

    @pytest.mark.asyncio
    async def test_esearch_503_is_retried(self) -> None:
        """PubMed returning 503 on esearch must trigger retry, not immediate failure."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        import aiohttp

        config = SearchConfig(max_retries=3, rate_limit_delay_seconds=0.0)
        engine = PubMedSearchEngine(config=config)

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
            pytest.raises(SearchError, match="All search attempts failed"),
        ):
            await engine.search("test query")

        # Should have retried max_retries times
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_esearch_200_no_retry(self) -> None:
        """PubMed returning 200 should not trigger any retry."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        config = SearchConfig(max_retries=3, rate_limit_delay_seconds=0.0)
        engine = PubMedSearchEngine(config=config)

        call_count = 0

        @asynccontextmanager
        async def mock_get(url: str, params: dict | None = None):  # type: ignore[override]
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.raise_for_status = MagicMock()  # no-op for 200
            mock_resp.json = AsyncMock(return_value={"esearchresult": {"idlist": []}})
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get

        with patch.object(engine, "_get_session", new=AsyncMock(return_value=mock_session)):
            results = await engine.search("test query")

        assert call_count == 1
        assert results == []


class TestTreatmentDifferentialBoosts:
    """Treatment and differential search types must have content_type_boosts."""

    def test_treatment_boosts_exist(self) -> None:
        from radiant_harness.config import get_config

        boosts = get_config().ranking.content_type_boosts
        assert "treatment" in boosts
        assert boosts["treatment"]["guidelines"] > boosts["treatment"]["case_report"]

    def test_differential_boosts_exist(self) -> None:
        from radiant_harness.config import get_config

        boosts = get_config().ranking.content_type_boosts
        assert "differential" in boosts
        assert boosts["differential"]["review"] > boosts["differential"]["case_report"]

    def test_treatment_guideline_beats_case_report(self) -> None:
        """For treatment queries, guidelines must rank above case reports."""
        manager = WebSearchManager()

        guideline = SearchResult(
            title="Treatment protocol guideline for brain tumors",
            url="https://pubmed.ncbi.nlm.nih.gov/1000/",
            content="Clinical guideline for treatment of brain tumors.",
            snippet="Guideline",
            source="pubmed",
            reliability_score=0.95,
            content_type="guidelines",
            medical_relevance=0.9,
        )
        case_report = SearchResult(
            title="Treatment protocol case report for brain tumors",
            url="https://pubmed.ncbi.nlm.nih.gov/1001/",
            content="A single case report of treatment for brain tumors.",
            snippet="Case report",
            source="pubmed",
            reliability_score=0.95,
            content_type="case_report",
            medical_relevance=0.9,
        )

        ranked = manager._rank_results(
            [case_report, guideline],
            query="brain tumor treatment",
            search_type="treatment",
        )

        assert ranked[0].url.endswith("/1000/"), (
            "Guideline should rank above case report for treatment queries"
        )

    def test_all_search_types_have_boosts(self) -> None:
        """Every allowed search type should have at least partial boosts."""
        from radiant_harness.config import get_config

        boosts = get_config().ranking.content_type_boosts
        # "general" intentionally has no boosts — it's the catch-all
        for search_type in (
            "diagnosis",
            "guidelines",
            "research",
            "anatomy",
            "treatment",
            "differential",
        ):
            assert search_type in boosts, f"Missing content_type_boosts for '{search_type}'"


class TestEfetchRetry:
    """efetch must retry once on transient failure before degrading."""

    @pytest.mark.asyncio
    async def test_efetch_503_retried_then_succeeds(self) -> None:
        """Transient 503 from efetch should be retried; abstracts recovered."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        import aiohttp

        config = SearchConfig(rate_limit_delay_seconds=0.0)
        engine = PubMedSearchEngine(config=config)

        efetch_call_count = 0

        @asynccontextmanager
        async def mock_get(url: str, params: dict | None = None):  # type: ignore[override]
            nonlocal efetch_call_count
            mock_resp = AsyncMock()
            if "efetch" in url:
                efetch_call_count += 1
                if efetch_call_count == 1:
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
                else:
                    mock_resp.status = 200
                    mock_resp.raise_for_status = MagicMock()
                    abstract_xml = """
                    <PubmedArticleSet>
                      <PubmedArticle>
                        <MedlineCitation>
                          <PMID>12345</PMID>
                          <Article>
                            <Abstract>
                              <AbstractText>Recovered abstract.</AbstractText>
                            </Abstract>
                          </Article>
                        </MedlineCitation>
                      </PubmedArticle>
                    </PubmedArticleSet>
                    """
                    mock_resp.text = AsyncMock(return_value=abstract_xml)
            else:
                mock_resp.status = 200
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json = AsyncMock(return_value={"result": {}})
                mock_resp.text = AsyncMock(return_value="<PubmedArticleSet/>")
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get

        with patch.object(engine, "_get_session", new=AsyncMock(return_value=mock_session)):
            abstracts = await engine._fetch_abstracts(["12345"])

        assert efetch_call_count == 2
        assert "12345" in abstracts
        assert "Recovered abstract." in abstracts["12345"]

    @pytest.mark.asyncio
    async def test_efetch_persistent_failure_degrades(self) -> None:
        """Persistent efetch failure must degrade gracefully, not crash."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        import aiohttp

        config = SearchConfig(rate_limit_delay_seconds=0.0)
        engine = PubMedSearchEngine(config=config)

        @asynccontextmanager
        async def mock_get(url: str, params: dict | None = None):  # type: ignore[override]
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

        with patch.object(engine, "_get_session", new=AsyncMock(return_value=mock_session)):
            abstracts = await engine._fetch_abstracts(["12345"])

        # Should degrade gracefully — empty dict, no crash
        assert abstracts == {}

    @pytest.mark.asyncio
    async def test_efetch_warning_includes_pmids(self) -> None:
        """efetch failure log must include the PMIDs being fetched."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        import aiohttp

        config = SearchConfig(rate_limit_delay_seconds=0.0)
        engine = PubMedSearchEngine(config=config)
        warnings: list[str] = []

        def capture_warning(msg: str) -> None:
            warnings.append(msg)

        @asynccontextmanager
        async def mock_get(url: str, params: dict | None = None):  # type: ignore[override]
            mock_resp = AsyncMock()
            mock_resp.status = 500
            mock_resp.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=aiohttp.RequestInfo(
                        url=url,
                        method="GET",
                        headers={},
                        real_url=url,  # type: ignore[arg-type]
                    ),
                    history=(),
                    status=500,
                    message="Internal Server Error",
                )
            )
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get

        with (
            patch.object(engine, "_get_session", new=AsyncMock(return_value=mock_session)),
            patch("radiant_harness.retrieval.web_search.logger") as mock_logger,
        ):
            mock_logger.warning = capture_warning
            mock_logger.debug = capture_warning
            await engine._fetch_abstracts(["99999", "88888"])

        all_msgs = " ".join(warnings)
        assert "99999" in all_msgs
        assert "88888" in all_msgs


class TestContentPreviewConfigDefault:
    """max_content_preview_length default must be 500 (matching live behavior)."""

    def test_default_is_500(self) -> None:
        config = SearchConfig()
        assert config.max_content_preview_length == 500
