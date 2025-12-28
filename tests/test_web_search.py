"""Tests for web search module."""

from __future__ import annotations

import pytest

from radiant_harness.config import SearchConfig
from radiant_harness.retrieval.web_search import PubMedSearchEngine
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
        # Default config should have rate_limit_delay_seconds = 0.5
        engine = PubMedSearchEngine()
        # Should match whatever the default config specifies
        assert engine._rate_limit_delay == engine.config.rate_limit_delay_seconds

    def test_custom_timeout(self) -> None:
        """Verify custom timeout is respected."""
        config = SearchConfig(timeout_seconds=60)
        engine = PubMedSearchEngine(config=config)
        assert engine.timeout.total == 60


class TestWebSearchManager:
    """Tests for WebSearchManager."""

    def test_invalid_engine_raises(self) -> None:
        """Test that invalid engine name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown search engine"):
            WebSearchManager(engines=["invalid_engine"])

    def test_empty_engines_raises(self) -> None:
        """Test that empty engines list still provides default."""
        # Default should be pubmed
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
