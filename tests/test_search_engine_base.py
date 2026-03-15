"""Tests for the shared search engine base class (PS-3 deduplication)."""

from __future__ import annotations

import pytest

from radiant_harness.exceptions import HarnessError
from radiant_harness.retrieval.base import BaseSearchEngine
from radiant_harness.retrieval.base import SearchEngineError
from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.web_search import SearchError


# ---------------------------------------------------------------------------
# SearchEngineError hierarchy
# ---------------------------------------------------------------------------
class TestSearchEngineErrorHierarchy:
    """SearchEngineError must be part of the HarnessError hierarchy."""

    def test_base_is_harness_error(self) -> None:
        err = SearchEngineError("engine", "message")
        assert isinstance(err, HarnessError)

    def test_search_error_is_search_engine_error(self) -> None:
        err = SearchError("PubMed", "fail")
        assert isinstance(err, SearchEngineError)

    def test_image_search_error_is_search_engine_error(self) -> None:
        err = ImageSearchError("Open-i", "fail")
        assert isinstance(err, SearchEngineError)

    def test_catch_both_with_single_except(self) -> None:
        """Callers can catch both error flavours with one except clause."""
        for err in (SearchError("A", "a"), ImageSearchError("B", "b")):
            with pytest.raises(SearchEngineError):
                raise err

    def test_preserves_fields(self) -> None:
        cause = RuntimeError("boom")
        err = SearchEngineError("eng", "msg", original_error=cause)
        assert err.engine_name == "eng"
        assert err.original_error is cause
        assert "eng" in str(err)


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------
class TestABCEnforcement:
    """Abstract base classes must not be instantiable."""

    def test_cannot_instantiate_base_search_engine(self) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            BaseSearchEngine(name="test")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete subclass can be created
# ---------------------------------------------------------------------------
class TestConcreteSubclass:
    """A minimal concrete subclass should work."""

    def test_concrete_web_engine_can_be_created(self) -> None:
        from radiant_harness.retrieval.web_search import PubMedSearchEngine

        engine = PubMedSearchEngine()
        assert engine.name == "PubMed"

    def test_concrete_image_engine_can_be_created(self) -> None:
        from radiant_harness.retrieval.image_search import OpenISearchEngine

        engine = OpenISearchEngine()
        assert engine.name == "Open-i"


# ---------------------------------------------------------------------------
# Retry delegates to _make_error
# ---------------------------------------------------------------------------
class TestRetryUsesConcreteError:
    """The retry wrapper must raise the engine-specific error type."""

    @pytest.mark.asyncio
    async def test_web_engine_raises_search_error(self) -> None:
        from radiant_harness.config import SearchConfig
        from radiant_harness.retrieval.web_search import PubMedSearchEngine

        engine = PubMedSearchEngine(config=SearchConfig(max_retries=1, timeout_seconds=1))

        # _search_impl will fail with a network error; retry exhaustion
        # should raise SearchError (not generic SearchEngineError).
        import aiohttp

        async def _fail(query: str, max_results: int) -> list:
            raise aiohttp.ClientError("boom")

        engine._search_impl = _fail  # type: ignore[assignment]

        with pytest.raises(SearchError):
            await engine.search("test")

    @pytest.mark.asyncio
    async def test_image_engine_raises_image_search_error(self) -> None:
        from radiant_harness.config import SearchConfig
        from radiant_harness.retrieval.image_search import OpenISearchEngine

        engine = OpenISearchEngine(config=SearchConfig(max_retries=1, timeout_seconds=1))

        import aiohttp

        async def _fail(query: str, max_results: int) -> list:
            raise aiohttp.ClientError("boom")

        engine._search_impl = _fail  # type: ignore[assignment]

        with pytest.raises(ImageSearchError):
            await engine.search("test")
