"""Tests for honest bot User-Agent across all search engines (PS-3).

All engines must identify themselves as ``radiant_harness/<version>``
rather than impersonating a browser.  Impersonation can violate API
terms of service and masks bot traffic from operators.
"""

from __future__ import annotations

from typing import Any

import radiant_harness
from radiant_harness.retrieval.base import BaseSearchEngine
from radiant_harness.retrieval.base import SearchEngineError
from radiant_harness.retrieval.image_search import ImageSearchResult
from radiant_harness.retrieval.image_search import OpenISearchEngine
from radiant_harness.retrieval.web_search import PubMedSearchEngine
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.retrieval.web_search import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BROWSER_STRINGS = ("Mozilla", "Chrome", "Safari", "AppleWebKit", "Gecko")


def _assert_honest_ua(ua: str) -> None:
    """Assert the User-Agent is an honest bot identifier."""
    assert "radiant_harness" in ua, f"UA must contain 'radiant_harness', got: {ua}"
    assert radiant_harness.__version__ in ua, f"UA must contain version, got: {ua}"
    for browser_str in _BROWSER_STRINGS:
        assert browser_str not in ua, f"UA must not contain '{browser_str}', got: {ua}"


def _assert_no_browser_headers(headers: dict[str, str]) -> None:
    """Assert headers don't contain browser-only fields."""
    browser_headers = ("DNT", "Upgrade-Insecure-Requests")
    for h in browser_headers:
        assert h not in headers, f"Header '{h}' is browser-specific and should not appear"


# ---------------------------------------------------------------------------
# Concrete stub for BaseSearchEngine (image variant)
# ---------------------------------------------------------------------------
class _StubImageEngine(BaseSearchEngine[ImageSearchResult, SearchEngineError]):
    async def _search_impl(self, *_args: Any, **_kwargs: Any) -> list:  # type: ignore[override]
        return []

    def _make_error(
        self, message: str, original_error: Exception | None = None
    ) -> SearchEngineError:
        return SearchEngineError("StubImage", message, original_error)


# Concrete stub for BaseSearchEngine (web variant)
class _StubSearchEngine(BaseSearchEngine[SearchResult, SearchError]):
    async def _search_impl(self, *_args: Any, **_kwargs: Any) -> list:  # type: ignore[override]
        return []

    def _make_error(self, message: str, original_error: Exception | None = None) -> SearchError:
        return SearchError("StubSearch", message, original_error)


# ---------------------------------------------------------------------------
# BaseSearchEngine (inherited by all engines)
# ---------------------------------------------------------------------------
class TestBaseSearchEngineUA:
    """BaseSearchEngine._get_headers() must use honest bot UA."""

    def test_ua_is_honest(self) -> None:
        engine = _StubImageEngine(name="test")
        headers = engine._get_headers()
        _assert_honest_ua(headers["User-Agent"])

    def test_no_browser_headers(self) -> None:
        engine = _StubImageEngine(name="test")
        headers = engine._get_headers()
        _assert_no_browser_headers(headers)


# ---------------------------------------------------------------------------
# SearchEngine (web search base)
# ---------------------------------------------------------------------------
class TestSearchEngineUA:
    """SearchEngine must inherit honest bot UA from base."""

    def test_ua_is_honest(self) -> None:
        engine = _StubSearchEngine(name="test")
        headers = engine._get_headers()
        _assert_honest_ua(headers["User-Agent"])

    def test_no_browser_headers(self) -> None:
        engine = _StubSearchEngine(name="test")
        headers = engine._get_headers()
        _assert_no_browser_headers(headers)


# ---------------------------------------------------------------------------
# PubMedSearchEngine (already honest — regression guard)
# ---------------------------------------------------------------------------
class TestPubMedSearchEngineUA:
    """PubMed must remain honest (regression guard)."""

    def test_ua_is_honest(self) -> None:
        engine = PubMedSearchEngine()
        headers = engine._get_headers()
        _assert_honest_ua(headers["User-Agent"])

    def test_no_browser_headers(self) -> None:
        engine = PubMedSearchEngine()
        headers = engine._get_headers()
        _assert_no_browser_headers(headers)


# ---------------------------------------------------------------------------
# ImageSearchEngine / OpenISearchEngine
# ---------------------------------------------------------------------------
class TestImageSearchEngineUA:
    """ImageSearchEngine must use honest bot UA (was spoofed Chrome)."""

    def test_ua_is_honest(self) -> None:
        engine = _StubImageEngine(name="test-img")
        headers = engine._get_headers()
        _assert_honest_ua(headers["User-Agent"])

    def test_no_browser_headers(self) -> None:
        engine = _StubImageEngine(name="test-img")
        headers = engine._get_headers()
        _assert_no_browser_headers(headers)


class TestOpenISearchEngineUA:
    """OpenISearchEngine (concrete) must inherit honest bot UA."""

    def test_ua_is_honest(self) -> None:
        engine = OpenISearchEngine()
        headers = engine._get_headers()
        _assert_honest_ua(headers["User-Agent"])

    def test_no_browser_headers(self) -> None:
        engine = OpenISearchEngine()
        headers = engine._get_headers()
        _assert_no_browser_headers(headers)

    def test_accept_includes_json(self) -> None:
        """Image search engines need JSON accept for API responses."""
        engine = OpenISearchEngine()
        headers = engine._get_headers()
        assert "application/json" in headers.get("Accept", "")
