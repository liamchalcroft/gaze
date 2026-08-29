"""Tests for credential scrubbing in search engine exception logging (PS-1)."""

from __future__ import annotations

import httpx
import pytest

from gaze.retrieval.base import _sanitize_exception_message


class TestSanitizeExceptionMessage:
    """_sanitize_exception_message must redact api_key values."""

    def test_redacts_api_key_in_url(self) -> None:
        raw_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&api_key=SECRET123&id=1234"
        )
        request = httpx.Request("GET", raw_url)
        exc = httpx.HTTPStatusError(
            f"Server error '500 Internal Server Error' for url '{raw_url}'",
            request=request,
            response=httpx.Response(500, request=request),
        )
        sanitized = _sanitize_exception_message(exc)
        assert "SECRET123" not in sanitized
        assert "api_key=[REDACTED]" in sanitized

    def test_redacts_api_key_in_plain_string_exception(self) -> None:
        """An httpx transport error can contain a plain URL string."""
        exc = httpx.RequestError(
            "Cannot connect to https://example.com/api?api_key=mysecretkey&db=pubmed"
        )
        sanitized = _sanitize_exception_message(exc)
        assert "mysecretkey" not in sanitized
        assert "api_key=[REDACTED]" in sanitized
        # Non-sensitive params should survive
        assert "db=pubmed" in sanitized

    def test_no_api_key_unchanged(self) -> None:
        """Messages without api_key should pass through unchanged."""
        exc = TimeoutError("Connection timed out after 30s")
        sanitized = _sanitize_exception_message(exc)
        assert sanitized == "Connection timed out after 30s"

    def test_multiple_api_key_occurrences(self) -> None:
        """All occurrences of api_key should be redacted."""
        exc = RuntimeError("Tried api_key=FIRST then api_key=SECOND")
        sanitized = _sanitize_exception_message(exc)
        assert "FIRST" not in sanitized
        assert "SECOND" not in sanitized
        assert sanitized.count("api_key=[REDACTED]") == 2

    def test_api_key_at_end_of_url(self) -> None:
        """api_key at end of query string (no trailing &)."""
        exc = httpx.RequestError("Error at https://example.com?db=pubmed&api_key=endkey")
        sanitized = _sanitize_exception_message(exc)
        assert "endkey" not in sanitized
        assert "api_key=[REDACTED]" in sanitized

    def test_api_key_with_special_chars(self) -> None:
        """API keys with alphanumeric and dash/underscore characters."""
        exc = httpx.RequestError("https://example.com?api_key=abc-123_XYZ.456&other=val")
        sanitized = _sanitize_exception_message(exc)
        assert "abc-123_XYZ.456" not in sanitized
        assert "api_key=[REDACTED]" in sanitized
        assert "other=val" in sanitized

    def test_empty_exception_message(self) -> None:
        exc = RuntimeError("")
        sanitized = _sanitize_exception_message(exc)
        assert sanitized == ""


class TestSanitizationIntegration:
    """Verify the helper is wired into the retry logging path."""

    @pytest.mark.asyncio
    async def test_base_search_engine_retry_redacts_api_key(self) -> None:
        """BaseSearchEngine.search() must redact api_key in retry warnings."""
        from loguru import logger

        from gaze.config import SearchConfig
        from gaze.retrieval.web_search import PubMedSearchEngine

        engine = PubMedSearchEngine(config=SearchConfig(max_retries=1, timeout_seconds=1))

        async def _fail(query: str, max_results: int) -> list:  # type: ignore[type-arg]
            raise httpx.RequestError(
                "GET https://eutils.ncbi.nlm.nih.gov?api_key=LEAKED_KEY&db=pubmed failed"
            )

        engine._search_impl = _fail  # type: ignore[assignment]

        # Capture loguru output via a custom sink
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")

        try:
            with pytest.raises(Exception):  # noqa: B017
                await engine.search("test query")
        finally:
            logger.remove(sink_id)

        # Check that the warning log does NOT contain the raw key
        assert any("api_key=[REDACTED]" in msg for msg in captured), (
            f"Expected redacted api_key in warnings, got: {captured}"
        )
        assert not any("LEAKED_KEY" in msg for msg in captured), (
            "Raw API key leaked into log messages!"
        )
