"""Shared base class for search engines.

Deduplicates the common session management, retry logic, and configuration
handling that was duplicated between ``SearchEngine`` (web) and
``ImageSearchEngine`` (image).
"""

from __future__ import annotations

import asyncio
from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

import aiohttp
from beartype import beartype
from loguru import logger

from radiant_harness.config import SearchConfig
from radiant_harness.config import get_config
from radiant_harness.exceptions import HarnessError

# ---------------------------------------------------------------------------
# Generic type variables
# ---------------------------------------------------------------------------
ResultT = TypeVar("ResultT")
"""Type variable for search result dataclasses."""

ErrorT = TypeVar("ErrorT", bound="SearchEngineError")
"""Type variable for engine-specific error types."""


# ---------------------------------------------------------------------------
# Shared error base
# ---------------------------------------------------------------------------
class SearchEngineError(HarnessError):
    """Base exception for all search-engine errors.

    Both :class:`~radiant_harness.retrieval.web_search.SearchError` and
    :class:`~radiant_harness.retrieval.image_search.ImageSearchError` inherit
    from this class so callers can catch either flavour with a single
    ``except SearchEngineError``.
    """

    def __init__(
        self,
        engine_name: str,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        self.engine_name = engine_name
        self.original_error = original_error
        super().__init__(f"{engine_name}: {message}")


# ---------------------------------------------------------------------------
# Abstract base search engine
# ---------------------------------------------------------------------------
class BaseSearchEngine(ABC, Generic[ResultT, ErrorT]):
    """Abstract base for search engines with retry / session management.

    Subclasses must implement :meth:`_search_impl` and :meth:`_make_error`.

    Type parameters:
        ResultT: The result dataclass returned by the engine (e.g.
            ``SearchResult``, ``ImageSearchResult``).
        ErrorT: The engine-specific error type raised on failure (e.g.
            ``SearchError``, ``ImageSearchError``).
    """

    @beartype
    def __init__(
        self,
        name: str,
        config: SearchConfig | None = None,
    ) -> None:
        self._config = config or get_config().search
        self.name = name
        self.timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        self.max_retries = self._config.max_retries
        self.headers = self._get_headers()
        self._session: aiohttp.ClientSession | None = None

    # -- configuration -------------------------------------------------------

    @property
    def config(self) -> SearchConfig:
        """Get the search configuration."""
        return self._config

    # -- session management --------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session for connection pooling."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout,
            )
        return self._session

    async def close(self) -> None:
        """Close the session and release resources."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- headers (overridable) -----------------------------------------------

    @beartype
    def _get_headers(self) -> dict[str, str]:
        """Return default HTTP headers.

        Subclasses may override to supply engine-specific headers.
        """
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    # -- retry wrapper -------------------------------------------------------

    @beartype
    async def search(self, query: str, max_results: int = 5) -> list[ResultT]:
        """Search with automatic retry and exponential back-off.

        Returns:
            List of results (may be empty if no matches).

        Raises:
            SearchEngineError (or a subclass): If all retry attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self._search_impl(query, max_results)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = e
                logger.warning(f"Search attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise self._make_error("All search attempts failed", last_error)

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    async def _search_impl(self, query: str, max_results: int) -> list[ResultT]:
        """Execute the actual search.  Implemented by concrete engines."""
        ...

    @abstractmethod
    def _make_error(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> SearchEngineError:
        """Construct the engine-specific error type.

        This avoids requiring the generic ``ErrorT`` at runtime while still
        letting each concrete engine raise its own error class.
        """
        ...
