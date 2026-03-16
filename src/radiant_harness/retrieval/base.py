"""Shared base class for search engines.

Deduplicates the common session management, retry logic, and configuration
handling shared by ``PubMedSearchEngine`` and ``OpenISearchEngine``.
"""

from __future__ import annotations

import asyncio
import re
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
# Credential scrubbing
# ---------------------------------------------------------------------------
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_api_field(value: str, *, max_length: int = 500) -> str:
    """Sanitize a text field from an external API response.

    Strips control characters and truncates to *max_length* to reduce
    prompt-injection surface when these values later appear in LLM
    conversations.
    """
    value = _CONTROL_CHAR_RE.sub("", value)
    if len(value) > max_length:
        value = value[:max_length]
    return value


_SENSITIVE_QS_RE = re.compile(r"(api_key=)[^&\s)\"']+")


def _sanitize_exception_message(exc: Exception) -> str:
    """Produce a safe string from *exc*, redacting sensitive URL query params.

    aiohttp exceptions may embed the full request URL (including query
    parameters like ``api_key``) in their string representation.  This helper
    replaces known sensitive parameter values with ``[REDACTED]`` so that
    credentials are never written to log files.
    """
    return _SENSITIVE_QS_RE.sub(r"\1[REDACTED]", str(exc))


# ---------------------------------------------------------------------------
# Generic type variables
# ---------------------------------------------------------------------------
_ResultT = TypeVar("_ResultT")
"""Module-private type variable for search result dataclasses."""

_ErrorT = TypeVar("_ErrorT", bound="SearchEngineError")
"""Module-private type variable for engine-specific error types."""


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
class BaseSearchEngine(ABC, Generic[_ResultT, _ErrorT]):
    """Abstract base for search engines with retry / session management.

    Subclasses must implement :meth:`_search_impl` and :meth:`_make_error`.

    Type parameters:
        _ResultT: The result dataclass returned by the engine (e.g.
            ``SearchResult``, ``ImageSearchResult``).
        _ErrorT: The engine-specific error type raised on failure (e.g.
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
        """Return default HTTP headers with an honest bot User-Agent.

        Automated tools should identify themselves honestly rather than
        impersonating browsers.  This helps API operators apply appropriate
        rate-limiting and avoids violating terms of service.

        Subclasses may override to supply engine-specific headers (e.g.
        PubMed adds ``mailto:`` and version information).
        """
        import radiant_harness

        return {
            "User-Agent": f"radiant_harness/{radiant_harness.__version__}",
            "Accept": "application/json, application/xml, text/html;q=0.9, */*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    # -- retry wrapper -------------------------------------------------------

    @beartype
    async def search(self, query: str, max_results: int = 5) -> list[_ResultT]:
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
                logger.warning(
                    f"Search attempt {attempt + 1} failed for {self.name}: "
                    f"{_sanitize_exception_message(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise self._make_error("All search attempts failed", last_error)

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    async def _search_impl(self, query: str, max_results: int) -> list[_ResultT]:
        """Execute the actual search.  Implemented by concrete engines."""
        ...

    @abstractmethod
    def _make_error(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> SearchEngineError:
        """Construct the engine-specific error type.

        This avoids requiring the generic ``_ErrorT`` at runtime while still
        letting each concrete engine raise its own error class.
        """
        ...
