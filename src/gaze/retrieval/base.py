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

import httpx
from beartype import beartype
from loguru import logger

from gaze.config import SearchConfig
from gaze.config import get_config
from gaze.exceptions import GazeError

# ---------------------------------------------------------------------------
# Credential scrubbing
# ---------------------------------------------------------------------------
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_api_field(value: object, *, max_length: int = 500) -> str:
    """Sanitize a text field from an external API response.

    Strips control characters and truncates to *max_length* to reduce
    prompt-injection surface when these values later appear in LLM
    conversations.

    Accepts any object because the caller is handling untrusted payloads: a
    field documented as a string can arrive as a number or null, and this is
    the single place best positioned to absorb that rather than raising a
    TypeError out of tool execution. Non-strings become "".
    """
    if not isinstance(value, str):
        return ""
    value = _CONTROL_CHAR_RE.sub("", value)
    if len(value) > max_length:
        value = value[:max_length]
    return value


_SENSITIVE_QS_RE = re.compile(r"(api_key=)[^&\s)\"']+")


def _sanitize_exception_message(exc: Exception) -> str:
    """Produce a safe string from *exc*, redacting sensitive URL query params.

    HTTP client exceptions may embed the full request URL (including query
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


def _is_retryable_status(status: int) -> bool:
    """Whether an HTTP status is worth retrying.

    429 (rate limited) and 5xx (server-side) are transient. Every other 4xx is
    a rejection of the request itself, so retrying only adds latency and, for
    rate-limited APIs, further requests against the quota.
    """
    return status == 429 or status >= 500


class AsyncRateLimiter:
    """Spaces out requests to at most one per *min_interval* seconds.

    A per-request gate rather than a sleep between requests: sleeping after a
    call does not bound concurrent callers, so two searches issued in the same
    turn (or two requests fired via ``gather``) can still burst past a
    provider's per-IP limit. Waiters reserve their slot in arrival order while
    holding the lock, so ordering is stable and no two requests share a slot.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    @property
    def min_interval(self) -> float:
        return self._min_interval

    async def acquire(self) -> None:
        """Block until the caller may issue its request."""
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = self._next_allowed
            self._next_allowed = now + self._min_interval


class SearchEngineError(GazeError):
    """Base exception for all search-engine errors.

    Both :class:`~gaze.retrieval.web_search.SearchError` and
    :class:`~gaze.retrieval.image_search.ImageSearchError` inherit
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
        self.timeout = httpx.Timeout(self._config.timeout_seconds)
        self.max_retries = self._config.max_retries
        self.headers = self._get_headers()
        self._session: httpx.AsyncClient | None = None

    # -- configuration -------------------------------------------------------

    @property
    def config(self) -> SearchConfig:
        """Get the search configuration."""
        return self._config

    # -- session management --------------------------------------------------

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create a reusable HTTP client for connection pooling."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._session

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._session is not None and not self._session.is_closed:
            await self._session.aclose()
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
        import gaze

        return {
            "User-Agent": f"gaze/{gaze.__version__}",
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
            except httpx.HTTPStatusError as e:
                # A 4xx is the server rejecting this request; repeating it
                # verbatim cannot help. Only 429 and 5xx are worth another go.
                status = e.response.status_code
                if not _is_retryable_status(status):
                    raise self._make_error(f"Search failed with HTTP {status}", e) from e
                last_error = e
                logger.warning(
                    f"Search attempt {attempt + 1} failed for {self.name} "
                    f"(HTTP {status}): {_sanitize_exception_message(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except (httpx.HTTPError, asyncio.TimeoutError, OSError) as e:
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
