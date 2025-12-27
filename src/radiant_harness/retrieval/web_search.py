"""PubMed search with result formatting and reliability scoring."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import os
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import aiohttp
from beartype import beartype
from loguru import logger

from radiant_harness.cache import TTLCache
from radiant_harness.config import CacheConfig
from radiant_harness.config import RankingWeights
from radiant_harness.config import SearchConfig
from radiant_harness.config import get_config

# Note: SSL verification is enabled by default for security.
# If you encounter SSL issues with specific endpoints, handle them explicitly
# per-request rather than globally disabling SSL warnings.


@dataclass
class SearchResult:
    """Enhanced search result with LLM-friendly formatting."""

    title: str
    url: str
    content: str  # Full content or detailed snippet
    snippet: str  # Brief description
    source: str
    reliability_score: float  # 0.0-1.0 - original source reliability
    publication_date: str | None = None
    author: str | None = None
    journal: str | None = None
    doi: str | None = None
    content_type: str = "unknown"  # article, guidelines, case_report, review
    medical_relevance: float = 0.0  # Medical relevance score
    extracted_entities: list[str] = field(default_factory=list)  # Medical entities found
    citation_count: int | None = None  # For academic sources
    open_access: bool = False
    ranking_score: float = 0.0  # Composite ranking score (set during ranking)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to LLM-friendly dictionary format."""
        config = get_config()
        max_preview = config.search.max_content_preview_length
        return {
            "title": self.title,
            "url": self.url,
            "summary": self.snippet,
            "content": self.content[:max_preview] + "..."
            if len(self.content) > max_preview
            else self.content,
            "source": self.source,
            "reliability": f"{self.reliability_score:.2f}",
            "medical_relevance": f"{self.medical_relevance:.2f}",
            "publication_date": self.publication_date,
            "content_type": self.content_type,
            "key_entities": self.extracted_entities[:10],  # Top 10 entities
            "open_access": self.open_access,
        }


class SearchError(Exception):
    """Raised when a search operation fails."""

    def __init__(self, engine_name: str, message: str, original_error: Exception | None = None):
        self.engine_name = engine_name
        self.original_error = original_error
        super().__init__(f"{engine_name}: {message}")


class SearchEngine:
    """Base class for search engines with common functionality."""

    @beartype
    def __init__(
        self,
        name: str,
        config: SearchConfig | None = None,
    ) -> None:
        """Initialize search engine.

        Args:
            name: Engine identifier
            config: Search configuration. If None, uses global default.
        """
        self._config = config or get_config().search
        self.name = name
        self.timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        self.max_retries = self._config.max_retries
        self.headers = self._get_headers()
        # Reusable session for connection pooling (lazy-initialized)
        self._session: aiohttp.ClientSession | None = None

    @property
    def config(self) -> SearchConfig:
        """Get the search configuration."""
        return self._config

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

    @beartype
    def _get_headers(self) -> dict[str, str]:
        """Get standard headers for web requests."""
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    @beartype
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search with retry logic.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results

        Raises:
            SearchError: If all retry attempts fail
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                results = await self._search_impl(query, max_results)
                if results:
                    return results
                # Empty results is valid - no matches found
                return []
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
            ) as e:
                last_error = e
                logger.warning(f"Search attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        raise SearchError(self.name, "All search attempts failed", last_error)

    @beartype
    async def _search_impl(self, query: str, max_results: int) -> list[SearchResult]:
        """Implement actual search logic in subclasses.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        raise NotImplementedError

    @beartype
    def _calculate_reliability(self, url: str) -> float:
        """Calculate source reliability score based on URL domain."""
        domain = urlparse(url).netloc.lower()

        # Exact domain matches for high-reliability medical sources
        high_reliability_domains = {
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "www.ncbi.nlm.nih.gov": 0.95,
            "ncbi.nlm.nih.gov": 0.95,
            "www.cochrane.org": 0.95,
            "cochrane.org": 0.95,
            "www.who.int": 0.95,
            "who.int": 0.95,
            "www.fda.gov": 0.90,
            "fda.gov": 0.90,
            "www.nice.org.uk": 0.90,
            "nice.org.uk": 0.90,
            "www.acr.org": 0.85,
            "acr.org": 0.85,
            "radiopaedia.org": 0.85,
            "www.radiopaedia.org": 0.85,
            "www.radiologyinfo.org": 0.85,
            "radiologyinfo.org": 0.85,
        }

        if domain in high_reliability_domains:
            return high_reliability_domains[domain]

        # Check TLD-based patterns for academic sources
        # Use endswith for proper TLD matching
        if domain.endswith(".edu") or domain.endswith(".ac.uk"):
            return 0.80
        if ".nih.gov" in domain or domain.endswith(".gov"):
            return 0.80

        # Medical publisher domains (check if domain contains publisher name)
        medical_publishers = ["elsevier", "wiley", "springer", "thelancet", "jamanetwork", "bmj"]
        for publisher in medical_publishers:
            if publisher in domain:
                return 0.85

        # General web sources
        return 0.60


@functools.lru_cache(maxsize=1)
def _get_ncbi_email() -> str | None:
    """Get NCBI_EMAIL from environment, logging once if missing.

    Uses lru_cache for thread-safe one-time initialization.
    """
    email = os.getenv("NCBI_EMAIL")
    if not email:
        logger.debug("NCBI_EMAIL not set; PubMed requests will proceed without it")
    return email


class PubMedSearchEngine(SearchEngine):
    """Enhanced PubMed search with better error handling and metadata extraction."""

    # Default medical entity patterns
    MEDICAL_ENTITY_PATTERNS: list[str] = [
        r"\b(?:tumor|mass|lesion|cyst|hemorrhage|infarct|edema)\b",
        r"\b(?:hyperintensity|hypointensity|enhancement|atrophy)\b",
        r"\b(?:mri|ct|pet|x-ray|ultrasound|mammography)\b",
        r"\b(?:cerebral|cortex|ventricle|white matter|gray matter)\b",
        r"\b(?:malignant|benign|metastatic|primary)\b",
    ]

    @beartype
    def __init__(self, config: SearchConfig | None = None) -> None:
        """Initialize PubMed search engine.

        Args:
            config: Search configuration. If None, uses global default.
        """
        super().__init__("PubMed", config=config)
        self.base_url = self._config.ncbi_base_url
        self.api_key = os.getenv("NCBI_API_KEY")
        self.email = _get_ncbi_email()

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        self._rate_limit_delay = self._config.rate_limit_delay_seconds

    async def _search_impl(self, query: str, max_results: int) -> list[SearchResult]:
        """Search PubMed with enhanced metadata extraction."""
        # Step 1: Search for articles
        search_url = f"{self.base_url}esearch.fcgi"
        search_params: dict[str, str | int] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "tool": "radiant_harness",
        }
        if self.email:
            search_params["email"] = self.email

        if self.api_key:
            search_params["api_key"] = self.api_key

        session = await self._get_session()
        async with session.get(search_url, params=search_params) as response:
            if response.status != 200:
                raise SearchError(
                    self.name,
                    f"PubMed API returned status {response.status}",
                )
            search_data = await response.json()

        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
            return []

        pmid_list = search_data["esearchresult"]["idlist"]
        if not pmid_list:
            return []

        return await self._fetch_article_details(pmid_list)

    async def _fetch_article_details(self, pmid_list: list[str]) -> list[SearchResult]:
        """Fetch detailed article information from PubMed."""
        fetch_url = f"{self.base_url}esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "json",
        }

        if self.api_key:
            fetch_params["api_key"] = self.api_key

        # Rate limiting to avoid overwhelming NCBI API
        await asyncio.sleep(self._rate_limit_delay)

        session = await self._get_session()
        async with session.get(fetch_url, params=fetch_params) as response:
            if response.status != 200:
                raise SearchError(
                    self.name,
                    f"Failed to fetch article details: status {response.status}",
                )
            data = await response.json()

        if "result" not in data:
            return []

        results: list[SearchResult] = []
        for pmid in pmid_list:
            if pmid not in data["result"]:
                continue

            article = data["result"][pmid]

            # Extract metadata
            title = article.get("title", "").strip()
            authors = article.get("authors", [])
            journal = article.get("fulljournalname", "")
            pub_date = article.get("pubdate", "")
            abstract = article.get("abstract", "")
            doi = article.get("doi", "")
            article_ids = article.get("articleids", [])

            # Check for open access
            open_access = any(
                aid.get("idtype") == "pmc" for aid in article_ids if isinstance(aid, dict)
            )

            # Determine content type
            publication_types = article.get("publicationtypes", [])
            content_type = "article"  # default
            if "Review" in publication_types:
                content_type = "review"
            elif "Case Reports" in publication_types:
                content_type = "case_report"
            elif "Guideline" in publication_types:
                content_type = "guidelines"

            # Create content
            content = abstract if abstract else title

            # Extract medical entities
            entities = self._extract_medical_entities(title + " " + content)

            # Create search result
            result = SearchResult(
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                content=content,
                snippet=self._create_snippet(title, content),
                source="pubmed",
                reliability_score=self._calculate_reliability(
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ),
                publication_date=pub_date,
                author=", ".join([a.get("name", "") for a in authors[:3]]),
                journal=journal,
                doi=doi,
                content_type=content_type,
                medical_relevance=0.9,  # PubMed is highly relevant for medical
                extracted_entities=entities,
                open_access=open_access,
            )

            results.append(result)

        return results

    @beartype
    def _create_snippet(self, title: str, content: str) -> str:
        """Create a concise snippet from title and content.

        Args:
            title: Article title
            content: Full content or abstract

        Returns:
            Concise snippet, preferring sentence boundaries
        """
        if not content or content == title:
            return title

        config = get_config()
        # First configured number of characters of content, or first sentence
        snippet = content[: config.search.max_snippet_length]
        sentence_end = snippet.rfind(".")
        if sentence_end > config.search.max_snippet_length // 2:  # Prefer sentence boundaries
            snippet = snippet[: sentence_end + 1]
        elif len(snippet) < len(content):
            snippet += "..."

        return snippet.strip()

    @beartype
    def _extract_medical_entities(self, text: str) -> list[str]:
        """Extract medical entities from text using configurable patterns.

        Override MEDICAL_ENTITY_PATTERNS class attribute to customize.

        Args:
            text: Text to extract entities from

        Returns:
            Sorted list of unique medical entities found
        """
        entities: set[str] = set()
        text_lower = text.lower()

        for pattern in self.MEDICAL_ENTITY_PATTERNS:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

        return sorted(entities)


class WebSearchManager:
    """Manager for web search operations with LLM agent integration.

    Manages multiple search engines, handles caching, rate limiting,
    and result ranking for medical literature search.

    Example:
        async with WebSearchManager() as manager:
            results = await manager.search("glioblastoma MRI features")
            formatted = manager.format_for_llm(results)
    """

    SUPPORTED_ENGINES = {"pubmed"}
    ALLOWED_SEARCH_TYPES = {
        "diagnosis",
        "guidelines",
        "research",
        "anatomy",
        "general",
        "treatment",
        "differential",
    }

    # Query enhancement templates by search type - can be overridden
    QUERY_ENHANCEMENTS: dict[str, str] = {
        "diagnosis": "{query} diagnosis imaging findings",
        "guidelines": "{query} clinical guidelines protocol",
        "research": "{query} recent research study findings",
        "anatomy": "{query} anatomy normal variants imaging",
        "treatment": "{query} treatment therapy response imaging",
        "differential": "{query} differential diagnosis imaging",
    }
    DEFAULT_ENHANCEMENT = "{query} medical imaging findings"

    @beartype
    def __init__(
        self,
        engines: list[str] | None = None,
        max_results_per_engine: int | None = None,
        max_total_results: int | None = None,
        search_config: SearchConfig | None = None,
        cache_config: CacheConfig | None = None,
        ranking_weights: RankingWeights | None = None,
    ) -> None:
        """Initialize web search manager.

        Args:
            engines: List of search engines to use (default: ["pubmed"])
            max_results_per_engine: Results to fetch per engine (overrides config)
            max_total_results: Maximum total results to return (overrides config)
            search_config: Search configuration. If None, uses global default.
            cache_config: Cache configuration. If None, uses global default.
            ranking_weights: Ranking weights. If None, uses global default.

        Raises:
            ValueError: If no valid engines are specified
        """
        config = get_config()
        self._search_config = search_config or config.search
        self._cache_config = cache_config or config.cache
        self._ranking_weights = ranking_weights or config.ranking

        self.max_results_per_engine = (
            max_results_per_engine or self._search_config.max_results_per_engine
        )
        self.max_total_results = max_total_results or self._search_config.max_total_results
        self.rate_limit_delay = self._search_config.rate_limit_delay_seconds

        # Use shared TTLCache instead of manual cache management
        self._cache: TTLCache[list[SearchResult]] = TTLCache(self._cache_config)

        # Initialize search engines
        self.engines: list[SearchEngine] = []
        engines = engines or ["pubmed"]  # Default to PubMed

        for engine in engines:
            if engine == "pubmed":
                self.engines.append(PubMedSearchEngine(config=self._search_config))
            elif engine not in self.SUPPORTED_ENGINES:
                raise ValueError(
                    f"Unknown search engine: {engine}. Supported: {self.SUPPORTED_ENGINES}"
                )

        if not self.engines:
            raise ValueError("No valid search engines configured")

    async def __aenter__(self) -> WebSearchManager:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close all engine sessions and release resources."""
        for engine in self.engines:
            await engine.close()
        self._cache.clear()

    @beartype
    async def search(
        self,
        query: str,
        search_type: str = "general",
        medical_focus: bool = True,
        enhance_query: bool = True,
    ) -> list[SearchResult]:
        """Perform web search with query enhancement and result ranking.

        Args:
            query: Search query
            search_type: Type of search (diagnosis/guidelines/research/anatomy/general)
            medical_focus: Whether to prioritize medical sources
            enhance_query: Whether to enhance the query automatically

        Returns:
            Ranked list of search results

        Raises:
            ValueError: If query is empty or search_type is invalid
            SearchError: If all search engines fail
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if search_type not in self.ALLOWED_SEARCH_TYPES:
            raise ValueError(
                f"search_type must be one of {sorted(self.ALLOWED_SEARCH_TYPES)}, got '{search_type}'"
            )

        # Enhance query if requested (do this before cache key construction)
        search_query = self._enhance_query(query, search_type) if enhance_query else query

        # Cache key must capture all knobs that change result sets
        engine_names = ",".join(engine.name for engine in self.engines)
        query_hash = hashlib.sha256(search_query.encode()).hexdigest()[:8]
        cache_key = f"{query_hash}:{search_type}:{medical_focus}:{self.max_results_per_engine}:{self.max_total_results}:{engine_names}"

        # Check cache using TTLCache (handles expiration automatically)
        cached_results = self._cache.get(cache_key)
        if cached_results is not None:
            logger.debug(f"Using cached results for: {search_query}")
            return cached_results

        logger.info(f"Searching for: '{search_query}' (enhanced from: '{query}')")

        # Search across all engines
        all_results: list[SearchResult] = []
        errors: list[SearchError] = []
        for i, engine in enumerate(self.engines):
            try:
                results = await engine.search(search_query, self.max_results_per_engine)
                all_results.extend(results)
                # Rate limit between engines (skip after last engine)
                if i < len(self.engines) - 1:
                    await asyncio.sleep(self.rate_limit_delay)
            except SearchError as e:
                errors.append(e)
                logger.error(f"Engine {engine.name} failed: {e}")

        # If all engines failed, raise an error
        if errors and not all_results:
            raise SearchError(
                "WebSearchManager",
                f"All search engines failed: {[str(e) for e in errors]}",
            )

        # Filter and rank results
        filtered_results = self._filter_results(all_results, medical_focus)
        ranked_results = self._rank_results(filtered_results, query, search_type)

        # Limit results
        final_results = ranked_results[: self.max_total_results]

        # Cache results using TTLCache (handles expiration automatically)
        self._cache.set(cache_key, final_results)

        logger.info(f"Search complete: {len(final_results)} results from {len(all_results)} total")
        return final_results

    @beartype
    def _enhance_query(self, query: str, search_type: str) -> str:
        """Enhance query based on search type using configurable templates."""
        if search_type in self.QUERY_ENHANCEMENTS:
            return self.QUERY_ENHANCEMENTS[search_type].format(query=query)
        return self.DEFAULT_ENHANCEMENT.format(query=query)

    @beartype
    def _filter_results(
        self, results: list[SearchResult], medical_focus: bool
    ) -> list[SearchResult]:
        """Filter and deduplicate results.

        Args:
            results: List of search results to filter
            medical_focus: Whether to prioritize medical sources

        Returns:
            Filtered and deduplicated list of results
        """
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        filtered_results: list[SearchResult] = []

        for result in results:
            # Skip duplicates
            url_key = result.url.lower().rstrip("/")
            title_key = result.title.lower().strip()

            if url_key in seen_urls or title_key in seen_titles:
                continue

            seen_urls.add(url_key)
            seen_titles.add(title_key)

            # Basic quality checks
            if not result.title or len(result.title) < 10:
                continue

            if not result.url or not result.url.startswith(("http://", "https://")):
                continue

            # Filter by medical focus if requested
            if medical_focus and result.medical_relevance < 0.3:
                continue

            filtered_results.append(result)

        return filtered_results

    @beartype
    def _rank_results(
        self, results: list[SearchResult], query: str, search_type: str
    ) -> list[SearchResult]:
        """Rank results by relevance and quality.

        Uses configurable ranking weights to score results based on:
        - Source reliability
        - Medical relevance
        - Publication recency
        - Open access status
        - Content type matching
        - Query term matching
        - Medical entity matching

        Args:
            results: List of search results to rank
            query: Original search query
            search_type: Type of search for content type boosts

        Returns:
            Results sorted by ranking score (descending)
        """
        query_terms = query.lower().split()
        weights = self._ranking_weights

        for result in results:
            # Base score from reliability
            score = result.reliability_score

            # Boost for medical relevance
            score += result.medical_relevance * weights.medical_relevance_weight

            # Boost for recent publications (newer = higher boost)
            if result.publication_date:
                match = re.search(r"\b(19|20)\d{2}\b", result.publication_date)
                if match:
                    year = int(match.group())
                    current_year = datetime.now().year
                    years_old = current_year - year
                    # Newer papers get higher boost, decays over configured years
                    recency_boost = max(
                        0.0,
                        weights.recency_max_boost
                        * (1 - min(1.0, years_old / weights.recency_decay_years)),
                    )
                    score += recency_boost

            # Boost for open access
            if result.open_access:
                score += weights.open_access_boost

            # Content type boosts based on search type
            content_type_boosts = weights.content_type_boosts.get(search_type, {})
            content_boost = content_type_boosts.get(result.content_type, 0.0)
            score += content_boost

            # Query term matching
            title_lower = result.title.lower()
            content_lower = result.content.lower()

            title_matches = sum(1 for term in query_terms if term in title_lower)
            content_matches = sum(1 for term in query_terms if term in content_lower)

            score += (title_matches * weights.title_match_weight) + (
                content_matches * weights.content_match_weight
            )

            # Entity matching
            entity_matches = sum(
                1 for entity in result.extracted_entities if entity in query.lower()
            )
            score += entity_matches * weights.entity_match_weight

            # Store in ranking_score, preserving original reliability_score
            result.ranking_score = min(1.0, score)

        # Sort by ranking score (descending)
        results.sort(key=lambda x: x.ranking_score, reverse=True)
        return results

    def format_for_llm(
        self, results: list[SearchResult], max_content_length: int | None = None
    ) -> str:
        """Format results for LLM consumption.

        Args:
            results: Search results to format
            max_content_length: Maximum total content length

        Returns:
            Formatted string for LLM
        """
        if max_content_length is None:
            max_content_length = get_config().search.max_content_for_llm

        if not results:
            return "No search results found."

        formatted_results = ["## Search Results\n"]
        total_length = len(formatted_results[0])

        for i, result in enumerate(results, 1):
            result_dict = result.to_llm_dict()

            # Create formatted result
            formatted = f"""
### Result {i}

**Title:** {result_dict["title"]}

**Source:** {result_dict["source"]} (Reliability: {result_dict["reliability"]})
**Type:** {result_dict["content_type"]} | Medical Relevance: {result_dict["medical_relevance"]}

**Summary:** {result_dict["summary"]}

**Key Information:**
{result_dict["content"]}

**Key Entities:** {", ".join(result_dict["key_entities"])}

**URL:** {result_dict["url"]}
"""

            # Check length limit
            if total_length + len(formatted) > max_content_length:
                formatted_results.append(
                    f"\n... ({len(results) - i + 1} more results omitted due to length limit)"
                )
                break

            formatted_results.append(formatted)
            total_length += len(formatted)

        return "\n".join(formatted_results)


# Convenience functions for common use cases
async def search_medical_literature(
    query: str, max_results: int = 5, search_type: str = "general"
) -> list[SearchResult]:
    """Search medical literature with optimized settings.

    Args:
        query: Search query
        max_results: Maximum number of results
        search_type: Type of search (diagnosis, guidelines, research, anatomy, general)

    Returns:
        List of search results

    Raises:
        SearchError: If the search fails
    """
    async with WebSearchManager(
        engines=["pubmed"], max_total_results=max_results, max_results_per_engine=max_results
    ) as manager:
        return await manager.search(query, search_type=search_type)


async def search_clinical_guidelines(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search specifically for clinical guidelines.

    Raises:
        SearchError: If the search fails
    """
    async with WebSearchManager(max_total_results=max_results) as manager:
        return await manager.search(query, search_type="guidelines")


async def search_diagnostic_information(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search for diagnostic information and case reports.

    Raises:
        SearchError: If the search fails
    """
    async with WebSearchManager(max_total_results=max_results) as manager:
        return await manager.search(query, search_type="diagnosis")
