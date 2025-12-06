"""
Enhanced Web Search Tool for LLM Agents

This module provides a production-ready web search tool designed specifically for LLM agents,
with proper error handling, result formatting, source verification, and reliability scoring.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp
from beartype import beartype
from loguru import logger

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
        return {
            "title": self.title,
            "url": self.url,
            "summary": self.snippet,
            "content": self.content[:2000] + "..." if len(self.content) > 2000 else self.content,
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

    def __init__(self, name: str, timeout: int = 30, max_retries: int = 3):
        self.name = name
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.headers = self._get_headers()
        # Reusable session for connection pooling (lazy-initialized)
        self._session: aiohttp.ClientSession | None = None

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

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search with retry logic. Raises SearchError on failure."""
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

    async def _search_impl(self, query: str, max_results: int) -> list[SearchResult]:
        """Implement actual search logic in subclasses."""
        raise NotImplementedError

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

    _email_warning_logged: bool = False

    def __init__(self, **kwargs):
        super().__init__("PubMed", **kwargs)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = os.getenv("NCBI_API_KEY")
        self.email = self._get_optional_email()

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        self._rate_limit_delay = 0.5

    @classmethod
    def _get_optional_email(cls) -> str | None:
        """Return NCBI_EMAIL if provided; log debug message once if missing."""
        email = os.getenv("NCBI_EMAIL")
        if not email and not cls._email_warning_logged:
            logger.debug("NCBI_EMAIL not set; PubMed requests will proceed without it")
            cls._email_warning_logged = True
        return email

    async def _search_impl(self, query: str, max_results: int) -> list[SearchResult]:
        """Search PubMed with enhanced metadata extraction."""
        # Step 1: Search for articles
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
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
            # No results found - this is valid, not an error
            return []

        pmid_list = search_data["esearchresult"]["idlist"]
        if not pmid_list:
            # No matching articles - valid empty result
            return []

        # Step 2: Fetch detailed information
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
            # No article details available - return empty (valid case)
            return []

        results = []
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
                aid["idtype"] == "pmc" for aid in article_ids if isinstance(aid, dict)
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

    def _create_snippet(self, title: str, content: str) -> str:
        """Create a concise snippet."""
        if not content or content == title:
            return title

        # First 200 characters of content, or first sentence
        snippet = content[:200]
        sentence_end = snippet.rfind(".")
        if sentence_end > 100:  # Prefer sentence boundaries
            snippet = snippet[: sentence_end + 1]
        elif len(snippet) < len(content):
            snippet += "..."

        return snippet.strip()

    def _extract_medical_entities(self, text: str) -> list[str]:
        """Extract medical entities from text using configurable patterns.

        Override MEDICAL_ENTITY_PATTERNS class attribute to customize.
        """
        entities = set()
        text_lower = text.lower()

        for pattern in self.MEDICAL_ENTITY_PATTERNS:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

        return sorted(entities)


class WebSearchManager:
    """Manager for web search operations with LLM agent integration."""

    SUPPORTED_ENGINES = {"pubmed"}
    MAX_CACHE_SIZE = 500  # Maximum number of cached queries
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

    def __init__(
        self,
        engines: list[str] | None = None,
        max_results_per_engine: int = 3,
        timeout: int = 30,
        max_total_results: int = 10,
        cache_duration: int = 300,  # 5 minutes
        rate_limit_delay: float = 1.0,  # seconds between engine calls
    ):
        """
        Initialize web search manager.

        Args:
            engines: List of search engines to use (default: ["pubmed"])
            max_results_per_engine: Results to fetch per engine
            timeout: Request timeout in seconds
            max_total_results: Maximum total results to return
            cache_duration: Cache duration in seconds
            rate_limit_delay: Delay between search engine calls in seconds

        Raises:
            ValueError: If no valid engines are specified
        """
        self.max_results_per_engine = max_results_per_engine
        self.max_total_results = max_total_results
        self.cache_duration = cache_duration
        self.rate_limit_delay = rate_limit_delay
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}

        # Initialize search engines
        self.engines: list[SearchEngine] = []
        engines = engines or ["pubmed"]  # Default to PubMed

        for engine in engines:
            if engine == "pubmed":
                self.engines.append(PubMedSearchEngine(timeout=timeout))
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
        exc_tb: Any,
    ) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close all engine sessions and release resources."""
        for engine in self.engines:
            await engine.close()

    def _evict_stale_cache(self) -> None:
        """Evict stale entries and enforce maximum cache size."""
        current_time = time.time()

        # First, remove expired entries
        expired_keys = [
            key for key, (timestamp, _) in self._cache.items()
            if current_time - timestamp > self.cache_duration
        ]
        for key in expired_keys:
            del self._cache[key]

        # If still over limit, evict oldest entries
        if len(self._cache) > self.MAX_CACHE_SIZE:
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])
            keys_to_remove = sorted_keys[: len(self._cache) - self.MAX_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del self._cache[key]

    @beartype
    async def search(
        self,
        query: str,
        search_type: str = "general",
        medical_focus: bool = True,
        enhance_query: bool = True,
    ) -> list[SearchResult]:
        """
        Perform web search with query enhancement and result ranking.

        Args:
            query: Search query
            search_type: Type of search (diagnosis/guidelines/research/anatomy/general)
            medical_focus: Whether to prioritize medical sources
            enhance_query: Whether to enhance the query automatically

        Returns:
            Ranked list of search results
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if search_type not in self.ALLOWED_SEARCH_TYPES:
            raise ValueError(
                f"search_type must be one of {sorted(self.ALLOWED_SEARCH_TYPES)}, got '{search_type}'"
            )

        # Evict stale cache entries periodically
        self._evict_stale_cache()

        # Enhance query if requested (do this before cache key construction)
        search_query = self._enhance_query(query, search_type) if enhance_query else query

        # Cache key must capture all knobs that change result sets
        engine_names = ",".join(engine.name for engine in self.engines)
        cache_key = (
            f"q={search_query}|orig={query}|type={search_type}|med={medical_focus}|"
            f"enh={enhance_query}|per={self.max_results_per_engine}|"
            f"total={self.max_total_results}|engines={engine_names}"
        )

        # Check cache
        if cache_key in self._cache:
            timestamp, cached_results = self._cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                logger.debug(f"Using cached results for: {search_query}")
                return cached_results

        logger.info(f"Searching for: '{search_query}' (enhanced from: '{query}')")

        # Search across all engines
        all_results = []
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

        # Cache results
        self._cache[cache_key] = (time.time(), final_results)

        logger.info(f"Search complete: {len(final_results)} results from {len(all_results)} total")
        return final_results

    def _enhance_query(self, query: str, search_type: str) -> str:
        """Enhance query based on search type using configurable templates."""
        if search_type in self.QUERY_ENHANCEMENTS:
            return self.QUERY_ENHANCEMENTS[search_type].format(query=query)

        # Check if query already contains enhancement terms
        enhancement_terms = {"diagnosis", "guidelines", "research", "anatomy", "treatment"}
        if any(term in query.lower() for term in enhancement_terms):
            return query

        return self.DEFAULT_ENHANCEMENT.format(query=query)

    def _filter_results(
        self, results: list[SearchResult], medical_focus: bool
    ) -> list[SearchResult]:
        """Filter and deduplicate results."""
        seen_urls = set()
        seen_titles = set()
        filtered_results = []

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

    def _rank_results(
        self, results: list[SearchResult], query: str, search_type: str
    ) -> list[SearchResult]:
        """Rank results by relevance and quality."""
        query_terms = query.lower().split()

        for result in results:
            # Base score from reliability
            score = result.reliability_score

            # Boost for medical relevance
            score += result.medical_relevance * 0.3

            # Boost for recent publications (newer = higher boost)
            if result.publication_date:
                match = re.search(r"\b(19|20)\d{2}\b", result.publication_date)
                if match:
                    year = int(match.group())
                    current_year = datetime.now().year
                    years_old = current_year - year
                    # Newer papers get higher boost: max 0.15 for current year, decays over 15 years
                    recency_boost = max(0.0, 0.15 * (1 - min(1.0, years_old / 15)))
                    score += recency_boost

            # Boost for open access
            if result.open_access:
                score += 0.1

            # Content type boosts based on search type
            type_boosts = {
                "diagnosis": {"case_report": 0.2, "article": 0.1},
                "guidelines": {"guidelines": 0.3, "review": 0.2},
                "research": {"article": 0.2, "review": 0.1},
                "anatomy": {"review": 0.2, "article": 0.1},
            }

            if search_type in type_boosts and result.content_type in type_boosts[search_type]:
                score += type_boosts[search_type][result.content_type]

            # Query term matching
            title_lower = result.title.lower()
            content_lower = result.content.lower()

            title_matches = sum(1 for term in query_terms if term in title_lower)
            content_matches = sum(1 for term in query_terms if term in content_lower)

            score += (title_matches * 0.2) + (content_matches * 0.05)

            # Entity matching
            entity_matches = sum(
                1 for entity in result.extracted_entities if entity in query.lower()
            )
            score += entity_matches * 0.1

            # Store in ranking_score, preserving original reliability_score
            result.ranking_score = min(1.0, score)

        # Sort by ranking score (descending)
        results.sort(key=lambda x: x.ranking_score, reverse=True)
        return results

    def format_for_llm(self, results: list[SearchResult], max_content_length: int = 5000) -> str:
        """
        Format results for LLM consumption.

        Args:
            results: Search results to format
            max_content_length: Maximum total content length

        Returns:
            Formatted string for LLM
        """
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
