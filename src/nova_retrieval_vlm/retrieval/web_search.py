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
from typing import Any
from urllib.parse import urlparse

import aiohttp
import requests
import urllib3
from beartype import beartype
from loguru import logger

# Suppress SSL warnings for web scraping
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class SearchResult:
    """Enhanced search result with LLM-friendly formatting."""

    title: str
    url: str
    content: str  # Full content or detailed snippet
    snippet: str  # Brief description
    source: str
    reliability_score: float  # 0.0-1.0
    publication_date: str | None = None
    author: str | None = None
    journal: str | None = None
    doi: str | None = None
    content_type: str = "unknown"  # article, guidelines, case_report, review
    medical_relevance: float = 0.0  # Medical relevance score
    extracted_entities: list[str] = field(default_factory=list)  # Medical entities found
    citation_count: int | None = None  # For academic sources
    open_access: bool = False

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


class SearchEngine:
    """Base class for search engines with common functionality."""

    def __init__(self, name: str, timeout: int = 30, max_retries: int = 3):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self._get_headers())

    def _get_headers(self) -> dict[str, str]:
        """Get standard headers for web requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search with retry logic."""
        for attempt in range(self.max_retries):
            try:
                results = await self._search_impl(query, max_results)
                if results:
                    return results

            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        logger.error(f"All search attempts failed for {self.name}")
        return []

    async def _search_impl(self, query: str, max_results: int) -> list[SearchResult]:
        """Implement actual search logic in subclasses."""
        raise NotImplementedError

    def _calculate_reliability(self, url: str, source: str) -> float:
        """Calculate source reliability score."""
        _ = source  # Unused parameter
        # High-reliability medical sources
        high_reliability = {
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "www.ncbi.nlm.nih.gov": 0.95,
            "www.cochrane.org": 0.95,
            "www.who.int": 0.95,
            "www.fda.gov": 0.90,
            "www.nice.org.uk": 0.90,
            "www.acr.org": 0.85,
            "radiopaedia.org": 0.85,
            "www.radiologyinfo.org": 0.85,
        }

        domain = urlparse(url).netloc.lower()
        if domain in high_reliability:
            return high_reliability[domain]

        # Academic sources
        if any(academic in domain for academic in ["edu", "nih.", "gov.", "ac.", "org."]):
            return 0.80

        # Medical publishers
        if any(
            publisher in domain
            for publisher in ["elsevier", "wiley", "springer", "thelancet", "jamanetwork"]
        ):
            return 0.85

        # General web sources
        return 0.60


class PubMedSearchEngine(SearchEngine):
    """Enhanced PubMed search with better error handling and metadata extraction."""

    def __init__(self, **kwargs):
        super().__init__("PubMed", **kwargs)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = os.getenv("NCBI_API_KEY")

    def _get_headers(self) -> dict[str, str]:
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _search_impl(self, query: str, max_results: int) -> list[SearchResult]:
        """Search PubMed with enhanced metadata extraction."""
        try:
            # Step 1: Search for articles
            search_url = f"{self.base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
                "tool": "nova_retrieval_vlm",
                "email": "research@example.com",
            }

            if self.api_key:
                search_params["api_key"] = self.api_key

            async with (
                aiohttp.ClientSession() as session,
                session.get(search_url, params=search_params, timeout=self.timeout) as response,
            ):
                if response.status != 200:
                    return []
                search_data = await response.json()

            if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
                return []

            pmid_list = search_data["esearchresult"]["idlist"]
            if not pmid_list:
                return []

            # Step 2: Fetch detailed information
            return await self._fetch_article_details(pmid_list)

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

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

        # Rate limiting
        await asyncio.sleep(0.5)

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(fetch_url, params=fetch_params, timeout=self.timeout) as response,
            ):
                if response.status != 200:
                    return []
                data = await response.json()

            results = []
            if "result" not in data:
                return results

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
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "pubmed"
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

        except Exception as e:
            logger.error(f"Failed to fetch article details: {e}")
            return []

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
        """Extract medical entities from text."""
        # Common medical terms and patterns
        medical_patterns = [
            r"\b(?:glioblastoma|meningioma|metastasis|lymphoma|astrocytoma|ependymoma)\b",
            r"\b(?:hyperintensity|hypointensity|enhancement|edema|hemorrhage)\b",
            r"\b(?:MRI|CT|PET|SPECT|X-ray|ultrasound)\b",
            r"\b(?:cerebral|cerebellar|brainstem|cortex|ventricle|meninges)\b",
            r"\b(?:contrast|gadolinium|FLAIR|T1|T2|DWI)\b",
            r"\b(?:radiology|neurology|neurosurgery|pathology)\b",
        ]

        entities = set()
        text_lower = text.lower()

        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities.update(matches)

        return sorted(entities)


class WebSearchManager:
    """Manager for web search operations with LLM agent integration."""

    def __init__(
        self,
        engines: list[str] | None = None,
        max_results_per_engine: int = 3,
        timeout: int = 30,
        max_total_results: int = 10,
        cache_duration: int = 300,  # 5 minutes
    ):
        """
        Initialize web search manager.

        Args:
            engines: List of search engines to use
            max_results_per_engine: Results to fetch per engine
            timeout: Request timeout in seconds
            max_total_results: Maximum total results to return
            cache_duration: Cache duration in seconds
        """
        self.max_results_per_engine = max_results_per_engine
        self.max_total_results = max_total_results
        self.cache_duration = cache_duration
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}

        # Initialize search engines
        self.engines: list[SearchEngine] = []
        if not engines:
            engines = ["pubmed"]  # Default to reliable sources only

        for engine in engines:
            if engine == "pubmed":
                self.engines.append(PubMedSearchEngine(timeout=timeout))
            else:
                logger.warning(f"Unknown search engine: {engine}")

        if not self.engines:
            logger.warning("No search engines configured, using PubMed as fallback")
            self.engines.append(PubMedSearchEngine(timeout=timeout))

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
            search_type: Type of search ('diagnosis', 'guidelines', 'research', 'anatomy', 'general')
            medical_focus: Whether to prioritize medical sources
            enhance_query: Whether to enhance the query automatically

        Returns:
            Ranked list of search results
        """
        # Check cache
        cache_key = f"{query}:{search_type}:{medical_focus}"
        if cache_key in self._cache:
            timestamp, cached_results = self._cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                logger.debug(f"Using cached results for: {query}")
                return cached_results

        # Enhance query if requested
        search_query = self._enhance_query(query, search_type) if enhance_query else query
        logger.info(f"Searching for: '{search_query}' (enhanced from: '{query}')")

        # Search across all engines
        all_results = []
        for engine in self.engines:
            try:
                results = await engine.search(search_query, self.max_results_per_engine)
                all_results.extend(results)

                # Rate limiting between engines
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Engine {engine.name} failed: {e}")
                continue

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
        """Enhance query based on search type."""
        enhancements = {
            "diagnosis": f"{query} diagnosis imaging findings radiology MRI CT",
            "guidelines": f"{query} clinical guidelines imaging protocol radiology",
            "research": f"{query} recent research MRI CT imaging findings study",
            "anatomy": f"{query} anatomy normal variants imaging radiology",
            "treatment": f"{query} treatment therapy imaging response radiology",
            "differential": f"{query} differential diagnosis imaging findings radiology",
        }

        if search_type in enhancements:
            return enhancements[search_type]
        elif any(
            term in query.lower() for term in ["diagnosis", "guidelines", "research", "anatomy"]
        ):
            # Query already seems enhanced
            return query
        else:
            # Default medical enhancement
            return f"{query} medical imaging radiology findings"

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

            # Boost for recent publications
            if result.publication_date:
                try:
                    year = int(re.search(r"\b(19|20)\d{2}\b", result.publication_date).group())
                    current_year = 2024
                    recency_boost = max(0, (current_year - year) / 10 * 0.1)
                    score += recency_boost
                except (AttributeError, ValueError, TypeError):
                    # No valid date found, skip recency boost
                    pass

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

            result.reliability_score = min(1.0, score)  # Cap at 1.0

        # Sort by score (descending)
        results.sort(key=lambda x: x.reliability_score, reverse=True)
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
                formatted.append(
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
    """Search medical literature with optimized settings."""
    manager = WebSearchManager(
        engines=["pubmed"], max_total_results=max_results, max_results_per_engine=max_results
    )
    return await manager.search(query, search_type=search_type)


def search_medical_literature_sync(
    query: str, max_results: int = 5, search_type: str = "general"
) -> list[SearchResult]:
    """Synchronous wrapper for medical literature search."""
    try:
        import asyncio

        # Try to get current event loop
        try:
            asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            # For now, return empty results to avoid blocking
            logger.warning("Called sync search from async context, returning empty results")
            return []
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(search_medical_literature(query, max_results, search_type))
    except Exception as e:
        logger.error(f"Synchronous medical literature search failed: {e}")
        return []


async def search_clinical_guidelines(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search specifically for clinical guidelines."""
    manager = WebSearchManager(max_total_results=max_results)
    return await manager.search(query, search_type="guidelines")


async def search_diagnostic_information(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search for diagnostic information and case reports."""
    manager = WebSearchManager(max_total_results=max_results)
    return await manager.search(query, search_type="diagnosis")
