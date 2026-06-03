"""PubMed search with result formatting and reliability scoring."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import aiohttp
import defusedxml.ElementTree as ET  # noqa: N817
from beartype import beartype
from defusedxml import DefusedXmlException
from loguru import logger

from gaze.cache import TTLCache
from gaze.config import CacheConfig
from gaze.config import SearchConfig
from gaze.config import get_config
from gaze.retrieval.base import BaseSearchEngine
from gaze.retrieval.base import SearchEngineError
from gaze.retrieval.base import _sanitize_api_field
from gaze.retrieval.base import _sanitize_exception_message

# Note: SSL verification is enabled by default for security.
# If you encounter SSL issues with specific endpoints, handle them explicitly
# per-request rather than globally disabling SSL warnings.

_PUBLICATION_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Ranking weights for search result scoring.
_MEDICAL_RELEVANCE_WEIGHT: float = 0.3
_RECENCY_MAX_BOOST: float = 0.15
_RECENCY_DECAY_YEARS: int = 15
_OPEN_ACCESS_BOOST: float = 0.1
_TITLE_MATCH_WEIGHT: float = 0.2
_CONTENT_MATCH_WEIGHT: float = 0.05
_ENTITY_MATCH_WEIGHT: float = 0.1
_PHRASE_MATCH_WEIGHT: float = 0.15
_CONTENT_TYPE_BOOSTS: dict[str, dict[str, float]] = {
    "diagnosis": {"guidelines": 0.3, "review": 0.2, "article": 0.1, "case_report": 0.05},
    "guidelines": {"guidelines": 0.3, "review": 0.2},
    "research": {"article": 0.2, "review": 0.1},
    "anatomy": {"review": 0.2, "article": 0.1},
    "treatment": {"guidelines": 0.3, "review": 0.2, "article": 0.1, "case_report": 0.05},
    "differential": {"review": 0.25, "guidelines": 0.2, "article": 0.1, "case_report": 0.05},
}


@dataclass(frozen=True)
class SearchResult:
    """Enhanced search result with ranking metadata."""

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
    extracted_entities: tuple[str, ...] = ()  # Medical entities found
    citation_count: int | None = None  # For academic sources
    open_access: bool = False
    ranking_score: float = 0.0  # Composite ranking score (set during ranking)

    def __post_init__(self) -> None:
        if not isinstance(self.extracted_entities, tuple):
            object.__setattr__(self, "extracted_entities", tuple(self.extracted_entities))


class SearchError(SearchEngineError):
    """Raised when a web search operation fails."""


@functools.lru_cache(maxsize=1)
def _get_ncbi_email() -> str | None:
    """Get NCBI_EMAIL from environment, logging once if missing.

    Uses lru_cache for thread-safe one-time initialization.
    """
    email = os.getenv("NCBI_EMAIL")
    if not email:
        logger.debug("NCBI_EMAIL not set; PubMed requests will proceed without it")
    return email


# Evidence-tier adjustments applied on top of domain-based reliability.
# Systematic reviews and guidelines represent higher evidence quality than
# individual case reports.  These offsets reflect the evidence hierarchy
# used in evidence-based medicine (EBM) pyramid.
EVIDENCE_TIER_ADJUSTMENTS: dict[str, float] = {
    "guidelines": 0.04,  # Highest: clinical practice guidelines
    "review": 0.02,  # Systematic reviews / meta-analyses
    "article": 0.0,  # Standard journal articles (baseline)
    "case_report": -0.05,  # Lower evidence: individual case reports
}


class PubMedSearchEngine(BaseSearchEngine[SearchResult, SearchError]):
    """Enhanced PubMed search with better error handling and metadata extraction."""

    def _make_error(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> SearchError:
        return SearchError(self.name, message, original_error)

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
        if domain.endswith(".nih.gov") or domain.endswith(".gov"):
            return 0.80

        # Medical publisher domains (check if domain contains publisher name)
        medical_publishers = ["elsevier", "wiley", "springer", "thelancet", "jamanetwork", "bmj"]
        for publisher in medical_publishers:
            if publisher in domain:
                return 0.85

        # General web sources
        return 0.60

    @beartype
    def _get_headers(self) -> dict[str, str]:
        """NCBI-compliant headers for PubMed E-utilities.

        NCBI guidelines require tools to identify themselves honestly via
        User-Agent rather than impersonating a browser.  The ``tool`` and
        ``email`` query parameters are already included in each request;
        the User-Agent header reinforces the identification.
        """
        import gaze

        ua = f"gaze/{gaze.__version__}"
        email = _get_ncbi_email()
        if email:
            ua += f" (mailto:{email})"
        return {
            "User-Agent": ua,
            "Accept": "application/json, application/xml, text/xml",
        }

    # Pre-compiled medical entity patterns — avoids re.compile overhead
    # on every call to _extract_medical_entities.
    MEDICAL_ENTITY_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(?:tumor|mass|lesion|cyst|hemorrhage|infarct|edema)\b"),
        re.compile(r"\b(?:hyperintensity|hypointensity|enhancement|atrophy)\b"),
        re.compile(r"\b(?:mri|ct|pet|x-ray|ultrasound|mammography)\b"),
        re.compile(r"\b(?:cerebral|cortex|ventricle|white matter|gray matter)\b"),
        re.compile(r"\b(?:malignant|benign|metastatic|primary)\b"),
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

        # NCBI E-utilities authenticate via the api_key query parameter,
        # not via Bearer token headers.  Do NOT inject the key into headers.
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
            "tool": "gaze",
        }
        if self.email:
            search_params["email"] = self.email

        if self.api_key:
            search_params["api_key"] = self.api_key

        session = await self._get_session()
        async with session.get(search_url, params=search_params) as response:
            # Let aiohttp raise ClientResponseError for 4xx/5xx so the
            # base-class retry wrapper can catch and retry transient failures
            # (e.g. 429, 503).  Previously this raised SearchError which
            # bypassed retry entirely.
            response.raise_for_status()
            search_data = await response.json()

        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
            return []

        pmid_list = search_data["esearchresult"]["idlist"]
        if not pmid_list:
            return []

        return await self._fetch_article_details(pmid_list)

    async def _fetch_article_details(self, pmid_list: list[str]) -> list[SearchResult]:
        """Fetch detailed article information from PubMed.

        Uses esummary for metadata and efetch for abstracts (esummary does
        not return abstract text).  The two requests are independent so we
        fire them concurrently via asyncio.gather to halve the wait time.
        """
        # Rate-limit: single delay after esearch, before the concurrent fetches.
        # Previously each fetch slept independently, but since they run via
        # asyncio.gather the sleeps overlapped and both requests fired at the
        # same instant — defeating the stagger intent.
        await asyncio.sleep(self._rate_limit_delay)

        # Run esummary and efetch concurrently — they share no data dependency
        summary_data, abstracts = await asyncio.gather(
            self._fetch_summary(pmid_list),
            self._fetch_abstracts(pmid_list),
        )

        if "result" not in summary_data:
            return []

        results: list[SearchResult] = []
        for pmid in pmid_list:
            if pmid not in summary_data["result"]:
                continue

            article = summary_data["result"][pmid]

            # Extract and sanitize metadata (defense-in-depth against
            # prompt injection via crafted PubMed records).
            title = _sanitize_api_field(article.get("title", "").strip(), max_length=300)
            authors = article.get("authors", [])
            journal = _sanitize_api_field(article.get("fulljournalname", ""), max_length=200)
            pub_date = _sanitize_api_field(article.get("pubdate", ""), max_length=30)
            doi = _sanitize_api_field(article.get("doi", ""), max_length=100)
            article_ids: list[dict[str, str]] = article.get("articleids", [])

            # Check for open access (PMC ID present → open access)
            open_access = any(
                aid.get("idtype") == "pmc" for aid in article_ids if isinstance(aid, dict)
            )

            # Determine content type from PubMed's pubtype field.
            publication_types = article.get("pubtype", [])
            content_type = self._classify_content_type(publication_types)

            # Use abstract from efetch if available, else title
            abstract = _sanitize_api_field(abstracts.get(pmid, ""), max_length=5000)
            content = abstract if abstract else title

            # Extract medical entities
            entities = self._extract_medical_entities(title + " " + content)

            # Base reliability from domain + evidence-tier adjustment
            base_reliability = self._calculate_reliability(
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            )
            tier_adj = EVIDENCE_TIER_ADJUSTMENTS.get(content_type, 0.0)
            reliability = max(0.0, min(1.0, base_reliability + tier_adj))

            # Derive medical relevance from content signals rather than
            # hardcoding.  Base 0.7 (all PubMed is medical) + entity
            # density bonus (up to 0.2) + abstract presence bonus (0.1).
            entity_bonus = min(0.2, len(entities) * 0.04)
            abstract_bonus = 0.1 if abstract else 0.0
            medical_relevance = min(1.0, 0.7 + entity_bonus + abstract_bonus)

            result = SearchResult(
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                content=content,
                snippet=self._create_snippet(title, content),
                source="pubmed",
                reliability_score=reliability,
                publication_date=pub_date,
                author=_sanitize_api_field(
                    ", ".join([a.get("name", "") for a in authors[:3]]),
                    max_length=200,
                ),
                journal=journal,
                doi=doi,
                content_type=content_type,
                medical_relevance=medical_relevance,
                extracted_entities=entities,
                open_access=open_access,
            )

            results.append(result)

        return results

    async def _fetch_summary(self, pmid_list: list[str]) -> dict[str, Any]:
        """Fetch article metadata via esummary JSON.

        Returns:
            The parsed JSON response dict (contains a ``"result"`` key on success).
        """
        ids_str = ",".join(pmid_list)
        summary_url = f"{self.base_url}esummary.fcgi"
        summary_params: dict[str, str] = {
            "db": "pubmed",
            "id": ids_str,
            "retmode": "json",
            "tool": "gaze",
        }
        if self.email:
            summary_params["email"] = self.email
        if self.api_key:
            summary_params["api_key"] = self.api_key

        session = await self._get_session()
        async with session.get(summary_url, params=summary_params) as response:
            # Let aiohttp raise ClientResponseError so transient HTTP errors
            # (429, 5xx) are retried by the base-class retry wrapper.
            response.raise_for_status()
            return await response.json()

    async def _fetch_abstracts(self, pmid_list: list[str]) -> dict[str, str]:
        """Fetch abstracts via efetch XML (esummary does not include them).

        Retries once on transient failure before degrading gracefully.

        Returns:
            Mapping of PMID → abstract text.  Missing abstracts are omitted.
        """
        fetch_url = f"{self.base_url}efetch.fcgi"
        fetch_params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": "gaze",
        }
        if self.email:
            fetch_params["email"] = self.email
        if self.api_key:
            fetch_params["api_key"] = self.api_key

        pmids_str = ",".join(pmid_list)
        session = await self._get_session()
        for attempt in range(2):
            try:
                async with session.get(fetch_url, params=fetch_params) as response:
                    response.raise_for_status()
                    xml_text = await response.text()
                return self._parse_abstracts_xml(xml_text)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == 0:
                    logger.debug(
                        f"efetch attempt 1 failed ({_sanitize_exception_message(exc)}) "
                        f"for PMIDs {pmids_str}; retrying"
                    )
                    await asyncio.sleep(1)
                    continue
                logger.warning(
                    f"efetch failed after retry ({_sanitize_exception_message(exc)}) "
                    f"for PMIDs {pmids_str}; proceeding without abstracts"
                )
                return {}
        return {}  # unreachable but satisfies type checker

    @beartype
    def _parse_abstracts_xml(self, xml_text: str) -> dict[str, str]:
        """Parse efetch PubMed XML to extract PMID → abstract mappings.

        Uses defusedxml.ElementTree for safe XML parsing (XXE protection).
        The PubMed XML structure wraps each article in <PubmedArticle> with
        a <PMID> element and <AbstractText> elements inside <Abstract>.
        """
        abstracts: dict[str, str] = {}

        try:
            root = ET.fromstring(xml_text)
        except (ET.ParseError, DefusedXmlException) as e:
            logger.warning("Failed to parse PubMed XML; proceeding without abstracts: %s", e)
            return {}

        for article in root.iter("PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                continue
            pmid = pmid_elem.text.strip()

            # Collect all AbstractText sections (structured abstracts have
            # multiple sections: Background, Methods, Results, Conclusions)
            sections: list[str] = []
            for at in article.iter("AbstractText"):
                # itertext() yields all text content, stripping child tags
                text = "".join(at.itertext()).strip()
                if text:
                    sections.append(text)

            if sections:
                abstracts[pmid] = " ".join(sections)

        return abstracts

    @beartype
    def _classify_content_type(self, publication_types: list[str]) -> str:
        """Classify PubMed article content type from pubtype strings.

        PubMed pubtype values are strings like "Practice Guideline",
        "Systematic Review", "Case Reports", "Journal Article", etc.
        We use substring matching because values can be compound
        (e.g. "Systematic Review" should still match "review").

        Args:
            publication_types: List of pubtype strings from esummary

        Returns:
            One of "guidelines", "review", "case_report", or "article"
        """
        lower_types = [pt.lower() for pt in publication_types]
        for pt in lower_types:
            if "guideline" in pt:
                return "guidelines"
        for pt in lower_types:
            if "meta-analysis" in pt or "meta analysis" in pt:
                return "review"
        for pt in lower_types:
            if "review" in pt:
                return "review"
        for pt in lower_types:
            if "case report" in pt:
                return "case_report"
        return "article"

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

        max_len = self._config.max_snippet_length
        snippet = content[:max_len]
        sentence_end = snippet.rfind(".")
        if sentence_end > max_len // 2:  # Prefer sentence boundaries
            snippet = snippet[: sentence_end + 1]
        elif len(snippet) < len(content):
            # Prefer word boundary over mid-word truncation
            space_end = snippet.rfind(" ")
            if space_end > max_len // 2:
                snippet = snippet[:space_end]
            snippet += "..."

        return snippet.strip()

    @beartype
    def _extract_medical_entities(self, text: str) -> tuple[str, ...]:
        """Extract medical entities from text using configurable patterns.

        Override MEDICAL_ENTITY_PATTERNS class attribute to customize.

        Args:
            text: Text to extract entities from

        Returns:
            Sorted tuple of unique medical entities found
        """
        entities: set[str] = set()
        text_lower = text.lower()

        for pattern in self.MEDICAL_ENTITY_PATTERNS:
            matches = pattern.findall(text_lower)
            entities.update(matches)

        return tuple(sorted(entities))


class WebSearchManager:
    """Manager for web search operations with LLM agent integration.

    Manages multiple search engines, handles caching, rate limiting,
    and result ranking for medical literature search.

    Example:
        async with WebSearchManager() as manager:
            results = await manager.search("glioblastoma MRI features")
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

    # Query enhancement templates by search type - can be overridden.
    # Limited to 2 appended terms to avoid diluting PubMed relevance.
    QUERY_ENHANCEMENTS: dict[str, str] = {
        "diagnosis": "{query} diagnosis imaging",
        "guidelines": "{query} clinical guidelines",
        "research": "{query} research findings",
        "anatomy": "{query} anatomy imaging",
        "treatment": "{query} treatment imaging",
        "differential": "{query} differential diagnosis",
    }
    DEFAULT_ENHANCEMENT = "{query} medical imaging"

    @beartype
    def __init__(
        self,
        engines: list[str] | None = None,
        max_results_per_engine: int | None = None,
        max_total_results: int | None = None,
        search_config: SearchConfig | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None:
        """Initialize web search manager.

        Args:
            engines: List of search engines to use (default: ["pubmed"])
            max_results_per_engine: Results to fetch per engine (default: 5)
            max_total_results: Maximum total results to return (default: 10)
            search_config: Search configuration. If None, uses global default.
            cache_config: Cache configuration. If None, uses global default.

        Raises:
            ValueError: If no valid engines are specified
        """
        config = get_config()
        self._search_config = search_config or config.search
        self._cache_config = cache_config or config.cache

        self.max_results_per_engine = (
            5 if max_results_per_engine is None else max_results_per_engine
        )
        self.max_total_results = 10 if max_total_results is None else max_total_results
        self.rate_limit_delay = self._search_config.rate_limit_delay_seconds

        if self.max_results_per_engine < 1:
            raise ValueError(
                f"max_results_per_engine must be >= 1, got {self.max_results_per_engine}"
            )
        if self.max_total_results < 1:
            raise ValueError(f"max_total_results must be >= 1, got {self.max_total_results}")

        # Use shared TTLCache instead of manual cache management
        self._cache: TTLCache[list[SearchResult]] = TTLCache(self._cache_config)

        # Initialize search engines
        self.engines: list[PubMedSearchEngine] = []
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
                f"search_type must be one of "
                f"{sorted(self.ALLOWED_SEARCH_TYPES)}, got '{search_type}'"
            )

        # Build query variants: enhanced first, then original, then
        # progressively shorter forms.  PubMed AND-joins all terms, so
        # long queries (8+ words) frequently return zero results.
        query_variants: list[str] = []
        if enhance_query:
            query_variants.append(self._enhance_query(query, search_type))
        query_variants.append(query)

        # For long queries, add progressively shorter variants by
        # dropping trailing words.  Keeps at least 3 terms.
        words = query.split()
        if len(words) > 5:
            query_variants.append(" ".join(words[:5]))
        if len(words) > 3:
            query_variants.append(" ".join(words[:3]))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_variants: list[str] = []
        for v in query_variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)

        engine_names = ",".join(engine.name for engine in self.engines)

        for variant_idx, search_query in enumerate(unique_variants):
            # Cache key must capture all knobs that change result sets
            query_hash = hashlib.sha256(search_query.encode()).hexdigest()[:16]
            cache_key = (
                f"{query_hash}:{search_type}:{medical_focus}"
                f":{self.max_results_per_engine}"
                f":{self.max_total_results}:{engine_names}"
            )

            # Check cache using TTLCache (handles expiration automatically)
            cached_results = self._cache.get(cache_key)
            if cached_results is not None:
                logger.debug(f"Using cached results for: {search_query}")
                return cached_results

            logger.info(f"Searching for: '{search_query}' (from: '{query}')")

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

            if all_results:
                # Filter and rank results
                filtered_results = self._filter_results(all_results, medical_focus)
                ranked_results = self._rank_results(filtered_results, query, search_type)

                # Limit results
                final_results = ranked_results[: self.max_total_results]

                # Cache results using TTLCache (handles expiration automatically)
                self._cache.set(cache_key, final_results)

                if variant_idx > 0:
                    logger.info(
                        f"Search complete: {len(final_results)} results "
                        f"(found on attempt {variant_idx + 1} with shortened query)"
                    )
                else:
                    logger.info(
                        f"Search complete: {len(final_results)} results "
                        f"from {len(all_results)} total"
                    )
                return final_results

            logger.debug(f"Zero results for variant '{search_query}', trying shorter query")

        # All variants exhausted — return empty
        logger.warning(f"All query variants returned zero results for: '{query}'")
        return []

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
        query_lower = query.lower()
        query_terms = query_lower.split()
        query_term_patterns = [re.compile(r"\b" + re.escape(t) + r"\b") for t in query_terms]
        # Bigram phrase patterns for compound medical terms (e.g. "white matter")
        query_bigrams = [
            f"{query_terms[i]} {query_terms[i + 1]}" for i in range(len(query_terms) - 1)
        ]
        bigram_patterns = [re.compile(r"\b" + re.escape(bg) + r"\b") for bg in query_bigrams]
        current_year = datetime.now().year
        content_type_boosts = _CONTENT_TYPE_BOOSTS.get(search_type, {})

        # Pre-compute word-boundary patterns for query terms present in
        # result entities.  This avoids re-compiling a regex for every
        # (result, entity) pair inside the scoring loop.
        all_entities: set[str] = set()
        for result in results:
            all_entities.update(result.extracted_entities)
        entity_in_query: set[str] = {
            entity
            for entity in all_entities
            if re.search(r"\b" + re.escape(entity) + r"\b", query_lower)
        }

        # Compute raw scores for each result.
        raw_scores: list[float] = []
        for result in results:
            score = result.reliability_score

            score += result.medical_relevance * _MEDICAL_RELEVANCE_WEIGHT

            if result.publication_date:
                match = _PUBLICATION_YEAR_RE.search(result.publication_date)
                if match:
                    year = int(match.group())
                    years_old = current_year - year
                    recency_boost = max(
                        0.0,
                        _RECENCY_MAX_BOOST * (1 - min(1.0, years_old / _RECENCY_DECAY_YEARS)),
                    )
                    score += recency_boost

            if result.open_access:
                score += _OPEN_ACCESS_BOOST

            content_boost = content_type_boosts.get(result.content_type, 0.0)
            score += content_boost

            # Query term matching (word-boundary to avoid substring false positives)
            title_lower = result.title.lower()
            content_lower = result.content.lower()
            title_matches = sum(1 for pat in query_term_patterns if pat.search(title_lower))
            content_matches = sum(1 for pat in query_term_patterns if pat.search(content_lower))
            score += (title_matches * _TITLE_MATCH_WEIGHT) + (
                content_matches * _CONTENT_MATCH_WEIGHT
            )

            # Bigram phrase matching — rewards results that contain compound
            # terms together (e.g. "white matter") over results that merely
            # contain the individual words in unrelated contexts.
            phrase_matches = sum(
                1 for pat in bigram_patterns if pat.search(title_lower) or pat.search(content_lower)
            )
            score += phrase_matches * _PHRASE_MATCH_WEIGHT

            entity_matches = sum(
                1 for entity in result.extracted_entities if entity in entity_in_query
            )
            score += entity_matches * _ENTITY_MATCH_WEIGHT

            raw_scores.append(score)

        # Normalize scores to [0, 1] relative to observed maximum.
        max_score = max(raw_scores) if raw_scores else 0.0
        normalized = [s / max_score for s in raw_scores] if max_score > 0 else raw_scores

        # Build new frozen results with ranking_score set, sorted descending.
        scored = sorted(
            (
                dataclasses.replace(result, ranking_score=ns)
                for result, ns in zip(results, normalized, strict=True)
            ),
            key=lambda r: r.ranking_score,
            reverse=True,
        )
        return scored
