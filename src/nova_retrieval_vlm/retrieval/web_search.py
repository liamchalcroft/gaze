"""
Web Search Component for Medical VLM

This module provides web search capabilities that can be integrated into the
visual multiturn pipeline, allowing the model to search for current medical
information, guidelines, and research while analyzing medical images.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests
import urllib3
from bs4 import BeautifulSoup
from loguru import logger

# Suppress SSL warnings for web scraping
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class WebSearchResult:
    """Result from web search with metadata."""

    title: str
    url: str
    snippet: str
    source: str  # 'google', 'duckduckgo', 'pubmed', 'scraped', etc.
    relevance_score: float
    medical_concepts: list[str]
    publication_date: str | None = None


class WebSearcher:
    """Web search component for medical information retrieval."""

    def __init__(
        self,
        search_engines: list[str] | None = None,
        medical_sites: list[str] | None = None,
        max_results: int = 5,
        timeout: int = 15,
    ):
        """
        Initialize web searcher.

        Args:
            search_engines: List of search engines to use
            medical_sites: List of medical websites to prioritize
            max_results: Maximum results per search
            timeout: Request timeout in seconds
        """
        self.search_engines = search_engines or ["duckduckgo_html", "pubmed"]
        self.medical_sites = medical_sites or [
            "pubmed.ncbi.nlm.nih.gov",
            "www.ncbi.nlm.nih.gov",
            "www.radiologyinfo.org",
            "www.nice.org.uk",
            "www.acr.org",
            "www.aan.com",
            "www.rcr.ac.uk",
            "radiopaedia.org",
            "mriquestions.com",
            "www.uptodate.com",
            "www.medscape.com",
            "emedicine.medscape.com",
            "www.nejm.org",
            "www.thelancet.com",
            "jamanetwork.com",
            "pubs.rsna.org",
            "link.springer.com",
            "onlinelibrary.wiley.com",
        ]
        self.max_results = max_results
        self.timeout = timeout

        # Enhanced user agent for better compatibility
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def search(self, query: str, medical_focus: bool = True) -> list[WebSearchResult]:
        """
        Perform web search across multiple engines.

        Args:
            query: Search query
            medical_focus: Whether to add medical context to the query

        Returns:
            List of web search results
        """
        results = []

        # Add medical context to query if medical_focus is True
        if medical_focus:
            # Enhance query with medical terms while keeping it natural
            # Avoid duplication if medical terms already present
            query_lower = query.lower()
            if not any(term in query_lower for term in ["medical", "radiology", "imaging"]):
                enhanced_query = f"{query} medical radiology imaging"
            else:
                enhanced_query = query
        else:
            enhanced_query = query

        logger.info(f"Starting web search for: '{enhanced_query}'")

        # Try the original query first
        results = self._try_search_engines(enhanced_query)

        # If no results and query is complex, try fallback strategies
        if not results and medical_focus:
            logger.info("No results from original query, trying fallback strategies...")
            results = self._try_fallback_searches(query)

        if not results:
            logger.warning("No results from any search engine")
            return []

        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(results)
        ranked_results = self._rank_results(unique_results, query)

        final_results = ranked_results[: self.max_results]
        logger.info(f"Returning {len(final_results)} final results")

        return final_results

    def _try_search_engines(self, query: str) -> list[WebSearchResult]:
        """Try all configured search engines with the given query."""
        results = []

        # Search across different engines
        for engine in self.search_engines:
            try:
                logger.info(f"Searching with {engine}...")

                if engine == "duckduckgo_html":
                    engine_results = self._search_duckduckgo_html(query)
                elif engine == "pubmed":
                    engine_results = self._search_pubmed(query)
                elif engine == "google_scrape":
                    engine_results = self._search_google_scrape(query)
                else:
                    logger.warning(f"Unknown search engine: {engine}")
                    continue

                if engine_results:
                    logger.info(f"Found {len(engine_results)} results from {engine}")
                    results.extend(engine_results)
                else:
                    logger.warning(f"No results from {engine}")

                # Rate limiting between engines
                time.sleep(2)

            except Exception as e:
                logger.error(f"Search failed for {engine}: {e}")
                continue

        return results

    def _try_fallback_searches(self, original_query: str) -> list[WebSearchResult]:
        """Try simplified versions of complex queries."""
        fallback_queries = self._generate_fallback_queries(original_query)

        for fallback_query in fallback_queries:
            logger.info(f"Trying fallback query: '{fallback_query}'")
            results = self._try_search_engines(fallback_query)
            if results:
                logger.info(f"Fallback query successful with {len(results)} results")
                return results

            # Small delay between fallback attempts
            time.sleep(1)

        return []

    def _generate_fallback_queries(self, query: str) -> list[str]:
        """Generate simpler versions of complex queries."""
        fallback_queries = []

        # Extract key medical terms
        important_terms = []
        medical_keywords = [
            "hyperintensity",
            "glioma",
            "metastasis",
            "abscess",
            "tumor",
            "lesion",
            "cerebellar",
            "brain",
            "mri",
            "ct",
            "flair",
            "t1",
            "t2",
            "diagnosis",
            "differential",
            "imaging",
            "radiology",
            "pathology",
            "anatomy",
        ]

        query_words = query.lower().split()
        for word in query_words:
            # Remove common connectors and extract meaningful terms
            clean_word = word.strip(".,!?()[]{}")
            if clean_word in medical_keywords:
                important_terms.append(clean_word)

        # Strategy 1: Use first few important terms
        if len(important_terms) >= 2:
            fallback_queries.append(" ".join(important_terms[:3]))

        # Strategy 2: Extract main anatomical/pathological terms
        anatomy_terms = ["cerebellar", "brain", "cerebral", "spinal"]
        pathology_terms = ["hyperintensity", "glioma", "metastasis", "abscess", "tumor", "lesion"]

        anatomy_found = [term for term in important_terms if term in anatomy_terms]
        pathology_found = [term for term in important_terms if term in pathology_terms]

        if anatomy_found and pathology_found:
            fallback_queries.append(f"{anatomy_found[0]} {pathology_found[0]}")

        # Strategy 3: Just use the most specific medical term
        if pathology_found:
            fallback_queries.append(pathology_found[0])
        elif anatomy_found:
            fallback_queries.append(anatomy_found[0])

        # Remove duplicates while preserving order
        unique_fallbacks = []
        for q in fallback_queries:
            if q and q not in unique_fallbacks:
                unique_fallbacks.append(q)

        return unique_fallbacks[:3]  # Limit to 3 fallback attempts

    def _search_duckduckgo_html(self, query: str) -> list[WebSearchResult]:
        """Search DuckDuckGo by scraping HTML results."""
        try:
            # Use DuckDuckGo HTML search
            search_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}

            # Use better headers to avoid detection
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            response = requests.get(
                search_url, params=params, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()

            if response.status_code != 200:
                logger.warning(f"DuckDuckGo returned status {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Try multiple selectors as DuckDuckGo structure changes
            selectors_to_try = [
                "div.result",  # Original selector
                'div[class*="result"]',  # Contains "result"
                "div.web-result",  # Alternative
                "article",  # Generic article tags
                "div.links_main",  # Another possible selector
            ]

            result_divs = []
            for selector in selectors_to_try:
                result_divs = soup.select(selector)
                if result_divs:
                    logger.debug(f"Found {len(result_divs)} results with selector: {selector}")
                    break

            if not result_divs:
                logger.warning("No result divs found with any selector")
                return []

            for div in result_divs[:5]:  # Limit to top 5 results
                try:
                    # Try multiple approaches to extract title and URL
                    title_link = None
                    title = ""
                    url = ""
                    snippet = ""

                    # Try different selectors for title/link
                    title_selectors = [
                        "a.result__a",
                        'a[class*="result"]',
                        "h2 a",
                        "h3 a",
                        'a[href*="http"]',
                    ]

                    for title_selector in title_selectors:
                        title_link = div.select_one(title_selector)
                        if title_link:
                            break

                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get("href", "")
                    else:
                        # Fallback: look for any link
                        all_links = div.find_all("a", href=True)
                        for link in all_links:
                            href = link.get("href", "")
                            if href.startswith("http") and not href.startswith(
                                "https://duckduckgo.com"
                            ):
                                title = link.get_text(strip=True)
                                url = href
                                break

                    # Try to extract snippet
                    snippet_selectors = [
                        "a.result__snippet",
                        "div.result__snippet",
                        "span.result__snippet",
                        'div[class*="snippet"]',
                        "p",
                    ]

                    for snippet_selector in snippet_selectors:
                        snippet_elem = div.select_one(snippet_selector)
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                            break

                    # If still no snippet, use any text content
                    if not snippet:
                        snippet = div.get_text(strip=True)[:200]  # First 200 chars

                    # Only add if we have at least title and URL
                    if title and url and url.startswith("http"):
                        results.append(
                            WebSearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="duckduckgo",
                                relevance_score=0.8,
                                medical_concepts=self._extract_medical_concepts(
                                    f"{title} {snippet}"
                                ),
                            )
                        )

                except Exception as e:
                    logger.debug(f"Error parsing result div: {e}")
                    continue

            logger.debug(f"DuckDuckGo extracted {len(results)} results")
            return results

        except requests.exceptions.Timeout:
            logger.warning("DuckDuckGo search timed out")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"DuckDuckGo request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []

    def _search_google_scrape(self, query: str) -> list[WebSearchResult]:
        """Search Google by scraping results (use carefully due to rate limits)."""
        try:
            # Use Google search URL
            search_url = "https://www.google.com/search"
            params = {"q": query, "num": 10, "hl": "en"}

            response = requests.get(
                search_url, params=params, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Find search result divs (Google's structure changes frequently)
            search_results = soup.find_all("div", class_="g")

            for result in search_results[:5]:
                try:
                    # Extract title and URL
                    title_elem = result.find("h3")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    # Find the link
                    link_elem = result.find("a")
                    if not link_elem:
                        continue

                    url = link_elem.get("href", "")

                    # Extract snippet
                    snippet_elem = result.find("span", class_="st") or result.find(
                        "div", class_="s"
                    )
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if title and url and url.startswith("http"):
                        results.append(
                            WebSearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="google",
                                relevance_score=0.9,
                                medical_concepts=self._extract_medical_concepts(
                                    f"{title} {snippet}"
                                ),
                            )
                        )

                except Exception as e:
                    logger.debug(f"Error parsing Google result: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Google scraping failed: {e}")
            return []

    def _search_pubmed(self, query: str) -> list[WebSearchResult]:
        """Search PubMed using E-utilities API."""
        try:
            # Use PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

            # Search for articles
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": 5,
                "retmode": "json",
                "sort": "relevance",
                "tool": "nova_retrieval_vlm",  # Identify our tool to NCBI
                "email": "research@example.com",  # Required by NCBI guidelines
            }

            # Add extra delay for NCBI rate limiting
            time.sleep(1)

            response = requests.get(search_url, params=search_params, timeout=self.timeout)
            response.raise_for_status()

            if response.status_code != 200:
                logger.warning(f"PubMed search returned status {response.status_code}")
                return []

            search_data = response.json()

            # Check for errors in the response
            if "error" in search_data:
                logger.warning(f"PubMed API error: {search_data['error']}")
                return []

            # Get article details
            if "esearchresult" in search_data and "idlist" in search_data["esearchresult"]:
                pmid_list = search_data["esearchresult"]["idlist"]

                if not pmid_list:
                    logger.debug("PubMed search returned empty ID list")
                    return []

                # Fetch article details
                fetch_url = f"{base_url}esummary.fcgi"
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(pmid_list),
                    "retmode": "json",
                    "tool": "nova_retrieval_vlm",
                    "email": "research@example.com",
                }

                # Add delay for rate limiting
                time.sleep(1)

                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=self.timeout)
                fetch_response.raise_for_status()

                if fetch_response.status_code != 200:
                    logger.warning(f"PubMed fetch returned status {fetch_response.status_code}")
                    return []

                fetch_data = fetch_response.json()

                # Check for errors in fetch response
                if "error" in fetch_data:
                    logger.warning(f"PubMed fetch error: {fetch_data['error']}")
                    return []

                results = []
                for pmid in pmid_list:
                    if pmid in fetch_data.get("result", {}):
                        article = fetch_data["result"][pmid]

                        title = article.get("title", "No title")
                        authors = article.get("authors", [])
                        journal = article.get("fulljournalname", "Unknown journal")
                        pub_date = article.get("pubdate", "Unknown date")

                        # Create snippet from title and journal
                        snippet = f"Journal: {journal}. Published: {pub_date}"
                        if authors:
                            author_list = authors[:3]  # First 3 authors
                            author_names = [author.get("name", "") for author in author_list]
                            snippet += f". Authors: {', '.join(filter(None, author_names))}"

                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                        results.append(
                            WebSearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="pubmed",
                                relevance_score=0.9,
                                medical_concepts=self._extract_medical_concepts(
                                    f"{title} {snippet}"
                                ),
                                publication_date=pub_date,
                            )
                        )

                return results
            else:
                logger.debug("PubMed search returned no esearchresult or idlist")
                return []

        except requests.exceptions.Timeout:
            logger.warning("PubMed search timed out")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"PubMed request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"PubMed search failed with unexpected error: {e}")
            return []

    def _extract_medical_concepts(self, text: str) -> list[str]:
        """Extract medical concepts from text using expanded terminology."""
        medical_terms = [
            # Imaging modalities
            "mri",
            "ct",
            "x-ray",
            "ultrasound",
            "pet",
            "spect",
            "mammography",
            "fluoroscopy",
            "angiography",
            "tomography",
            # General medical terms
            "radiology",
            "diagnosis",
            "prognosis",
            "treatment",
            "therapy",
            "pathology",
            "histology",
            "biopsy",
            "screening",
            # Anatomical terms
            "brain",
            "cerebral",
            "cerebellum",
            "brainstem",
            "ventricle",
            "cortex",
            "white matter",
            "gray matter",
            "csf",
            "meninges",
            "temporal",
            "frontal",
            "parietal",
            "occipital",
            # Pathological findings
            "tumor",
            "lesion",
            "mass",
            "nodule",
            "cyst",
            "edema",
            "hemorrhage",
            "infarct",
            "stroke",
            "ischemia",
            "necrosis",
            "inflammation",
            "infection",
            "abscess",
            "hematoma",
            # Specific conditions
            "glioma",
            "meningioma",
            "metastasis",
            "lymphoma",
            "adenoma",
            "hydrocephalus",
            "atrophy",
            "dementia",
            "alzheimer",
            "multiple sclerosis",
            "epilepsy",
            "seizure",
            # Clinical terms
            "contrast",
            "enhancement",
            "signal",
            "intensity",
            "artifact",
            "protocol",
            "sequence",
            "acquisition",
            "reconstruction",
        ]

        text_lower = text.lower()
        found_concepts = [term for term in medical_terms if term in text_lower]

        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in found_concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)

        return unique_concepts

    def _deduplicate_results(self, results: list[WebSearchResult]) -> list[WebSearchResult]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        unique_results = []

        for result in results:
            # Normalize URL for comparison
            url_key = result.url.lower().rstrip("/")
            title_key = result.title.lower().strip()

            if url_key not in seen_urls and title_key not in seen_titles:
                seen_urls.add(url_key)
                seen_titles.add(title_key)
                unique_results.append(result)

        return unique_results

    def _rank_results(
        self, results: list[WebSearchResult], original_query: str
    ) -> list[WebSearchResult]:
        """Rank results by relevance, medical site priority, and content quality."""
        query_terms = original_query.lower().split()

        for result in results:
            # Base score from source reliability
            if result.source == "pubmed":
                result.relevance_score = 0.9
            elif result.source == "google":
                result.relevance_score = 0.8
            elif result.source == "duckduckgo":
                result.relevance_score = 0.7

            # Boost medical sites
            if any(site in result.url.lower() for site in self.medical_sites):
                result.relevance_score *= 1.3

            # Boost based on medical concepts
            concept_boost = min(len(result.medical_concepts) * 0.1, 0.3)
            result.relevance_score += concept_boost

            # Boost based on query term overlap in title and snippet
            title_lower = result.title.lower()
            snippet_lower = result.snippet.lower()

            title_matches = sum(1 for term in query_terms if term in title_lower)
            snippet_matches = sum(1 for term in query_terms if term in snippet_lower)

            # Weight title matches more heavily
            overlap_boost = (title_matches * 0.3) + (snippet_matches * 0.1)
            result.relevance_score += overlap_boost

            # Boost for longer, more informative snippets
            if len(result.snippet) > 100:
                result.relevance_score += 0.1

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results


class MedicalWebSearcher(WebSearcher):
    """Specialized web searcher for medical information."""

    def __init__(self, **kwargs: Any):
        # Use medical-optimized search engines - prioritize PubMed
        kwargs.setdefault("search_engines", ["pubmed"])  # Start with just PubMed for reliability
        kwargs.setdefault("timeout", 20)  # Longer timeout for medical searches
        super().__init__(**kwargs)

        # Medical query templates
        self.query_templates = {
            "diagnosis": "{condition} diagnosis differential radiology imaging findings",
            "guidelines": "{condition} clinical guidelines imaging protocol recommendations",
            "research": "{condition} recent research MRI CT imaging findings literature",
            "anatomy": "{structure} anatomy MRI normal variants pathology",
            "pathology": "{condition} pathology imaging characteristics MRI CT findings",
        }

    def search_medical(
        self, query: str, search_type: str = "general", condition: str | None = None
    ) -> list[WebSearchResult]:
        """
        Perform medical-specific web search.

        Args:
            query: Base search query
            search_type: Type of medical search ('diagnosis', 'guidelines', 'research', 'anatomy', 'pathology', 'general')
            condition: Medical condition or anatomical structure

        Returns:
            List of medical web search results
        """
        # Enhance query with medical context
        if search_type in self.query_templates and condition:
            try:
                enhanced_query = self.query_templates[search_type].format(condition=condition)
                final_query = f"{query} {enhanced_query}"
            except (KeyError, ValueError) as e:
                logger.warning(f"Query template formatting failed: {e}")
                final_query = f"{query} medical radiology imaging"
        else:
            final_query = f"{query} medical imaging radiology"

        # Perform search with medical focus
        results = self.search(final_query, medical_focus=True)

        # Filter and prioritize medical sources
        medical_results = []
        general_results = []

        for result in results:
            if self._is_medical_source(result.url):
                medical_results.append(result)
            else:
                general_results.append(result)

        # Return medical sources first, then general results if needed
        final_results = medical_results
        if len(medical_results) < self.max_results:
            remaining_slots = self.max_results - len(medical_results)
            final_results.extend(general_results[:remaining_slots])

        return final_results[: self.max_results]

    def general_search(self, query: str) -> list[WebSearchResult]:
        """
        Perform general web search (non-medical focused).

        Args:
            query: Search query

        Returns:
            List of web search results
        """
        return self.search(query, medical_focus=False)

    def _is_medical_source(self, url: str) -> bool:
        """Check if URL is from a reputable medical source."""
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.medical_sites)

    def search_pubmed(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        """Direct PubMed search - wrapper for the internal method."""
        old_max = self.max_results
        self.max_results = max_results
        try:
            results = self._search_pubmed(query)
            return results
        finally:
            self.max_results = old_max

    def guidelines_search(self, query: str) -> list[WebSearchResult]:
        """
        Search for clinical guidelines and protocols.

        Args:
            query: Search query for guidelines

        Returns:
            List of web search results focused on clinical guidelines
        """
        return self.search_medical(query, search_type="guidelines")

    def research_search(self, query: str) -> list[WebSearchResult]:
        """
        Search for recent research and studies.

        Args:
            query: Search query for research

        Returns:
            List of web search results focused on recent research
        """
        return self.search_medical(query, search_type="research")

    def anatomy_search(self, query: str) -> list[WebSearchResult]:
        """
        Search for anatomical information.

        Args:
            query: Search query for anatomical information

        Returns:
            List of web search results focused on anatomy
        """
        return self.search_medical(query, search_type="anatomy")
