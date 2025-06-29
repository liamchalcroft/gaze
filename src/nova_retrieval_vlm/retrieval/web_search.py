"""
Web Search Component for Medical VLM

This module provides web search capabilities that can be integrated into the
visual multiturn pipeline, allowing the model to search for current medical
information, guidelines, and research while analyzing medical images.
"""

from __future__ import annotations
import requests
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin
import re
from loguru import logger
from bs4 import BeautifulSoup
import urllib3

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
    medical_concepts: List[str]
    publication_date: Optional[str] = None


class WebSearcher:
    """Web search component for medical information retrieval."""
    
    def __init__(self, 
                 search_engines: List[str] = None,
                 medical_sites: List[str] = None,
                 max_results: int = 5,
                 timeout: int = 15):
        """
        Initialize web searcher.
        
        Args:
            search_engines: List of search engines to use
            medical_sites: List of medical websites to prioritize
            max_results: Maximum results per search
            timeout: Request timeout in seconds
        """
        self.search_engines = search_engines or ['duckduckgo_html', 'pubmed']
        self.medical_sites = medical_sites or [
            'pubmed.ncbi.nlm.nih.gov',
            'www.ncbi.nlm.nih.gov',
            'www.radiologyinfo.org',
            'www.nice.org.uk',
            'www.acr.org',
            'www.aan.com',
            'www.rcr.ac.uk',
            'radiopaedia.org',
            'mriquestions.com',
            'www.uptodate.com',
            'www.medscape.com',
            'emedicine.medscape.com',
            'www.nejm.org',
            'www.thelancet.com',
            'jamanetwork.com',
            'pubs.rsna.org',
            'link.springer.com',
            'onlinelibrary.wiley.com'
        ]
        self.max_results = max_results
        self.timeout = timeout
        
        # Enhanced user agent for better compatibility
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def search(self, query: str, medical_focus: bool = True) -> List[WebSearchResult]:
        """
        Perform web search with medical focus.
        
        Args:
            query: Search query
            medical_focus: Whether to prioritize medical sites
            
        Returns:
            List of web search results
        """
        results = []
        
        # Add medical context to query if medical_focus is True
        if medical_focus:
            # Enhance query with medical terms while keeping it natural
            enhanced_query = f"{query} medical radiology imaging"
        else:
            enhanced_query = query
        
        logger.info(f"Starting web search for: '{enhanced_query}'")
        
        # Search across different engines
        for engine in self.search_engines:
            try:
                logger.info(f"Searching with {engine}...")
                
                if engine == 'duckduckgo_html':
                    engine_results = self._search_duckduckgo_html(enhanced_query)
                elif engine == 'pubmed':
                    engine_results = self._search_pubmed(enhanced_query)
                elif engine == 'google_scrape':
                    engine_results = self._search_google_scrape(enhanced_query)
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
        
        if not results:
            logger.warning("No results from any search engine")
            return []
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(results)
        ranked_results = self._rank_results(unique_results, query)
        
        final_results = ranked_results[:self.max_results]
        logger.info(f"Returning {len(final_results)} final results")
        
        return final_results
    
    def _search_duckduckgo_html(self, query: str) -> List[WebSearchResult]:
        """Search DuckDuckGo by scraping HTML results."""
        try:
            # Use DuckDuckGo HTML search
            search_url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find search result divs
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:5]:  # Limit to top 5 results
                try:
                    # Extract title and URL
                    title_link = div.find('a', class_='result__a')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_div = div.find('a', class_='result__snippet')
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                    
                    if title and url:
                        results.append(WebSearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source='duckduckgo',
                            relevance_score=0.8,
                            medical_concepts=self._extract_medical_concepts(f"{title} {snippet}")
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing result div: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []
    
    def _search_google_scrape(self, query: str) -> List[WebSearchResult]:
        """Search Google by scraping results (use carefully due to rate limits)."""
        try:
            # Use Google search URL
            search_url = f"https://www.google.com/search"
            params = {
                'q': query,
                'num': 10,
                'hl': 'en'
            }
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find search result divs (Google's structure changes frequently)
            search_results = soup.find_all('div', class_='g')
            
            for result in search_results[:5]:
                try:
                    # Extract title and URL
                    title_elem = result.find('h3')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # Find the link
                    link_elem = result.find('a')
                    if not link_elem:
                        continue
                    
                    url = link_elem.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = result.find('span', class_='st') or result.find('div', class_='s')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and url and url.startswith('http'):
                        results.append(WebSearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source='google',
                            relevance_score=0.9,
                            medical_concepts=self._extract_medical_concepts(f"{title} {snippet}")
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing Google result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Google scraping failed: {e}")
            return []
    
    def _search_pubmed(self, query: str) -> List[WebSearchResult]:
        """Search PubMed using E-utilities API."""
        try:
            # Use PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            
            # Search for articles
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': 5,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            response = requests.get(search_url, params=search_params, timeout=self.timeout)
            response.raise_for_status()
            search_data = response.json()
            
            # Get article details
            if 'esearchresult' in search_data and 'idlist' in search_data['esearchresult']:
                pmid_list = search_data['esearchresult']['idlist']
                
                if pmid_list:
                    # Fetch article details
                    fetch_url = f"{base_url}esummary.fcgi"
                    fetch_params = {
                        'db': 'pubmed',
                        'id': ','.join(pmid_list),
                        'retmode': 'json'
                    }
                    
                    fetch_response = requests.get(fetch_url, params=fetch_params, timeout=self.timeout)
                    fetch_response.raise_for_status()
                    fetch_data = fetch_response.json()
                    
                    results = []
                    for pmid in pmid_list:
                        if pmid in fetch_data.get('result', {}):
                            article = fetch_data['result'][pmid]
                            
                            # Get authors
                            authors = []
                            if 'authors' in article:
                                authors = [author.get('name', '') for author in article['authors'][:3]]
                            
                            title = article.get('title', 'No title')
                            abstract = article.get('abstract', '')
                            
                            results.append(WebSearchResult(
                                title=title,
                                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                snippet=abstract[:300] + "..." if len(abstract) > 300 else abstract,
                                source='pubmed',
                                relevance_score=0.9,  # High relevance for PubMed medical results
                                medical_concepts=self._extract_medical_concepts(f"{title} {abstract}"),
                                publication_date=article.get('pubdate', '')
                            ))
                    
                    return results
            
            return []
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def _extract_medical_concepts(self, text: str) -> List[str]:
        """Extract medical concepts from text using expanded terminology."""
        medical_terms = [
            # Imaging modalities
            'mri', 'ct', 'x-ray', 'ultrasound', 'pet', 'spect', 'mammography',
            'fluoroscopy', 'angiography', 'tomography',
            
            # General medical terms
            'radiology', 'diagnosis', 'prognosis', 'treatment', 'therapy',
            'pathology', 'histology', 'biopsy', 'screening',
            
            # Anatomical terms
            'brain', 'cerebral', 'cerebellum', 'brainstem', 'ventricle',
            'cortex', 'white matter', 'gray matter', 'csf', 'meninges',
            'temporal', 'frontal', 'parietal', 'occipital',
            
            # Pathological findings
            'tumor', 'lesion', 'mass', 'nodule', 'cyst', 'edema', 
            'hemorrhage', 'infarct', 'stroke', 'ischemia', 'necrosis',
            'inflammation', 'infection', 'abscess', 'hematoma',
            
            # Specific conditions
            'glioma', 'meningioma', 'metastasis', 'lymphoma', 'adenoma',
            'hydrocephalus', 'atrophy', 'dementia', 'alzheimer',
            'multiple sclerosis', 'epilepsy', 'seizure',
            
            # Clinical terms
            'contrast', 'enhancement', 'signal', 'intensity', 'artifact',
            'protocol', 'sequence', 'acquisition', 'reconstruction'
        ]
        
        found_concepts = []
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                found_concepts.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in found_concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _deduplicate_results(self, results: List[WebSearchResult]) -> List[WebSearchResult]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            # Normalize URL for comparison
            url_key = result.url.lower().rstrip('/')
            title_key = result.title.lower().strip()
            
            if url_key not in seen_urls and title_key not in seen_titles:
                seen_urls.add(url_key)
                seen_titles.add(title_key)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[WebSearchResult], original_query: str) -> List[WebSearchResult]:
        """Rank results by relevance, medical site priority, and content quality."""
        query_terms = original_query.lower().split()
        
        for result in results:
            # Base score from source reliability
            if result.source == 'pubmed':
                result.relevance_score = 0.9
            elif result.source == 'google':
                result.relevance_score = 0.8
            elif result.source == 'duckduckgo':
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
    
    def __init__(self, **kwargs):
        # Use medical-optimized search engines
        kwargs.setdefault('search_engines', ['pubmed', 'duckduckgo_html'])
        super().__init__(**kwargs)
        
        # Medical query templates
        self.query_templates = {
            'diagnosis': '{condition} diagnosis differential radiology imaging findings',
            'guidelines': '{condition} clinical guidelines imaging protocol recommendations',
            'research': '{condition} recent research MRI CT imaging findings literature',
            'anatomy': '{structure} anatomy MRI normal variants pathology',
            'pathology': '{condition} pathology imaging characteristics MRI CT findings'
        }
    
    def search_medical(self, 
                      query: str, 
                      search_type: str = 'general',
                      condition: str = None) -> List[WebSearchResult]:
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
        
        return final_results[:self.max_results]
    
    def general_search(self, query: str) -> List[WebSearchResult]:
        """
        Perform general web search (non-medical focused).
        
        Args:
            query: Search query
            
        Returns:
            List of web search results
        """
        return self.search(query, medical_focus=False)
    
    def medical_search(self, query: str) -> List[WebSearchResult]:
        """
        Alias for search_medical for backward compatibility.
        
        Args:
            query: Search query
            
        Returns:
            List of medical web search results
        """
        return self.search_medical(query)
    
    def _is_medical_source(self, url: str) -> bool:
        """Check if URL is from a reputable medical source."""
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.medical_sites)
    
    def search_pubmed(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        """Direct PubMed search - wrapper for the internal method."""
        old_max = self.max_results
        self.max_results = max_results
        try:
            results = self._search_pubmed(query)
            return results
        finally:
            self.max_results = old_max 