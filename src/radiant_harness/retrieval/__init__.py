"""Retrieval tools for evidence-based analysis.

Provides web search and image search capabilities for LLM agents.
"""

from __future__ import annotations

from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.image_search import ImageSearchResult
from radiant_harness.retrieval.image_search import MedicalImageSearchManager
from radiant_harness.retrieval.image_search import search_medical_images
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.retrieval.web_search import SearchResult
from radiant_harness.retrieval.web_search import WebSearchManager
from radiant_harness.retrieval.web_search import search_clinical_guidelines
from radiant_harness.retrieval.web_search import search_diagnostic_information
from radiant_harness.retrieval.web_search import search_medical_literature

__all__ = [
    # Web search
    "SearchResult",
    "SearchError",
    "WebSearchManager",
    "search_medical_literature",
    "search_clinical_guidelines",
    "search_diagnostic_information",
    # Image search
    "ImageSearchResult",
    "ImageSearchError",
    "MedicalImageSearchManager",
    "search_medical_images",
]
