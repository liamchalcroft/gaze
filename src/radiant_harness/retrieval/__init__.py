"""Retrieval tools for evidence-based analysis.

Provides web search and image search capabilities for LLM agents.
"""

from __future__ import annotations

from radiant_harness.retrieval.base import SearchEngineError
from radiant_harness.retrieval.image_search import ImageDownloadError
from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.image_search import ImageSearchResult
from radiant_harness.retrieval.image_search import MedicalImageSearchManager
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.retrieval.web_search import SearchResult
from radiant_harness.retrieval.web_search import WebSearchManager

__all__ = [
    # Base
    "SearchEngineError",
    # Web search
    "SearchResult",
    "SearchError",
    "WebSearchManager",
    # Image search
    "ImageSearchResult",
    "ImageSearchError",
    "ImageDownloadError",
    "MedicalImageSearchManager",
]
