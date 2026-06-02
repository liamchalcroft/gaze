"""Retrieval tools for evidence-based analysis.

Provides web search and image search capabilities for LLM agents.
"""

from __future__ import annotations

from gaze.retrieval.base import SearchEngineError
from gaze.retrieval.image_search import ImageDownloadError
from gaze.retrieval.image_search import ImageSearchError
from gaze.retrieval.image_search import ImageSearchResult
from gaze.retrieval.image_search import MedicalImageSearchManager
from gaze.retrieval.web_search import SearchError
from gaze.retrieval.web_search import SearchResult
from gaze.retrieval.web_search import WebSearchManager

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
