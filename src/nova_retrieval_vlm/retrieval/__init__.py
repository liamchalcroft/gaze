"""Enhanced web search system for medical knowledge retrieval."""

from .web_search import SearchResult
from .web_search import WebSearchManager
from .web_search import search_medical_literature
from .web_search import search_medical_literature_sync

__all__ = [
    "SearchResult",
    "WebSearchManager",
    "search_medical_literature",
    "search_medical_literature_sync",
]
