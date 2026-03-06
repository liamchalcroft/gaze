"""PubmedQA example implementation using Radiant Harness.

Demonstrates text-only (no images) agentic analysis with web search support.
"""

from __future__ import annotations

from .processor import PubmedQAProcessor
from .schemas import normalize_pubmedqa_answer

__all__ = ["PubmedQAProcessor", "normalize_pubmedqa_answer"]
