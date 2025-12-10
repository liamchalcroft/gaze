"""Public model interfaces for the VLM harness.

Provides two equal-priority adapter implementations:
- OpenAIAdapter: For OpenAI API compatible services (OpenAI, OpenRouter, etc.)
- HuggingFaceAdapter: For local HuggingFace models (requires torch, transformers)
- HuggingFaceVLMAdapter: For local HuggingFace vision-language models

The HuggingFace adapters are lazily imported to avoid requiring torch/transformers
when only using the OpenAI adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from radiant_harness.models._types import GenerationLog
from radiant_harness.models.adapter_protocol import AdapterProtocol
from radiant_harness.models.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter
    from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter

__all__ = [
    "AdapterProtocol",
    "GenerationLog",
    "OpenAIAdapter",
    "HuggingFaceAdapter",
    "HuggingFaceVLMAdapter",
]


def __getattr__(name: str):
    """Lazy import for HuggingFace adapters to avoid torch dependency."""
    if name == "HuggingFaceAdapter":
        from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter

        return HuggingFaceAdapter
    if name == "HuggingFaceVLMAdapter":
        from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter

        return HuggingFaceVLMAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
