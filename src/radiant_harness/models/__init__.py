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

from radiant_harness.models.adapter_protocol import AdapterProtocol
from radiant_harness.models.adapter_protocol import GenerationLog
from radiant_harness.models.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter
    from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter
    from radiant_harness.models.lmstudio_adapter import LMStudioAdapter

__all__ = [
    "AdapterProtocol",
    "GenerationLog",
    "OpenAIAdapter",
    "HuggingFaceAdapter",
    "HuggingFaceVLMAdapter",
    "LMStudioAdapter",
]


def __getattr__(name: str):
    """Lazy import for optional adapters to avoid heavy dependencies."""
    if name == "HuggingFaceAdapter":
        from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter

        return HuggingFaceAdapter
    if name == "HuggingFaceVLMAdapter":
        from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter

        return HuggingFaceVLMAdapter
    if name == "LMStudioAdapter":
        from radiant_harness.models.lmstudio_adapter import LMStudioAdapter

        return LMStudioAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
