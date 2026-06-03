"""Public model interfaces for GAZE.

Provides two equal-priority adapter implementations:
- OpenAIAdapter: For OpenAI API compatible services (OpenAI, OpenRouter, etc.)
- HuggingFaceAdapter: For local HuggingFace models (requires torch, transformers)
- HuggingFaceVLMAdapter: For local HuggingFace vision-language models

The HuggingFace adapters are lazily imported to avoid requiring torch/transformers
when only using the OpenAI adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gaze.models.adapter_protocol import AdapterProtocol
from gaze.models.adapter_protocol import GenerationLog
from gaze.models.lmstudio_adapter import LMStudioAdapter
from gaze.models.lmstudio_adapter import list_lmstudio_model_ids
from gaze.models.lmstudio_adapter import require_lmstudio_model
from gaze.models.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from gaze.models.huggingface_adapter import HuggingFaceAdapter
    from gaze.models.huggingface_adapter import HuggingFaceVLMAdapter

__all__ = [
    "AdapterProtocol",
    "GenerationLog",
    "OpenAIAdapter",
    "HuggingFaceAdapter",
    "HuggingFaceVLMAdapter",
    "LMStudioAdapter",
    "list_lmstudio_model_ids",
    "require_lmstudio_model",
]


def __getattr__(name: str):
    """Lazy import for optional adapters to avoid heavy dependencies."""
    if name == "HuggingFaceAdapter":
        from gaze.models.huggingface_adapter import HuggingFaceAdapter

        return HuggingFaceAdapter
    if name == "HuggingFaceVLMAdapter":
        from gaze.models.huggingface_adapter import HuggingFaceVLMAdapter

        return HuggingFaceVLMAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
