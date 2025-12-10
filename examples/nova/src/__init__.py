"""NOVA Retrieval VLM - Medical Image Analysis Framework.

A comprehensive framework for comparing vision-language models on
the NOVA brain-MRI benchmark with retrieval-augmented generation.

Features:
- Multi-model support (OpenAI, OpenRouter with 100+ models)
- Retrieval-augmented generation with medical guidelines
- Comprehensive evaluation metrics for localization, captioning, and diagnosis
- Interactive visualization tools and Streamlit GUI
- Multi-turn reasoning with adaptive analysis
- Advanced prompt engineering with Jinja2 templates

This package provides tools for researchers working on medical AI,
specifically focusing on brain MRI analysis and diagnostic tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Core imports (no heavy dependencies)
from .config import NOVAConfig
from .config import TaskType
from .processor import NOVAAgenticProcessor
from .rewards import NOVARewardWeights
from .rewards import NOVAVerifiersReward
from .schemas import NOVA_SCHEMA
from .schemas import get_required_fields
from .schemas import validate_nova_response

# Lazy imports for torch-dependent modules
if TYPE_CHECKING:
    from .data import NovaDataset

__version__ = "0.1.0"

__all__ = [
    "NOVAAgenticProcessor",
    "NOVAConfig",
    "NOVARewardWeights",
    "NOVAVerifiersReward",
    "TaskType",
    "NOVA_SCHEMA",
    "get_required_fields",
    "validate_nova_response",
    "NovaDataset",
]


def __getattr__(name: str):
    """Lazy import for torch-dependent modules."""
    if name == "NovaDataset":
        from .data import NovaDataset

        return NovaDataset
    if name in {"evaluate_caption", "evaluate_detection", "evaluate_diagnosis_nova_official"}:
        from .evaluation import caption
        from .evaluation import detection
        from .evaluation import diagnosis

        return {
            "evaluate_caption": caption.evaluate_caption,
            "evaluate_detection": detection.evaluate_detection,
            "evaluate_diagnosis_nova_official": diagnosis.evaluate_diagnosis_nova_official,
        }[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
