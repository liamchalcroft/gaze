"""
NOVA Retrieval VLM - Medical Image Analysis Framework

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

from .config import NOVAConfig
from .config import TaskType
from .data import NovaDataset
from .evaluation.caption import evaluate_caption
from .evaluation.detection import evaluate_detection
from .evaluation.diagnosis import evaluate_diagnosis_nova_official
from .processor import NOVAAgenticProcessor
from .schemas import NOVA_SCHEMA
from .schemas import get_required_fields
from .schemas import validate_nova_response

__version__ = "0.1.0"

__all__ = [
    "NOVAAgenticProcessor",
    "NOVAConfig",
    "TaskType",
    "NOVA_SCHEMA",
    "get_required_fields",
    "validate_nova_response",
    "evaluate_caption",
    "evaluate_detection",
    "evaluate_diagnosis_nova_official",
    "NovaDataset",
]
