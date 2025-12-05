"""Modular processing system for NOVA VLM tasks."""

from nova_retrieval_vlm.types import JSONParseError

from .base import BaseProcessor
from .base import ProcessorConfig
from .caption import CaptionProcessor
from .diagnosis import DiagnosisProcessor
from .localization import LocalizationProcessor

__all__ = [
    "BaseProcessor",
    "JSONParseError",
    "ProcessorConfig",
    "CaptionProcessor",
    "DiagnosisProcessor",
    "LocalizationProcessor",
]
