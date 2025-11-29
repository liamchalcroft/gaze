"""Modular processing system for NOVA VLM tasks."""

from .base import BaseProcessor
from .base import ProcessorConfig
from .caption import CaptionProcessor
from .diagnosis import DiagnosisProcessor
from .localization import LocalizationProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorConfig",
    "CaptionProcessor",
    "DiagnosisProcessor",
    "LocalizationProcessor",
]
