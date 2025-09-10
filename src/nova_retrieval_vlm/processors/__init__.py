"""Modular processing system for NOVA VLM tasks."""

from .base import BaseProcessor
from .base import ProcessorConfig
from .caption import CaptionProcessor
from .detection import DetectionProcessor
from .diagnosis import DiagnosisProcessor
from .localization import LocalizationProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorConfig",
    "CaptionProcessor",
    "DetectionProcessor",
    "DiagnosisProcessor",
    "LocalizationProcessor",
]
