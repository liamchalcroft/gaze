"""Utility modules for gaze."""

import math

from beartype import beartype

from gaze.utils.iou import compute_iou
from gaze.utils.json_coerce import coerce_json_types
from gaze.utils.json_extract import extract_json_from_text

_CONFIDENCE_WORD_MAP: dict[str, float] = {
    "very high": 0.95,
    "high": 0.85,
    "medium-high": 0.75,
    "medium": 0.65,
    "moderate": 0.65,
    "medium-low": 0.45,
    "low": 0.3,
    "very low": 0.15,
    "none": 0.0,
}


@beartype
def clamp_confidence(value: object) -> float | None:
    """Clamp a confidence value to [0.0, 1.0].

    Local models often emit confidence on non-standard scales (e.g. 0-100
    or 0-5) or as word labels (e.g. "high", "medium").  Rather than
    rejecting these as invalid, we normalize to the expected range so
    the rest of the pipeline can proceed.

    Returns None for boolean, NaN, or infinite inputs.
    """
    if isinstance(value, bool):
        return None
    # Handle word labels from local models (e.g. "high", "medium")
    if isinstance(value, str):
        label = value.strip().lower()
        if label in _CONFIDENCE_WORD_MAP:
            return _CONFIDENCE_WORD_MAP[label]
        # Try parsing as a numeric string (e.g. "0.85")
        try:
            f = float(label)
        except (ValueError, OverflowError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return max(0.0, min(1.0, f))
    if not isinstance(value, int | float):
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return max(0.0, min(1.0, f))


__all__ = ["clamp_confidence", "coerce_json_types", "compute_iou", "extract_json_from_text"]
