"""Utility modules for radiant_harness."""

import math

from radiant_harness.utils.iou import compute_iou
from radiant_harness.utils.json_coerce import coerce_json_types
from radiant_harness.utils.json_extract import extract_json_from_text


def clamp_confidence(value: object) -> float | None:
    """Clamp a confidence value to [0.0, 1.0].

    Local models often emit confidence on non-standard scales (e.g. 0-100
    or 0-5).  Rather than rejecting these as invalid, we clamp to the
    expected range so the rest of the pipeline can proceed.

    Returns None for non-numeric, boolean, NaN, or infinite inputs.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return max(0.0, min(1.0, f))


__all__ = ["clamp_confidence", "coerce_json_types", "compute_iou", "extract_json_from_text"]
