"""Extraction and IoU helpers, re-exported from the GAZE framework.

This module is the single point where the environment depends on
``gaze-vlm``. The scoring code in ``rewards.py`` imports these three
helpers from here, so wiring them to GAZE keeps the reward logic and the
GAZE pipeline byte-for-byte consistent.

- ``compute_iou(box1, box2) -> float`` over ``[x1, y1, x2, y2]`` boxes.
- ``extract_json_from_text(text) -> dict | None``.
- ``extract_completion_text(completion) -> str``.
"""

from __future__ import annotations

from gaze.utils import compute_iou, extract_json_from_text
from gaze.verifiers.rewards import extract_completion_text

__all__ = ["compute_iou", "extract_completion_text", "extract_json_from_text"]
