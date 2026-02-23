"""Inlined utilities for standalone operation.

These are local copies of functions from radiant_harness.utils and
radiant_harness.verifiers.rewards so that the nova-brain-mri package
can be installed and evaluated without depending on radiant_harness.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box as [x1, y1, x2, y2]
        box2: Second bounding box as [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1

    Raises:
        ValueError: If boxes don't have exactly 4 coordinates
    """
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Bounding boxes must have exactly 4 coordinates")

    # Normalize coordinates so x1 <= x2, y1 <= y2.
    # VLMs commonly emit coordinates in arbitrary order.
    x1_1, y1_1, x2_1, y2_1 = (
        min(box1[0], box1[2]),
        min(box1[1], box1[3]),
        max(box1[0], box1[2]),
        max(box1[1], box1[3]),
    )
    x1_2, y1_2, x2_2, y2_2 = (
        min(box2[0], box2[2]),
        min(box2[1], box2[3]),
        max(box2[0], box2[2]),
        max(box2[1], box2[3]),
    )

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract JSON object from model output text.

    Handles common formats:
    - Markdown code blocks (```json ... ```)
    - Raw JSON objects
    - JSON embedded in surrounding text

    Args:
        text: Text that may contain a JSON object

    Returns:
        Parsed JSON dict, or None if no valid JSON found
    """
    text = text.strip()
    if not text:
        return None

    # Handle markdown code block
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline == -1:
            return None
        closing = text.rfind("```")
        if closing <= first_newline:
            return None
        text = text[first_newline + 1 : closing].strip()

    # Try to parse directly first (most common case)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            typed: dict[str, Any] = result
            return typed
        return None
    except json.JSONDecodeError:
        pass

    # Use raw_decode to find JSON object
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text, i)
            if isinstance(result, dict):
                typed: dict[str, Any] = result
                return typed
        except json.JSONDecodeError:
            continue

    return None


def extract_completion_text(completion: Any) -> str:
    """Extract text content from a verifiers completion.

    Handles multiple formats:
    - Plain string
    - Message list with assistant role
    - Multimodal content lists

    Args:
        completion: Model completion in any supported format

    Returns:
        Extracted text content
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # Find last assistant message
        for msg in reversed(completion):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                return content

            # Handle multimodal content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")

    return str(completion or "")
