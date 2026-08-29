"""Shared IoU calculation utilities."""

from __future__ import annotations

from collections.abc import Sequence

from beartype import beartype
from loguru import logger


@beartype
def compute_iou(
    box1: Sequence[float | int],
    box2: Sequence[float | int],
    *,
    lenient: bool = False,
) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box as [x1, y1, x2, y2] (ints accepted:
            JSON coordinates arrive as ints)
        box2: Second bounding box as [x1, y1, x2, y2]
        lenient: If True, silently reorder inverted coordinates so that
            ``x1 <= x2`` and ``y1 <= y2``. This awards full credit to a box
            whose corners are transposed, which is gameable when the score
            feeds an RL reward, so it is off by default: a malformed box
            scores 0.0 and is logged.

    Returns:
        IoU score between 0 and 1

    Raises:
        ValueError: If boxes don't have exactly 4 coordinates
    """
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Bounding boxes must have exactly 4 coordinates")

    if lenient:
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
    else:
        for name, box in (("box1", box1), ("box2", box2)):
            if box[2] <= box[0] or box[3] <= box[1]:
                logger.warning(
                    "compute_iou: {} {} is degenerate or inverted "
                    "(expected x1 < x2 and y1 < y2); scoring 0.0. "
                    "Pass lenient=True to reorder instead.",
                    name,
                    list(box),
                )
                return 0.0
        x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
        x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Check if boxes intersect
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area
