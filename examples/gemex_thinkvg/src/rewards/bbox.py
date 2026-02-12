"""Bounding box verification reward for GEMeX-ThinkVG.

Computes reward based on IoU (Intersection over Union) between
predicted and ground truth bounding boxes.
"""

from __future__ import annotations

from collections.abc import Sequence

from beartype import beartype

# GEMeX image dimensions after preprocessing
IMAGE_SIZE = 336

# IoU thresholds for reward tiers
IOU_THRESHOLD_STRICT = 0.5  # Standard detection threshold
IOU_THRESHOLD_LOOSE = 0.3  # Permissive for approximate grounding
IOU_THRESHOLD_MINIMAL = 0.1  # Any overlap


@beartype
def validate_bbox(bbox: list[int | float], image_size: int = IMAGE_SIZE) -> bool:
    """Validate bounding box format and values.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_size: Image dimension for validation

    Returns:
        True if valid bbox
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    # Check all coordinates are numeric
    if not all(isinstance(x, int | float) for x in bbox):
        return False

    # Check coordinate ordering
    if x2 <= x1 or y2 <= y1:
        return False

    # Check bounds (allow small margin for floating point)
    margin = 5
    if x1 < -margin or y1 < -margin:
        return False
    if x2 > image_size + margin or y2 > image_size + margin:
        return False

    return True


@beartype
def clamp_bbox(
    bbox: list[int | float],
    image_size: int = IMAGE_SIZE,
) -> list[int]:
    """Clamp bounding box to valid image coordinates.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_size: Image dimension

    Returns:
        Clamped bbox with integer coordinates
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), image_size - 1))
    y1 = max(0, min(int(y1), image_size - 1))
    x2 = max(x1 + 1, min(int(x2), image_size))
    y2 = max(y1 + 1, min(int(y2), image_size))

    return [x1, y1, x2, y2]


@beartype
def compute_iou(
    bbox1: list[int | float],
    bbox2: list[int | float],
) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]

    Returns:
        IoU score in [0, 1]
    """
    # Clamp to valid coordinates
    b1 = clamp_bbox(bbox1)
    b2 = clamp_bbox(bbox2)

    # Compute intersection
    x1_inter = max(b1[0], b2[0])
    y1_inter = max(b1[1], b2[1])
    x2_inter = min(b1[2], b2[2])
    y2_inter = min(b1[3], b2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height

    # Compute union
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


@beartype
def compute_giou(
    bbox1: list[int | float],
    bbox2: list[int | float],
) -> float:
    """Compute Generalized IoU between two bounding boxes.

    GIoU handles non-overlapping boxes better than standard IoU.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]

    Returns:
        GIoU score in [-1, 1], higher is better
    """
    b1 = clamp_bbox(bbox1)
    b2 = clamp_bbox(bbox2)

    # Standard IoU computation
    x1_inter = max(b1[0], b2[0])
    y1_inter = max(b1[1], b2[1])
    x2_inter = min(b1[2], b2[2])
    y2_inter = min(b1[3], b2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0.0

    # Compute enclosing box
    x1_encl = min(b1[0], b2[0])
    y1_encl = min(b1[1], b2[1])
    x2_encl = max(b1[2], b2[2])
    y2_encl = max(b1[3], b2[3])
    encl_area = (x2_encl - x1_encl) * (y2_encl - y1_encl)

    # GIoU
    if encl_area <= 0:
        return iou

    giou = iou - (encl_area - union) / encl_area

    return giou


@beartype
def compute_center_distance(
    bbox1: list[int | float],
    bbox2: list[int | float],
    image_size: int = IMAGE_SIZE,
) -> float:
    """Compute normalized distance between bbox centers.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        image_size: Image size for normalization

    Returns:
        Normalized distance in [0, 1], 0 = same center
    """
    b1 = clamp_bbox(bbox1, image_size)
    b2 = clamp_bbox(bbox2, image_size)

    # Compute centers
    cx1 = (b1[0] + b1[2]) / 2
    cy1 = (b1[1] + b1[3]) / 2
    cx2 = (b2[0] + b2[2]) / 2
    cy2 = (b2[1] + b2[3]) / 2

    # Euclidean distance normalized by diagonal
    max_dist = (2 ** 0.5) * image_size
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    return min(1.0, dist / max_dist)


@beartype
def compute_bbox_reward(
    prediction: list[int | float],
    reference: list[int | float],
    image_size: int = IMAGE_SIZE,
) -> dict[str, float]:
    """Compute bounding box verification reward.

    Combines:
    - IoU (primary metric)
    - GIoU (handles non-overlapping boxes)
    - Center distance (partial credit for nearby boxes)

    Args:
        prediction: Predicted bbox [x1, y1, x2, y2]
        reference: Ground truth bbox [x1, y1, x2, y2]
        image_size: Image dimension

    Returns:
        Dict with component scores and final reward
    """
    # Validate inputs
    pred_valid = validate_bbox(prediction, image_size)
    ref_valid = validate_bbox(reference, image_size)

    if not pred_valid:
        return {
            "iou": 0.0,
            "giou": -1.0,
            "center_distance": 1.0,
            "reward": 0.0,
            "valid_prediction": False,
            "iou_50": False,
            "iou_30": False,
        }

    if not ref_valid:
        # No ground truth bbox - cannot evaluate
        return {
            "iou": 0.0,
            "giou": 0.0,
            "center_distance": 0.0,
            "reward": 0.0,
            "valid_prediction": True,
            "valid_reference": False,
            "iou_50": False,
            "iou_30": False,
        }

    # Compute metrics
    iou = compute_iou(prediction, reference)
    giou = compute_giou(prediction, reference)
    center_dist = compute_center_distance(prediction, reference, image_size)

    # Threshold indicators
    iou_50 = iou >= IOU_THRESHOLD_STRICT
    iou_30 = iou >= IOU_THRESHOLD_LOOSE

    # Penalise degenerate "full-image" predictions.
    # A box covering most of the image trivially overlaps everything,
    # so we scale the reward toward 0 as the predicted area grows.
    pred = clamp_bbox(prediction, image_size)
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    image_area = image_size * image_size
    area_ratio = pred_area / image_area if image_area > 0 else 0.0
    # Linear penalty: full credit at ≤50% coverage, zero at 100%.
    area_penalty_start = 0.5
    area_penalty = max(0.0, min(1.0, (1.0 - area_ratio) / (1.0 - area_penalty_start))) if area_ratio > area_penalty_start else 1.0

    # Weighted reward
    # Primary: IoU (most important)
    # Secondary: Center distance (for partial credit when IoU is 0)
    if iou >= IOU_THRESHOLD_MINIMAL:
        # Has some overlap - use IoU directly
        reward = iou * area_penalty
    else:
        # No overlap - use center distance for partial credit
        # Closer = better, but cap the reward since no overlap
        proximity_reward = (1.0 - center_dist) * 0.2
        reward = proximity_reward * area_penalty

    return {
        "iou": iou,
        "giou": giou,
        "center_distance": center_dist,
        "reward": reward,
        "valid_prediction": True,
        "valid_reference": True,
        "iou_50": iou_50,
        "iou_30": iou_30,
    }


@beartype
def compute_batch_bbox_rewards(
    predictions: Sequence[list[int | float]],
    references: Sequence[list[int | float]],
    image_size: int = IMAGE_SIZE,
) -> list[dict[str, float]]:
    """Compute bbox rewards for a batch of samples.

    Args:
        predictions: List of predicted bboxes
        references: List of reference bboxes
        image_size: Image dimension

    Returns:
        List of reward dicts for each sample
    """
    return [
        compute_bbox_reward(pred, ref, image_size)
        for pred, ref in zip(predictions, references, strict=True)
    ]
