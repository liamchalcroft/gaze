from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from beartype import beartype

# Import shared IoU utility
from gaze.utils.iou import compute_iou

logger = logging.getLogger(__name__)

# NOVA benchmark IoU thresholds
#
# Clinical context for brain MRI localization:
# At IOU_THRESHOLD_LOOSE = 0.3, a predicted box needs only 30% overlap with
# ground truth to count as a match.  For a small brain lesion (e.g. 5 mm in a
# 240 mm FOV, ~10 px box), this tolerates ~15 mm spatial error — roughly the
# width of a gyrus.  While appropriate for screening-level "did the model find
# the right hemisphere/lobe?", it is too permissive for lesion-level precision.
# IOU_THRESHOLD_STANDARD = 0.5 is the recommended threshold for clinical
# evaluation; the 0.3 threshold is retained for NOVA protocol compatibility and
# as a lenient secondary metric.
IOU_THRESHOLD_LOOSE = 0.3  # Permissive — see clinical context above
IOU_THRESHOLD_STANDARD = 0.5  # Standard COCO-style threshold

# mAP@[50:95] range as floats — produces [0.5, 0.55, ..., 0.95] (10 thresholds)
_MAP_RANGE_IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


@beartype
def clamp_and_validate_box(
    box: list[int | float], width: int | float, height: int | float
) -> list[float]:
    """Clamp a bounding box to image dimensions and ensure x1 < x2, y1 < y2.

    Args:
        box: [x1, y1, x2, y2] bounding box coordinates.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Clamped [x1, y1, x2, y2] with coordinates within [0, width] / [0, height].
    """
    x1, y1, x2, y2 = (float(c) for c in box)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    return [x1, y1, x2, y2]


@beartype
def rescale_and_clamp_box(
    box: list[int | float], width: int | float, height: int | float
) -> list[float]:
    """Rescale out-of-bounds coordinates per-axis, then clamp.

    When a model outputs coordinates in a different coordinate space (e.g.
    [238, 223, 760, 479] for a 480x480 image), simple clamping squishes the
    box.  This function detects the implicit coordinate range on each axis
    independently and rescales accordingly.

    Per-axis scaling is used instead of uniform scaling because the model may
    be operating in a non-square coordinate space (e.g. it thinks the image
    is 1000x500 when it is actually 480x480).  Uniform scaling with
    ``min(scale_x, scale_y)`` would needlessly shift coordinates on the axis
    that was already within bounds, degrading spatial accuracy for lesion
    localization.

    Args:
        box: [x1, y1, x2, y2] bounding box coordinates.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Rescaled and clamped [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = (float(c) for c in box)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    w = float(width)
    h = float(height)

    # Detect if coordinates exceed image bounds — rescale per-axis
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    if max_x > w or max_y > h:
        scale_x = w / max_x if max_x > w else 1.0
        scale_y = h / max_y if max_y > h else 1.0
        logger.debug(
            "rescale_and_clamp_box: box [%.1f,%.1f,%.1f,%.1f] exceeds %dx%d, "
            "rescaling x=%.3f y=%.3f",
            x1,
            y1,
            x2,
            y2,
            int(w),
            int(h),
            scale_x,
            scale_y,
        )
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

    # Final clamp to be safe
    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    x2 = max(0.0, min(x2, w))
    y2 = max(0.0, min(y2, h))
    return [x1, y1, x2, y2]


@beartype
def _convert_to_tensors(
    data: dict[str, Any] | list[list[int | float]],
) -> dict[str, torch.Tensor]:
    """Convert detection data to proper tensor format for torchmetrics.

    Handles both dict format ({"boxes": [...], "scores": [...], "labels": [...]})
    and list format (raw list of boxes).

    Expects boxes in (x1, y1, x2, y2) format.
    """
    # Handle list format (raw boxes without scores/labels)
    if isinstance(data, list):
        data = {"boxes": data, "scores": [], "labels": []}

    boxes = data.get("boxes", [])
    scores = data.get("scores", [])
    labels = data.get("labels", [])

    if isinstance(boxes, torch.Tensor):
        boxes_len = boxes.shape[0] if boxes.ndim > 0 else 0
    else:
        boxes_len = len(boxes)
    if boxes_len == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    if isinstance(boxes, torch.Tensor):
        boxes_tensor = boxes.float()
    else:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

    if boxes_tensor.ndim == 1:
        boxes_tensor = boxes_tensor.unsqueeze(0)

    if isinstance(scores, torch.Tensor):
        scores_tensor = scores.float()
    else:
        scores_tensor = (
            torch.tensor(scores, dtype=torch.float32) if scores else torch.ones(len(boxes_tensor))
        )

    if isinstance(labels, torch.Tensor):
        labels_tensor = labels.long()
    else:
        labels_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros(len(boxes_tensor), dtype=torch.int64)
        )

    return {
        "boxes": boxes_tensor,
        "scores": scores_tensor,
        "labels": labels_tensor,
    }


@beartype
def _compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    # Convert tensors to lists for the shared IoU function
    box1_list = [float(box1[i].item()) for i in range(4)]
    box2_list = [float(box2[i].item()) for i in range(4)]
    return compute_iou(box1_list, box2_list)


@beartype
def _compute_acc_and_counts(
    preds_tensors: list[dict[str, torch.Tensor]],
    refs_tensors: list[dict[str, torch.Tensor]],
    iou_threshold: float,
) -> tuple[float, int, int, int]:
    """Compute detection accuracy and TP/FP/FN counts at given IoU threshold.

    Args:
        preds_tensors: List of prediction dicts with 'boxes' tensor
        refs_tensors: List of reference dicts with 'boxes' tensor
        iou_threshold: IoU threshold for matching (e.g., 0.3 or 0.5)

    Returns:
        Tuple of (accuracy, true_positives, false_positives, false_negatives)
    """
    hits = 0
    total = len(refs_tensors)
    tp = 0  # True positives: predictions matched to ground truth
    fp = 0  # False positives: predictions not matched to ground truth
    fn = 0  # False negatives: ground truth boxes not matched by predictions

    for pred, ref in zip(preds_tensors, refs_tensors, strict=True):
        pred_boxes = pred["boxes"]
        ref_boxes = ref["boxes"]

        # Track which ground truth boxes have been matched
        matched_refs: set[int] = set()

        # If no ground truth boxes, count as hit if no predictions
        if len(ref_boxes) == 0:
            if len(pred_boxes) == 0:
                hits += 1
            else:
                fp += len(pred_boxes)  # All predictions are false positives
            continue

        # Check if any prediction matches any ground truth at IoU >= threshold
        sample_hit = False
        for pred_box in pred_boxes:
            best_iou = 0.0
            best_ref_idx = -1
            for ref_idx, ref_box in enumerate(ref_boxes):
                if ref_idx in matched_refs:
                    continue
                iou = _compute_iou(pred_box, ref_box)
                if iou > best_iou:
                    best_iou = iou
                    best_ref_idx = ref_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_refs.add(best_ref_idx)
                sample_hit = True
            else:
                fp += 1

        # Unmatched ground truth boxes are false negatives
        fn += len(ref_boxes) - len(matched_refs)

        if sample_hit:
            hits += 1

    accuracy = hits / total if total > 0 else 0.0
    return accuracy, tp, fp, fn


def _per_image_ap(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> float:
    """Compute Average Precision for a single image at a given IoU threshold.

    Uses 11-point interpolation (PASCAL VOC style), which is the standard
    in medical imaging benchmarks including NOVA.

    Args:
        pred_boxes: (N, 4) predicted boxes
        pred_scores: (N,) confidence scores
        gt_boxes: (M, 4) ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        AP value in [0, 1].
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by confidence descending
    sorted_indices = pred_scores.argsort(descending=True)
    matched_gt: set[int] = set()
    tp_list: list[int] = []

    for si in sorted_indices:
        pred_box = pred_boxes[si]
        best_iou = 0.0
        best_gt_idx = -1
        for gi in range(n_gt):
            if gi in matched_gt:
                continue
            iou = _compute_iou(pred_box, gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            matched_gt.add(best_gt_idx)
        else:
            tp_list.append(0)

    # Compute precision-recall curve and 11-point interpolated AP
    tp_cumsum = torch.tensor(tp_list).cumsum(0).float()
    fp_cumsum = torch.arange(1, len(tp_list) + 1).float() - tp_cumsum
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt

    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        mask = recall >= t
        if mask.any():
            ap += float(precision[mask].max()) / 11.0
    return ap


def _mean_per_image_ap(
    preds_tensors: list[dict[str, torch.Tensor]],
    refs_tensors: list[dict[str, torch.Tensor]],
    iou_threshold: float,
) -> float:
    """Compute mean AP across all images at a given IoU threshold."""
    aps = []
    for pred, ref in zip(preds_tensors, refs_tensors, strict=True):
        ap = _per_image_ap(
            pred["boxes"],
            pred["scores"],
            ref["boxes"],
            iou_threshold,
        )
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


@beartype
def evaluate_detection(
    preds: Sequence[dict[str, Any] | list[list[int | float]]],
    refs: Sequence[dict[str, Any] | list[list[int | float]]],
) -> dict[str, float | int]:
    """
    Compute detection metrics following NOVA benchmark protocol.

    mAP is computed as the mean of per-image AP values (11-point interpolation),
    matching the NOVA paper's evaluation methodology.

    Metrics:
    - mAP@0.3: Mean Average Precision at IoU threshold 0.3
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@[50:95]: Mean AP averaged across IoU thresholds 0.5 to 0.95 (step 0.05)
    - ACC50: Detection accuracy at IoU 0.5 (proportion of samples with at least one hit)
    - recall50: Box-level recall at IoU 0.5 (fraction of GT boxes matched)
    - precision50: Box-level precision at IoU 0.5 (fraction of predictions matched)
    - TP30/FP30: True positive and false positive counts at IoU 0.3

    Args:
        preds: List of prediction dicts with 'boxes', 'scores', 'labels', or raw box lists
        refs: List of reference dicts with 'boxes', 'scores', 'labels', or raw box lists

    Returns:
        Dictionary with keys 'map30', 'map50', 'map50_95', 'acc50', 'recall50',
        'precision50', 'tp30', 'fp30'.

    Raises:
        ValueError: If preds and refs have different lengths or are empty.
    """
    if not preds:
        raise ValueError("Cannot evaluate empty predictions list")
    if not refs:
        raise ValueError("Cannot evaluate empty references list")

    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have same length, got {len(preds)} vs {len(refs)}")

    # Convert inputs to proper tensor format
    preds_tensors = [_convert_to_tensors(pred) for pred in preds]
    refs_tensors = [_convert_to_tensors(ref) for ref in refs]

    # ACC50 and box-level counts at standard threshold
    acc50, tp50, fp50, fn50 = _compute_acc_and_counts(
        preds_tensors, refs_tensors, IOU_THRESHOLD_STANDARD
    )

    # Box-level recall and precision at standard threshold
    recall50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
    precision50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0

    # TP30/FP30 at loose threshold (per NOVA protocol)
    _, tp30, fp30, _ = _compute_acc_and_counts(preds_tensors, refs_tensors, IOU_THRESHOLD_LOOSE)

    # Per-image mean AP (NOVA paper methodology: 11-point interpolation)
    map30 = _mean_per_image_ap(preds_tensors, refs_tensors, IOU_THRESHOLD_LOOSE)
    map50 = _mean_per_image_ap(preds_tensors, refs_tensors, IOU_THRESHOLD_STANDARD)
    map50_95 = sum(
        _mean_per_image_ap(preds_tensors, refs_tensors, t) for t in _MAP_RANGE_IOU_THRESHOLDS
    ) / len(_MAP_RANGE_IOU_THRESHOLDS)

    return {
        "map30": map30,
        "map50": map50,
        "map50_95": map50_95,
        "acc50": acc50,
        "recall50": recall50,
        "precision50": precision50,
        "tp30": tp30,
        "fp30": fp30,
    }
