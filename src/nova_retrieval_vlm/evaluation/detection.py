from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def _convert_to_tensors(data: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert detection data to proper tensor format for torchmetrics."""
    boxes = data.get("boxes", [])
    scores = data.get("scores", [])
    labels = data.get("labels", [])

    # Handle empty detections - check length to avoid tensor ambiguity
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

    # Convert boxes
    if isinstance(boxes, torch.Tensor):
        boxes_tensor = boxes.float()
    else:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

    # Ensure 2D shape for boxes
    if boxes_tensor.ndim == 1:
        boxes_tensor = boxes_tensor.unsqueeze(0)

    # Convert scores
    if isinstance(scores, torch.Tensor):
        scores_tensor = scores.float()
    else:
        scores_tensor = (
            torch.tensor(scores, dtype=torch.float32) if scores else torch.ones(len(boxes_tensor))
        )

    # Convert labels
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


def _compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1.item() + area2.item() - intersection

    return intersection / union if union > 0 else 0.0


def _compute_acc50(preds_tensors: list[dict], refs_tensors: list[dict]) -> float:
    """Compute detection accuracy at IoU 0.5 (per-sample hit rate)."""
    hits = 0
    total = len(refs_tensors)

    for pred, ref in zip(preds_tensors, refs_tensors, strict=False):
        pred_boxes = pred["boxes"]
        ref_boxes = ref["boxes"]

        # If no ground truth boxes, count as hit if no predictions
        if len(ref_boxes) == 0:
            if len(pred_boxes) == 0:
                hits += 1
            continue

        # Check if any prediction matches any ground truth at IoU >= 0.5
        sample_hit = False
        for ref_box in ref_boxes:
            for pred_box in pred_boxes:
                if _compute_iou(pred_box, ref_box) >= 0.5:
                    sample_hit = True
                    break
            if sample_hit:
                break

        if sample_hit:
            hits += 1

    return hits / total if total > 0 else 0.0


def evaluate_detection(
    preds: Sequence[dict[str, Any]],
    refs: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute detection metrics following NOVA benchmark protocol.

    Metrics:
    - mAP@0.3: Mean Average Precision at IoU threshold 0.3
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@[50:95]: Mean AP averaged across IoU thresholds 0.5 to 0.95 (step 0.05)
    - ACC50: Detection accuracy at IoU 0.5 (proportion of samples with at least one hit)

    Args:
        preds: List of prediction dictionaries with 'boxes', 'scores', 'labels'
        refs: List of reference dictionaries with 'boxes', 'scores', 'labels'

    Returns:
        Dictionary with keys 'map30', 'map50', 'map50_95', 'acc50'.
    """
    if not preds or not refs:
        return {"map30": 0.0, "map50": 0.0, "map50_95": 0.0, "acc50": 0.0}

    # Convert inputs to proper tensor format
    preds_tensors = [_convert_to_tensors(pred) for pred in preds]
    refs_tensors = [_convert_to_tensors(ref) for ref in refs]

    # ACC50 (detection accuracy at IoU 0.5)
    acc50 = _compute_acc50(preds_tensors, refs_tensors)

    # mAP@30
    m30 = MeanAveragePrecision(iou_thresholds=[0.3])
    m30.update(preds_tensors, refs_tensors)
    res30 = m30.compute()
    map30 = float(res30["map"])

    # mAP@50
    m50 = MeanAveragePrecision(iou_thresholds=[0.5])
    m50.update(preds_tensors, refs_tensors)
    res50 = m50.compute()
    map50 = float(res50["map"])

    # mAP@[50:95] per NOVA protocol
    ious = [th / 100 for th in range(50, 100, 5)]
    m5095 = MeanAveragePrecision(iou_thresholds=ious)
    m5095.update(preds_tensors, refs_tensors)
    res5095 = m5095.compute()
    map50_95 = float(res5095["map"])

    return {"map30": map30, "map50": map50, "map50_95": map50_95, "acc50": acc50}
