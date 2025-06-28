from __future__ import annotations
from typing import Sequence, Dict, Any
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def _convert_to_tensors(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert detection data to proper tensor format."""
    converted = {}
    
    # Convert boxes
    if "boxes" in data:
        if isinstance(data["boxes"], torch.Tensor):
            converted["boxes"] = data["boxes"]
        else:
            converted["boxes"] = torch.tensor(data["boxes"], dtype=torch.float32)
    
    # Convert scores
    if "scores" in data:
        if isinstance(data["scores"], torch.Tensor):
            converted["scores"] = data["scores"]
        else:
            converted["scores"] = torch.tensor(data["scores"], dtype=torch.float32)
    
    # Convert labels
    if "labels" in data:
        if isinstance(data["labels"], torch.Tensor):
            converted["labels"] = data["labels"]
        else:
            converted["labels"] = torch.tensor(data["labels"], dtype=torch.int64)
    
    return converted


def evaluate_detection(
    preds: Sequence[Dict[str, Any]],
    refs: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute mAP at IoU thresholds 0.3, 0.5, and averaged 0.50:0.95.

    Args:
        preds: List of prediction dictionaries with 'boxes', 'scores', 'labels'
        refs: List of reference dictionaries with 'boxes', 'scores', 'labels'

    Returns:
        Dictionary with keys 'map30', 'map50', 'map50_95'.
    """
    # Convert inputs to proper tensor format
    preds_tensors = [_convert_to_tensors(pred) for pred in preds]
    refs_tensors = [_convert_to_tensors(ref) for ref in refs]
    
    # mAP@30
    m30 = MeanAveragePrecision(iou_thresholds=[0.3])
    m30.update(preds_tensors, refs_tensors)
    res30 = m30.compute()
    map30 = float(res30['map'])
    
    # mAP@50
    m50 = MeanAveragePrecision(iou_thresholds=[0.5])
    m50.update(preds_tensors, refs_tensors)
    res50 = m50.compute()
    map50 = float(res50['map'])
    
    # mAP@[50:95]
    ious = [th/100 for th in range(50, 100, 5)]
    m5095 = MeanAveragePrecision(iou_thresholds=ious)
    m5095.update(preds_tensors, refs_tensors)
    res5095 = m5095.compute()
    map50_95 = float(res5095['map'])
    
    return {'map30': map30, 'map50': map50, 'map50_95': map50_95}
