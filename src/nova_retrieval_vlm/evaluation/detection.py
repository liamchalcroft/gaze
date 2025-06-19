from __future__ import annotations
from typing import Sequence, Dict, Any
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate_detection(
    preds: Sequence[Dict[str, Any]],
    refs: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute mAP at IoU thresholds 0.3, 0.5, and averaged 0.50:0.95.

    Returns:
        Dictionary with keys 'map30', 'map50', 'map50_95'.
    """
    # mAP@30
    m30 = MeanAveragePrecision(iou_thresholds=[0.3])
    m30.update(preds, refs)
    res30 = m30.compute()
    map30 = float(res30['map'])
    # mAP@50
    m50 = MeanAveragePrecision(iou_thresholds=[0.5])
    m50.update(preds, refs)
    res50 = m50.compute()
    map50 = float(res50['map'])
    # mAP@[50:95]
    ious = [th/100 for th in range(50, 100, 5)]
    m5095 = MeanAveragePrecision(iou_thresholds=ious)
    m5095.update(preds, refs)
    res5095 = m5095.compute()
    map50_95 = float(res5095['map'])
    return {'map30': map30, 'map50': map50, 'map50_95': map50_95}
