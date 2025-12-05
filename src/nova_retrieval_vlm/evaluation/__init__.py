"""Evaluation module for NOVA retrieval VLM tasks."""

from __future__ import annotations

import json
from pathlib import Path

from beartype import beartype

from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official


@beartype
def evaluate(
    preds_jsonl: str | Path, refs_jsonl: str | Path, task: str = "localization"
) -> dict[str, float]:
    """
    Run evaluation based on the specified task and return relevant scores.

    Args:
        preds_jsonl: Path to predictions JSONL.
        refs_jsonl: Path to reference JSONL.
        task: The specific task to evaluate ('localization', 'caption', or 'diagnosis').

    Returns:
        Dictionary of metric names to scores relevant to the specified task.
    """
    preds_path = Path(preds_jsonl)
    refs_path = Path(refs_jsonl)

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    if not refs_path.exists():
        raise FileNotFoundError(f"References file not found: {refs_path}")

    with preds_path.open() as f:
        preds = [json.loads(line) for line in f]
    with refs_path.open() as f:
        refs = [json.loads(line) for line in f]
    result_metrics = {}

    if task == "localization":
        # Validate required fields - fail fast on missing data
        pred_boxes = []
        for i, p in enumerate(preds):
            if "boxes" not in p:
                raise ValueError(f"Prediction {i} missing required 'boxes' field")
            pred_boxes.append(p["boxes"])

        ref_boxes = []
        for i, r in enumerate(refs):
            if "boxes" not in r:
                raise ValueError(f"Reference {i} missing required 'boxes' field")
            ref_boxes.append(r["boxes"])

        det_metrics = evaluate_detection(pred_boxes, ref_boxes)
        # Validate evaluation returned expected metrics
        required_metrics = ["map30", "map50", "map50_95", "acc50", "tp30", "fp30"]
        for metric in required_metrics:
            if metric not in det_metrics:
                raise KeyError(f"evaluate_detection missing required metric: {metric}")

        result_metrics.update(
            {
                "detection_mAP30": det_metrics["map30"],
                "detection_mAP50": det_metrics["map50"],
                "detection_mAP50_95": det_metrics["map50_95"],
                "detection_ACC50": det_metrics["acc50"],
                "detection_TP30": det_metrics["tp30"],
                "detection_FP30": det_metrics["fp30"],
            }
        )

    elif task == "caption":
        from nova_retrieval_vlm.evaluation.caption import evaluate_caption

        # Validate required fields - fail fast on missing data
        pred_caps = []
        for i, p in enumerate(preds):
            if "caption" not in p:
                raise ValueError(f"Prediction {i} missing required 'caption' field")
            pred_caps.append(p["caption"])

        ref_caps = []
        for i, r in enumerate(refs):
            if "caption" not in r:
                raise ValueError(f"Reference {i} missing required 'caption' field")
            ref_caps.append(r["caption"])

        cap_scores = evaluate_caption(pred_caps, ref_caps)
        # Validate evaluation returned expected metrics (radgraph_f1 may be None but key exists)
        required_metrics = [
            "bleu",
            "bert_f1",
            "meteor",
            "radgraph_f1",
            "modality_f1",
            "clinical_f1",
            "binary_accuracy",
            "binary_f1",
        ]
        for metric in required_metrics:
            if metric not in cap_scores:
                raise KeyError(f"evaluate_caption missing required metric: {metric}")

        result_metrics.update(
            {
                "caption_bleu": cap_scores["bleu"],
                "caption_bert_f1": cap_scores["bert_f1"],
                "caption_radgraph_f1": cap_scores["radgraph_f1"],  # May be None if radgraph not installed
                "caption_meteor": cap_scores["meteor"],
                "caption_modality_f1": cap_scores["modality_f1"],
                "caption_clinical_f1": cap_scores["clinical_f1"],
                "caption_binary_accuracy": cap_scores["binary_accuracy"],
                "caption_binary_f1": cap_scores["binary_f1"],
            }
        )

    elif task == "diagnosis":
        # Validate required fields - fail fast on missing data
        pred_diags = []
        for i, p in enumerate(preds):
            if "diagnosis" not in p:
                raise ValueError(f"Prediction {i} missing required 'diagnosis' field")
            pred_diags.append(p["diagnosis"])

        ref_diags = []
        for i, r in enumerate(refs):
            if "diagnosis" not in r:
                raise ValueError(f"Reference {i} missing required 'diagnosis' field")
            ref_diags.append(r["diagnosis"])

        diag_scores = evaluate_diagnosis_nova_official(pred_diags, ref_diags)
        # Validate evaluation returned expected metrics
        required_metrics = ["top1", "top5", "coverage", "entropy"]
        for metric in required_metrics:
            if metric not in diag_scores:
                raise KeyError(f"evaluate_diagnosis_nova_official missing required metric: {metric}")

        result_metrics.update(
            {
                "diagnosis_top1": diag_scores["top1"],
                "diagnosis_top5": diag_scores["top5"],
                "diagnosis_coverage": diag_scores["coverage"],
                "diagnosis_entropy": diag_scores["entropy"],
            }
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    return result_metrics


# Public API
__all__ = [
    "evaluate",
    "evaluate_detection",
    "evaluate_diagnosis_nova_official",
]
