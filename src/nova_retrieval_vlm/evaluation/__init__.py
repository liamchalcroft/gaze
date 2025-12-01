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
        pred_boxes = [p.get("boxes", []) for p in preds]
        ref_boxes = [r.get("boxes", []) for r in refs]

        det_metrics = evaluate_detection(pred_boxes, ref_boxes)
        result_metrics.update(
            {
                "detection_mAP30": det_metrics.get("map30", 0.0),
                "detection_mAP50": det_metrics.get("map50", 0.0),
                "detection_mAP50_95": det_metrics.get("map50_95", 0.0),
                "detection_ACC50": det_metrics.get("acc50", 0.0),
                "detection_TP30": det_metrics.get("tp30", 0),
                "detection_FP30": det_metrics.get("fp30", 0),
            }
        )

    elif task == "caption":
        from nova_retrieval_vlm.evaluation.caption import evaluate_caption

        pred_caps = [p.get("caption", "") for p in preds]
        ref_caps = [r.get("caption", "") for r in refs]

        cap_scores = evaluate_caption(pred_caps, ref_caps)
        result_metrics.update(
            {
                "caption_bleu": cap_scores.get("bleu", 0.0),
                "caption_bert_f1": cap_scores.get("bert_f1", 0.0),
                "caption_radgraph_f1": cap_scores.get("radgraph_f1"),
                "caption_meteor": cap_scores.get("meteor", 0.0),
                "caption_modality_f1": cap_scores.get("modality_f1", 0.0),
                "caption_clinical_f1": cap_scores.get("clinical_f1", 0.0),
                "caption_binary_accuracy": cap_scores.get("binary_accuracy", 0.0),
                "caption_binary_f1": cap_scores.get("binary_f1", 0.0),
            }
        )

    elif task == "diagnosis":
        pred_diags = [p.get("diagnosis", "") for p in preds]
        ref_diags = [r.get("diagnosis", "") for r in refs]

        diag_scores = evaluate_diagnosis_nova_official(pred_diags, ref_diags)
        result_metrics.update(
            {
                "diagnosis_top1": diag_scores.get("top1", 0.0),
                "diagnosis_top5": diag_scores.get("top5", 0.0),
                "diagnosis_coverage": diag_scores.get("coverage", 0.0),
                "diagnosis_entropy": diag_scores.get("entropy", 0.0),
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
