"""Evaluation metrics for PubmedQA.

Computes accuracy and other metrics for yes/no/maybe classification.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from beartype import beartype


@beartype
def evaluate_pubmedqa(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Evaluate PubmedQA predictions against ground truth.

    Args:
        predictions: List of predicted answers ("yes", "no", "maybe")
        references: List of ground truth answers

    Returns:
        Dictionary with evaluation metrics:
        - accuracy: Overall accuracy
        - accuracy_yes: Accuracy on "yes" samples
        - accuracy_no: Accuracy on "no" samples
        - accuracy_maybe: Accuracy on "maybe" samples
        - macro_f1: Macro-averaged F1 score

    Raises:
        ValueError: If predictions and references have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    if not predictions:
        raise ValueError("Cannot evaluate empty predictions")

    # Normalize to lowercase
    preds = [p.lower().strip() for p in predictions]
    refs = [r.lower().strip() for r in references]

    # Overall accuracy
    correct = sum(1 for p, r in zip(preds, refs, strict=True) if p == r)
    accuracy = correct / len(preds)

    # Per-class metrics
    classes = ["yes", "no", "maybe"]
    class_metrics: dict[str, dict[str, float]] = {}

    for cls in classes:
        # True positives, false positives, false negatives
        tp = sum(1 for p, r in zip(preds, refs, strict=True) if p == cls and r == cls)
        fp = sum(1 for p, r in zip(preds, refs, strict=True) if p == cls and r != cls)
        fn = sum(1 for p, r in zip(preds, refs, strict=True) if p != cls and r == cls)

        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Class-specific accuracy
        class_total = sum(1 for r in refs if r == cls)
        class_correct = sum(1 for p, r in zip(preds, refs, strict=True) if p == r and r == cls)
        class_acc = class_correct / class_total if class_total > 0 else 0.0

        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": class_acc,
            "support": class_total,
        }

    # Macro-averaged F1
    macro_f1 = sum(class_metrics[cls]["f1"] for cls in classes) / len(classes)

    # Distribution analysis
    pred_dist = Counter(preds)
    ref_dist = Counter(refs)

    return {
        "accuracy": accuracy,
        "accuracy_yes": class_metrics["yes"]["accuracy"],
        "accuracy_no": class_metrics["no"]["accuracy"],
        "accuracy_maybe": class_metrics["maybe"]["accuracy"],
        "precision_yes": class_metrics["yes"]["precision"],
        "precision_no": class_metrics["no"]["precision"],
        "precision_maybe": class_metrics["maybe"]["precision"],
        "recall_yes": class_metrics["yes"]["recall"],
        "recall_no": class_metrics["no"]["recall"],
        "recall_maybe": class_metrics["maybe"]["recall"],
        "f1_yes": class_metrics["yes"]["f1"],
        "f1_no": class_metrics["no"]["f1"],
        "f1_maybe": class_metrics["maybe"]["f1"],
        "macro_f1": macro_f1,
        "support_yes": float(class_metrics["yes"]["support"]),
        "support_no": float(class_metrics["no"]["support"]),
        "support_maybe": float(class_metrics["maybe"]["support"]),
        "pred_dist_yes": pred_dist.get("yes", 0) / len(preds),
        "pred_dist_no": pred_dist.get("no", 0) / len(preds),
        "pred_dist_maybe": pred_dist.get("maybe", 0) / len(preds),
        "ref_dist_yes": ref_dist.get("yes", 0) / len(refs),
        "ref_dist_no": ref_dist.get("no", 0) / len(refs),
        "ref_dist_maybe": ref_dist.get("maybe", 0) / len(refs),
    }
