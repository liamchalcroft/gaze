from __future__ import annotations
from typing import Sequence, Any, Union
from collections import Counter
import math


def evaluate_diagnosis(
    preds: Sequence[Union[Any, list[Any]]],
    refs: Sequence[Any],
) -> dict[str, float]:
    """
    Compute Top-1 and Top-5 accuracy for diagnosis predictions,
    plus coverage and prediction entropy.

    Args:
        preds: List of predicted diagnosis or list of predictions (for top-5).
        refs: List of reference diagnoses.

    Returns:
        Dictionary with keys 'top1', 'top5', 'coverage', 'entropy'.
    """
    n = len(refs)
    if n == 0:
        return {'top1': 0.0, 'top5': 0.0, 'coverage': 0.0, 'entropy': 0.0}
    # Top-1 and Top-5
    top1_count = 0
    top5_count = 0
    all_preds = []
    for p, r in zip(preds, refs):
        if isinstance(p, list):
            top1 = p[0] if p else None
            if top1 == r:
                top1_count += 1
            if r in p:
                top5_count += 1
            all_preds.extend(p)
        else:
            if p == r:
                top1_count += 1
                top5_count += 1
            all_preds.append(p)
    top1 = top1_count / n
    top5 = top5_count / n
    # Coverage: unique preds vs unique refs
    uniq_preds = len(set(all_preds))
    uniq_refs = len(set(refs))
    coverage = uniq_preds / uniq_refs if uniq_refs > 0 else 0.0
    # Entropy of prediction distribution
    cnt = Counter(all_preds)
    entropy = 0.0
    for freq in cnt.values():
        p_i = freq / len(all_preds)
        entropy -= p_i * math.log(p_i + 1e-12, 2)
    return {'top1': top1, 'top5': top5, 'coverage': coverage, 'entropy': entropy}
