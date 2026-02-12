"""Reward functions for NOVA Brain MRI MedMarks environment.

Provides verifiers-compatible reward functions for:
- Caption: Token F1 similarity
- Diagnosis: Top-k accuracy with medical term normalization
- Localization: IoU-based detection reward

These can be used individually or combined for multi-task evaluation.

All shared utilities (IoU, JSON extraction, completion text extraction) are
imported from radiant_harness to maintain parity with the examples/ rewards.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import verifiers as vf

from radiant_harness.utils.iou import compute_iou
from radiant_harness.utils.json_extract import extract_json_from_text
from radiant_harness.verifiers.rewards import extract_completion_text

NOVATask = Literal["caption", "diagnosis", "localization", "all"]


def _normalize_diagnosis(diagnosis: str) -> str:
    """Normalize diagnosis text for comparison.

    Strips hedging modifiers (possible, probable, likely, suspected) that
    indicate uncertainty but NOT severity qualifiers (mild, moderate, severe)
    which are clinically meaningful and change the diagnosis.
    """
    normalized = diagnosis.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    # Remove hedging modifiers only — severity qualifiers are clinically meaningful
    for modifier in ["possible", "probable", "likely", "suspected"]:
        normalized = normalized.replace(modifier, "")
    return " ".join(normalized.split())


def caption_reward(
    prompt: str,  # noqa: ARG001 - Required by verifiers interface
    completion: Any,
    info: dict[str, Any],
) -> float:
    """Compute caption reward using token F1 similarity.

    Args:
        prompt: Input prompt (unused)
        completion: Model completion
        info: Case info with ground truth

    Returns:
        Token F1 score in [0.0, 1.0]
    """
    text = extract_completion_text(completion)
    response = extract_json_from_text(text)

    if response is None:
        return 0.0

    pred_caption = response.get("caption", "")
    if isinstance(pred_caption, dict):
        pred_caption = pred_caption.get("description", "")

    ref_caption = info.get("caption", info.get("gold_caption", ""))
    if isinstance(ref_caption, dict):
        ref_caption = ref_caption.get("description", "")

    if not pred_caption or not ref_caption:
        return 0.0

    pred_tokens = set(str(pred_caption).lower().split())
    ref_tokens = set(str(ref_caption).lower().split())

    if not ref_tokens:
        return 0.0

    intersection = pred_tokens & ref_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def diagnosis_reward(
    prompt: str,  # noqa: ARG001 - Required by verifiers interface
    completion: Any,
    info: dict[str, Any],
) -> float:
    """Compute diagnosis reward using top-k accuracy.

    Args:
        prompt: Input prompt (unused)
        completion: Model completion
        info: Case info with ground truth

    Returns:
        Accuracy score in [0.0, 1.0]
    """
    text = extract_completion_text(completion)
    response = extract_json_from_text(text)

    if response is None:
        return 0.0

    diagnosis = response.get("diagnosis", {})
    if isinstance(diagnosis, str):
        pred_primary = diagnosis
        pred_list = [diagnosis]
    else:
        pred_primary = diagnosis.get("primary_diagnosis", "")
        pred_list = [pred_primary]

        differentials = diagnosis.get("differential_diagnoses", [])
        for diff in differentials:
            if isinstance(diff, dict):
                pred_list.append(diff.get("diagnosis", ""))
            elif isinstance(diff, str):
                pred_list.append(diff)

    ref_diagnosis = info.get("diagnosis", info.get("gold_diagnosis", ""))
    if isinstance(ref_diagnosis, dict):
        ref_diagnosis = ref_diagnosis.get("primary", ref_diagnosis.get("diagnosis", ""))
    ref_list = [ref_diagnosis] if isinstance(ref_diagnosis, str) else list(ref_diagnosis)

    if not pred_list or not ref_list:
        return 0.0

    pred_normalized = {_normalize_diagnosis(d) for d in pred_list if d}
    ref_normalized = {_normalize_diagnosis(d) for d in ref_list if d}

    if not ref_normalized:
        return 0.0

    top1_match = _normalize_diagnosis(pred_primary) in ref_normalized

    matches = pred_normalized & ref_normalized
    coverage = len(matches) / len(ref_normalized)

    return 0.6 * float(top1_match) + 0.4 * coverage


def localization_reward_factory(iou_threshold: float = 0.5):
    """Create localization reward function with configurable IoU threshold.

    Default threshold is 0.5 to align with NOVA evaluation metric
    (ACC50 / mAP@0.5), preventing reward-eval misalignment.

    Args:
        iou_threshold: Minimum IoU for positive match (default 0.5 per NOVA eval)

    Returns:
        Reward function
    """

    def localization_reward(
        prompt: str,  # noqa: ARG001 - Required by verifiers interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute localization reward using IoU matching.

        Args:
            prompt: Input prompt (unused)
            completion: Model completion
            info: Case info with ground truth

        Returns:
            Detection F1 score in [0.0, 1.0]
        """
        text = extract_completion_text(completion)
        response = extract_json_from_text(text)

        if response is None:
            return 0.0

        localization = response.get("localization", {})
        pred_boxes = []

        if isinstance(localization, list):
            for loc in localization:
                if isinstance(loc, dict) and "bounding_box" in loc:
                    pred_boxes.append(loc["bounding_box"])
                elif isinstance(loc, dict) and "bbox" in loc:
                    pred_boxes.append(loc["bbox"])
                elif isinstance(loc, list) and len(loc) == 4:
                    pred_boxes.append(loc)
        elif isinstance(localization, dict):
            localizations = localization.get("localizations", [])
            for loc in localizations:
                if isinstance(loc, dict):
                    box = loc.get("bounding_box") or loc.get("bbox")
                    if box:
                        pred_boxes.append(box)

        ref_boxes = info.get("boxes", info.get("gold_boxes", []))
        if isinstance(ref_boxes, dict):
            ref_boxes = ref_boxes.get("boxes", [])

        if not pred_boxes and not ref_boxes:
            return 1.0
        if not pred_boxes or not ref_boxes:
            return 0.0

        matched_refs = set()
        true_positives = 0

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_ref_idx = -1

            for ref_idx, ref_box in enumerate(ref_boxes):
                if ref_idx in matched_refs:
                    continue
                # Convert to float for compute_iou (handles int coords from JSON)
                pred_floats = [float(c) for c in pred_box[:4]]
                ref_floats = [float(c) for c in ref_box[:4]]
                iou = compute_iou(pred_floats, ref_floats)
                if iou > best_iou:
                    best_iou = iou
                    best_ref_idx = ref_idx

            if best_iou >= iou_threshold and best_ref_idx >= 0:
                matched_refs.add(best_ref_idx)
                true_positives += 1

        precision = true_positives / len(pred_boxes)
        recall = true_positives / len(ref_boxes)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    return localization_reward


def combined_reward_factory(
    caption_weight: float = 0.33,
    diagnosis_weight: float = 0.34,
    localization_weight: float = 0.33,
    iou_threshold: float = 0.5,
):
    """Create combined reward function for all NOVA tasks.

    Args:
        caption_weight: Weight for caption reward
        diagnosis_weight: Weight for diagnosis reward
        localization_weight: Weight for localization reward
        iou_threshold: IoU threshold for localization (default 0.5 per NOVA eval)

    Returns:
        Combined reward function
    """
    loc_reward = localization_reward_factory(iou_threshold)

    def combined_reward(
        prompt: str,
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute combined NOVA reward.

        Args:
            prompt: Input prompt
            completion: Model completion
            info: Case info with ground truth

        Returns:
            Weighted combined score in [0.0, 1.0]
        """
        cap_score = caption_reward(prompt, completion, info)
        diag_score = diagnosis_reward(prompt, completion, info)
        loc_score = loc_reward(prompt, completion, info)

        return (
            caption_weight * cap_score
            + diagnosis_weight * diag_score
            + localization_weight * loc_score
        )

    return combined_reward


def create_nova_rubric(
    task: NOVATask = "all",
    iou_threshold: float = 0.5,
) -> vf.Rubric:
    """Create verifiers rubric for NOVA evaluation.

    Args:
        task: Which task(s) to evaluate
        iou_threshold: IoU threshold for localization (default 0.5 per NOVA eval)

    Returns:
        Configured rubric
    """
    if task == "caption":
        return vf.Rubric(funcs=[caption_reward])

    if task == "diagnosis":
        return vf.Rubric(funcs=[diagnosis_reward])

    if task == "localization":
        return vf.Rubric(funcs=[localization_reward_factory(iou_threshold)])

    # Combined reward for "all" task
    return vf.Rubric(
        funcs=[
            caption_reward,
            diagnosis_reward,
            localization_reward_factory(iou_threshold),
        ],
        weights=[0.33, 0.34, 0.33],
    )


__all__ = [
    "caption_reward",
    "combined_reward_factory",
    "create_nova_rubric",
    "diagnosis_reward",
    "localization_reward_factory",
]
