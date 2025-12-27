"""Reward functions for NOVA Brain MRI MedMarks environment.

Provides verifiers-compatible reward functions for:
- Caption: Token F1 similarity
- Diagnosis: Top-k accuracy with medical term normalization
- Localization: IoU-based detection reward

These can be used individually or combined for multi-task evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import verifiers as vf

NOVATask = Literal["caption", "diagnosis", "localization", "all"]


def extract_completion_text(completion: Any) -> str:
    """Extract text content from a verifiers completion.

    Handles multiple formats:
    - Plain string
    - Message list with assistant role
    - Multimodal content lists

    Args:
        completion: Model completion in any supported format

    Returns:
        Extracted text content
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        for msg in reversed(completion):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")

    return str(completion or "")


def extract_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON object from text using robust parsing.

    Uses JSONDecoder.raw_decode() to correctly handle JSON strings
    that may contain braces, avoiding fragile brace-matching.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON dict or None
    """
    decoder = json.JSONDecoder()

    for i, c in enumerate(text):
        if c != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text, i)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    return None


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union for two bounding boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value in [0.0, 1.0]
    """
    if len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


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
    response = extract_json_response(text)

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


def _normalize_diagnosis(diagnosis: str) -> str:
    """Normalize diagnosis text for comparison."""
    normalized = diagnosis.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    for modifier in ["possible", "probable", "likely", "suspected", "mild", "moderate", "severe"]:
        normalized = normalized.replace(modifier, "")

    return " ".join(normalized.split())


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
    response = extract_json_response(text)

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


def localization_reward_factory(iou_threshold: float = 0.3):
    """Create localization reward function with configurable IoU threshold.

    Args:
        iou_threshold: Minimum IoU for positive match

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
        response = extract_json_response(text)

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
                iou = compute_iou(pred_box, ref_box)
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
    iou_threshold: float = 0.3,
):
    """Create combined reward function for all NOVA tasks.

    Args:
        caption_weight: Weight for caption reward
        diagnosis_weight: Weight for diagnosis reward
        localization_weight: Weight for localization reward
        iou_threshold: IoU threshold for localization

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
    iou_threshold: float = 0.3,
) -> vf.Rubric:
    """Create verifiers rubric for NOVA evaluation.

    Args:
        task: Which task(s) to evaluate
        iou_threshold: IoU threshold for localization

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
