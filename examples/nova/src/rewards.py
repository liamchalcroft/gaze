"""Verifiers-compatible reward functions for NOVA brain-MRI tasks.

Provides task-specific rewards for:
- Caption: BLEU + semantic similarity
- Diagnosis: Top-1/Top-5 accuracy with medical term normalization
- Localization: IoU-based detection reward

Also provides combined reward for multi-task training.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from typing import Literal

from beartype import beartype

from radiant_harness.utils import compute_iou
from radiant_harness.utils import extract_json_from_text
from radiant_harness.verifiers import BaseRewardFunction
from radiant_harness.verifiers import extract_completion_text

NOVATask = Literal["caption", "diagnosis", "localization", "all"]


@dataclass(frozen=True)
class NOVARewardWeights:
    """Weights for combining NOVA task rewards."""

    caption: float = 0.33
    diagnosis: float = 0.34
    localization: float = 0.33

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.caption + self.diagnosis + self.localization
        if abs(total - 1.0) > 1e-6:
            msg = f"Reward weights must sum to 1.0, got {total}"
            raise ValueError(msg)


DEFAULT_WEIGHTS = NOVARewardWeights()


@beartype
def compute_caption_reward(prediction: str, reference: str) -> float:
    """Compute caption reward using token overlap.

    Simple token F1 score without heavy dependencies.
    For full evaluation, use evaluation/caption.py with BLEU, BERT, etc.

    Args:
        prediction: Predicted caption text
        reference: Ground truth caption

    Returns:
        Token F1 score in [0.0, 1.0]
    """
    if not prediction or not reference:
        return 0.0

    pred_tokens = Counter(prediction.lower().split())
    ref_tokens = Counter(reference.lower().split())

    if not ref_tokens:
        return 0.0

    # Multiset intersection: min count for each shared token preserves frequency
    intersection = sum((pred_tokens & ref_tokens).values())
    pred_total = sum(pred_tokens.values())
    ref_total = sum(ref_tokens.values())

    precision = intersection / pred_total if pred_total else 0.0
    recall = intersection / ref_total

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


@beartype
def compute_diagnosis_reward(
    prediction: str | list[str],
    reference: str | list[str],
) -> float:
    """Compute diagnosis reward using top-k matching.

    Supports both single diagnosis and differential diagnosis lists.

    Args:
        prediction: Predicted diagnosis (single or list)
        reference: Ground truth diagnosis (single or list)

    Returns:
        Accuracy score in [0.0, 1.0]
    """
    # Normalize to lists
    pred_list = [prediction] if isinstance(prediction, str) else list(prediction)
    ref_list = [reference] if isinstance(reference, str) else list(reference)

    if not pred_list or not ref_list:
        return 0.0

    # Normalize diagnoses
    pred_normalized = {_normalize_diagnosis(d) for d in pred_list}
    ref_normalized = {_normalize_diagnosis(d) for d in ref_list}

    # Top-1: does first prediction match any reference?
    top1_match = _normalize_diagnosis(pred_list[0]) in ref_normalized

    # Coverage: what fraction of references are matched?
    matches = pred_normalized & ref_normalized
    coverage = len(matches) / len(ref_normalized)

    # Combined score: top-1 is weighted more
    return 0.6 * float(top1_match) + 0.4 * coverage


# Must stay in sync with evaluation/diagnosis.py._ABBREVIATION_MAPPING.
_ABBREVIATION_MAPPING: dict[str, str] = {
    "sod": "septo-optic dysplasia",
    "acc": "agenesis of corpus callosum",
    "cpa": "cerebellopontine angle",
    "avm": "arteriovenous malformation",
    "pnet": "primitive neuroectodermal tumor",
    "gbm": "glioblastoma multiforme",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "dwi": "diffusion weighted imaging",
    "flair": "fluid attenuated inversion recovery",
    "dc": "dermoid cyst",
    "ec": "epidermoid cyst",
    "ac": "arachnoid cyst",
    "cm": "cavernous malformation",
    "vs": "vestibular schwannoma",
    "an": "acoustic neuroma",
    "da": "diffuse axonal injury",
    "sah": "subarachnoid hemorrhage",
    "ich": "intracerebral hemorrhage",
    "ms": "multiple sclerosis",
    "nph": "normal pressure hydrocephalus",
}

_DASH_PATTERN = re.compile(r"\s*[–—]\s*")


def _normalize_diagnosis(diagnosis: str) -> str:
    """Normalize diagnosis text for reward comparison.

    MORE aggressive than evaluation/diagnosis.py:normalize_diagnosis_string()
    — also strips punctuation and hedging modifiers for simple string matching.
    """
    if not diagnosis:
        return ""

    # Lowercase and normalize dashes
    normalized = diagnosis.lower().strip()
    normalized = _DASH_PATTERN.sub("-", normalized)
    # Remove punctuation (except hyphens, which are clinically meaningful)
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    # Collapse whitespace
    normalized = " ".join(normalized.split())

    # Expand common abbreviations as whole words — aligned with evaluation/diagnosis.py
    for abbrev, full in _ABBREVIATION_MAPPING.items():
        normalized = re.sub(r"\b" + re.escape(abbrev) + r"\b", full, normalized)

    # Remove hedging modifiers only — severity qualifiers are clinically meaningful.
    # Use word boundaries to avoid corrupting substrings (e.g. "improbable").
    for modifier in ["possible", "probable", "likely", "suspected"]:
        normalized = re.sub(r"\b" + modifier + r"\b", "", normalized)

    # Final whitespace cleanup
    return " ".join(normalized.split())


@beartype
def _area_penalty(
    box: list[float],
    image_area: float,
    penalty_start: float = 0.5,
) -> float:
    """Compute penalty for degenerate full-image boxes.

    A box covering most of the image trivially overlaps any ground-truth,
    so we scale the reward toward 0 as predicted area grows.
    Linear ramp: full credit at <= penalty_start coverage, zero at 100%.

    Args:
        box: [x1, y1, x2, y2] in absolute pixels.
        image_area: Total image area (width * height).
        penalty_start: Area ratio above which penalty begins (default 0.5).

    Returns:
        Multiplier in [0.0, 1.0].
    """
    if image_area <= 0:
        return 1.0
    if penalty_start >= 1.0:
        return 1.0
    pred_area = abs(box[2] - box[0]) * abs(box[3] - box[1])
    area_ratio = pred_area / image_area
    if area_ratio <= penalty_start:
        return 1.0
    return max(0.0, (1.0 - area_ratio) / (1.0 - penalty_start))


@beartype
def compute_localization_reward(
    prediction: list[list[int | float]],
    reference: list[list[int | float]],
    iou_threshold: float = 0.5,
    image_area: float = 0.0,
    area_penalty_start: float = 0.5,
) -> float:
    """Compute localization reward using IoU matching.

    Uses IoU threshold of 0.5 by default to align with NOVA evaluation
    metric (ACC50 / mAP@0.5), preventing reward-eval misalignment.

    When image_area > 0, applies an area penalty to each predicted box
    to discourage degenerate full-image predictions that trivially
    overlap any ground-truth.

    Args:
        prediction: List of predicted bounding boxes [x1, y1, x2, y2]
        reference: List of ground truth bounding boxes
        iou_threshold: IoU threshold for positive match (default 0.5 per NOVA eval)
        image_area: Total image area (width * height) for area penalty.
            When 0 (default), no area penalty is applied.
        area_penalty_start: Area ratio above which penalty begins (default 0.5).

    Returns:
        Detection score in [0.0, 1.0]
    """
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    matched_refs = set()
    true_positives = 0
    penalty_sum = 0.0

    for pred_box in prediction:
        if len(pred_box) < 4:
            continue
        pred_box_f = [float(coord) for coord in pred_box[:4]]
        best_iou = 0.0
        best_ref_idx = -1

        for ref_idx, ref_box in enumerate(reference):
            if ref_idx in matched_refs:
                continue
            if len(ref_box) < 4:
                continue
            ref_box_f = [float(coord) for coord in ref_box[:4]]
            iou = compute_iou(pred_box_f, ref_box_f)
            if iou > best_iou:
                best_iou = iou
                best_ref_idx = ref_idx

        if best_iou >= iou_threshold and best_ref_idx >= 0:
            matched_refs.add(best_ref_idx)
            true_positives += 1
            # Accumulate area penalty for matched predictions
            if image_area > 0:
                penalty_sum += _area_penalty(pred_box_f, image_area, area_penalty_start)
            else:
                penalty_sum += 1.0

    # F1-like score
    precision = true_positives / len(prediction) if prediction else 0.0
    recall = true_positives / len(reference) if reference else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    # Apply average area penalty across matched predictions
    if true_positives > 0 and image_area > 0:
        avg_penalty = penalty_sum / true_positives
        f1 *= avg_penalty

    return f1


class NOVAVerifiersReward(BaseRewardFunction):
    """Verifiers-compatible reward function for NOVA tasks.

    Adapts NOVA evaluation metrics to the radiant_harness BaseRewardFunction
    interface for use with verifiers training and evaluation.

    Example:
        reward_fn = NOVAVerifiersReward(task="diagnosis")
        reward = reward_fn(prompt, completion, info)

        # Or via processor:
        processor = NOVAAgenticProcessor(task="localization")
        env_cls = processor.as_verifiers_env(dataset_path="train.jsonl")
    """

    def __init__(
        self,
        task: NOVATask = "all",
        weights: NOVARewardWeights | None = None,
    ) -> None:
        """Initialize NOVA reward function.

        Args:
            task: Which NOVA task(s) to compute reward for
            weights: Component weights for combined reward
        """
        self.task = task
        self.weights = weights or DEFAULT_WEIGHTS

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute NOVA reward for a completion.

        Args:
            prompt: The input prompt (unused, required by interface)
            completion: Model completion (string or message list)
            info: Case information with ground truth

        Returns:
            Task-specific reward value in [0.0, 1.0]
        """
        # Extract completion text and parse response
        comp_text = self._extract_text(completion)
        response = self._extract_json_response(comp_text)

        if response is None:
            return 0.0

        # Compute task-specific rewards
        if self.task == "caption":
            return self._compute_caption_task(response, info)
        if self.task == "diagnosis":
            return self._compute_diagnosis_task(response, info)
        if self.task == "localization":
            return self._compute_localization_task(response, info)

        # Combined reward for all tasks
        caption_reward = self._compute_caption_task(response, info)
        diagnosis_reward = self._compute_diagnosis_task(response, info)
        localization_reward = self._compute_localization_task(response, info)

        return (
            self.weights.caption * caption_reward
            + self.weights.diagnosis * diagnosis_reward
            + self.weights.localization * localization_reward
        )

    def _compute_caption_task(
        self,
        response: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """Compute caption reward."""
        pred_caption = response.get("caption", "")
        if isinstance(pred_caption, dict):
            pred_caption = pred_caption.get("description", pred_caption.get("text", ""))

        ref_caption = info.get("caption", info.get("gold_caption", ""))
        if isinstance(ref_caption, dict):
            ref_caption = ref_caption.get("description", ref_caption.get("text", ""))

        return compute_caption_reward(str(pred_caption), str(ref_caption))

    def _compute_diagnosis_task(
        self,
        response: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """Compute diagnosis reward."""
        diagnosis = response.get("diagnosis", {})
        if isinstance(diagnosis, str):
            pred_diag = diagnosis
        else:
            pred_diag = diagnosis.get(
                "primary_diagnosis",
                diagnosis.get("primary", diagnosis.get("diagnosis", "")),
            )

        ref_diagnosis = info.get("diagnosis", info.get("gold_diagnosis", ""))
        if isinstance(ref_diagnosis, dict):
            ref_diagnosis = ref_diagnosis.get(
                "primary_diagnosis",
                ref_diagnosis.get("primary", ref_diagnosis.get("diagnosis", "")),
            )

        return compute_diagnosis_reward(pred_diag, ref_diagnosis)

    def _compute_localization_task(
        self,
        response: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """Compute localization reward."""

        def _extract_boxes(raw: Any, *, is_prediction: bool) -> list[list[int | float]]:
            boxes: list[list[int | float]] = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        # Predictions must use "bounding_box" (matches NOVA schema).
                        # Ground truth may use "bbox" (dataset convention).
                        if is_prediction:
                            box = item.get("bounding_box")
                        else:
                            box = item.get("bounding_box", item.get("bbox"))
                        if isinstance(box, list) and len(box) >= 4:
                            boxes.append(box[:4])
                    elif isinstance(item, list) and len(item) >= 4:
                        boxes.append(item[:4])
                return boxes

            if isinstance(raw, dict):
                if isinstance(raw.get("boxes"), list):
                    return _extract_boxes(raw["boxes"], is_prediction=is_prediction)
                if isinstance(raw.get("localizations"), list):
                    return _extract_boxes(raw["localizations"], is_prediction=is_prediction)
                if is_prediction:
                    box = raw.get("bounding_box")
                else:
                    box = raw.get("bounding_box", raw.get("bbox"))
                if isinstance(box, list) and len(box) >= 4:
                    boxes.append(box[:4])
            return boxes

        pred_boxes = _extract_boxes(response.get("localization", {}), is_prediction=True)

        # Accept benchmark info in either {"boxes": ...} or {"localizations": [{"bbox": ...}]}
        ref_source = info.get("boxes", info.get("gold_boxes"))
        if ref_source is None:
            ref_source = info.get("localizations", info.get("gold_localizations", []))
        ref_boxes = _extract_boxes(ref_source, is_prediction=False)

        width = info.get("image_width", info.get("width", 0))
        height = info.get("image_height", info.get("height", 0))
        image_area = float(width) * float(height)

        return compute_localization_reward(pred_boxes, ref_boxes, image_area=image_area)

    def _extract_text(self, completion: Any) -> str:
        """Extract text from completion."""
        return extract_completion_text(completion)

    def _extract_json_response(self, text: str) -> dict[str, Any] | None:
        """Extract JSON response from text."""
        return extract_json_from_text(text)


__all__ = [
    "NOVARewardWeights",
    "NOVAVerifiersReward",
    "NOVATask",
    "compute_caption_reward",
    "compute_diagnosis_reward",
    "compute_localization_reward",
]
