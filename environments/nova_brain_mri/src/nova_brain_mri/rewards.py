"""Reward functions for NOVA Brain MRI MedMarks environment.

Provides verifiers-compatible reward functions for:
- Caption: Token F1 similarity (multiset intersection)
- Diagnosis: Top-1 accuracy with medical term normalization
- Localization: IoU-based detection reward with area penalty

These can be used individually or combined for multi-task evaluation.

Utilities are inlined in _utils.py so this package is fully standalone.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal

import verifiers as vf

from ._utils import compute_iou, extract_completion_text, extract_json_from_text

NOVATask = Literal["caption", "diagnosis", "localization", "all"]

# Stopwords that inflate token overlap without carrying semantic content.
# Must stay in sync with examples/nova/src/rewards.py._CAPTION_STOPWORDS.
_CAPTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "his",
        "her",
        "its",
        "their",
        "this",
        "that",
        "these",
        "those",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "during",
        "before",
        "after",
        "and",
        "or",
        "but",
        "nor",
        "not",
        "no",
        "so",
        "if",
        "then",
    }
)

# Must stay in sync with examples/nova/src/rewards.py._ABBREVIATION_MAPPING.
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
    """Normalize diagnosis text for comparison.

    Strips hedging modifiers (possible, probable, likely, suspected) that
    indicate uncertainty but NOT severity qualifiers (mild, moderate, severe)
    which are clinically meaningful and change the diagnosis.

    Preserves hyphens (clinically meaningful, e.g. "septo-optic") and
    expands common medical abbreviations for robust matching.
    """
    if not diagnosis:
        return ""

    # Lowercase and normalize dashes
    normalized = diagnosis.lower().strip()
    normalized = _DASH_PATTERN.sub("-", normalized)
    # Remove punctuation EXCEPT hyphens (clinically meaningful)
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    # Collapse whitespace
    normalized = " ".join(normalized.split())

    # Expand common abbreviations as whole words
    for abbrev, full in _ABBREVIATION_MAPPING.items():
        normalized = re.sub(r"\b" + re.escape(abbrev) + r"\b", full, normalized)

    # Remove hedging modifiers only, with word boundaries to avoid
    # corrupting substrings (e.g. "improbable" should not become "im able")
    for modifier in ["possible", "probable", "likely", "suspected"]:
        normalized = re.sub(r"\b" + modifier + r"\b", "", normalized)

    # Final whitespace cleanup
    return " ".join(normalized.split())


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


def caption_reward(
    prompt: str,  # noqa: ARG001 - Required by verifiers interface
    completion: Any,
    info: dict[str, Any],
) -> float:
    """Compute caption reward using token F1 similarity.

    Uses multiset (Counter) intersection to preserve token frequency,
    matching the examples/nova reference implementation.

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

    pred_tokens = Counter(
        t for t in re.findall(r"\b\w+\b", str(pred_caption).lower()) if t not in _CAPTION_STOPWORDS
    )
    ref_tokens = Counter(
        t for t in re.findall(r"\b\w+\b", str(ref_caption).lower()) if t not in _CAPTION_STOPWORDS
    )

    # Both empty after stopword filtering = trivial match (aligned with core TokenF1Reward)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
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
    else:
        pred_primary = diagnosis.get(
            "primary_diagnosis",
            diagnosis.get("primary", diagnosis.get("diagnosis", "")),
        )

    ref_diagnosis = info.get("diagnosis", info.get("gold_diagnosis", ""))
    if isinstance(ref_diagnosis, dict):
        ref_diagnosis = ref_diagnosis.get(
            "primary_diagnosis",
            ref_diagnosis.get("primary", ref_diagnosis.get("diagnosis", "")),
        )

    # Match examples/nova: score primary prediction against reference only.
    # Differentials are not included to avoid inflating scores relative to
    # the examples/nova pipeline.
    pred_list = [pred_primary] if isinstance(pred_primary, str) else list(pred_primary)
    ref_list = [ref_diagnosis] if isinstance(ref_diagnosis, str) else list(ref_diagnosis)

    if not pred_list or not ref_list:
        return 0.0

    pred_normalized = {_normalize_diagnosis(d) for d in pred_list if d}
    ref_normalized = {_normalize_diagnosis(d) for d in ref_list if d}

    if not ref_normalized:
        return 0.0

    top1_match = _normalize_diagnosis(pred_primary) in ref_normalized

    # Coverage: F1 over matched diagnoses (penalises false positives)
    matches = pred_normalized & ref_normalized
    if not matches:
        coverage = 0.0
    else:
        cov_precision = len(matches) / len(pred_normalized)
        cov_recall = len(matches) / len(ref_normalized)
        coverage = 2 * cov_precision * cov_recall / (cov_precision + cov_recall)

    return 0.6 * float(top1_match) + 0.4 * coverage


def _extract_pred_boxes(localization: Any) -> list[list[float]]:
    """Extract predicted bounding boxes from localization response.

    Only accepts "bounding_box" key from predictions (NOVA schema).
    The "bbox" key is a ground-truth dataset convention and should not
    be accepted from model predictions to enforce schema compliance.
    """
    pred_boxes: list[list[float]] = []

    if isinstance(localization, list):
        for loc in localization:
            if isinstance(loc, dict) and "bounding_box" in loc:
                box = loc["bounding_box"]
                if isinstance(box, list) and len(box) >= 4:
                    pred_boxes.append([float(c) for c in box[:4]])
            elif isinstance(loc, list) and len(loc) >= 4:
                pred_boxes.append([float(c) for c in loc[:4]])
    elif isinstance(localization, dict):
        localizations = localization.get("localizations", [])
        for loc in localizations:
            if isinstance(loc, dict):
                box = loc.get("bounding_box")
                if isinstance(box, list) and len(box) >= 4:
                    pred_boxes.append([float(c) for c in box[:4]])

    return pred_boxes


def _extract_ref_boxes(info: dict[str, Any]) -> list[list[float]]:
    """Extract reference bounding boxes from info dict.

    Accepts both "bounding_box" and "bbox" keys (dataset conventions).
    Falls back to "localizations"/"gold_localizations" keys.
    """
    ref_source = info.get("boxes", info.get("gold_boxes"))
    if ref_source is None:
        ref_source = info.get("localizations", info.get("gold_localizations", []))
    if isinstance(ref_source, dict):
        ref_source = ref_source.get("boxes", ref_source.get("localizations", []))

    ref_boxes: list[list[float]] = []
    if isinstance(ref_source, list):
        for item in ref_source:
            if isinstance(item, dict):
                box = item.get("bounding_box", item.get("bbox"))
                if isinstance(box, list) and len(box) >= 4:
                    ref_boxes.append([float(c) for c in box[:4]])
            elif isinstance(item, list) and len(item) >= 4:
                ref_boxes.append([float(c) for c in item[:4]])

    return ref_boxes


def localization_reward_factory(
    iou_threshold: float = 0.5,
    area_penalty_start: float = 0.5,
):
    """Create localization reward function with configurable IoU threshold.

    Default threshold is 0.5 to align with NOVA evaluation metric
    (ACC50 / mAP@0.5), preventing reward-eval misalignment.

    Args:
        iou_threshold: Minimum IoU for positive match (default 0.5 per NOVA eval)
        area_penalty_start: Area ratio above which penalty begins (default 0.5)

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

        pred_boxes = _extract_pred_boxes(response.get("localization", {}))
        ref_boxes = _extract_ref_boxes(info)

        if not pred_boxes and not ref_boxes:
            return 1.0
        if not pred_boxes or not ref_boxes:
            return 0.0

        # Compute image area for area penalty
        width = info.get("image_width", info.get("width", 0))
        height = info.get("image_height", info.get("height", 0))
        image_area = float(width) * float(height)

        if image_area <= 0 and pred_boxes:
            import warnings

            warnings.warn(
                "localization_reward: image_area=0 disables area penalty. "
                "Full-image boxes will receive full IoU credit. Pass "
                "image_width/image_height in info when degenerate-box "
                "gaming matters (e.g. RL training).",
                stacklevel=2,
            )

        matched_refs: set[int] = set()
        true_positives = 0
        penalty_sum = 0.0

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
                if image_area > 0:
                    penalty_sum += _area_penalty(pred_box, image_area, area_penalty_start)
                else:
                    penalty_sum += 1.0

        precision = true_positives / len(pred_boxes)
        recall = true_positives / len(ref_boxes)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)

        # Apply average area penalty across matched predictions
        if true_positives > 0 and image_area > 0:
            avg_penalty = penalty_sum / true_positives
            f1 *= avg_penalty

        return f1

    return localization_reward


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
    "create_nova_rubric",
    "diagnosis_reward",
    "localization_reward_factory",
]
