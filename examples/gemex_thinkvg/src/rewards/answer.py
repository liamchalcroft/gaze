"""Answer verification reward for GEMeX-ThinkVG.

Computes reward based on semantic matching between predicted
and ground truth medical findings/diagnoses.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from beartype import beartype

# Common medical abbreviations and their expansions
MEDICAL_ABBREVIATIONS: dict[str, list[str]] = {
    "cxr": ["chest x-ray", "chest x ray", "chest radiograph"],
    "pa": ["posteroanterior", "posterior anterior"],
    "ap": ["anteroposterior", "anterior posterior"],
    "lat": ["lateral"],
    "lll": ["left lower lobe"],
    "rll": ["right lower lobe"],
    "lul": ["left upper lobe"],
    "rul": ["right upper lobe"],
    "rml": ["right middle lobe"],
    "bil": ["bilateral"],
    "r/o": ["rule out"],
    "c/w": ["consistent with", "compatible with"],
    "s/p": ["status post"],
    "wrt": ["with respect to"],
    "hx": ["history"],
    "dx": ["diagnosis"],
    "tx": ["treatment"],
    "rx": ["prescription", "treatment"],
    "fx": ["fracture"],
    "ptx": ["pneumothorax"],
    "chf": ["congestive heart failure", "heart failure"],
    "copd": ["chronic obstructive pulmonary disease"],
    "ards": ["acute respiratory distress syndrome"],
    "pe": ["pulmonary embolism"],
    "tb": ["tuberculosis"],
    "ca": ["carcinoma", "cancer"],
}


@beartype
def normalize_medical_text(text: str) -> str:
    """Normalize medical text for comparison.

    - Lowercase
    - Expand common abbreviations
    - Remove punctuation
    - Collapse whitespace

    Args:
        text: Raw medical text

    Returns:
        Normalized text
    """
    text = text.lower().strip()

    # Expand abbreviations
    for abbrev, expansions in MEDICAL_ABBREVIATIONS.items():
        # Match abbreviation as whole word
        pattern = rf"\b{re.escape(abbrev)}\b"
        text = re.sub(pattern, expansions[0], text)

    # Remove punctuation except hyphens in medical terms
    text = re.sub(r"[^\w\s-]", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text


@beartype
def compute_token_overlap(pred: str, ref: str) -> float:
    """Compute token-level overlap between prediction and reference.

    Args:
        pred: Predicted answer (normalized)
        ref: Reference answer (normalized)

    Returns:
        F1-style overlap score (0.0-1.0)
    """
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = pred_tokens & ref_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


@beartype
def compute_exact_match(pred: str, ref: str) -> float:
    """Check exact match after normalization.

    Args:
        pred: Predicted answer
        ref: Reference answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if normalize_medical_text(pred) == normalize_medical_text(ref) else 0.0


@beartype
def compute_contains_match(pred: str, ref: str) -> float:
    """Check if prediction contains or is contained by reference.

    Returns a score discounted by the length ratio so that a model
    cannot game the reward by dumping all possible answers into a
    single long prediction string.

    Score = containment_flag * min(len_short, len_long) / max(len_short, len_long)

    Args:
        pred: Predicted answer (normalized)
        ref: Reference answer (normalized)

    Returns:
        Score in [0.0, 1.0], discounted by length ratio
    """
    pred_norm = normalize_medical_text(pred)
    ref_norm = normalize_medical_text(ref)

    if not pred_norm or not ref_norm:
        return 1.0 if pred_norm == ref_norm else 0.0

    if pred_norm in ref_norm or ref_norm in pred_norm:
        shorter = min(len(pred_norm), len(ref_norm))
        longer = max(len(pred_norm), len(ref_norm))
        return shorter / longer

    return 0.0


@beartype
def compute_answer_reward(
    prediction: str,
    reference: str,
    question_type: str = "open_ended",
) -> dict[str, Any]:
    """Compute answer verification reward.

    Combines multiple matching strategies:
    - Exact match (highest weight for closed questions)
    - Containment match
    - Token overlap (F1-style)

    Args:
        prediction: Model's answer
        reference: Ground truth answer
        question_type: Type of question for weighting

    Returns:
        Dict with component scores and final reward
    """
    pred_norm = normalize_medical_text(prediction)
    ref_norm = normalize_medical_text(reference)

    # Compute component scores
    exact = compute_exact_match(prediction, reference)
    contains = compute_contains_match(prediction, reference)
    token_f1 = compute_token_overlap(pred_norm, ref_norm)

    # Weight based on question type
    if question_type in ("closed_ended", "single_choice"):
        # Exact match is critical for closed questions
        weights = {"exact": 0.7, "contains": 0.2, "token_f1": 0.1}
    elif question_type == "multi_choice":
        # Need to match multiple correct options
        weights = {"exact": 0.5, "contains": 0.3, "token_f1": 0.2}
    else:  # open_ended
        # More flexibility for open-ended answers
        weights = {"exact": 0.3, "contains": 0.3, "token_f1": 0.4}

    reward = (
        weights["exact"] * exact
        + weights["contains"] * contains
        + weights["token_f1"] * token_f1
    )

    return {
        "exact_match": exact,
        "contains_match": contains,
        "token_f1": token_f1,
        "reward": reward,
        "question_type": question_type,
    }


@beartype
def compute_batch_answer_rewards(
    predictions: Sequence[str],
    references: Sequence[str],
    question_types: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute answer rewards for a batch of samples.

    Args:
        predictions: List of predicted answers
        references: List of reference answers
        question_types: Optional list of question types

    Returns:
        List of reward dicts for each sample
    """
    if question_types is None:
        question_types = ["open_ended"] * len(predictions)

    return [
        compute_answer_reward(pred, ref, q_type)
        for pred, ref, q_type in zip(predictions, references, question_types, strict=True)
    ]
