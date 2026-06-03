"""Evaluation metrics for VQA-RAD.

Computes accuracy and other metrics for visual question answering.
Handles both closed (yes/no) and open-ended questions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from beartype import beartype

_BINARY_YES = frozenset({"yes", "y", "true", "1"})
_BINARY_NO = frozenset({"no", "n", "false", "0"})


@beartype
def normalize_binary(answer: str) -> str | None:
    """Normalize a closed-ended answer to 'yes' or 'no'.

    Handles common variations (y/n, true/false, 1/0) after stripping
    punctuation and whitespace.

    Args:
        answer: Raw answer string

    Returns:
        'yes', 'no', or None if the answer is not a recognized binary value
    """
    normalized = normalize_answer(answer)
    # After normalization, take the first token only so that
    # "yes, the image shows..." still maps to "yes".
    first_token = normalized.split()[0] if normalized else ""
    if first_token in _BINARY_YES:
        return "yes"
    if first_token in _BINARY_NO:
        return "no"
    return None


# Common medical synonyms in VQA-RAD answers — normalize to the most
# frequent form in the dataset so that semantically equivalent answers
# are not penalised by exact/token matching.
_MEDICAL_SYNONYMS: dict[str, str] = {
    "hemorrhage": "bleeding",
    "haemorrhage": "bleeding",
    "carcinoma": "cancer",
    "neoplasm": "tumor",
    "tumour": "tumor",
    "oedema": "edema",
    "myocardial infarction": "heart attack",
    "cerebrovascular accident": "stroke",
    "fracture": "broken bone",
    "pneumothorax": "collapsed lung",
    "pulmonary embolism": "pe",
    "ct scan": "ct",
    "computed tomography": "ct",
    "magnetic resonance imaging": "mri",
    "x ray": "xray",
    "chest x ray": "chest xray",
    "abdominal": "abdomen",
}


@beartype
def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized lowercase answer with medical synonym substitution
    """
    # Lowercase and strip whitespace
    answer = answer.lower().strip()

    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)

    # Collapse whitespace
    answer = " ".join(answer.split())

    # Apply medical synonym normalization (longer phrases first)
    for term, replacement in sorted(_MEDICAL_SYNONYMS.items(), key=lambda x: -len(x[0])):
        answer = re.sub(r"\b" + re.escape(term) + r"\b", replacement, answer)

    return answer


@beartype
def compute_exact_match(pred: str, ref: str) -> bool:
    """Check if prediction exactly matches reference after normalization.

    Args:
        pred: Predicted answer
        ref: Reference answer

    Returns:
        True if exact match
    """
    return normalize_answer(pred) == normalize_answer(ref)


@beartype
def compute_token_f1(pred: str, ref: str) -> float:
    """Compute token-level F1 score between prediction and reference.

    Args:
        pred: Predicted answer
        ref: Reference answer

    Returns:
        F1 score (0.0-1.0)
    """
    pred_tokens = set(normalize_answer(pred).split())
    ref_tokens = set(normalize_answer(ref).split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


@beartype
def evaluate_vqa_rad(
    predictions: Sequence[str],
    references: Sequence[str],
    answer_types: Sequence[str] | None = None,
) -> dict[str, float]:
    """Evaluate VQA-RAD predictions against ground truth.

    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
        answer_types: Optional list of answer types ("closed" or "open")
                     for per-type evaluation

    Returns:
        Dictionary with evaluation metrics:
        - exact_match: Overall exact match accuracy
        - token_f1: Average token-level F1 score
        - closed_accuracy: Accuracy on closed questions (if types provided)
        - open_accuracy: Accuracy on open questions (if types provided)
        - open_f1: F1 on open questions (if types provided)

    Raises:
        ValueError: If predictions and references have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    if not predictions:
        raise ValueError("Cannot evaluate empty predictions")

    # Compute overall metrics
    exact_matches = [
        compute_exact_match(p, r) for p, r in zip(predictions, references, strict=True)
    ]
    token_f1s = [compute_token_f1(p, r) for p, r in zip(predictions, references, strict=True)]

    metrics: dict[str, float] = {
        "exact_match": sum(exact_matches) / len(exact_matches),
        "token_f1": sum(token_f1s) / len(token_f1s),
        "num_samples": float(len(predictions)),
    }

    # Per-type evaluation if answer types provided
    if answer_types is not None:
        if len(answer_types) != len(predictions):
            raise ValueError("answer_types length must match predictions")

        closed_matches: list[bool] = []
        open_matches: list[bool] = []
        open_f1s: list[float] = []

        for i, (pred, ref, t) in enumerate(zip(predictions, references, answer_types, strict=True)):
            if t == "closed":
                # Use normalize_binary for closed questions (first-token
                # yes/no extraction) to match the reward function and
                # evaluate_closed_only, rather than full-string exact match.
                pred_bin = normalize_binary(pred)
                ref_bin = normalize_binary(ref)
                closed_matches.append(
                    pred_bin is not None and ref_bin is not None and pred_bin == ref_bin
                )
            else:
                open_matches.append(exact_matches[i])
                open_f1s.append(token_f1s[i])

        if closed_matches:
            metrics["closed_accuracy"] = sum(closed_matches) / len(closed_matches)
            metrics["num_closed"] = float(len(closed_matches))
        else:
            metrics["closed_accuracy"] = 0.0
            metrics["num_closed"] = 0.0

        if open_matches:
            metrics["open_accuracy"] = sum(open_matches) / len(open_matches)
            metrics["open_f1"] = sum(open_f1s) / len(open_f1s)
            metrics["num_open"] = float(len(open_matches))
        else:
            metrics["open_accuracy"] = 0.0
            metrics["open_f1"] = 0.0
            metrics["num_open"] = 0.0

    return metrics


@beartype
def evaluate_closed_only(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Evaluate only closed (yes/no) questions.

    Args:
        predictions: List of predicted answers
        references: List of ground truth answers

    Returns:
        Dictionary with yes/no accuracy metrics
    """
    preds_binary = [normalize_binary(p) for p in predictions]
    refs_binary = [normalize_binary(r) for r in references]

    # Filter to valid yes/no pairs.  We require the *reference* to be a
    # recognized binary value (otherwise the sample isn't truly closed).
    # Predictions that don't normalize to yes/no (e.g. "possibly") stay as
    # None and will simply fail the equality check, counting as incorrect.
    valid_pairs = [(p, r) for p, r in zip(preds_binary, refs_binary, strict=True) if r is not None]

    if not valid_pairs:
        return {"accuracy": 0.0, "num_samples": 0.0}

    valid_preds, valid_refs = zip(*valid_pairs, strict=True)

    correct = sum(1 for p, r in zip(valid_preds, valid_refs, strict=True) if p == r)

    # Per-class accuracy
    yes_refs = [(p, r) for p, r in zip(valid_preds, valid_refs, strict=True) if r == "yes"]
    no_refs = [(p, r) for p, r in zip(valid_preds, valid_refs, strict=True) if r == "no"]

    yes_correct = sum(1 for p, r in yes_refs if p == r) if yes_refs else 0
    no_correct = sum(1 for p, r in no_refs if p == r) if no_refs else 0

    return {
        "accuracy": correct / len(valid_pairs),
        "yes_accuracy": yes_correct / len(yes_refs) if yes_refs else 0.0,
        "no_accuracy": no_correct / len(no_refs) if no_refs else 0.0,
        "num_samples": float(len(valid_pairs)),
        "num_yes": float(len(yes_refs)),
        "num_no": float(len(no_refs)),
    }
