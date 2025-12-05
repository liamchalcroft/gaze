"""Caption evaluation metrics for medical image analysis."""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Sequence

import nltk
import sacrebleu
from beartype import beartype
from bert_score import score as bert_score_fn
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score

# Check for optional RadGraph dependency at module load
RADGRAPH_AVAILABLE = importlib.util.find_spec("radgraph") is not None


@functools.lru_cache(maxsize=1)
def _ensure_nltk_data() -> bool:
    """Download required NLTK data if not already present.

    Uses lru_cache for thread-safe one-time initialization without global mutable state.
    The cache ensures this runs exactly once per process.

    Returns:
        True when initialization is complete (value unused, just for caching).
    """
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return True


@beartype
def _calculate_radgraph_f1(refs: Sequence[str], preds: Sequence[str]) -> float | None:
    """Calculate RadGraph F1 score.

    RadGraph is an optional heavy dependency for clinical NLP evaluation.
    Returns None if not installed.

    Raises:
        ImportError: If radgraph is specified in pyproject.toml but not properly installed.
    """
    if not RADGRAPH_AVAILABLE:
        return None

    from radgraph.radgraph import F1RadGraph

    rg = F1RadGraph(reward_level="partial")
    radgraph_f1_result, *_ = rg.forward(refs, preds)
    return float(radgraph_f1_result)


@functools.lru_cache(maxsize=128)
def _cached_word_tokenize(text: str) -> tuple[str, ...]:
    """Cached word tokenization to avoid repeated processing.

    Returns tuple instead of list for hashability (required by lru_cache).
    """
    return tuple(word_tokenize(text))


@beartype
def evaluate_caption(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float | None]:
    """Evaluate generated captions using multiple metrics.

    Args:
        preds: List of predicted captions.
        refs: List of reference captions.

    Returns:
        Dictionary with keys 'bleu', 'bert_f1', 'radgraph_f1', 'meteor',
        'modality_f1', 'clinical_f1', 'binary_f1'. radgraph_f1 may be None
        if radgraph is not installed.

    Raises:
        ValueError: If preds and refs have different lengths.
    """
    _ensure_nltk_data()

    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have same length, got {len(preds)} vs {len(refs)}")
    bleu = sacrebleu.corpus_bleu(preds, [refs])

    _, _, f1_scores = bert_score_fn(cands=preds, refs=refs, lang="en", rescale_with_baseline=True)
    bert_f1_score = float(f1_scores.mean()) * 100

    # METEOR calculation with proper tokenization
    # Prepare references for METEOR (list of lists format)
    # Convert tuples back to lists for METEOR compatibility
    ref_tokens = [[list(_cached_word_tokenize(ref))] for ref in refs]
    pred_tokens = [list(_cached_word_tokenize(pred)) for pred in preds]

    # Calculate METEOR scores - fail on invalid inputs
    meteor_scores = [
        nltk_meteor_score(ref_tokens[i], pred_token) for i, pred_token in enumerate(pred_tokens)
    ]

    meteor = float(sum(meteor_scores) / len(meteor_scores) * 100)

    # RadGraph F1 (optional heavy dependency)
    radgraph_f1 = _calculate_radgraph_f1(refs, preds)

    # Keyword-based F1 (case-insensitive exact keyword matching per NOVA protocol)
    # NOVA modality terms: "flair, axial, sagittal, t1, t2, coronal, dwi, t1w, t2w, weighted"
    modality_terms = {
        "flair",
        "t1",
        "t2",
        "t1w",
        "t2w",
        "axial",
        "sagittal",
        "coronal",
        "dwi",
        "weighted",
    }
    clinical_terms = {
        "lesion",
        "tumor",
        "tumour",
        "hemorrhage",
        "haemorrhage",
        "infarct",
        "infarction",
        "cyst",
        "atrophy",
        "metastasis",
        "metastases",
        "edema",
        "oedema",
        "mass",
        "enhancement",
        "abnormal",
        "abnormality",
    }
    # Single pass through predictions and references for all keyword-based metrics
    mod_f1s: list[float] = []
    clin_f1s: list[float] = []
    binary_correct = 0
    tp = 0  # True positives for binary F1
    fp = 0  # False positives for binary F1
    fn = 0  # False negatives for binary F1

    for p, r in zip(preds, refs, strict=True):
        p_words = {w.lower().strip(".,;:!?()[]") for w in p.split()}
        r_words = {w.lower().strip(".,;:!?()[]") for w in r.split()}

        # Modality F1
        p_mod = p_words & modality_terms
        r_mod = r_words & modality_terms
        mod_intersection = p_mod & r_mod
        prec_mod = len(mod_intersection) / len(p_mod) if p_mod else 0.0
        rec_mod = len(mod_intersection) / len(r_mod) if r_mod else 0.0
        f1_mod = (
            (2 * prec_mod * rec_mod / (prec_mod + rec_mod)) if (prec_mod + rec_mod) > 0 else 0.0
        )
        mod_f1s.append(f1_mod)

        # Clinical F1
        p_clin = p_words & clinical_terms
        r_clin = r_words & clinical_terms
        clin_intersection = p_clin & r_clin
        prec_clin = len(clin_intersection) / len(p_clin) if p_clin else 0.0
        rec_clin = len(clin_intersection) / len(r_clin) if r_clin else 0.0
        f1_clin = (
            (2 * prec_clin * rec_clin / (prec_clin + rec_clin))
            if (prec_clin + rec_clin) > 0
            else 0.0
        )
        clin_f1s.append(f1_clin)

        # Binary abnormality classification (normal vs abnormal)
        pred_abnormal = bool(p_clin)
        ref_abnormal = bool(r_clin)
        if pred_abnormal == ref_abnormal:
            binary_correct += 1

        # Binary F1 TP/FP/FN calculation (reuse p_clin, r_clin from above)
        if pred_abnormal and ref_abnormal:
            tp += 1
        elif pred_abnormal and not ref_abnormal:
            fp += 1
        elif not pred_abnormal and ref_abnormal:
            fn += 1

    modality_f1 = float(sum(mod_f1s) / len(mod_f1s) if mod_f1s else 0.0)
    clinical_f1 = float(sum(clin_f1s) / len(clin_f1s) if clin_f1s else 0.0)
    binary_accuracy = float(binary_correct / len(preds) if preds else 0.0)

    # Binary F1 from accumulated TP/FP/FN
    precision_bin = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_bin = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    binary_f1 = (
        (2 * precision_bin * recall_bin / (precision_bin + recall_bin))
        if (precision_bin + recall_bin) > 0
        else 0.0
    )

    return {
        "bleu": float(bleu.score) / 100.0,  # Normalize from 0-100 to 0-1
        "bert_f1": bert_f1_score / 100.0,  # Normalize from 0-100 to 0-1
        "radgraph_f1": radgraph_f1,  # Already in 0-1 range
        "meteor": meteor / 100.0,  # Normalize from 0-100 to 0-1
        "modality_f1": modality_f1,  # Already in 0-1 range
        "clinical_f1": clinical_f1,  # Already in 0-1 range
        "binary_accuracy": binary_accuracy,  # Accuracy of abnormal/normal classification
        "binary_f1": binary_f1,  # F1 of abnormal/normal classification
    }
