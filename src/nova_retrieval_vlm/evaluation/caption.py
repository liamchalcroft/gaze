"""Caption evaluation metrics for medical image analysis."""

from __future__ import annotations

import functools
from collections.abc import Sequence

import nltk
import sacrebleu
from bert_score import score as bert_score_fn
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

# Ensure NLTK data is available at module load
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def _calculate_radgraph_f1(refs: Sequence[str], preds: Sequence[str]) -> float | None:
    """Calculate RadGraph F1 score if radgraph is installed.

    RadGraph is an optional heavy dependency for clinical NLP evaluation.
    Returns None if not installed, allowing callers to handle accordingly.
    """
    try:
        from radgraph.radgraph import F1RadGraph
    except ImportError:
        return None

    rg = F1RadGraph(reward_level="partial")
    radgraph_f1_result, *_ = rg.forward(refs, preds)
    return float(radgraph_f1_result)


@functools.lru_cache(maxsize=128)
def _cached_word_tokenize(text: str) -> list:
    """Cached word tokenization to avoid repeated processing."""
    return word_tokenize(text)


def evaluate_caption(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float | None]:
    """Evaluate generated captions using multiple metrics.

    Args:
        preds: List of predicted captions.
        refs: List of reference captions.

    Returns:
        Dictionary with keys 'bleu', 'bert_f1', 'radgraph_f1', 'meteor',
        'modality_f1', 'clinical_f1', 'binary_f1'. radgraph_f1 may be None
        if radgraph is not installed.
    """
    bleu = sacrebleu.corpus_bleu(preds, [refs])

    _, _, f1_scores = bert_score_fn(cands=preds, refs=refs, lang="en", rescale_with_baseline=True)
    bert_f1_score = float(f1_scores.mean()) * 100

    # METEOR - Fixed calculation with caching
    # Prepare references for METEOR (list of lists format)
    ref_tokens = [[_cached_word_tokenize(ref)] for ref in refs]
    pred_tokens = [_cached_word_tokenize(pred) for pred in preds]

    # Pre-calculate METEOR scores to avoid try-except in loop
    valid_meteor_scores = []
    failed_indices = []

    for i, pred_token in enumerate(pred_tokens):
        try:
            score = meteor_score(ref_tokens[i], pred_token)
            valid_meteor_scores.append(score)
        except (ValueError, RuntimeError) as e:
            failed_indices.append(i)
            logger.warning(f"METEOR calculation failed for sample {i}: {e}")

    # Handle failed calculations
    total_samples = len(pred_tokens)
    if failed_indices:
        logger.info(f"METEOR failed for {len(failed_indices)}/{total_samples} samples")

    meteor = float(sum(valid_meteor_scores) / total_samples * 100) if valid_meteor_scores else 0.0

    # RadGraph F1 (optional heavy dependency)
    radgraph_f1 = _calculate_radgraph_f1(refs, preds)

    # Keyword-based F1 (case-insensitive exact keyword matching per NOVA protocol)
    modality_terms = {"flair", "t1", "t2", "t1w", "t2w", "axial", "sagittal", "coronal", "weighted"}
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
    mod_f1s = []
    clin_f1s = []
    binary_correct = 0
    for p, r in zip(preds, refs, strict=False):
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

    modality_f1 = float(sum(mod_f1s) / len(mod_f1s) if mod_f1s else 0.0)
    clinical_f1 = float(sum(clin_f1s) / len(clin_f1s) if clin_f1s else 0.0)
    binary_accuracy = float(binary_correct / len(preds) if preds else 0.0)

    # Calculate binary F1 with TP/FP/FN (per NOVA protocol)
    tp = sum(
        1
        for p, r in zip(preds, refs, strict=False)
        if bool({w.lower().strip(".,;:!?()[]") for w in p.split()} & clinical_terms)
        and bool({w.lower().strip(".,;:!?()[]") for w in r.split()} & clinical_terms)
    )
    fp = sum(
        1
        for p, r in zip(preds, refs, strict=False)
        if bool({w.lower().strip(".,;:!?()[]") for w in p.split()} & clinical_terms)
        and not bool({w.lower().strip(".,;:!?()[]") for w in r.split()} & clinical_terms)
    )
    fn = sum(
        1
        for p, r in zip(preds, refs, strict=False)
        if not bool({w.lower().strip(".,;:!?()[]") for w in p.split()} & clinical_terms)
        and bool({w.lower().strip(".,;:!?()[]") for w in r.split()} & clinical_terms)
    )
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
