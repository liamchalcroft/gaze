"""Caption evaluation metrics for medical image analysis."""

from __future__ import annotations

import functools
import importlib.util
import re
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

    In air-gapped/offline environments, downloads will silently fail if data is
    already installed locally. Pre-install with:
        python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"

    Returns:
        True when initialization is complete (value unused, just for caching).
    """
    for resource in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass  # Already installed or offline — NLTK will error later if truly missing
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


_WORD_PATTERN = re.compile(r"\b[\w][\w-]*[\w]\b|\b\w\b")


def _extract_keyword_tokens(text: str) -> set[str]:
    """Extract keyword tokens from text, splitting hyphenated compounds.

    "T2-weighted axial FLAIR" yields {"t2-weighted", "t2", "weighted", "axial", "flair"}.
    This ensures that both the compound term and its parts can match keyword sets.
    """
    tokens: set[str] = set()
    for match in _WORD_PATTERN.finditer(text.lower()):
        word = match.group()
        tokens.add(word)
        # Also add hyphen-split sub-tokens for compound terms
        if "-" in word:
            tokens.update(word.split("-"))
    return tokens


def _rouge_l_sentence(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute ROUGE-L F1 between two token lists using longest common subsequence."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    # DP table for LCS length
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1].lower() == pred_tokens[j - 1].lower():
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / n
    recall = lcs_len / m
    return (2 * precision * recall) / (precision + recall)


_NEGATION_WORDS = frozenset({
    "no", "not", "without", "absent", "negative", "none", "nor",
    "unremarkable", "normal", "deny", "denies", "denied",
})

_NEGATION_WINDOW = 3  # max tokens before a clinical term to check for negation

# Clause-boundary tokens that stop the following-window negation search.
# Prevents "tumor with no edema" from incorrectly negating "tumor" —
# the "no" belongs to the next clause (after "with").
_SCOPE_TERMINATORS = frozenset({
    "with", "and", "but", "or", "while", "although", "however",
    ",", ";", ".", ":", "—",
})


def _has_clinical_terms(text: str) -> bool:
    """Check if text contains non-negated clinical abnormality terms.

    A clinical term is considered negated if a negation word (e.g. "no",
    "not", "without", "absent", "normal") appears within
    ``_NEGATION_WINDOW`` tokens **before or after** it.  This handles
    both pre-negation ("no evidence of tumor") and post-negation
    ("tumor is not seen", "lesion absent").

    The following-window search stops at clause-boundary tokens (commas,
    conjunctions like "with", "and", "but") to prevent negation words
    from one clause leaking into a neighbouring clinical term.
    """
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
    tokens = text.lower().split()
    for i, token in enumerate(tokens):
        # Strip punctuation for matching but keep position
        clean = token.strip(".,;:!?()[]\"'")
        if clean not in clinical_terms:
            continue
        # Check preceding window for negation
        window_start = max(0, i - _NEGATION_WINDOW)
        preceding = tokens[window_start:i]
        if any(w.strip(".,;:!?()[]\"'") in _NEGATION_WORDS for w in preceding):
            continue  # negated — skip this term
        # Check following window for post-negation ("tumor not seen")
        # Stop at clause-boundary tokens to prevent cross-clause negation.
        window_end = min(len(tokens), i + 1 + _NEGATION_WINDOW)
        following = tokens[i + 1 : window_end]
        negated_following = False
        for w in following:
            stripped = w.strip(".,;:!?()[]\"'")
            if stripped in _SCOPE_TERMINATORS or w in _SCOPE_TERMINATORS:
                break  # clause boundary — stop searching
            if stripped in _NEGATION_WORDS:
                negated_following = True
                break
        if negated_following:
            continue  # negated — skip this term
        return True
    return False


@beartype
def evaluate_caption(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float | str | None]:
    """Evaluate generated captions using multiple metrics.

    Args:
        preds: List of predicted captions.
        refs: List of reference captions.

    Returns:
        Dictionary with keys 'bleu', 'bert_f1', 'radgraph_f1', 'meteor',
        'rouge_l', 'modality_f1', 'clinical_f1', 'binary_f1',
        'binary_accuracy', 'abnormal_prevalence'. radgraph_f1 may be None
        if radgraph is not installed.

    Raises:
        ValueError: If preds and refs have different lengths or are empty.
    """
    _ensure_nltk_data()

    if not preds:
        raise ValueError("Cannot evaluate empty predictions list")
    if not refs:
        raise ValueError("Cannot evaluate empty references list")
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have same length, got {len(preds)} vs {len(refs)}")
    # Pin smoothing method and tokenizer explicitly for reproducibility
    bleu = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp", tokenize="13a")

    _, _, f1_scores, bert_hash = bert_score_fn(
        cands=preds, refs=refs, model_type="roberta-large",
        lang="en", rescale_with_baseline=True, return_hash=True,
    )
    # Baseline-rescaled BERTScore F1 can be negative for very poor candidates.
    # Clamp to [0, 1] before reporting — negative scores are not meaningful
    # and would violate CaptionMetrics(ge=0.0) if validated.
    bert_f1_score = max(0.0, min(1.0, float(f1_scores.mean())))

    # METEOR calculation with proper tokenization
    # Prepare references for METEOR (list of lists format)
    # Convert tuples back to lists for METEOR compatibility
    # METEOR expects list of reference lists, not list of lists of lists
    ref_tokens = [list(_cached_word_tokenize(ref)) for ref in refs]
    pred_tokens = [list(_cached_word_tokenize(pred)) for pred in preds]

    # Calculate METEOR scores - fail on invalid inputs
    # METEOR scores are in 0-1 range natively
    # Wrap each reference in a list since METEOR expects multiple references per prediction
    meteor_scores = [
        nltk_meteor_score([ref_tokens[i]], pred_token) for i, pred_token in enumerate(pred_tokens)
    ]

    meteor = float(sum(meteor_scores) / len(meteor_scores))

    # ROUGE-L (robust to length differences unlike BLEU)
    rouge_l_scores = [
        _rouge_l_sentence(list(pred_tokens[i]), list(ref_tokens[i])) for i in range(len(preds))
    ]
    rouge_l = float(sum(rouge_l_scores) / len(rouge_l_scores))

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
        # Extract word tokens and also split hyphenated terms into components.
        # "T2-weighted" yields {"t2-weighted", "t2", "weighted"} so that
        # both the compound and its parts can match keyword sets.
        p_words = _extract_keyword_tokens(p)
        r_words = _extract_keyword_tokens(r)

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

        # Binary abnormality classification (normal vs abnormal).
        # Use _has_clinical_terms() (negation-aware) so that "no evidence
        # of tumor" is classified as normal, consistent with abnormal_prevalence.
        pred_abnormal = _has_clinical_terms(p)
        ref_abnormal = _has_clinical_terms(r)
        if pred_abnormal == ref_abnormal:
            binary_correct += 1

        # Binary F1 TP/FP/FN calculation
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

    # Abnormal prevalence in ground truth — documents class balance.
    # NOVA is almost entirely pathological cases, so binary accuracy is
    # trivially high when the model always predicts "abnormal".
    abnormal_prevalence = float(sum(1 for r in refs if _has_clinical_terms(r)) / len(refs))

    return {
        "bleu": float(bleu.score) / 100.0,  # sacrebleu reports 0-100; normalized to 0-1
        "bleu_raw": float(bleu.score),  # Original sacrebleu scale (0-100) for literature comparison
        "bert_f1": bert_f1_score,  # Already clamped to [0, 1]
        "bert_model": str(bert_hash),  # Model variant + hash for reproducibility
        "radgraph_f1": radgraph_f1,  # Already in 0-1 range
        "meteor": meteor,  # Already in 0-1 range (NLTK native)
        "rouge_l": rouge_l,  # LCS-based, robust to length differences
        "modality_f1": modality_f1,  # Already in 0-1 range
        "clinical_f1": clinical_f1,  # Already in 0-1 range
        "binary_accuracy": binary_accuracy,  # Accuracy of abnormal/normal classification
        "binary_f1": binary_f1,  # F1 of abnormal/normal classification
        "abnormal_prevalence": abnormal_prevalence,  # GT class balance (high = trivial binary task)
    }
