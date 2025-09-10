from __future__ import annotations

import functools
from collections.abc import Sequence

import nltk
import sacrebleu
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

# Lazy import bert_score to avoid torch import issues during collection
bert_score = None

# Cache for NLTK downloads to avoid repeated downloads
_nltk_downloaded = False


def _get_bert_score():
    """Lazy import bert_score to avoid torch import issues during collection."""
    global bert_score
    if bert_score is None:
        try:
            from bert_score import score as bert_score_module
            bert_score = bert_score_module
        except ImportError as e:
            logger.warning(f"BERTScore not available: {e}")
            bert_score = False
    return bert_score


def _ensure_nltk_downloads():
    """Ensure NLTK data is downloaded (cached)."""
    global _nltk_downloaded
    if not _nltk_downloaded:
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        _nltk_downloaded = True


@functools.lru_cache(maxsize=128)
def _cached_word_tokenize(text: str) -> list:
    """Cached word tokenization to avoid repeated processing."""
    return word_tokenize(text)


def evaluate_caption(
    preds: Sequence[str],
    refs: Sequence[str],
) -> dict[str, float]:
    """
    Evaluate generated captions using SacreBLEU, BERTScore, and RadGraph F1.

    Args:
        preds: List of predicted captions.
        refs: List of reference captions.

    Returns:
        Dictionary with keys 'bleu', 'bert_f1', 'radgraph_f1', 'meteor', 'modality_f1', 'clinical_f1', 'binary_f1'.
    """
    # Ensure NLTK data is available
    _ensure_nltk_downloads()

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs])

    # BERTScore
    bert_score_fn = _get_bert_score()
    if bert_score_fn is False:
        # BERTScore not available, return 0.0
        bert_f1 = 0.0
    else:
        P, R, F1 = bert_score_fn(cands=preds, refs=refs, lang="en", rescale_with_baseline=True)
        bert_f1 = float(F1.mean()) * 100

    # METEOR - Fixed calculation with caching
    # Prepare references for METEOR (list of lists format)
    ref_tokens = [[_cached_word_tokenize(ref)] for ref in refs]
    pred_tokens = [_cached_word_tokenize(pred) for pred in preds]

    # Calculate METEOR scores
    meteor_scores = []
    for i, pred_token in enumerate(pred_tokens):
        try:
            score = meteor_score(ref_tokens[i], pred_token)
            meteor_scores.append(score)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"METEOR calculation failed for sample {i}: {e}")
            meteor_scores.append(0.0)

    meteor = float(sum(meteor_scores) / len(meteor_scores) * 100) if meteor_scores else 0.0

    # RadGraph F1 (optional heavy dependency)
    radgraph_f1 = 0.0
    # RadGraph F1 - fail fast if not available
    from radgraph.radgraph import F1RadGraph

    rg = F1RadGraph(reward_level="partial")
    radgraph_f1, *_ = rg.forward(refs, preds)
    radgraph_f1 = float(radgraph_f1)

    # Keyword-based F1
    modality_terms = {"flair", "t1", "t2", "t1w", "t2w", "axial", "sagittal", "coronal", "weighted"}
    clinical_terms = {
        "lesion",
        "tumor",
        "hemorrhage",
        "infarct",
        "cyst",
        "atrophy",
        "metastasis",
        "edema",
    }
    mod_f1s = []
    clin_f1s = []
    abnormal_preds = 0
    for p, r in zip(preds, refs, strict=False):
        p_words = {w.lower().strip(".,") for w in p.split()}
        r_words = {w.lower().strip(".,") for w in r.split()}
        # modality
        p_mod = p_words & modality_terms
        r_mod = r_words & modality_terms
        prec_mod = len(p_mod) / len(p_mod) if p_mod else 0.0
        rec_mod = len(p_mod) / len(r_mod) if r_mod else 0.0
        f1_mod = (
            (2 * prec_mod * rec_mod / (prec_mod + rec_mod)) if (prec_mod + rec_mod) > 0 else 0.0
        )
        mod_f1s.append(f1_mod)
        # clinical
        p_clin = p_words & clinical_terms
        r_clin = r_words & clinical_terms
        prec_clin = len(p_clin) / len(p_clin) if p_clin else 0.0
        rec_clin = len(p_clin) / len(r_clin) if r_clin else 0.0
        f1_clin = (
            (2 * prec_clin * rec_clin / (prec_clin + rec_clin))
            if (prec_clin + rec_clin) > 0
            else 0.0
        )
        clin_f1s.append(f1_clin)
        # binary abnormality: assume any clinical term => abnormal
        if p_clin:
            abnormal_preds += 1
    modality_f1 = float(sum(mod_f1s) / len(mod_f1s) if mod_f1s else 0.0)
    clinical_f1 = float(sum(clin_f1s) / len(clin_f1s) if clin_f1s else 0.0)
    binary_f1 = float(
        (2 * (abnormal_preds / len(preds)) * (1.0) / ((abnormal_preds / len(preds)) + 1.0))
        if preds
        else 0.0
    )
    return {
        "bleu": float(bleu.score),
        "bert_f1": float(F1.mean()),
        "radgraph_f1": radgraph_f1,
        "meteor": meteor,
        "modality_f1": modality_f1,
        "clinical_f1": clinical_f1,
        "binary_f1": binary_f1,
    }
