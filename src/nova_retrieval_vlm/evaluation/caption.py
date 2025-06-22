from __future__ import annotations

from typing import Dict, Sequence

import nltk
import sacrebleu
from bert_score import score as bert_score
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


def evaluate_caption(
    preds: Sequence[str],
    refs: Sequence[str],
) -> Dict[str, float]:
    """
    Evaluate generated captions using SacreBLEU, BERTScore, and RadGraph F1.

    Args:
        preds: List of predicted captions.
        refs: List of reference captions.

    Returns:
        Dictionary with keys 'bleu', 'bert_f1', 'radgraph_f1', 'meteor', 'modality_f1', 'clinical_f1', 'binary_f1'.
    """
    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    # BERTScore
    P, R, F1 = bert_score(cands=preds, refs=refs, lang="en", rescale_with_baseline=True)
    # METEOR
    nltk.download("punkt")
    pred_tokens = [word_tokenize(pred) for pred in preds]
    ref_tokens = [[word_tokenize(ref) for ref in refs[i]] for i in range(len(refs))]
    meteor_scores = [meteor_score(ref_tokens[i], pred_tokens[i]) for i in range(len(preds))]
    meteor = float(sum(meteor_scores) / len(meteor_scores) * 100)

    # RadGraph F1 (optional heavy dependency)
    radgraph_f1 = 0.0
    try:
        from radgraph.radgraph import F1RadGraph

        rg = F1RadGraph(reward_level="partial")
        radgraph_f1, *_ = rg.forward(refs, preds)
        radgraph_f1 = float(radgraph_f1)
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("RadGraph evaluation unavailable: %s", exc)
        radgraph_f1 = 0.0
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
    for p, r in zip(preds, refs):
        p_words = set(w.lower().strip(".,") for w in p.split())
        r_words = set(w.lower().strip(".,") for w in r.split())
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
