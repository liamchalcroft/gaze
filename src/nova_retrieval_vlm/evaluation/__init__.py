# NOTE:
# -----
# Importing *bert_score*, *transformers*, or *sentence_transformers* at module
# import time can drag in heavyweight dependencies such as **PyTorch** which
# might not be present in all environments (e.g. during lightweight CI runs).
# To make the library more robust we *lazy-load* the task-specific evaluator
# only when it is actually required.

from importlib import import_module
from types import ModuleType
from typing import Dict


_TASK_TO_MODULE = {
    "localization": "nova_retrieval_vlm.evaluation.detection",
    "caption": "nova_retrieval_vlm.evaluation.caption",
    "diagnosis": "nova_retrieval_vlm.evaluation.diagnosis",
}


def _lazy_import(module_path: str, symbol: str):
    """Import *symbol* from *module_path* on demand.

    Returns *None* if the import fails so that the caller can decide how to
    proceed (e.g. raise a clear error or return an empty metric dict).
    """
    try:
        module: ModuleType = import_module(module_path)
        return getattr(module, symbol)
    except (ImportError, AttributeError):
        return None


def evaluate(preds_jsonl: str, refs_jsonl: str, task: str = 'localization') -> Dict[str, float]:
    """
    Run detection, caption, and diagnosis evaluation based on the specified task and return relevant scores.

    Args:
        preds_jsonl: Path to predictions JSONL.
        refs_jsonl: Path to reference JSONL.
        task: The specific task to evaluate ('localization', 'caption', or 'diagnosis').

    Returns:
        Dictionary of metric names to scores relevant to the specified task.
    """
    import json
    preds = [json.loads(line) for line in open(preds_jsonl)]
    refs = [json.loads(line) for line in open(refs_jsonl)]
    result_metrics = {}
    
    if task == 'localization':
        # Import will raise ImportError with a normal traceback if torch or
        # torchvision are missing – we let it propagate so the user sees the
        # concrete root-cause instead of a custom wrapper.
        from nova_retrieval_vlm.evaluation.detection import evaluate_detection  # noqa: E501

        def _maybe_xywh_to_xyxy(boxes: list[list[float]]) -> list[list[float]]:
            """Convert boxes from [x,y,w,h] to [x1,y1,x2,y2] **in place** if needed.

            Heuristic: if any box has *x2 ≤ x1* or *y2 ≤ y1* we assume the
            format is xywh.  We then transform all boxes accordingly.
            """
            if not boxes:
                return boxes
            flag_xywh = False
            for b in boxes:
                if len(b) == 4 and (b[2] <= b[0] or b[3] <= b[1]):
                    flag_xywh = True
                    break
            if flag_xywh:
                converted = [[x, y, x + w, y + h] for x, y, w, h in boxes]
                return converted
            return boxes

        # Ensure consistent box format
        for rec in preds:
            rec['boxes'] = _maybe_xywh_to_xyxy(rec['boxes'])
        for rec in refs:
            rec['boxes'] = _maybe_xywh_to_xyxy(rec['boxes'])

        import torch  # pylint: disable=import-error
        pred_det = [
            {
                'boxes': torch.tensor(p['boxes'], dtype=torch.float),
                'labels': torch.tensor([0] * len(p.get('labels', [])), dtype=torch.long),
                'scores': torch.tensor(p.get('scores', [1.0] * len(p['boxes'])), dtype=torch.float),
            } for p in preds
        ]
        ref_det = [
            {
                'boxes': torch.tensor(r['boxes'], dtype=torch.float),
                'labels': torch.tensor([0] * len(r.get('labels', [])), dtype=torch.long),
                'scores': torch.tensor(r.get('scores', [1.0] * len(r['boxes'])), dtype=torch.float),
            } for r in refs
        ]
        det_metrics = evaluate_detection(pred_det, ref_det)
        result_metrics.update({
            'detection_mAP30': det_metrics['map30'],
            'detection_mAP50': det_metrics['map50'],
            'detection_mAP50_95': det_metrics['map50_95'],
        })
    
    elif task == 'caption':
        from nova_retrieval_vlm.evaluation.caption import evaluate_caption
        pred_caps = [p.get('caption', '') for p in preds]
        ref_caps = [r.get('caption', '') for r in refs]
        cap_scores = evaluate_caption(pred_caps, ref_caps)
        result_metrics.update({
            'caption_bleu': cap_scores['bleu'],
            'caption_bert_f1': cap_scores['bert_f1'],
            'caption_radgraph_f1': cap_scores['radgraph_f1'],
            'caption_meteor': cap_scores['meteor'],
            'caption_modality_f1': cap_scores['modality_f1'],
            'caption_clinical_f1': cap_scores['clinical_f1'],
            'caption_binary_f1': cap_scores['binary_f1'],
        })
    
    elif task == 'diagnosis':
        from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis
        pred_diags = [p.get('diagnosis', '') for p in preds]
        ref_diags = [r.get('diagnosis', '') for r in refs]
        diag_scores = evaluate_diagnosis(pred_diags, ref_diags)
        result_metrics.update({
            'diagnosis_top1': diag_scores['top1'],
            'diagnosis_top5': diag_scores['top5'],
            'diagnosis_coverage': diag_scores['coverage'],
            'diagnosis_entropy': diag_scores['entropy'],
        })
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return result_metrics


# No fallback stubs – missing optional dependencies will raise the original
# ImportError so that issues surface immediately during testing/runs. 