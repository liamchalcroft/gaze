import importlib

import pytest

# Ensure we have a *real* torch install. If 'torch' is a stub/mock without __spec__, skip.
try:
    torch = importlib.import_module("torch")
    _HAS_TORCH = getattr(torch, "__file__", None) is not None
except Exception:  # pragma: no cover
    _HAS_TORCH = False

if not _HAS_TORCH:
    pytest.skip(
        "Real torch package not available – skipping caption metric test", allow_module_level=True
    )

bert = pytest.importorskip("bert_score", reason="bert_score not installed")

from nova_retrieval_vlm.evaluation.caption import evaluate_caption


def test_evaluate_caption_identical():
    preds = [
        "There is a small lesion in the left temporal lobe.",
        "Normal MRI brain scan without abnormalities.",
    ]
    refs = [
        "There is a small lesion in the left temporal lobe.",
        "Normal MRI brain scan without abnormalities.",
    ]

    metrics = evaluate_caption(preds, refs)
    # BLEU should be high for identical captions (normalized to 0-1)
    assert metrics["bleu"] == pytest.approx(1.0, rel=1e-2)
    # BERT F1 should be high (normalized to 0-1)
    assert metrics["bert_f1"] >= 0.95
    # METEOR should be reasonable for identical texts (normalized to 0-1)
    assert metrics["meteor"] >= 0.25
    # Radgraph and other F1s may be zero if dependency missing but keys exist
    for key in [
        "radgraph_f1",
        "modality_f1",
        "clinical_f1",
        "binary_f1",
    ]:
        assert key in metrics


def test_evaluate_caption_different():
    """Test caption evaluation with different texts."""
    preds = [
        "There is a mass in the brain.",
        "The scan shows normal findings.",
    ]
    refs = [
        "There is a lesion in the temporal lobe.",
        "Normal MRI brain scan without abnormalities.",
    ]

    metrics = evaluate_caption(preds, refs)
    # All metrics should be present
    required_metrics = [
        "bleu",
        "bert_f1",
        "meteor",
        "radgraph_f1",
        "modality_f1",
        "clinical_f1",
        "binary_f1",
    ]
    for metric in required_metrics:
        assert metric in metrics

    # Scores should be reasonable for different texts (all normalized to 0-1)
    assert 0.0 <= metrics["bleu"] <= 1.0
    assert 0.0 <= metrics["bert_f1"] <= 1.0
    assert 0.0 <= metrics["meteor"] <= 1.0
