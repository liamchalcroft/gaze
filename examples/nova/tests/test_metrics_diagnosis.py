from unittest.mock import patch

import pytest

from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official


@patch("nova_retrieval_vlm.evaluation.diagnosis.llm_semantic_match")
def test_evaluate_diagnosis_simple(mock_semantic_match):
    """Test diagnosis evaluation with mocked semantic matching."""

    def semantic_match_side_effect(pred, ref, model_name=None):
        return pred == ref

    mock_semantic_match.side_effect = semantic_match_side_effect

    preds = [
        "dx_a",  # correct top1
        ["dx_x", "dx_c", "dx_b"],  # correct in top5
    ]
    refs = [
        "dx_a",
        "dx_c",
    ]
    metrics = evaluate_diagnosis_nova_official(preds, refs)
    # top1: first correct (1), second incorrect (0) => 0.5
    assert metrics["top1"] == pytest.approx(0.5, rel=1e-4)
    # top5: both included => 1.0
    assert metrics["top5"] == pytest.approx(1.0, rel=1e-4)
    # coverage should be >=1 because we predicted at least same unique as refs
    assert metrics["coverage"] >= 1.0
    # entropy non-negative
    assert metrics["entropy"] >= 0.0
