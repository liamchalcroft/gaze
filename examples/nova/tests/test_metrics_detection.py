import pytest


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch required for detection metrics") is None,
    reason="torch not available",
)
@pytest.mark.skipif(
    pytest.importorskip("torchmetrics", reason="torchmetrics required") is None,
    reason="torchmetrics not available",
)
def test_evaluate_detection_perfect_match():
    from nova_retrieval_vlm.evaluation.detection import evaluate_detection

    preds = [
        {
            "boxes": [[10.0, 10.0, 20.0, 20.0]],
            "scores": [0.9],
            "labels": [0],
        }
    ]
    refs = [
        {
            "boxes": [[10.0, 10.0, 20.0, 20.0]],
            "scores": [1.0],
            "labels": [0],
        }
    ]
    metrics = evaluate_detection(preds, refs)
    assert "map30" in metrics and "map50" in metrics and "map50_95" in metrics
    # All metrics should be perfect because the prediction matches GT exactly
    assert metrics["map30"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["map50"] == pytest.approx(1.0, rel=1e-4)
    # mAP50-95 should also be 1.0 for perfect match at all iou thresholds
    assert metrics["map50_95"] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch required for detection metrics") is None,
    reason="torch not available",
)
@pytest.mark.skipif(
    pytest.importorskip("torchmetrics", reason="torchmetrics required") is None,
    reason="torchmetrics not available",
)
def test_evaluate_detection_tensor_input():
    """Test detection evaluation with tensor inputs."""
    import torch

    from nova_retrieval_vlm.evaluation.detection import evaluate_detection

    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    refs = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "scores": torch.tensor([1.0]),
            "labels": torch.tensor([0]),
        }
    ]
    metrics = evaluate_detection(preds, refs)
    assert "map30" in metrics and "map50" in metrics and "map50_95" in metrics
    # Should work with tensor inputs
    assert metrics["map30"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["map50"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["map50_95"] == pytest.approx(1.0, rel=1e-4)
