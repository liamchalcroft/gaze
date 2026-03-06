"""Unit tests for evaluation metrics.

Tests the corrected bounding box coordinate conversion and evaluation metrics.
"""

from __future__ import annotations

import pytest

# Optional torch import
try:
    import torch

    from examples.nova.src.data.nova_ground_truth import GroundTruthLocalization
    from examples.nova.src.evaluation.detection import _compute_iou
    from examples.nova.src.evaluation.detection import _convert_to_tensors
    from examples.nova.src.evaluation.detection import evaluate_detection

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestBoundingBoxConversion:
    """Test bounding box coordinate system conversion."""

    def test_convert_boxes_xyxy_format(self):
        """Test that boxes in (x1, y1, x2, y2) format are converted correctly."""
        # Input in (x1, y1, x2, y2) format - the expected format per docstring
        boxes = [
            [10, 20, 40, 60],  # x1=10, y1=20, x2=40, y2=60
            [0, 0, 100, 100],  # x1=0, y1=0, x2=100, y2=100
        ]

        result = _convert_to_tensors(boxes)

        assert result["boxes"].shape == (2, 4)
        expected = torch.tensor([[10, 20, 40, 60], [0, 0, 100, 100]], dtype=torch.float32)
        torch.testing.assert_close(result["boxes"], expected)

    def test_convert_boxes_already_xyxy(self):
        """Test that boxes already in (x1, y1, x2, y2) format pass through unchanged."""
        # Input already in (x1, y1, x2, y2) format
        boxes = [
            [10, 20, 40, 60],
            [0, 0, 100, 100],
        ]

        result = _convert_to_tensors(boxes)

        assert result["boxes"].shape == (2, 4)
        expected = torch.tensor([[10, 20, 40, 60], [0, 0, 100, 100]], dtype=torch.float32)
        torch.testing.assert_close(result["boxes"], expected)

    def test_empty_boxes(self):
        """Test empty box list."""
        result = _convert_to_tensors([])

        assert result["boxes"].shape == (0, 4)
        assert result["scores"].shape == (0,)
        assert result["labels"].shape == (0,)

    def test_boxes_with_scores_and_labels(self):
        """Test boxes with scores and labels."""
        data = {
            "boxes": [[10, 20, 40, 60]],
            "scores": [0.9],
            "labels": [1],
        }

        result = _convert_to_tensors(data)

        assert result["boxes"].shape == (1, 4)
        assert result["scores"].shape == (1,)
        assert result["labels"].shape == (1,)
        assert result["scores"][0] == 0.9
        assert result["labels"][0] == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestIoUCalculation:
    """Test IoU calculation with corrected coordinate system."""

    def test_perfect_overlap(self):
        """Test IoU with identical boxes."""
        box1 = torch.tensor([10, 10, 20, 20], dtype=torch.float32)
        box2 = torch.tensor([10, 10, 20, 20], dtype=torch.float32)

        iou = _compute_iou(box1, box2)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        box1 = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
        box2 = torch.tensor([20, 20, 30, 30], dtype=torch.float32)

        iou = _compute_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU with partially overlapping boxes."""
        # Box1: 10x10 box, Box2: 10x10 box overlapping by 5x5 area
        box1 = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
        box2 = torch.tensor([5, 5, 15, 15], dtype=torch.float32)

        iou = _compute_iou(box1, box2)
        # Intersection: 25, Union: 175, IoU: 25/175 = 0.142857...
        assert abs(iou - 0.142857) < 1e-6

    def test_edge_case_touching(self):
        """Test IoU when boxes just touch edges."""
        box1 = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
        box2 = torch.tensor([10, 0, 20, 10], dtype=torch.float32)

        iou = _compute_iou(box1, box2)
        assert iou == 0.0  # No overlap, just touching


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestDetectionEvaluation:
    """Test complete detection evaluation pipeline."""

    def test_simple_evaluation(self):
        """Test basic detection evaluation with perfect matches."""
        preds = [
            {"boxes": [[10, 10, 20, 20]], "scores": [0.9], "labels": [1]},
            {"boxes": [[0, 0, 50, 50]], "scores": [0.8], "labels": [2]},
        ]
        refs = [
            {"boxes": [[10, 10, 20, 20]], "scores": [1.0], "labels": [1]},
            {"boxes": [[0, 0, 50, 50]], "scores": [1.0], "labels": [2]},
        ]

        results = evaluate_detection(preds, refs)

        assert "map50" in results
        assert "map30" in results
        assert "acc50" in results
        assert results["acc50"] == 1.0  # All predictions match ground truth

    def test_evaluation_with_mismatches(self):
        """Test evaluation with incorrect predictions."""
        preds = [
            {"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [1]},  # Wrong location
            {"boxes": [], "scores": [], "labels": []},  # No prediction
        ]
        refs = [
            {"boxes": [[10, 10, 20, 20]], "scores": [1.0], "labels": [1]},
            {"boxes": [[0, 0, 50, 50]], "scores": [1.0], "labels": [2]},
        ]

        results = evaluate_detection(preds, refs)

        assert results["acc50"] < 1.0  # Not all predictions match
        assert results["fp30"] > 0  # False positives at IoU 0.3

    def test_evaluation_error_handling(self):
        """Test error handling in evaluation."""
        with pytest.raises(ValueError, match="Cannot evaluate empty predictions"):
            evaluate_detection([], [{"boxes": []}])

        with pytest.raises(ValueError, match="Cannot evaluate empty references"):
            evaluate_detection([{"boxes": []}], [])

        with pytest.raises(ValueError, match="preds and refs must have same length"):
            evaluate_detection([{"boxes": []}], [{"boxes": []}, {"boxes": []}])

    def test_map50_reuses_map50_95_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """mAP@0.5 should come from the COCO-range pass, not a separate third pass."""
        from examples.nova.src.evaluation import detection as detection_module

        calls: list[tuple[float, ...]] = []

        class _FakeMap:
            def __init__(self, *, iou_thresholds):
                calls.append(tuple(float(t) for t in iou_thresholds))

            def update(self, preds_tensors, refs_tensors) -> None:
                assert preds_tensors
                assert refs_tensors

            def compute(self) -> dict[str, float]:
                return {"map": 0.4, "map_50": 0.6}

        monkeypatch.setattr(detection_module, "MeanAveragePrecision", _FakeMap)

        preds = [{"boxes": [[10, 10, 20, 20]], "scores": [0.9], "labels": [1]}]
        refs = [{"boxes": [[10, 10, 20, 20]], "scores": [1.0], "labels": [1]}]

        results = evaluate_detection(preds, refs)

        assert calls == [
            (detection_module.IOU_THRESHOLD_LOOSE,),
            tuple(detection_module._MAP_RANGE_IOU_THRESHOLDS),
        ]
        assert results["map30"] == 0.4
        assert results["map50"] == 0.6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestGroundTruthData:
    """Test ground truth data structure."""

    def test_ground_truth_localization_coordinates(self):
        """Test that ground truth localizations use correct coordinate format."""
        # Create a ground truth localization with converted coordinates
        bbox = GroundTruthLocalization(bbox=(10, 20, 40, 60))  # (x1, y1, x2, y2)

        # The coordinates should already be in the correct format
        x1, y1, x2, y2 = bbox.bbox
        assert x1 == 10
        assert y1 == 20
        assert x2 == 40  # x1 + width (10 + 30)
        assert y2 == 60  # y1 + height (20 + 40)

        # Validate coordinate relationships
        assert x2 > x1
        assert y2 > y1
        assert (x2 - x1) == 30  # width
        assert (y2 - y1) == 40  # height
