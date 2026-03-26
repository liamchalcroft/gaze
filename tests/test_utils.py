"""Tests for utility modules."""

from __future__ import annotations

import pytest

from radiant_harness.utils import clamp_confidence
from radiant_harness.utils.iou import compute_iou


class TestClampConfidenceStrings:
    """Cover clamp_confidence with numeric string inputs (lines 44-46)."""

    def test_numeric_string_in_range(self) -> None:
        assert clamp_confidence("0.85") == 0.85

    def test_numeric_string_zero(self) -> None:
        assert clamp_confidence("0") == 0.0

    def test_numeric_string_one(self) -> None:
        assert clamp_confidence("1.0") == 1.0

    def test_numeric_string_clamped_high(self) -> None:
        assert clamp_confidence("1.5") == 1.0

    def test_numeric_string_clamped_low(self) -> None:
        assert clamp_confidence("-0.5") == 0.0

    def test_numeric_string_nan_returns_none(self) -> None:
        assert clamp_confidence("nan") is None

    def test_numeric_string_inf_returns_none(self) -> None:
        assert clamp_confidence("inf") is None

    def test_numeric_string_negative_inf_returns_none(self) -> None:
        assert clamp_confidence("-inf") is None

    def test_numeric_string_large_scale_clamped(self) -> None:
        assert clamp_confidence("75") == 1.0

    def test_unparseable_string_returns_none(self) -> None:
        assert clamp_confidence("not_a_number") is None


class TestIoU:
    """Test IoU calculation utility."""

    def test_perfect_overlap(self):
        """Test IoU with identical boxes."""
        box1 = [10.0, 10.0, 20.0, 20.0]
        box2 = [10.0, 10.0, 20.0, 20.0]

        iou = compute_iou(box1, box2)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]

        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU with partially overlapping boxes."""
        # Box 1: 10x10 = 100 area
        # Box 2: 10x10 = 100 area
        # Intersection: 5x5 = 25 area
        # Union: 100 + 100 - 25 = 175 area
        # IoU: 25 / 175 = 0.142857...
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 5.0, 15.0, 15.0]

        iou = compute_iou(box1, box2)
        assert abs(iou - 0.142857142857) < 1e-10

    def test_edge_touching(self):
        """Test IoU when boxes only touch at edges."""
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [10.0, 0.0, 20.0, 10.0]  # Touches at x=10

        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_one_box_inside_another(self):
        """Test IoU when one box is completely inside another."""
        box1 = [0.0, 0.0, 20.0, 20.0]  # 20x20 = 400 area
        box2 = [5.0, 5.0, 15.0, 15.0]  # 10x10 = 100 area, fully inside

        iou = compute_iou(box1, box2)
        assert iou == 0.25  # 100 / 400

    def test_invalid_box_length(self):
        """Test error with wrong number of coordinates."""
        box1 = [0.0, 0.0, 10.0]  # Only 3 coordinates
        box2 = [0.0, 0.0, 10.0, 10.0]

        with pytest.raises(ValueError, match="Bounding boxes must have exactly 4 coordinates"):
            compute_iou(box1, box2)

        with pytest.raises(ValueError, match="Bounding boxes must have exactly 4 coordinates"):
            compute_iou(box2, box1)

    def test_zero_area_box(self):
        """Test IoU with zero-area box."""
        box1 = [0.0, 0.0, 0.0, 10.0]  # Zero width
        box2 = [0.0, 0.0, 10.0, 10.0]

        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_large_coordinates(self):
        """Test IoU with large coordinate values."""
        box1 = [1000.0, 1000.0, 2000.0, 2000.0]  # 1000x1000
        box2 = [1500.0, 1500.0, 2500.0, 2500.0]  # 1000x1000, 50% overlap

        iou = compute_iou(box1, box2)
        assert abs(iou - 0.142857142857) < 1e-10  # Same ratio as smaller test

    def test_inverted_box_normalized(self):
        """Test that inverted coordinates (x2<x1) are auto-normalized."""
        # [100,100,50,50] should be treated as [50,50,100,100]
        box_normal = [50.0, 50.0, 100.0, 100.0]
        box_inverted = [100.0, 100.0, 50.0, 50.0]

        assert compute_iou(box_normal, box_inverted) == 1.0

    def test_inverted_box_partial_overlap(self):
        """Inverted coords with partial overlap compute correctly."""
        # Normal: [0,0,10,10] vs [5,5,15,15] → IoU ≈ 0.1429
        iou_normal = compute_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])
        # Same with first box inverted
        iou_inverted = compute_iou([10.0, 10.0, 0.0, 0.0], [5.0, 5.0, 15.0, 15.0])
        assert abs(iou_normal - iou_inverted) < 1e-10

    def test_both_boxes_inverted(self):
        """Both boxes inverted still produces correct IoU."""
        iou = compute_iou([10.0, 10.0, 0.0, 0.0], [15.0, 15.0, 5.0, 5.0])
        expected = compute_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])
        assert abs(iou - expected) < 1e-10
